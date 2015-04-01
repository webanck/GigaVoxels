/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2013 INRIA <http://www.inria.fr/>
 *
 * Authors : GigaVoxels Team
 *
 * GigaVoxels is distributed under a dual-license scheme.
 * You can obtain a specific license from Inria at gigavoxels-licensing@inria.fr.
 * Otherwise the default license is the GPL version 3.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @version 1.0
 */

#ifndef _RLE_COMPRESSOR_KERNEL_H_
#define _RLE_COMPRESSOR_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// CUDA
#include <cuda_runtime.h>
#include <stdio.h>

// System
#include <cassert>

// Project
#include "macros.h"

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Copy input array into output array.
 * With pined memory, ~10% slower than cudaMemcopy
 *
 * @param pSize number of elements to process
 * @param pInput input array
 * @param pOutput output array
 ******************************************************************************/
__global__
void Kernel_PassThrough( const int pSize, const unsigned int* pInput, unsigned int* pOutput )
{
	for( int index = threadIdx.x + blockIdx.x * blockDim.x;
			index < pSize;
			index += gridDim.x * blockDim.x ) {
		pOutput[index] = pInput[index];
	}
}

/******************************************************************************
 * Copy input array into output array.
 * With pined memory, ~10% slower than cudaMemcopy
 *
 * @param pSize number of elements to process
 * @param pInput input array
 * @param pOutput output array
 ******************************************************************************/
__global__
void Kernel_PassThrough_Pascal( const int pSize, const unsigned int* pInput, unsigned int* pOutput )
{
    // Global index
    const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

    // Check bounds
    if ( index < pSize )
    {
        // One thread write one unique data
        pOutput[ index ] = pInput[ index ];
    }
}

/******************************************************************************
 * Uncompress data from input to output
 * - it uses shared memory to compute offsets where to write in output
 *
 * Input data meta information :
 * First
 *	- input[ 0 ] : total nb of elements
 * Then, couple of data (nb,value)
 *	- input[ N ] : number of reiteration of a value
 *	- input[ N + 1 ] : value
 *
 * Note : input[ 2 ] : it has a flag hidden in the least significant bit to tell whether or not data compression occured (1 : compression, 0 : no compression)
 *
 * Ex :
 * input => 12 24 24 150 16 16 16 8
 * turns to :
 * 11 1 13 2 24 1 150 3 16 1 8
 * with : 11 (<= nb total elements) 1 13 (<= hidden flag in 12) - (nb,value =>) 2 24 - 1 150 - 3 16 - 1 8
 *
 * @param pSize number of elements to process
 * @param pInput input array
 * @param pOutput output array
 ******************************************************************************/
__global__
void RLE_kernel( const int pSize, const unsigned char* pInput, unsigned char* pOutput )
{
	// Input data current index
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	// Equivalent 1D linear offset
	const int offset = x + y * blockDim.x * gridDim.x;

	// Check 3rd value to see if there is non compaction
	//
	// - then, just copy input into output
	if ( ! ( pInput[ 2 ] & ( 1 << 0 ) )	// hide 1 flag in the least significant bit
		&& offset < pSize )				// Check bounds
	{
		pOutput[ offset ] = pInput[ offset ];

		//printf("%i %i\n", pOutput[offset], pInput[offset] );

		// Exit
		return;
	}

	// Shared Memory
	__shared__ unsigned int sNbElem;
	__shared__ unsigned int* sTabAddElem;

	// Catch the number of elements in brick compressed with RLE
	// - it is done by only one thread
	if ( threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 )
	{
		sNbElem = static_cast< unsigned int >( pInput[ 0 ] );
		sTabAddElem = new unsigned int[ sNbElem ];

		// Fill array with offset values
		//
		// - each memory slot contains the sum all previous input data, i.e the number of reiteration of values
		sTabAddElem[ 0 ] = 0;
		for ( unsigned int i = 1; i < sNbElem; i++ )
		{
			sTabAddElem[ i ] = sTabAddElem[ i - 1 ] + static_cast< unsigned int >( pInput[ 2 * ( i - 1 ) + 1 ] );
		}
	}

	// Thread synchronization
	__syncthreads();

	// Check bounds
	if ( offset < sNbElem )
	{
		// Odd index : used to retrieve how many times a value is repeated
		const unsigned int localOffsetRLE = 2 * offset + 1;

		// Catch the value of the RLE compression
		const unsigned char nbValues = pInput[ localOffsetRLE ];

		// Retrieve user data previously generated on CPU
		const unsigned char data = pInput[ localOffsetRLE + 1 ];

		// Retrieve the offset where to write in output data
		//uint localOffset = atomicAdd( &addElem, static_cast< unsigned int >( nbValues ) ) + processID;
		const unsigned int localOffset = sTabAddElem[ offset ];

		// Iterate through all reiteration of a value
		for ( unsigned int i = 0; i < static_cast< unsigned int >( nbValues ); i++ )
		{
			// Current offset where to write each repeated value from input to output
			const unsigned int expOffset = localOffset + i;

			// Check bounds
			if ( expOffset < pSize )
			{
				// Write data
				pOutput[ expOffset ] = data;
			}
		}
	}
}

/******************************************************************************
 * Uncompress data from input to output
 * - it uses ATOMIC ADD to compute offsets where to write in output
 *
 * Input data meta information :
 * First
 *	- input[ 0 ] : total nb of elements
 * Then, couple of data (nb,value)
 *	- input[ i-th ] : number of reiteration of a value
 *	- input[ i-th + 1 ] : value
 *
 * Ex :
 * input => 12 24 24 150 16 16 16 8
 * turns to :
 * 11 1 12 2 24 1 150 3 16 1 8
 * with : 11 (<= nb total elements) - (nb,value =>) 1 12 - 2 24 - 1 150 - 3 16 - 1 8
 *
 * @param pSize number of elements to process
 * @param pInput input array
 * @param pOutput output array
 ******************************************************************************/
__global__
void RLE_atomic_kernel( int pSize, const unsigned char* pInput, unsigned char* pOutput )
{
	// Shared Memory
	__shared__ unsigned int sNbElem;
	__shared__ unsigned int sId;
//	__shared__ unsigned int sNb;

	// Catch the number of element in brick compress with RLE
	// - it is done by only one thread
	if ( threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 )
	{
		sId = 0;
//		sNb = 0;

		// Retrieve total number of elements
		sNbElem = static_cast< unsigned int >( pInput[ 0 ] );
	}

	// Thread synchronization
	__syncthreads();

	// Here, the idea is :
	// - given an index of a value
	// - retrieve the associated offset value
	// ====> for each thread, given an index, compute the sum all previous input data, i.e the number of reiteration of all values, before its index
	const unsigned int index = atomicAdd( &sId, 1 );
	unsigned int indexInCache = 0;
	unsigned int k = index;
	while ( k != 0 )
	{
		// Odd index : used to retrieve how many times a value is repeated
		indexInCache += pInput[ 2 * k + 1 ];

		// Update loop
		k--;
	}

	// Check bounds
	if ( index < sNbElem )
	{
		// Odd index : used to retrieve how many times a value is repeated
		const unsigned int it = pInput[ 2 * index + 1 ];

		// Even index : used to retrieve a value
		const unsigned int value = pInput[ 2 * index + 2 ];

		// Iterate through all reiteration of the value
		for ( unsigned int i = 0; i < it; i++ )
		{
			// Check bounds
			if ( indexInCache + i < pSize )
			{
				// Write data
				pOutput[ indexInCache + i ] = value;
			}
		}
	}
}

/******************************************************************************
 * Uncompress data from input to output
 * - it uses ATOMIC ADD to compute offsets where to write in output
 *
 * Input data meta information :
 * First
 *	- pOffset[ 0 ] : total nb of elements
 * Then, offsets to tell where to write in output
 *	- pOffset[ i-th ] : offsets (sum of all previous offsets, i.e the number of reiteration of all values, before its index)
 *
 * Ex :
 * input => 12 24 24 150 16 16 16 8
 * turns to :
 * 11 1 12 2 24 1 150 3 16 1 8
 * with : 11 (<= nb total elements) - (nb,value =>) 1 12 - 2 24 - 1 150 - 3 16 - 1 8
 *
 * @param pSize number of elements to process
 * @param pInput input array
 * @param pOffset offset array associated to input
 * @param pOutput output array
 ******************************************************************************/
 __global__
void RLE_atomic_kernel( int pSize, const unsigned char* pInput, unsigned int* pOffset, unsigned char* pOutput )
{
	// Shared Memory
	__shared__ unsigned int sNbElem;
	__shared__ unsigned int sId;
//	__shared__ unsigned int sNb;

	// Catch the number of element in brick compress with RLE
	// - it is done by only one thread
	if ( threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 )
	{
		sId = 0;
//		sNb = 0;

		// Retrieve total number of elements
		sNbElem = static_cast< unsigned int >( pOffset[ 0 ] );
	}

	// Thread synchronization
	__syncthreads();

	const unsigned int index = atomicAdd( &sId, 1 );

	const unsigned int indexInCache = static_cast< unsigned int >( pOffset[ index + 1 ] );

	// Check bounds
	if ( index < sNbElem )
	{
		// Retrieve how many times a value is repeated
		// - it is difference of two consecutive offset values
		const unsigned int it = pOffset[ index + 2 ] - indexInCache;

		// Retrieve a value
		const unsigned int value = pInput[ index ];

		// Iterate through all reiteration of the value
		for ( int i = 0; i < it; i++ )
		{
			// Check bounds
			if ( indexInCache + i < pSize )
			{
				// Write data
				pOutput[ indexInCache + i ] = value;
			}
		}
	}
}

 /******************************************************************************
  * ...
  ******************************************************************************/
__global__
void RLE_double_kernel(  const unsigned char* __restrict__ pInput, unsigned char* __restrict__ pOutput )
{
	//printf("Done!\n");
	// Input data current index


	const int offset = threadIdx.x+threadIdx.y * blockDim.x;
	int offset2;
	const int block_size = blockDim.x * blockDim.y;
	//__shared__ bool uncompressed;

	//__shared__  unsigned int nb_elements;
	unsigned int nb_elements;
	__shared__  unsigned char shared_input [BRICK_SIZE];
	//printf("done!\n");
	//if (threadIdx.x+threadIdx.y == 0)
	//{
	//
	//	nb_elements = pInput[blockIdx.x];
	//	//printf("%d\n",block_size);
	//	//shared_input = new unsigned char[nb_elements*2];
	//}

	nb_elements = pInput[blockIdx.x];
	//__syncthreads();


	offset2 = offset;

	if (nb_elements==0) {
		while (offset2  < BRICK_SIZE)
		{
			pOutput[offset2+(blockIdx.x) *BRICK_SIZE]=pInput[offset2+(blockIdx.x+1) *BRICK_SIZE ];
			offset2 +=block_size;

		}

		//uncompressed= true;
	} else {
		while (offset2  < nb_elements*2)
		{

			shared_input[offset2]=pInput[offset2+(blockIdx.x+1) *BRICK_SIZE ];

			offset2 +=block_size;
		}
	}

	__syncthreads();


	offset2 = offset;

	int indice = 0;
	int indiceEq = 0.f;
	while (offset2  < BRICK_SIZE)
	{

		while (indiceEq<=offset2 &&indice<nb_elements*2) {
			indiceEq+=shared_input[indice];
			indice+=2;
		}

		if (nb_elements != 0)
			pOutput[(blockIdx.x) *BRICK_SIZE + offset2] =shared_input[indice-1];

		offset2 +=block_size;
	}


	/*
	__syncthreads();

	if (threadIdx.x+threadIdx.y == 0)
	{
		delete[] shared_input;
	}
	*/
	//printf("Done!\n");
}

/******************************************************************************
 * Simple kernel doing the decompression without using parallelism.
 ******************************************************************************/
__global__
void basicRleKernel( const unsigned char* __restrict__ map_idata_sizes,
					 const unsigned char* __restrict__ map_idata_values,
	   				 const unsigned int* __restrict__ map_idata_header ,
					 unsigned char* __restrict__ pOutput )
{
	unsigned int brickId = threadIdx.x + blockIdx.x * blockDim.x;

	// Loop through the bricks
	while( brickId < N_BRICKS ) {
		const unsigned int brickStart = brickId > 0 ? map_idata_header[brickId - 1] : 0;
		if( brickStart == UINT_MAX ) {
			// Data are not compressed => we just need to copy
			assert( false ); // TODO
		} else {
			// Data are compressed
			// Index in the compressed array
			unsigned int rleIndex = brickStart;

			// Index in the brick
			unsigned int brickIndex = 0;

			// Copy + uncompress data
			while( brickIndex < BRICK_SIZE ) {
				const unsigned char plateauSize = map_idata_sizes[rleIndex];
				const unsigned char plateauValue = map_idata_values[rleIndex];
				for( unsigned int i = brickIndex; i < ( plateauSize + brickIndex ); ++i ) {
					pOutput[brickId * BRICK_SIZE + i] = plateauValue;
				}
				brickIndex += plateauSize;
				++rleIndex;
			}
		}

		brickId += blockDim.x * gridDim.x;
	}
}

/******************************************************************************
 * KERNEL : simpleRleKernel
 *
 * Note : pBricksEnds, pPlateausValues and pPlateausStarts are memory-mapped arrays
 *
 * @param pBricksEnds (output) RLE-encoded array => prefix-sum array of pPlateausValues. For each brick in pInput, it stores the number of compressed elements (i.e the size of the compressed brick, hence the "end" of the compressed brick) [max size is N_BRICKS, i.e total nb bricks pNbBricks]
 * @param pPlateausValues (output) RLE-encoded array => it store all values (compressed array without repetition) [max size is STR_SIZE, i.e as input data]
 * @param pPlateausStarts (output) RLE-encoded array => for each value in pPlateausValues array, it stores associated number of repetitions [max size is STR_SIZE, i.e as input data]
 * @param pOutput (output) decompressed data array
 ******************************************************************************/
__global__
void simpleRleKernel( 
        const unsigned int* __restrict__ pBrickEnds,
        const unsigned int* __restrict__ pPlateausValues,
        const unsigned char* __restrict__ pPlateausStarts,
        unsigned int* __restrict__ pOutput )
{
    // Thread ID in a warp
	const unsigned int threadId = threadIdx.x % WARP_SIZE;
    // Warp ID in a block
	const unsigned int warpId = threadIdx.x / WARP_SIZE;

	// Loop through the bricks
    for ( unsigned int brickId = blockIdx.x * N_WARPS_PER_BLOCS + warpId;
		   	brickId < N_BRICKS;
            brickId += gridDim.x * N_WARPS_PER_BLOCS )
    {
        // First, retrieve "start" and "stop" index of current brick data
        const unsigned int brickStart = brickId > 0 ? pBrickEnds[ brickId - 1 ] : 0;
        const unsigned int brickStop = pBrickEnds[ brickId ];

        // Check if brick is compressed or not
        if ( brickStop - brickStart == BRICK_SIZE )
        {
			// Data are not compressed => we just need to copy them
			// They are stored in the plateausValues array.
            for ( unsigned int index = threadId + brickStart;
					index < brickStop;
                    index += WARP_SIZE )
            {
                // Copy value
                pOutput[ BRICK_SIZE * brickId + index - brickStart ] = pPlateausValues[ index ];
			}
        }
        else
        {
            // The brick is compressed, we need to uncompress data

            // Shared Memory
            // - used as buffer for input arrays
            __shared__ unsigned char smStartsArr[ MAX_COMPRESSED_BRICK_SIZE * N_WARPS_PER_BLOCS ];
            __shared__ unsigned int smValuesArr[ MAX_COMPRESSED_BRICK_SIZE * N_WARPS_PER_BLOCS ];

			// Each warp works on one brick, so we don't need synchronization
            // - each threads as job to do
            unsigned char* smStarts = &smStartsArr[ MAX_COMPRESSED_BRICK_SIZE * warpId ];
            unsigned int* smValues = &smValuesArr[ MAX_COMPRESSED_BRICK_SIZE * warpId ];

            // Load the relevant part of the array in shared
            for ( unsigned int rleIndex = threadId + brickStart;
				   	rleIndex < brickStop;
                    rleIndex += WARP_SIZE )
            {
                const unsigned char start = pPlateausStarts[ rleIndex ];
                const unsigned int value = pPlateausValues[ rleIndex ];
                smStarts[ rleIndex - brickStart ] = start;
                smValues[ rleIndex - brickStart ] = value;
			}

            // Uncompress data
			unsigned int rleIndex =  1;
			unsigned int brickIndex = threadId;
            while ( brickIndex < BRICK_SIZE )
            {
                if ( rleIndex >= brickStop - brickStart
                        || smStarts[ rleIndex ] > brickIndex )
                {
                    // Copy value
                    pOutput[ brickIndex + BRICK_SIZE * brickId ] = smValues[ rleIndex - 1 ];

                    brickIndex += WARP_SIZE;
                }
                else
                {
					++rleIndex;
				}
			}
		}
	}
}

/******************************************************************************
 * KERNEL : dichoRleKernel
 *
 * Note : pBricksEnds, pPlateausValues and pPlateausStarts are memory-mapped arrays
 *
 * @param pBricksEnds (output) RLE-encoded array => prefix-sum array of pPlateausValues. For each brick in pInput, it stores the number of compressed elements (i.e the size of the compressed brick, hence the "end" of the compressed brick) [max size is N_BRICKS, i.e total nb bricks pNbBricks]
 * @param pPlateausValues (output) RLE-encoded array => it store all values (compressed array without repetition) [max size is STR_SIZE, i.e as input data]
 * @param pPlateausStarts (output) RLE-encoded array => for each value in pPlateausValues array, it stores associated number of repetitions [max size is STR_SIZE, i.e as input data]
 * @param pOutput (output) decompressed data array
 ******************************************************************************/
__global__
void dichoRleKernel( 
		const unsigned int* __restrict__ brickEnds,
		const unsigned int* __restrict__ plateausValues,
		const unsigned char* __restrict__ plateausStarts,
		unsigned int* __restrict__ output )
{
    // Thread ID in a warp
	const unsigned int threadId = threadIdx.x % WARP_SIZE;
    // Warp ID in a block
	const unsigned int warpId = threadIdx.x / WARP_SIZE;

	// Loop through the bricks
    for ( unsigned int brickId = blockIdx.x * N_WARPS_PER_BLOCS + warpId;
		   	brickId < N_BRICKS;
            brickId += gridDim.x * N_WARPS_PER_BLOCS )
    {
        // First, retrieve "start" and "stop" index of current brick data
        const unsigned int brickStart = brickId > 0 ? brickEnds[ brickId - 1 ] : 0;
        const unsigned int brickStop = brickEnds[ brickId ];

        // Check if brick is compressed or not
        if ( brickStop - brickStart == BRICK_SIZE )
        {
			// Data are not compressed => we just need to copy them
			// They are stored in the plateausValues array.
            for ( unsigned int index = threadId + brickStart;
					index < brickStop;
                    index += WARP_SIZE )
            {
                 // Copy value
                output[ BRICK_SIZE * brickId + index - brickStart ] = plateausValues[index];
			}
        }
        else
        {
           // The brick is compressed, we need to uncompress data

           // Shared memory used as buffer for input arrays
            __shared__ unsigned char startsArr[ MAX_COMPRESSED_BRICK_SIZE * N_WARPS_PER_BLOCS ];
            __shared__ unsigned int valuesArr[ MAX_COMPRESSED_BRICK_SIZE * N_WARPS_PER_BLOCS ];

			// Each warp works on one brick, so we don't need synchronization
			// (and each threads as job to do).
            unsigned char* starts = &startsArr[ MAX_COMPRESSED_BRICK_SIZE * warpId ];
            unsigned int* values = &valuesArr[ MAX_COMPRESSED_BRICK_SIZE * warpId ];

			// Load the relevant part of the array in shared.
            for ( unsigned int rleIndex = threadId + brickStart;
				   	rleIndex < brickStop;
                    rleIndex += WARP_SIZE )
            {
                const unsigned char start = plateausStarts[ rleIndex ];
                const unsigned int value = plateausValues[ rleIndex ];
                starts[ rleIndex - brickStart ] = start;
                values[ rleIndex - brickStart ] = value;
			}

			// Uncompress data
            for ( unsigned int brickIndex = threadId;
					brickIndex < BRICK_SIZE;
                    brickIndex += WARP_SIZE )
            {
				// Search the value to copy
                // - dichotomic search
				unsigned int start = 0;
				unsigned int stop = brickStop - brickStart;
				unsigned int middle = ( start + stop ) / 2;
                while ( start < stop )
                {
                    if ( starts[ middle ] <= brickIndex )
                    {
						start = middle + 1;
                    }
                    else
                    {
						stop = middle;
					}
					middle = ( start + stop ) / 2;
				}
				--middle;

                // Copy value
                output[ brickIndex + BRICK_SIZE * brickId ] = values[ middle ];
			}
		}
	}
}

/******************************************************************************
 * KERNEL : combinedRleKernel
 *
 * Note : pBricksEnds, pPlateausValues and pPlateausStarts are memory-mapped arrays
 *
 * @param pBricksEnds (output) RLE-encoded array => prefix-sum array of pPlateausValues. For each brick in pInput, it stores the number of compressed elements (i.e the size of the compressed brick, hence the "end" of the compressed brick) [max size is N_BRICKS, i.e total nb bricks pNbBricks]
 * @param pPlateausValues (output) RLE-encoded array => it store all values (compressed array without repetition) [max size is STR_SIZE, i.e as input data]
 * @param pPlateausStarts (output) RLE-encoded array => for each value in pPlateausValues array, it stores associated number of repetitions [max size is STR_SIZE, i.e as input data]
 * @param pOutput (output) decompressed data array
 ******************************************************************************/
__global__
void combinedRleKernel(
        const unsigned int* __restrict__ brickEnds,
        const unsigned int* __restrict__ plateausValues,
        const unsigned char* __restrict__ plateausStarts,
        unsigned int* __restrict__ output )
{
    // Thread ID in a warp
    const unsigned int threadId = threadIdx.x % WARP_SIZE;
    // Warp ID in a block
    const unsigned int warpId = threadIdx.x / WARP_SIZE;

    // Loop through the bricks
    for ( unsigned int brickId = blockIdx.x * N_WARPS_PER_BLOCS + warpId;
            brickId < N_BRICKS;
            brickId += gridDim.x * N_WARPS_PER_BLOCS )
    {
        // First, retrieve "start" and "stop" index of current brick data
        const unsigned int brickStart = brickId > 0 ? brickEnds[ brickId - 1 ] : 0;
        const unsigned int brickStop = brickEnds[ brickId ];

        // Check if brick is compressed or not
        if ( brickStop - brickStart == BRICK_SIZE )
        {
            // Data are not compressed => we just need to copy them
            // They are stored in the plateausValues array.
            for ( unsigned int index = threadId + brickStart;
                    index < brickStop;
                    index += WARP_SIZE )
            {
                // Copy value
                output[ BRICK_SIZE * brickId + index - brickStart ] = plateausValues[ index ];
            }
        }
        else
        {
            // Shared memory used as buffer for input arrays.
            __shared__ unsigned char startsArr[ MAX_COMPRESSED_BRICK_SIZE * N_WARPS_PER_BLOCS ];
            __shared__ unsigned int valuesArr[ MAX_COMPRESSED_BRICK_SIZE * N_WARPS_PER_BLOCS ];

            // Each warp works on one brick, so we don't need synchronization
            // (and each threads as job to do).
            unsigned char* starts = &startsArr[ MAX_COMPRESSED_BRICK_SIZE * warpId ];
            unsigned int* values = &valuesArr[ MAX_COMPRESSED_BRICK_SIZE * warpId ];

            // Load the relevant part of the array in shared.
            for ( unsigned int rleIndex = threadId + brickStart;
                    rleIndex < brickStop;
                    rleIndex += WARP_SIZE )
            {
                const unsigned char start = plateausStarts[ rleIndex ];
                const unsigned int value = plateausValues[ rleIndex ];
                starts[ rleIndex - brickStart ] = start;
                values[ rleIndex - brickStart ] = value;
            }

            // Uncompress data
            if ( brickStop - brickStart < BRICK_SIZE / 15 ) // 15 : hard coded value ?
            {
                unsigned int rleIndex =  1;
                unsigned int brickIndex = threadId;
                while( brickIndex < BRICK_SIZE )
                {
                    if ( rleIndex >= brickStop - brickStart
                            || starts[rleIndex] > brickIndex )
                    {
                        output[ brickIndex + BRICK_SIZE * brickId ] = values[ rleIndex - 1 ];
                        brickIndex += WARP_SIZE;
                    }
                    else
                    {
                        ++rleIndex;
                    }
                }
            }
            else
            {
                for ( unsigned int brickIndex = threadId;
                        brickIndex < BRICK_SIZE;
                        brickIndex += WARP_SIZE )
                {
                    // Search the value to copy
                    // Dichotomic search
                    unsigned int start = 0;
                    unsigned int stop = brickStop - brickStart;
                    unsigned int middle = ( start + stop ) / 2;
                    while ( start < stop )
                    {
                        if ( starts[middle] <= brickIndex )
                        {
                            start = middle + 1;
                        }
                        else
                        {
                            stop = middle;
                        }
                        middle = ( start + stop ) / 2;
                    }
                    --middle;

                    // Copy value
                    output[ brickIndex + BRICK_SIZE * brickId ] = values[ middle ];
                }
            }
        }
    }
}

/**
 * ...
 */
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

/******************************************************************************
 * ...
 ******************************************************************************/
__device__ void prefix_sum_exclusive(float *g_odata, float *g_idata, int n,int max)
{
	__shared__ float temp[MAX_COMPRESSED_BRICK_SIZE];// allocated on invocation

	int thid = threadIdx.x+threadIdx.y * blockDim.x;
	if (thid<n)
	{
		int offset = 1;

		int ai = thid;
		int bi = thid + (n/2);
		int bankOffsetA = CONFLICT_FREE_OFFSET(ai)  ;
		int bankOffsetB = CONFLICT_FREE_OFFSET(bi)  ;
		temp[ai + bankOffsetA] = g_idata[ai]  ;
		temp[bi + bankOffsetB] = g_idata[bi] ;

		for (int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree
		{
			__syncthreads();

			if (thid < d)
			{

				int ai = offset*(2*thid+1)-1;
				int bi = offset*(2*thid+2)-1;
				ai += CONFLICT_FREE_OFFSET(ai)  ;
				bi += CONFLICT_FREE_OFFSET(bi)  ;

				temp[bi] += temp[ai];
			}
			offset *= 2;
		}

		if (thid==0) { temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;}

		for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
		{
			offset >>= 1;
			__syncthreads();

			if (thid < d)
			{
				int ai = offset*(2*thid+1)-1;
				int bi = offset*(2*thid+2)-1;
				ai += CONFLICT_FREE_OFFSET(ai)  ;
				bi += CONFLICT_FREE_OFFSET(bi)  ;

				float t = temp[ai];
				temp[ai] = temp[bi];
				temp[bi] += t;
			}
		}

		__syncthreads();

		g_odata[ai] = temp[ai + bankOffsetA];
		g_odata[bi] = temp[bi + bankOffsetB];
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
__device__ void prefix_sum_exclusive2(unsigned int *g_idata, int n)
{

	int thid = threadIdx.x+threadIdx.y * blockDim.x;
	if (thid<n)
	{
		int offset = 1;

		for (int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree
		{


			if (thid < d)
			{

				int ai = offset*(2*thid+1)-1;
				int bi = offset*(2*thid+2)-1;
				ai += CONFLICT_FREE_OFFSET(ai)  ;
				bi += CONFLICT_FREE_OFFSET(bi)  ;

				g_idata[bi] += g_idata[ai];
			}
			offset *= 2;
		}

		if (thid==0) { g_idata[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;}

		for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
		{
			offset >>= 1;
			__syncthreads();

			if (thid < d)
			{
				int ai = offset*(2*thid+1)-1;
				int bi = offset*(2*thid+2)-1;
				ai += CONFLICT_FREE_OFFSET(ai)  ;
				bi += CONFLICT_FREE_OFFSET(bi)  ;

				float t = g_idata[ai];
				g_idata[ai] = g_idata[bi];
				g_idata[bi] += t;
			}
		}


	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
__device__ void prescan(unsigned int *g_odata, unsigned int *g_idata, int n)
{
 __shared__ unsigned int temp[128];// allocated on invocation

 int thid = threadIdx.x;
 int offset = 1;

 temp[2*thid] = g_idata[2*thid]; // load input into shared memory
 temp[2*thid+1] = g_idata[2*thid+1];

 for (int d = n>>1; d > 0; d >>= 1) // build sum in place up the tree
 {
 __syncthreads();

 if (thid < d)
 {
int ai = offset*(2*thid+1)-1;
 int bi = offset*(2*thid+2)-1;

 temp[bi] += temp[ai];
 }
 offset *= 2;
 }

 if (thid == 0) { temp[n - 1] = 0; } // clear the last element

 for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
 {
 offset >>= 1;
 __syncthreads();

 if (thid < d)
 {
int ai = offset*(2*thid+1)-1;
 int bi = offset*(2*thid+2)-1;

 unsigned int t = temp[ai];
 temp[ai] = temp[bi];
 temp[bi] += t;
 }
 }

 __syncthreads();

 g_odata[2*thid] = temp[2*thid]; // write results to device memory
 g_odata[2*thid+1] = temp[2*thid+1];
}

/******************************************************************************
 * Uncompress data from input to output
 * - it uses ATOMIC ADD to compute offsets where to write in output
 *
 * Input data meta information :
 * First
 *	- pOffset[ 0 ] : total nb of elements
 * Then, offsets to tell where to write in output
 *	- pOffset[ i-th ] : offsets (sum of all previous offsets, i.e the number of reiteration of all values, before its index)
 *
 * Ex :
 * input => 12 24 24 150 16 16 16 8
 * turns to :
 * 11 1 12 2 24 1 150 3 16 1 8
 * with : 11 (<= nb total elements) - (nb,value =>) 1 12 - 2 24 - 1 150 - 3 16 - 1 8
 *
 * @param pSize number of elements to process
 * @param pInput input array
 * @param pOffset offset array associated to input
 * @param pOutput output array
 ******************************************************************************/
 __global__
void RLE_reduction_kernel(const unsigned char* pInput, unsigned char* pOutput )
{
	__shared__ unsigned int shared_offsets[MAX_COMPRESSED_BRICK_SIZE];
	__shared__ unsigned char shared_values[MAX_COMPRESSED_BRICK_SIZE];

	//__shared__ float shared_sum_offsets[MAX_COMPRESSED_BRICK_SIZE];
	__shared__ unsigned int closest_power_of_2;
	const int offset = threadIdx.x+threadIdx.y * blockDim.x;
	int offset2;
	const int block_size = blockDim.x * blockDim.y;

	unsigned int nb_elements;
	nb_elements = pInput[blockIdx.x];

	if (threadIdx.x+threadIdx.y == 0 )
	{


		if (nb_elements>0)
		{
			closest_power_of_2=1;
			while (closest_power_of_2<nb_elements)
				closest_power_of_2<<=1;
			/*
			closest_power_of_2 = nb_elements ;// Get input
			closest_power_of_2--;
			closest_power_of_2 = (closest_power_of_2 >> 1) | closest_power_of_2;
			closest_power_of_2 = (closest_power_of_2 >> 2) | closest_power_of_2;
			closest_power_of_2 = (closest_power_of_2 >> 4) | closest_power_of_2;
			closest_power_of_2 = (closest_power_of_2 >> 8) | closest_power_of_2;
			closest_power_of_2 = (closest_power_of_2 >> 16) | closest_power_of_2;
			closest_power_of_2++; // Val is now the next highest power of 2.
			*/
			//shared_offsets[nb_elements]=static_cast<float>(BRICK_SIZE);
			//closest_power_of_2>>=1;
		}
	}

	__syncthreads();


	offset2 = offset;

	if (nb_elements==0) {
		while (offset2  < BRICK_SIZE)
		{
			pOutput[offset2+(blockIdx.x) *BRICK_SIZE]=pInput[offset2+(blockIdx.x+1) *BRICK_SIZE ];
			offset2 +=block_size;

		}

	} else {
		while (offset2  < nb_elements)
		{

			shared_offsets[offset2]=static_cast<unsigned int>(pInput[2*offset2+(blockIdx.x+1) *BRICK_SIZE ]);
			shared_values[offset2]=(pInput[2*offset2+ 1 +(blockIdx.x+1) *BRICK_SIZE ]);
			offset2 +=block_size;
		}
	}
	__syncthreads();

	/*if (threadIdx.x+threadIdx.y == 0 && blockIdx.x ==0)
	  {
		printf("ind : ");
		for (int k = 0; k < nb_elements;k++) {
			printf("%d;",shared_offsets[k]);
			}
		printf("\n\n");

	}
	__syncthreads();*/


	prefix_sum_exclusive2(shared_offsets,closest_power_of_2);
	//prescan(shared_offsets, shared_offsets,nb_elements) ;

	__syncthreads();

	if (threadIdx.x+threadIdx.y == 0 )
	{
		shared_offsets[nb_elements]=(BRICK_SIZE);

		/*if (blockIdx.x==256)
		{
			int uu=0;
			for (int k =0; k<nb_elements+1;k++)
			{
				if (uu > shared_offsets[k])
				{
					for (int kt =0; kt<nb_elements+1;kt++)
					{
						printf("%d ",shared_offsets[kt]);
					} printf("\n");
					printf("error\n");
				}
				uu=shared_offsets[k];
			}
		}*/

		/*if (blockIdx.x==0)
		{
			printf("sum : ");
			for (int k = 0; k <= nb_elements;k++) {
				printf("%d,",shared_offsets[k]);
			}
			printf("\n\n\n\n");
		}*/

	}
	__syncthreads();

	if (nb_elements != 0)
	{
		offset2 = offset;
		//unsigned char val;
		nb_elements++;

		while (offset2  < BRICK_SIZE)
		{

			//val = shared_values[offset2];
			//for (int k = shared_offsets[offset2]; k < shared_offsets[offset2+1]  ; k++)
			//{
			//
			//	pOutput[ k + (blockIdx.x) * BRICK_SIZE ] = val;
			//}


			int middle_index = nb_elements/2;
			int left_chunk_size = middle_index;
			int right_chunk_size = nb_elements - left_chunk_size;
			int old_left_chunk_size;
			bool out = false;
			int index_to_write=0;

			while(out != true) {
				if (shared_offsets[middle_index]==offset2)
				{
					index_to_write = middle_index;
					out =true;
				} else if ( shared_offsets[middle_index]>offset2){
					if (left_chunk_size <= 1)
					{
						index_to_write = middle_index-1;
						out = true;
					}
					old_left_chunk_size = left_chunk_size;
					left_chunk_size = left_chunk_size/2;
					right_chunk_size = old_left_chunk_size - left_chunk_size;
					middle_index -= right_chunk_size;
				} else {
					if (right_chunk_size <= 1)
					{
						index_to_write = middle_index;
						out = true;
					}
					left_chunk_size = right_chunk_size/2 ;
					middle_index += left_chunk_size ;
					right_chunk_size = right_chunk_size - left_chunk_size;

				}

			}
			/*if (offset2<shared_offsets[index_to_write] ||offset2>shared_offsets[index_to_write+1])
			{
				printf ("%d found between %d and %d index :%d nb_elems : %d\n",offset2,shared_offsets[index_to_write],shared_offsets[index_to_write+1],index_to_write,nb_elements);


			}*/
				/*
			int left = 0;
			int right = nb_elements;
			int middle = (left + right) >> 1;

			while( left < right ) {
				unsigned int temp = shared_offsets[middle];
				if( temp == offset2 ) {
					++middle;
					break;
				} else if ( temp < offset2 ) {
					left = middle +1;

				} else {
					right = middle ;
				}
				middle = (right + left) >> 1;
			}
			int index_to_write = middle;*/

			//__syncthreads();
			pOutput[offset2 + (blockIdx.x) * BRICK_SIZE] = shared_values[  index_to_write ];

			//pOutput[offset2 + (blockIdx.x) * BRICK_SIZE] = shared_values[  0 ];


			offset2 +=block_size;
		}


	}

}

//
// __global__
//void RLE_reduction_kernel2(const unsigned char* pInput, unsigned char* pOutput )
//{
//	__shared__ unsigned int shared_offsets[MAX_COMPRESSED_BRICK_SIZE];
//	__shared__ unsigned char shared_values[MAX_COMPRESSED_BRICK_SIZE];
//
//	//__shared__ float shared_sum_offsets[MAX_COMPRESSED_BRICK_SIZE];
//	__shared__ unsigned int closest_power_of_2;
//	const int offset = threadIdx.x+threadIdx.y * blockDim.x;
//	int offset2;
//	const int block_size = blockDim.x * blockDim.y;
//
//	__shared__  unsigned int nb_elements;
//
//	if (threadIdx.x+threadIdx.y == 0)
//	{
//
//		nb_elements = pInput[blockIdx.x];
//		if (nb_elements>0)
//		{
//			closest_power_of_2=1;
//			while (closest_power_of_2<nb_elements)
//				closest_power_of_2<<=1;
//			/*
//			closest_power_of_2 = nb_elements ;// Get input
//			closest_power_of_2--;
//			closest_power_of_2 = (closest_power_of_2 >> 1) | closest_power_of_2;
//			closest_power_of_2 = (closest_power_of_2 >> 2) | closest_power_of_2;
//			closest_power_of_2 = (closest_power_of_2 >> 4) | closest_power_of_2;
//			closest_power_of_2 = (closest_power_of_2 >> 8) | closest_power_of_2;
//			closest_power_of_2 = (closest_power_of_2 >> 16) | closest_power_of_2;
//			closest_power_of_2++; // Val is now the next highest power of 2.
//			*/
//			//shared_offsets[nb_elements]=static_cast<float>(BRICK_SIZE);
//		}
//	}
//
//	__syncthreads();
//
//
//	offset2 = offset;
//
//	if (nb_elements==0) {
//		while (offset2  < BRICK_SIZE)
//		{
//			pOutput[offset2+(blockIdx.x) *BRICK_SIZE]=pInput[offset2+(blockIdx.x+1) *BRICK_SIZE ];
//			offset2 +=block_size;
//
//		}
//
//	} else {
//		while (offset2  < nb_elements)
//		{
//
//			shared_offsets[offset2]=(pInput[2*offset2+(blockIdx.x+1) *BRICK_SIZE ]);
//			shared_values[offset2]=(pInput[2*offset2+ 1 +(blockIdx.x+1) *BRICK_SIZE ]);
//			offset2 +=block_size;
//		}
//	}
//	__syncthreads();
//
//	if (threadIdx.x+threadIdx.y == 0)
//	{
//		printf("ind : ");
//		for (int k = 0; k < nb_elements;k++) {
//			printf("%f,,;;,;;;,",shared_offsets[k]);
//		}
//		printf("\n\n");
//
//	}
//
//
//	prefix_sum_exclusive2(shared_offsets,closest_power_of_2);
//	__syncthreads();
//
//	if (threadIdx.x+threadIdx.y == 0)
//	{
//		shared_offsets[nb_elements]=(BRICK_SIZE);
//
//		printf("sum : \n");
//		for (int k = 0; k <= nb_elements;k++) {
//			printf("%ds;",shared_offsets[k]);
//		}
//		printf("\n\n\n\n");
//
//	}
//	__syncthreads();
//
//	if (nb_elements != 0)
//	{
//		offset2 = offset;
//		unsigned char val;
//
//		while (offset2  < nb_elements)
//		{
//
//			//val = shared_values[offset2];
//			//for (int k = shared_offsets[offset2]; k < shared_offsets[offset2+1]  ; k++)
//			//{
//			//
//			//	pOutput[ k + (blockIdx.x) * BRICK_SIZE ] = val;
//			//}
//
//
//
//			offset2 +=block_size;
//		}
//
//
//	}
//
//}

/*
__global__ void prescan(float *g_odata, float *g_idata, int n)
{



}*/


/*
__global__ void prescan(float *g_odata, float *g_idata, int n)
{

__shared__ float temp[100];
// allocated on invocation

int thid = threadIdx.x;

int offset = 1;

temp[2*thid] = g_idata[2*thid];
// load input into shared memory

temp[2*thid+1] = g_idata[2*thid+1];


for (int d = n>>1; d > 0; d >>= 1)
// build sum in place up the tree
 {

__syncthreads();


if (thid < d)
 {

int ai = offset*(2*thid+1)-1;

int bi = offset*(2*thid+2)-1;


temp[bi] += temp[ai];
 }

offset *= 2;
 }


if (thid == 0) { temp[n - 1] = 0; }
// clear the last element


for (int d = 1; d < n; d *= 2)
// traverse down tree & build scan
{

offset >>= 1;

__syncthreads();


if (thid < d)
 {

int ai = offset*(2*thid+1)-1;

int bi = offset*(2*thid+2)-1;


float t = temp[ai];

temp[ai] = temp[bi];

temp[bi] += t;
 }
 }


__syncthreads();


g_odata[2*thid] = temp[2*thid];
// write results to device memory

g_odata[2*thid+1] = temp[2*thid+1];
}

*/



// __device__ int locked = 0;
// __device__
// bool try_lock ()
// {
// 	int prev = atomicExch(&locked, 1);
// 	if(prev == 0)
// 		return true;
// 	return false;
// }
 //
 //__device__
 //unsigned int atomicAddRLE(unsigned int* nbElem, unsigned int* id, /*offset*/unsigned int val, /*where to read*/unsigned int &idRet)
 //{
	// // store old value
 //	unsigned int* address_as_ull = (unsigned int*)nbElem;
 //	unsigned int old = *address_as_ull, assumed;

 //	while(try_lock() == false)
 //		;
 //
	//do
	//{
 //		assumed = old;
 //		old = atomicCAS( /*valeur à tester*/address_as_ull, /*comp*/assumed, /*new*/(val + assumed) );
 //
	//	idRet = (*id);
 //		(*id)++;
 //	}
 //   while (assumed != old);
 //
	//unlock();
 //
	//return old;
 //}

 //__global__
 //void RLE_atomic_kernel(int size, const unsigned char* input, unsigned char *output)
 //{

 //	__shared__ unsigned int nbElem;
 //	__shared__ unsigned int id;
 //	__shared__ unsigned int nb;
 //	// Catch the number of element in brick compress with RLE

 //	if ( threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 ) {
 //		id = 0;
 //		nb = 0;
 //		nbElem = (unsigned int)input[0];
 //	}

 //	__syncthreads();

 //	unsigned int index;
 //	unsigned int indexInCache = atomicAddRLE(&nb, &id, /*offset*/(unsigned int)input[2 * id + 1], /*index where to read*/index);

 //	if (index < nbElem) {
 //		unsigned int it = input[ 2 * index + 1 ];
 //		unsigned int value = input[ 2 * index + 2 ];

 //		for (int i = 0; i < it; i++) {
 //			if ( indexInCache < size )
 //				output[indexInCache + i] = value;
 //		}
 //	}
 //}


 /******************************************************************************
  * ...
  ******************************************************************************/
__global__ void prescan2(unsigned int *g_odata, unsigned int *g_idata)
{
	__shared__ unsigned int rle_temp;

	unsigned int rle_info;

	if ( threadIdx.x == 0) {
		rle_temp=0;
	}
	if ( threadIdx.x < 5) {
		//unsigned int ind=0;

		rle_info = atomicAdd(&rle_temp, g_idata[ rle_temp>>16 ] + 1<<16);

		unsigned int rle_write = rle_info & 0x0000FFFF;
		unsigned int rle_read  = rle_info>>16;

		int nb_repetitions = g_idata[2*rle_read];
		int valeur = g_idata[2*rle_read+1];

		for (int k = 0;k<nb_repetitions; k++)
		{
			g_odata[k + rle_write] = valeur;
		}
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
__global__
void RLE_thrust_kernel(const unsigned int* scan ,const unsigned char * values, const unsigned int * header, unsigned char* pOutput )
{
	__shared__ unsigned int shared_offsets[MAX_COMPRESSED_BRICK_SIZE];
	__shared__ unsigned char shared_values[MAX_COMPRESSED_BRICK_SIZE];

	//__shared__ float shared_sum_offsets[MAX_COMPRESSED_BRICK_SIZE];

	const int offset = threadIdx.x+threadIdx.y * blockDim.x;
	int offset2;
	const int block_size = blockDim.x * blockDim.y;


	unsigned int ind_begin;

	unsigned int ind_end =header[blockIdx.x];
	unsigned int nb_elements;
	if (blockIdx.x==0)
	{
		ind_begin = 0;
	} else {
		ind_begin =  header[blockIdx.x-1];
	}

	nb_elements = ind_end -  ind_begin;

	//
	//
	//

	offset2 = offset;

	while (offset2  < nb_elements)
	{

		shared_offsets[offset2]= scan[ind_begin+offset2] - (BRICK_SIZE*blockIdx.x);
		shared_values[offset2]=values[ind_begin+offset2];

		offset2 +=block_size;
	}
	//__syncthreads();

	/*if (threadIdx.x+threadIdx.y ==0)
	{
		printf("%d\n",nb_elements);
		for (int k =0;k<nb_elements;k++)
			printf("%d %u ; ",shared_offsets[k],shared_values[k]);
		printf("\n");

	}*/

	__syncthreads();


	if (nb_elements != 0)
	{
		offset2 = offset;
		//unsigned char val;


		while (offset2  < BRICK_SIZE)
		{

			/*int middle_index = nb_elements/2;
			int left_chunk_size = middle_index;
			int right_chunk_size = nb_elements - left_chunk_size;
			int old_left_chunk_size;
			bool out = false;
			int index_to_write=0;

			while(out != true) {
				if (shared_offsets[middle_index]==offset2)
				{
					index_to_write = middle_index;
					out =true;
				} else if ( shared_offsets[middle_index]>offset2){
					if (left_chunk_size <= 1)
					{
						index_to_write = middle_index-1;
						out = true;
					}
					old_left_chunk_size = left_chunk_size;
					left_chunk_size = left_chunk_size/2;
					right_chunk_size = old_left_chunk_size - left_chunk_size;
					middle_index -= right_chunk_size;

				} else {
					if (right_chunk_size <= 1)
					{
						index_to_write = middle_index;
						out = true;
					}
					left_chunk_size = right_chunk_size/2 ;
					middle_index += left_chunk_size ;
					right_chunk_size = right_chunk_size - left_chunk_size;

				}

			}*/

			int left = 0;
			int right = nb_elements;
			int middle = (left + right) >> 1;

			while( left < right ) {
				unsigned int temp = shared_offsets[middle];
				if( temp == offset2 ) {
					++middle;
					break;
				} else if ( temp < offset2 ) {
					left = middle +1;

				} else {
					right = middle ;
				}
				middle = (right + left) >> 1;
			}
			int index_to_write = middle;


			pOutput[offset2 + (blockIdx.x) * BRICK_SIZE] = shared_values[  index_to_write ];

			offset2 +=block_size;
		}
	}
}

#endif
