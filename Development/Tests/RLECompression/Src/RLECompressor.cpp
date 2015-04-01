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

#include "RLECompressor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// System
#include <cassert>
#include <iostream>
#include <climits>

#include "macros.h"


/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * ...
 */
#define BIT_MOD 2

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
RLECompressor::RLECompressor()
{
}

/******************************************************************************
 * Desconstructor
 ******************************************************************************/
RLECompressor::~RLECompressor()
{
	// Finalize
	finalize();
}

/******************************************************************************
 * Initiliaze
 *
 * @param pWidth width
 * @param pHeight height
 ******************************************************************************/
void RLECompressor::initialize()
{
}

/******************************************************************************
 * Finalize
 ******************************************************************************/
void RLECompressor::finalize()
{
}

/******************************************************************************
 * ...
 *
 * @param ... ...
 * @param ... ...
 ******************************************************************************/
void RLECompressor::sameArray( const unsigned char* pInput, const unsigned char* pOutput )
{
	bool isCop = true;

	// Iterate through all elements
	for ( int i = 0; i < STR_SIZE; i++ )
	{
		if ( ( pInput[ i ] != pOutput[ i ] )
			&& ( pOutput[ i ] != 1 )
			&& ( i != 2 ) )
		{
			std::cout << "/!\\ Difference au " << i << "eme element " << std::endl;

			//for (int k = 0; k < 100; k++)
			std::cout << "/!\\ " << static_cast< int >( pInput[ i ] ) << " != " << static_cast< int >( pOutput[ i ] ) << std::endl;
			
			isCop = false;
			
			// Exit loop
			break;
		}
	}

	if ( isCop )
	{
		std::cout << "- copy OK" << std::endl;
	}
	else
	{ 
		std::cout << "- copy ERROR" << std::endl;
	}
}

/******************************************************************************
 * Given an input data array, write output array with RLE encoding (compression)
 *
 * @param pInput input data
 * @param pOutput output data (RLE encoding - compression)
 ******************************************************************************/
/*
int RLECompressor::RLEcompBis( const unsigned char* pInput, unsigned char* pOutput )
{
	unsigned int brickOffsetLinear = 1;
	unsigned int count = 0;
	unsigned char oldDensity = pInput[ 0 ];
	unsigned int nbElem = 0;
	bool outOfMem = false;

	// Iterate through all data elements
	for ( unsigned int offset = 0; offset < STR_SIZE; offset++ )
	{
		// Check bounds
		if ( brickOffsetLinear < STR_SIZE && nbElem < 255 )
		{
			// RLE version
			if ( pInput[ offset ] == oldDensity && count < 255 )
			{
				count++;
			}
			else
			{
				// Write the number if iteration of data
				pOutput[ brickOffsetLinear ] = count;
				brickOffsetLinear++;

				// Write the data value
				pOutput[ brickOffsetLinear ] = oldDensity;
				brickOffsetLinear++;

				// Update old value
				oldDensity = pInput[ offset ];

				// Reset number of iteration to 1
				count = 1;

				// Update global number of elements
				nbElem++;
			}
		}
		else
		{
			// LOG info
			std::cout << "No RLE compression" << std::endl;

			offset = STR_SIZE;
			memcpy( pOutput, pInput, STR_SIZE * sizeof( unsigned char ) );

			// Set LSB to 0 (Least Significant Bit) to tell that a RLE compression encoding has not occured
			// - in the 3rd indexed value, i.e the 1st data value (loss of resolution)
			pOutput[ BIT_MOD ] &= ~(1 << 0);

			outOfMem = true;

			// Exit loop
			break;
		}
	}
	
	// Finalize algorithm
	//
	// - write last value
	// - write total number of values
	if ( ! outOfMem )
	{
		// Save the last value

		// Write the number if iteration of data
		pOutput[ brickOffsetLinear ] = count;
		brickOffsetLinear++;

		// Write the data value
		pOutput[ brickOffsetLinear ] = oldDensity;
		brickOffsetLinear++;
		
		// Reset number of iteration to 1
		count = 1;

		// Update global number of elements
		nbElem++;

		// Store the total number of elements in the first memory slot
		pOutput[ 0 ] = nbElem;

		// Set LSB to 1 (Least Significant Bit) to tell that a RLE compression encoding has occured
		// - in the 3rd indexed value, i.e the 1st data value (loss of resolution)
		pOutput[ BIT_MOD ] ^= (1 << 0);

		// LOG info
		std::cout << "RLE compression encoding has been applied on " << nbElem << " elements"  << std::endl;
	}
	return brickOffsetLinear;
}
*/

//void RLECompressor::RLEcomp( const unsigned int* pInput, unsigned int* pOutput )
//{
//	// Fill the output array with UCHAR_MAX
//	//for ( int k = 0; k < BRICK_SIZE + STR_SIZE; ++k ) {
//	//	pOutput[k] = UCHAR_MAX;
//	//}
//
//	// Do the compression for each brick
//	for ( unsigned int brickNum = 0; brickNum < N_BRICKS; ++brickNum ) {
//
//		// Useful variables
//		const unsigned int brickIndex = brickNum * BRICK_SIZE;
//		unsigned int current_position_in_rle_array = N_BRICKS + brickIndex;
//		bool compression_flag = true;
//		unsigned int numberOfCompressedElements = 0;
//	
//		// Compress a plateau of value
//		unsigned int index = 0;
//		while ( index < BRICK_SIZE && compression_flag ) {
//			// Search a plateau
//			unsigned int current_value = pInput[brickIndex + index];
//			unsigned int current_value_plateau=current_value;
//			unsigned int current_plateauLength = 0;
//			do {
//				++current_plateauLength;
//				++index;
//			} while ( index < BRICK_SIZE 
//					&& current_plateauLength < UCHAR_MAX
//					&& pInput[brickIndex + index] == current_value_plateau );
//
//			// Copy information about the plateau in output array
//			pOutput[current_position_in_rle_array] = current_plateauLength;
//			pOutput[current_position_in_rle_array + 1] = current_value_plateau;
//			current_position_in_rle_array += 2;
//			++numberOfCompressedElements;
//
//			compression_flag = numberOfCompressedElements * 2 < BRICK_SIZE ;
//		}
//
//		if( compression_flag ) {
//			// Write the number of element in the compressed brick at the start of the output array
//			pOutput[brickNum] = numberOfCompressedElements;
//		} else {
//			// Don't try to compress, recopy the array as is
//			for ( unsigned int index = 0; index < BRICK_SIZE; ++index ) {
//				pOutput[BRICK_SIZE + brickNum * BRICK_SIZE + index] = pInput[brickNum * BRICK_SIZE +index];
//			}
//			pOutput[brickNum] = UCHAR_MAX;
//		}
//	}
//}

/******************************************************************************
 * RLE compression : HOST-side pre-process
 * - it writes RLE-encoding arrays (with a prefix-sum)
 *
 * @param pInput input data [max size is STR_SIZE]
 * @param pNbBricks number of bricks of data to process
 * @param pBricksEnds (output) RLE-encoded array => prefix-sum array of pPlateausValues. For each brick in pInput, it stores the number of compressed elements (i.e the size of the compressed brick, hence the "end" of the compressed brick) [max size is N_BRICKS, i.e total nb bricks pNbBricks]
 * @param pPlateausValues (output) RLE-encoded array => it store all values (compressed array without repetition) [max size is STR_SIZE, i.e as input data]
 * @param pPlateausStarts (output) RLE-encoded array => for each value in pPlateausValues array, it stores associated number of repetitions [max size is STR_SIZE, i.e as input data]
 ******************************************************************************/
void RLECompressor::compressionPrefixSum( const unsigned int* pInput,
        const unsigned int pNbBricks,
        unsigned int* pBricksEnds,
        unsigned int* pPlateausValues,
        unsigned char* pPlateausStarts )
{
	// Maximum number of plateaus in a compressed brick (after this number, compressed brick is 
	// bigger than non compressed one).
    const unsigned int maxNbPlateaus = MAX_COMPRESSED_BRICK_SIZE;

	// Number of plateaus stored in array so far.
	unsigned int totalPlateaus = 0;

    // Iterate throught bricks
    //
    // - do the compression for each brick
    for ( unsigned int brickId = 0; brickId < pNbBricks; ++brickId )
    {
        const unsigned int brickIndex = brickId * BRICK_SIZE;
		unsigned int currentPositionInRleArray = totalPlateaus;
		bool compressionFlag = true;
		unsigned int nPlateaus = 0;
	
		// Browse the brick as long as we are bellow the maximum number of plateaus
        unsigned int i = 0;
		unsigned int startPrec = 0;
        while ( i < BRICK_SIZE && compressionFlag )
        {
			// Search a plateau
            unsigned int currentValue = pInput[ brickIndex + i ];
            unsigned int currentValuePlateau = currentValue;
			unsigned int currentPlateauLength = 0;
            do
            {
				++currentPlateauLength;
                ++i;
            } while( i < BRICK_SIZE
                    && pInput[ brickIndex + i ] == currentValuePlateau );

			// Store information about the plateaus
            pPlateausStarts[ currentPositionInRleArray ] = startPrec;
            pPlateausValues[ currentPositionInRleArray ] = currentValuePlateau;
			++currentPositionInRleArray;
			startPrec += currentPlateauLength;

			// Check if compression is useful
			++nPlateaus;
            compressionFlag = nPlateaus < maxNbPlateaus;
		}

        unsigned int brickStart = brickId > 0 ? pBricksEnds[ brickId - 1 ] : 0;

		// Check if the brick needs to be compressed
        if ( compressionFlag )
        {
			// Store the size of the compressed brick
            pBricksEnds[ brickId ] = nPlateaus + brickStart;

            totalPlateaus += nPlateaus;
        }
        else
        {
			// Don't try to compress, recopy the array as is
            for ( unsigned int j = 0; j < BRICK_SIZE; ++j )
            {
                pPlateausValues[ totalPlateaus + j ] = pInput[ brickId * BRICK_SIZE + j ];
			}

            // Store the size of the brick
            pBricksEnds[ brickId ] = brickStart + BRICK_SIZE;

            totalPlateaus += BRICK_SIZE;
		}
	}
}


/******************************************************************************
 * Given an input data array, write output array with RLE encoding (compression)
 * - data
 * - and offsets
 *
 * @param pInput input data
 * @param pOutputData output data (RLE encoding - compression)
 * @param pOutputOffset output offsets (RLE encoding - compression)
 ******************************************************************************/
bool RLECompressor::RLEcompOffset( const unsigned char* pInput, unsigned char* pOutputData, unsigned int * pOutputOffset )
{
	unsigned int brickLinear = 0;
	unsigned int brickOffsetLinear = 1;
	unsigned int count = 0;
	unsigned char oldDensity = pInput[ 0 ];
	unsigned int nbElem = 0;
	bool outOfMem = false;

	// Iterate through all data elements
	for ( unsigned int offset = 0; offset < STR_SIZE; offset++ )
	{
		// Check bounds
		if ( brickOffsetLinear < STR_SIZE )
		{
			// RLE version
			if ( pInput[ offset ] == oldDensity )
			{
				count++;
			}
			else
			{
				// Write the number if iteration of data
				pOutputOffset[ brickOffsetLinear ] = count;
				brickOffsetLinear++;

				// Write the data value
				pOutputData[ brickLinear ] = oldDensity;
				brickLinear++;

				// Update old value
				oldDensity = pInput[ offset ];

				// Reset number of iteration to 1
				count = 1;

				// Update global number of elements (maximum of 255)
				nbElem++;
			}
		}
		else
		{
			offset = STR_SIZE;

			outOfMem = true;
		}
	}

	// Save the last value
	// Set the iteration of the color
	pOutputOffset[ brickOffsetLinear ] = count;
	brickOffsetLinear++;
	
	// Set the color value
	pOutputData[ brickLinear ] = oldDensity;
	brickLinear++;

	// Reset number of iteration to 1
	count = 1;

	// Update global number of elements
	nbElem++;

	// Store the total number of elements in the first memory slot
	pOutputOffset[ 0 ] = nbElem;
	
	return outOfMem;
}
