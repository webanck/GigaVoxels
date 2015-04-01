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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvStructure/GvNode.h>
//#include <GvStructure/GvVolumeTree.h>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Initialize the producer
 * 
 * @param volumeTreeKernel Reference on a volume tree data structure
 ******************************************************************************/
template< typename TDataStructureType >
inline void ProducerKernel< TDataStructureType >
::initialize( DataStructureKernel& pDataStructure )
{
	//_dataStructureKernel = pDataStructure;
}

/******************************************************************************
 * Produce data on device.
 * Implement the produceData method for the channel 0 (nodes)
 *
 * Producing data mecanism works element by element (node tile or brick) depending on the channel.
 *
 * In the function, user has to produce data for a node tile or a brick of voxels :
 * - for a node tile, user has to defined regions (i.e nodes) where lies data, constant values,
 * etc...
 * - for a brick, user has to produce data (i.e voxels) at for each of the channels
 * user had previously defined (color, normal, density, etc...)
 *
 * @param pGpuPool The device side pool (nodes or bricks)
 * @param pRequestID The current processed element coming from the data requests list (a node tile or a brick)
 * @param pProcessID Index of one of the elements inside a node tile or a voxel bricks
 * @param pNewElemAddress The address at which to write the produced data in the pool
 * @param pParentLocInfo The localization info used to locate an element in the pool
 * @param Loki::Int2Type< 0 > corresponds to the index of the node pool
 *
 * @return A feedback value that the user can return.
 ******************************************************************************/
template< typename TDataStructureType >
template< typename GPUPoolKernelType >
__device__
inline uint ProducerKernel< TDataStructureType >
::produceData( GPUPoolKernelType& nodePool, uint requestID, uint processID, uint3 newElemAddress,
			  const GvCore::GvLocalizationInfo& parentLocInfo, Loki::Int2Type< 0 > )
{
	// Check bounds
	if ( processID < NodeRes::getNumElements() )
	{
		// Declare a child node
		GvStructure::GvNode newnode;
		newnode.childAddress = 0;
		newnode.brickAddress = 0;

		// Initialization child node info
		//
		// - no need to call an oracle, by default we say that there is data everywhere
		newnode.setStoreBrick();
		newnode.setTerminal( false );
	
		// Write node info into the node pool
		nodePool.getChannel( Loki::Int2Type< 0 >() ).set( newElemAddress.x + processID, newnode.childAddress );
		nodePool.getChannel( Loki::Int2Type< 1 >() ).set( newElemAddress.x + processID, newnode.brickAddress );
	}

	return 0;
}

/******************************************************************************
 * Produce data on device.
 * Implement the produceData method for the channel 1 (bricks)
 *
 * Producing data mecanism works element by element (node tile or brick) depending on the channel.
 *
 * In the function, user has to produce data for a node tile or a brick of voxels :
 * - for a node tile, user has to defined regions (i.e nodes) where lies data, constant values,
 * etc...
 * - for a brick, user has to produce data (i.e voxels) at for each of the channels
 * user had previously defined (color, normal, density, etc...)
 *
 * @param pGpuPool The device side pool (nodes or bricks)
 * @param pRequestID The current processed element coming from the data requests list (a node tile or a brick)
 * @param pProcessID Index of one of the elements inside a node tile or a voxel bricks
 * @param pNewElemAddress The address at which to write the produced data in the pool
 * @param pParentLocInfo The localization info used to locate an element in the pool
 * @param Loki::Int2Type< 1 > corresponds to the index of the brick pool
 *
 * @return A feedback value that the user can return.
 ******************************************************************************/
template< typename TDataStructureType >
template< typename GPUPoolKernelType >
__device__
inline uint ProducerKernel< TDataStructureType >
::produceData( GPUPoolKernelType& dataPool, uint requestID, uint processID, uint3 newElemAddress,
			  const GvCore::GvLocalizationInfo& parentLocInfo, Loki::Int2Type< 1 > )
{
	// Range bounds
	const float alpha1 = -0.2f;
	const float alpha2 = 0.05f;

	const GvCore::GvLocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;

	// TO DO
	// - check the " 1 << 1 << parentLocDepth.get()" !??
	float3 levelResInv = make_float3( 1.f ) / make_float3( make_uint3( 1 << 1 << parentLocDepth.get() ) * BrickRes::get() );

	// Real brick size (with borders)
	uint3 elemSize = BrickRes::get() + make_uint3( 2 * BorderSize );

	// To know if there is at least one voxel in the range [alpha1, alpha2].
	__shared__ uint count;
	if ( threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 )
	{
		count = 0;
	}
	// Thread Synchronization
	__syncthreads();

	// The original KERNEL execution configuration on the HOST has a 2D block size :
	// dim3 blockSize( 16, 8, 1 );
	//
	// Each block process one brick of voxels.
	//
	// One thread iterate in 3D space given a pattern defined by the 2D block size
	// and the following "for" loops. Loops take into account borders.
	// In fact, each thread of the current 2D block compute elements layer by layer
	// on the z axis.
	//
	// One thread process only a subset of the voxels of the brick.
	//
	// Iterate through z axis step by step as blockDim.z is equal to 1
	uint3 elemOffset;
	for ( elemOffset.z = 0; elemOffset.z < elemSize.z; elemOffset.z += blockDim.z )
	{
		for ( elemOffset.y = 0; elemOffset.y < elemSize.y; elemOffset.y += blockDim.y )
		{
			for ( elemOffset.x = 0; elemOffset.x < elemSize.x; elemOffset.x += blockDim.x )
			{
				// Compute position index
				uint3 locOffset = elemOffset + make_uint3( threadIdx.x, threadIdx.y, threadIdx.z );

				// Test if the computed position index is inside the brick (with borders)
				if ( locOffset.x < elemSize.x && locOffset.y < elemSize.y && locOffset.z < elemSize.z )
				{
					// Compute the new element's address
					uint3 destAddress = newElemAddress + locOffset;
					
					// Compute normal from distance function ( interpreted as a potential function )
					float3 normal = this->template getNormal< GPUPoolKernelType >( dataPool, newElemAddress, locOffset, levelResInv, elemSize );

					// Write the voxel's normal 
					dataPool.template setValue< 1 >( destAddress, normal.x );
					dataPool.template setValue< 2 >( destAddress, normal.y );
					dataPool.template setValue< 3 >( destAddress, normal.z );

					// We test if this voxel is in the range
					const float dist = dataPool.template getValue< 0, float >( newElemAddress + locOffset );
					if ( ( dist < alpha2 ) && ( dist > alpha1 ) )
					{
						// Don't need to use atomic add here
						count++;
					}
				}
			}
		}
	}

	// Thread Synchronization
	__syncthreads();
	
	// Retrun 0 if there is information in this brick, 2 to stop production of this brick
	if ( count > 0 )
	{
		return 0;
	}
	else
	{
		return 2;
	}
}

/******************************************************************************
 * Helper fonction to compute the normal from potential function
 *
 * @param dataPool The device side pool (nodes or bricks)
 * @param newElemAddress The address at which to write the produced data in the pool
 * @param locOffset Position index
 * @param levelResInc Inv of the number of voxels at the current brick resolution
 * @param elemSize 
 *
 * @return the normal
 ******************************************************************************/
template< typename TDataStructureType >
template< typename GPUPoolKernelType >
__device__
inline float3 ProducerKernel< TDataStructureType >
::getNormal( GPUPoolKernelType& dataPool, uint3 newElemAddress, uint3 locOffset, float3 levelResInv, uint3 elemSize )
{
	// Compute the grad
	const uint3 zero = make_uint3( 0, 0, 0 );
	elemSize -= make_uint3( 1, 1, 1 );

	// We translate the offset to repeat information in the border
	// Comment this line if you want to compute normal in the border as inside the brick
	// but because to compute normal we use +1 and -1 value of the distance funtion that produce discontinuity
	tranlateOffset( locOffset, elemSize );

	float3 normal;
	const float aux = dataPool.template getValue< 0, float >( newElemAddress + locOffset );

	// Gradient
	// - x component
	float aux1 = dataPool.template getValue< 0, float >( newElemAddress + clamp( locOffset - make_uint3( 1, 0, 0 ), zero, elemSize ) );
	float aux2 = dataPool.template getValue< 0, float >( newElemAddress + clamp( locOffset + make_uint3( 1, 0, 0 ), zero, elemSize ) );
	// If the distance is 1.0, this value is not a real value of distance ( limitation signed distance field algorithm ) 
	if ( aux1 == 1.0 && aux2 == 1.0 )
	{
		normal.x = 0.0;
	}
	else if ( aux1 == 1.0 )
	{
		normal.x = ( aux2 - aux ) / ( levelResInv.x );
	}
	else if ( aux2 == 1.0 )
	{
		normal.x = ( aux - aux1 ) / ( levelResInv.x );
	}
	else
	{
		normal.x = ( aux2 - aux1 ) / ( 2 * levelResInv.x );
	}
		
	// Gradient
	// - y component
	aux1 = dataPool.template getValue< 0, float >( newElemAddress + clamp( locOffset - make_uint3( 0, 1, 0 ), zero, elemSize ) );
	aux2 = dataPool.template getValue< 0, float >( newElemAddress + clamp( locOffset + make_uint3( 0, 1, 0 ), zero, elemSize ) );
	if ( aux1 == 1.0 && aux2 == 1.0 )
	{
		normal.y = 0.0;
	}
	else if ( aux1 == 1.0 )
	{
		normal.y = ( aux2 - aux ) / ( levelResInv.y );
	}
	else if ( aux2 == 1.0 )
	{
		normal.y = ( aux - aux1 ) / ( levelResInv.y );
	}
	else
	{
		normal.y = ( aux2 - aux1 ) / ( 2 * levelResInv.y );
	}
	
	// Gradient
	// - z component
	aux1 = dataPool.template getValue< 0, float >( newElemAddress + clamp( locOffset - make_uint3( 0, 0, 1 ), zero, elemSize ) );
	aux2 = dataPool.template getValue< 0, float >( newElemAddress + clamp( locOffset + make_uint3( 0, 0, 1 ), zero, elemSize ) );
	if ( aux1 == 1.0 && aux2 == 1.0 )
	{
		normal.z = 0.0;
	}
	else if ( aux1 == 1.0 )
	{
		normal.z = ( aux2 - aux ) / ( levelResInv.z );
	}
	else if ( aux2 == 1.0 )
	{
		normal.z = ( aux - aux1 ) / ( levelResInv.z );
	}
	else
	{
		normal.z = ( aux2 - aux1 ) / ( 2 * levelResInv.z );
	}
	
	// Normalization
	normal = normalize( normal );
	
	return normal;
}

/*****************************************************************************
 * Helper function to translate offset if we are in the border
 *
 * @param offset
 *
 * @return offset translated
 ******************************************************************************/
template< typename TDataStructureType >
__device__
inline void ProducerKernel< TDataStructureType >
::tranlateOffset( uint3& offset, const uint3& elemSize )
{
	// x border
	if( offset.x == 0 )
	{
		offset.x++;
	}
	else if ( offset.x == elemSize.x )
	{
		offset.x--;
	}

	// y border
	if ( offset.y == 0 )
	{
		offset.y++;
	}
	else if ( offset.y == elemSize.y )
	{
		offset.y--;
	}

	// z border
	if ( offset.z == 0 )
	{
		offset.z++;
	}
	else if ( offset.z == elemSize.z )
	{
		offset.z--;
	}
}
