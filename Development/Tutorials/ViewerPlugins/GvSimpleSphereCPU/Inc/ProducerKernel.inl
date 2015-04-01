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
 * Initialize
 *
 * @param pNodesBuffer node buffer
 * @param pBricksPool bricks buffer
 ******************************************************************************/
template< typename TDataStructureType >
inline void ProducerKernel< TDataStructureType >
::init( const GvCore::Array3DKernelLinear< uint >& pNodesBuffer, const BricksPoolKernelType& pBricksPool )
{
	_nodesCache = pNodesBuffer;
	_bricksCache = pBricksPool;
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
	const GvCore::GvLocalizationInfo::CodeType parentLocCode = parentLocInfo.locCode;
	const GvCore::GvLocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;

	if ( processID < NodeRes::getNumElements() )
	{
		uint3 subOffset = NodeRes::toFloat3( processID );

		uint3 regionCoords = parentLocCode.addLevel< NodeRes >( subOffset ).get();
		uint regionDepth = parentLocDepth.addLevel().get();

		GvStructure::GvNode newnode;
		newnode.childAddress = 0;
		newnode.brickAddress = 0;

		GPUVoxelProducer::GPUVPRegionInfo nodeinfo = getRegionInfo( regionCoords, regionDepth, requestID, processID );

		if ( nodeinfo == GPUVoxelProducer::GPUVP_CONSTANT )
		{
			newnode.setTerminal( true );
		}
		else if ( nodeinfo == GPUVoxelProducer::GPUVP_DATA )
		{
			newnode.setStoreBrick();
			newnode.setTerminal( false );
		}
		else if ( nodeinfo == GPUVoxelProducer::GPUVP_DATA_MAXRES )
		{
			newnode.setStoreBrick();
			newnode.setTerminal( true );
		}

		// Write node info into the node pool
		nodePool.getChannel( Loki::Int2Type< 0 >() ).set( newElemAddress.x + processID, newnode.childAddress );
		nodePool.getChannel( Loki::Int2Type< 1 >() ).set( newElemAddress.x + processID, newnode.brickAddress );
	}

	return (0);
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
	// Here we are in a brick
	//
	// Each bloc is in charge of the production of one brick

	// Nb of elements in a brick
	const uint brickNumVoxels = 1000;

	// Brick offset in the mapped memory array where data has been produced on host
	//
	// requestID is the index of the brick
	const uint brickAddress = requestID * brickNumVoxels;

	// Iterate through elements of the brick
	//
	// Each thread compute several elements spaced by blockDim.x (see the FOR loop offset)
	// - thread 0 => 0 | blockDim.x * blockDim.y | 2 x blockDim.x * blockDim.y | 3 x blockDim.x * blockDim.y | etc...
	// - thread 1 => 1 | 1 + blockDim.x * blockDim.y | 1 + 2 x blockDim.x * blockDim.y| 1 + 3 x blockDim.x * blockDim.y | etc...
	// - thread 2 => 2 | 2 + blockDim.x * blockDim.y | 2 + 2 x blockDim.x * blockDim.y| 2 + 3 x blockDim.x * blockDim.y| etc...
	//
	// note : here we have a 2D kernel block size
	const uint threadAccessOffset = blockDim.x * blockDim.y;
	for ( uint dataOffset = processID; dataOffset < brickNumVoxels; dataOffset += threadAccessOffset )
	{
		// Convert 1D element index into 3D offset (to be able to write data in cache)
		uint3 voxelOffset;
		voxelOffset.x = dataOffset % BrickFullRes::x;
		voxelOffset.y = (dataOffset / BrickFullRes::x) % BrickFullRes::y;
		voxelOffset.z = (dataOffset / (BrickFullRes::x * BrickFullRes::y));

		// Address where to write data in cache
		const uint3 destAddress = newElemAddress + make_uint3( voxelOffset );

		typedef typename GvCore::DataChannelType< DataTList, 0 >::Result ColorType;
		typedef typename GvCore::DataChannelType< DataTList, 1 >::Result NormalType;

		// Retrieve voxel color previously generated on CPU
		// ColorType color = make_uchar4( 255, 0, 0, 255 );
		ColorType color = _bricksCache.getChannel( Loki::Int2Type< 0 >() ).get( brickAddress + dataOffset );

		// Retrieve voxel normal previously generated on CPU
		// NormalType normal = make_half4( 0.0f, 1.0f, 0.0f, 1.0f );
		NormalType normal = _bricksCache.getChannel( Loki::Int2Type< 1 >() ).get( brickAddress + dataOffset );

		// Write voxel data to data pool
		dataPool.setValue< 0 >( destAddress, color );
		dataPool.setValue< 1 >( destAddress, normal );
	}

	return 0;
}

/******************************************************************************
 * Helper function used to determine the type of zones in the data structure.
 *
 * The data structure is made of regions containing data, empty or constant regions.
 * Besides, this function can tell if the maximum resolution is reached in a region.
 *
 * @param regionCoords region coordinates
 * @param regionDepth region depth
 * @param nodeTileIndex ...
 * @param nodeTileOffset ...
 *
 * @return the type of the region
 ******************************************************************************/
template< typename TDataStructureType >
__device__
inline GPUVoxelProducer::GPUVPRegionInfo ProducerKernel< TDataStructureType >
::getRegionInfo( uint3 regionCoords, uint regionDepth, uint nodeTileIndex, uint nodeTileOffset )
{
	if ( regionDepth >= 32 )
	{
		return GPUVoxelProducer::GPUVP_DATA_MAXRES;
	}

	// Retrieve node info previously generated on CPU
	uint nodeInfo = _nodesCache.get( nodeTileIndex * NodeRes::getNumElements() + nodeTileOffset );
	if ( nodeInfo )
	{
		return GPUVoxelProducer::GPUVP_DATA;
	}
	else
	{
		return GPUVoxelProducer::GPUVP_CONSTANT;
	}
}
