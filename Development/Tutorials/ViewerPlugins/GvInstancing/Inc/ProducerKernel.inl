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
 ****************************** KERNEL DEFINITION *****************************
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
 * Inititialize
 *
 * @param maxdepth max depth
 * @param nodescache nodes cache
 * @param datacachepool data cache pool
 ******************************************************************************/
template< typename TDataStructureType >
inline void ProducerKernel< TDataStructureType >
::init( uint maxdepth, const GvCore::Array3DKernelLinear< uint >& nodescache, const DataCachePoolKernelType& datacachepool )
{
	_maxDepth = maxdepth;
	_cpuNodesCache = nodescache;
	_cpuDataCachePool = datacachepool;
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
::produceData( GPUPoolKernelType& nodePool, uint requestID, uint processID,
				uint3 newElemAddress, const GvCore::GvLocalizationInfo& parentLocInfo,
				Loki::Int2Type< 0 > )
{
	// NOTE :
	// In this method, you are inside a node tile.
	// A pre-process step on HOST has previously determined, for each node of the node tile, which type of data it holds.
	// Data type can be :
	// - a constant region,
	// - a region with data,
	// - a region where max resolution is reached.
	// So, one thread is responsible of the production of one node of a node tile.

	// Get localization info (code and depth)
	uint3 parentLocCode = parentLocInfo.locCode.get();
	uint parentLocDepth = parentLocInfo.locDepth.get();

	// Check bound
	if ( processID < NodeRes::getNumElements() )
	{
		// Create a new node
		GvStructure::GvNode newnode;

		// Initialize the child address with the HOST nodes cache
		newnode.childAddress = _cpuNodesCache.get( requestID * NodeRes::getNumElements() + processID );

		// Initialize the brick address
		newnode.brickAddress = 0;

		// Finally, write the new node information into the node pool by selecting channels :
		// - Loki::Int2Type< 0 >() points to node information
		// - Loki::Int2Type< 1 >() points to brick information
		//
		// newElemAddress.x + processID : is the adress of the new node in the node pool
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
template< typename TGPUPoolKernelType >
__device__
inline uint ProducerKernel< TDataStructureType >
::produceData( TGPUPoolKernelType& pDataPool, uint pRequestID, uint pProcessID,
				uint3 pNewElemAddress, const GvCore::GvLocalizationInfo& pParentLocInfo,
				Loki::Int2Type< 1 > )
{
	// parentLocDepth++; //Shift needed, to be corrected
	bool nonNull = ProducerKernel_ChannelLoad< TDataStructureType, TGPUPoolKernelType, GvCore::DataNumChannels< DataTList >::value - 1>::produceDataChannel( *this, pDataPool, pNewElemAddress, pParentLocInfo, pRequestID, pProcessID );

	return 0;
	//if (nonNull)
	//	return 0;
	//else
	//	return 2;
}

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Produce data at the specified channel
 *
 * @param gpuVPLK reference on the volume producer load kernel
 * @param dataPool the data pool in which to write data
 * @param elemAddress The address at which to write the produced data in the pool
 * @param parentLocInfo The localization info used to locate an element in the pool
 * @param pRequestID The current processed element coming from the data requests list (a brick)
 * @param pProcessID Index of one of the elements inside a voxel bricks
******************************************************************************/
template< typename TDataStructureType, typename TGPUPoolKernelType, int channel >
__device__
inline bool ProducerKernel_ChannelLoad< TDataStructureType, TGPUPoolKernelType, channel >
::produceDataChannel( ProducerKernel< TDataStructureType >& gpuVPLK,
					 TGPUPoolKernelType& dataPool, uint3 elemAddress, const GvCore::GvLocalizationInfo& parentLocInfo, uint requestID, uint processID )
{
	uint blockIndex = requestID;

	// Number of voxels
	uint brickNumVoxels = BrickFullRes::numElements;
	uint blockStartAddress = blockIndex * ProducerKernel< TDataStructureType >::BrickVoxelAlignment;

	uint blockNumThreads = blockDim.x * blockDim.y * blockDim.z;

	// Iterate through voxels of the current brick
	uint decal;
	for ( decal = 0; decal < brickNumVoxels; decal += blockNumThreads )
	{
		uint locDecal = decal + processID;

		if ( locDecal < brickNumVoxels )
		{
			typedef typename GvCore::DataChannelType< DataTList, channel >::Result VoxelType;
			VoxelType voxelData;

			uint locDecalOffset = locDecal;
			voxelData = gpuVPLK._cpuDataCachePool.getChannel( Loki::Int2Type< channel >() ).get( blockStartAddress + locDecalOffset );

			uint3 voxelOffset;
			voxelOffset.x = locDecal % BrickFullRes::x;
			voxelOffset.y = ( locDecal / BrickFullRes::x ) % BrickFullRes::y;
			voxelOffset.z = ( locDecal / ( BrickFullRes::x * BrickFullRes::y ) );
			uint3 destAddress = elemAddress + make_uint3( voxelOffset );

			// Write the voxel's data for the specified channel index
			dataPool.setValue< channel >( destAddress, voxelData );
		}
	}

	// Recursive call to produce data until the last channel is reached
	return ProducerKernel_ChannelLoad< TDataStructureType, TGPUPoolKernelType, channel - 1 >::produceDataChannel( gpuVPLK, dataPool, elemAddress, parentLocInfo, requestID, processID );
}

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Produce data at the specified channel
 *
 * @param gpuVPLK reference on the volume producer load kernel
 * @param dataPool the data pool in which to write data
 * @param elemAddress The address at which to write the produced data in the pool
 * @param parentLocInfo The localization info used to locate an element in the pool
 * @param pRequestID The current processed element coming from the data requests list (a brick)
 * @param pProcessID Index of one of the elements inside a voxel bricks
******************************************************************************/
template< typename TDataStructureType, typename TGPUPoolKernelType >
__device__
inline bool ProducerKernel_ChannelLoad< TDataStructureType, TGPUPoolKernelType, -1 >
::produceDataChannel( ProducerKernel< TDataStructureType >& gpuVPLK,
					  TGPUPoolKernelType& dataPool, uint3 elemAddress, const GvCore::GvLocalizationInfo& parentLocInfo, uint requestID, uint processID )
{
	return false;
}
