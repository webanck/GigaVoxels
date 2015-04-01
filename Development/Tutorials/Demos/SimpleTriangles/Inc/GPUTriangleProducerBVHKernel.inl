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

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Initialize
 *
 * @param h_nodesbufferarray node buffer
 * @param h_vertexbufferpool data buffer
 ******************************************************************************/
template< typename TDataStructureType, uint TDataPageSize >
inline void GPUTriangleProducerBVHKernel< TDataStructureType, TDataPageSize >
::init( VolTreeBVHNodeStorage* h_nodesbufferarray, GvCore::GPUPoolKernel< GvCore::Array3DKernelLinear, DataTypeList > h_vertexbufferpool )
{
	_nodesBufferKernel = h_nodesbufferarray;
	_dataBufferKernel = h_vertexbufferpool;
}

/******************************************************************************
 * Produce node tiles
 *
 * @param nodePool ...
 * @param requestID ...
 * @param processID ...
 * @param pNewElemAddress ...
 * @param parentLocInfo ...
 *
 * @return ...
 ******************************************************************************/
template< typename TDataStructureType, uint TDataPageSize >
template< typename GPUPoolKernelType >
__device__
inline uint GPUTriangleProducerBVHKernel< TDataStructureType, TDataPageSize >
::produceData( GPUPoolKernelType& nodePool, uint requestID, uint processID,
				uint3 pNewElemAddress, const VolTreeBVHNodeUser& node, Loki::Int2Type< 0 > )
{
	return 0;
}

/******************************************************************************
 * Produce bricks of data
 *
 * @param dataPool ...
 * @param requestID ...
 * @param processID ...
 * @param pNewElemAddress ...
 * @param parentLocInfo ...
 *
 * @return ...
 ******************************************************************************/
template< typename TDataStructureType, uint TDataPageSize >
template< typename GPUPoolKernelType >
__device__
inline uint GPUTriangleProducerBVHKernel< TDataStructureType, TDataPageSize >
::produceData( GPUPoolKernelType& dataPool, uint requestID, uint processID,
				uint3 pNewElemAddress, VolTreeBVHNodeUser& pNode, Loki::Int2Type< 1 > )
{
	const uint hostElementAddress = pNode.getDataIdx() * BVH_DATA_PAGE_SIZE;

	// Check bounds
	if ( processID < TDataPageSize )
	{
		// Retrieve data
		const float4 position = _dataBufferKernel.getChannel( Loki::Int2Type< 0 >() ).get( hostElementAddress + processID );
		const uchar4 color = _dataBufferKernel.getChannel( Loki::Int2Type< 1 >() ).get( hostElementAddress + processID );

		// Write data in data pool
		dataPool.getChannel( Loki::Int2Type< 0 >() ).set( pNewElemAddress.x + processID, position );
		dataPool.getChannel( Loki::Int2Type< 1 >() ).set( pNewElemAddress.x + processID, color );
	}

	// Update the node's data field and set the gpu flag
	pNode.setDataIdx( pNewElemAddress.x / BVH_DATA_PAGE_SIZE );
	pNode.setGPULink();

	return 0;
}

/******************************************************************************
 * Seems to be unused anymore...
 ******************************************************************************/
template< typename TDataStructureType, uint TDataPageSize >
template< class GPUTreeBVHType >
__device__
inline uint GPUTriangleProducerBVHKernel< TDataStructureType, TDataPageSize >
::produceNodeTileData( GPUTreeBVHType& gpuTreeBVH, uint requestID, uint processID, VolTreeBVHNodeUser& node, uint newNodeTileAddressNode )
{
	// Shared Memory
	__shared__ VolTreeBVHNodeStorage newNodeStorage[ 2 ];	// Not needed, can try without shared memory later

	// TODO: loop to deal with multiple pages per block
	if ( processID < VolTreeBVHNodeStorage::numWords * 2 )
	{
		uint tileElemNum = processID / VolTreeBVHNodeStorage::numWords;	// TODO:check perfos !
		uint tileElemWord = processID % VolTreeBVHNodeStorage::numWords;

		uint cpuBufferPageAddress = node.getSubNodeIdx() + tileElemNum;

		// Parallel coalesced read
		newNodeStorage[ tileElemNum ].words[ tileElemWord ] = _nodesBufferKernel[ cpuBufferPageAddress ].words[ tileElemWord ];	// Can be optimized for address comp

		// Parallel write
		gpuTreeBVH.parallelWriteBVHNode( tileElemWord, newNodeStorage[ tileElemNum ], newNodeTileAddressNode + tileElemNum );
	}

	node.setSubNodeIdx( newNodeTileAddressNode );
	node.setGPULink();

	return 0;
}
