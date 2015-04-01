/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2012 INRIA <http://www.inria.fr/>
 *
 * Authors : GigaVoxels Team
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
#include "GvStructure/GvNode.h"

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvStructure
{

/******************************************************************************
 * Update buffer with a subdivision request for a given node.
 *
 * @param nodeAddressEnc the encoded node address
 ******************************************************************************/
template< class NodeTileRes, class BrickFullRes, class NodeAddressType, class BrickAddressType >
__device__
__forceinline__ void GvDataProductionManagerKernel< NodeTileRes, BrickFullRes, NodeAddressType, BrickAddressType >
::subDivRequest( uint nodeAddressEnc )
{
	// Retrieve 3D node address
	const uint3 nodeAddress = NodeAddressType::unpackAddress( nodeAddressEnc );

	// Update buffer with a subdivision request for that node
	_updateBufferArray.set( nodeAddress, ( nodeAddressEnc & 0x3FFFFFFF ) | VTC_REQUEST_SUBDIV );
}

/******************************************************************************
 * Update buffer with a load request for a given node.
 *
 * @param nodeAddressEnc the encoded node address
 ******************************************************************************/
template< class NodeTileRes, class BrickFullRes, class NodeAddressType, class BrickAddressType >
__device__
__forceinline__ void GvDataProductionManagerKernel< NodeTileRes, BrickFullRes, NodeAddressType, BrickAddressType >
::loadRequest( uint nodeAddressEnc )
{
	// Retrieve 3D node address
	const uint3 nodeAddress = NodeAddressType::unpackAddress( nodeAddressEnc );

	// Update buffer with a load request for that node
	_updateBufferArray.set( nodeAddress, ( nodeAddressEnc & 0x3FFFFFFF ) | VTC_REQUEST_LOAD );
}

} // namespace GvStructure

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

namespace GvStructure
{

/******************************************************************************
 * KERNEL ClearVolTreeRoot
 *
 * This clears node pool child and brick 1st nodetile after root node.
 *
 * @param pDataStructure data structure
 * @param pRootAddress root node address from which to clear data
 ******************************************************************************/
template< typename VolTreeKernelType >
__global__
// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
void ClearVolTreeRoot( VolTreeKernelType pDataStructure, const uint pRootAddress )
{
	//uint lineSize = __uimul( blockDim.x, gridDim.x );
	const uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x );// + __uimul( blockIdx.y, lineSize );

	if ( elem < VolTreeKernelType::NodeResolution::getNumElements() )
	{
		GvStructure::GvNode node;
		node.childAddress = 0;
		node.brickAddress = 0;

		pDataStructure.setNode( node, pRootAddress + elem );
	}
}

// Updates
/******************************************************************************
 * KERNEL UpdateBrickUsage
 *
 * @param volumeTree ...
 * @param pRootAddress ...
 ******************************************************************************/
template< typename ElementRes, typename GPUCacheType >
__global__
// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
void UpdateBrickUsage( uint numElems, uint* lruElemAddressList, GPUCacheType gpuCache )
{
	uint elemNum = blockIdx.x;

	if ( elemNum < numElems )
	{
		uint elemIndexEnc = lruElemAddressList[ elemNum ];
		uint3 elemIndex = GvStructure::VolTreeBrickAddress::unpackAddress( elemIndexEnc );
		uint3 elemAddress = elemIndex * ElementRes::get();

		// FIXME: fixed border size !
		uint3 brickAddress = elemAddress + make_uint3( 1 );
		gpuCache._brickCacheManager.setElementUsage( brickAddress );
	}
}

/******************************************************************************
* KERNEL GvKernel_PreProcessRequests
*
* This kernel is used as first pass a stream compaction algorithm
* in order to create the masks of valid requests
* (i.e. the ones that have been requested during the N3-Tree traversal).
*
* @param pRequests Array of requests (i.e. subdivide nodes or load/produce bricks)
* @param pIsValidMask Resulting array of isValid masks
* @param pNbElements Number of elememts to process
******************************************************************************/
__global__
// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
void GvKernel_PreProcessRequests( const uint* pRequests, unsigned int* pIsValidMasks, const uint pNbElements )
{
	// Retrieve global data index
	uint lineSize = __uimul( blockDim.x, gridDim.x );
	uint index = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );

	// Check bounds
	if ( index < pNbElements )
	{
		// Set the associated isValid mask, knowing that
		// the input requests buffer is reset to 0 at each frame
		// (i.e. a value different of zero means that a request has been emitted).
		if ( pRequests[ index ] == 0 )
		{
			pIsValidMasks[ index ] = 0;
		}
		else
		{
			pIsValidMasks[ index ] = 1;
		}
	}
}

} // namespace GvStructure
