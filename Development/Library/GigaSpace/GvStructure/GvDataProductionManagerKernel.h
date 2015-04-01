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

#ifndef _GV_DATA_PRODUCTION_MANAGER_KERNEL_H_
#define _GV_DATA_PRODUCTION_MANAGER_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <helper_math.h>

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/GPUVoxelProducer.h"
#include "GvCore/GPUPool.h"
#include "GvCore/Array3DKernelLinear.h"
#include "GvCore/StaticRes3D.h"
#include "GvCore/GvLocalizationInfo.h"
#include "GvCache/GvCacheManagerKernel.h"
#include "GvStructure/GvVolumeTree.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvStructure
{

/** 
 * @struct GvDataProductionManagerKernel
 *
 * @brief The GvDataProductionManagerKernel struct provides methods to update buffer
 * of requests on device.
 *
 * Device-side object used to update the buffer of requests emitted by the renderer
 * during the data structure traversal. Requests can be either "node subdivision"
 * or "load brick of voxels".
 */
template< class NodeTileRes, class BrickFullRes, class NodeAddressType, class BrickAddressType >
struct GvDataProductionManagerKernel
{
	
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Bit mask for subdivision request (30th bit)
	 */
	static const unsigned int VTC_REQUEST_SUBDIV = 0x40000000U;

	/**
	 * Bit mask for load request (31th bit)
	 */
	static const unsigned int VTC_REQUEST_LOAD = 0x80000000U;

	/**
	 * Buffer used to store node addresses updated with subdivision or load requests
	 */
	GvCore::Array3DKernelLinear< uint > _updateBufferArray;

	/**
	 * Node cache manager
	 *
	 * Used to update timestamp usage information of nodes
	 */
	GvCache::GvCacheManagerKernel< NodeTileRes, NodeAddressType > _nodeCacheManager;

	/**
	 * Brick cache manager
	 *
	 * Used to update timestamp usage information of bricks
	 */
	GvCache::GvCacheManagerKernel< BrickFullRes, BrickAddressType > _brickCacheManager;

	/******************************** METHODS *********************************/

	/**
	 * Update buffer with a subdivision request for a given node.
	 *
	 * @param nodeAddressEnc the encoded node address
	 */
	__device__
	__forceinline__ void subDivRequest( uint nodeAddressEnc );

	/**
	 * Update buffer with a load request for a given node.
	 *
	 * @param nodeAddressEnc the encoded node address
	 */
	__device__
	__forceinline__ void loadRequest( uint nodeAddressEnc );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} // namespace GvStructure

/******************************************************************************
 ***************************** KERNEL DEFINITION ******************************
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
void ClearVolTreeRoot( VolTreeKernelType volumeTree, const uint rootAddress );

// Updates
/******************************************************************************
 * KERNEL UpdateBrickUsage
 *
 * @param volumeTree ...
 * @param rootAddress ...
 ******************************************************************************/
template< typename ElementRes, typename GPUCacheType >
__global__
void UpdateBrickUsage( uint numElems, uint* lruElemAddressList, GPUCacheType gpuCache );

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
void GvKernel_PreProcessRequests( const uint* pRequests, unsigned int* pIsValidMasks, const uint pNbElements );

///******************************************************************************
// * ...
// ******************************************************************************/
//template< typename TDataStructure, typename TPageTable >
//__global__ void GVKernel_TrackLeafNodes( TDataStructure pDataStructure, TPageTable pPageTable, const unsigned int pNbNodes, const unsigned int pMaxDepth, unsigned int* pLeafNodes )
//{
//	// Retrieve global index
//	const uint index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
//
//	// Check bounds
//	if ( index < pNbNodes )
//	{
//		// Try to retrieve node from the node pool given its address
//		GvStructure::GvNode node;
//		pDataStructure.fetchNode( node, index );
//
//		// Check if node has been initialized
//		// - maybe its safer to check only for childAddress
//		//if ( node.isInitializated() )
//		if ( node.childAddress != 0 )
//		{
//			// Retrieve node depth
//			const GvCore::GvLocalizationInfo localizationInfo = pPageTable.getLocalizationInfo( index );
//			
//			// Check node depth
//			if ( nodeDepth < pMaxDepth )
//			{
//				// Check is node is a leaf
//				if ( ! node.hasSubNodes() )
//				{
//					// Check is node is empty
//					if ( ! node.hasBrick() )
//					{
//						// Empty node
//						pLeafNodes[ index ] = 1;
//					}
//					else
//					{
//						// Node has data
//						pLeafNodes[ index ] = 0;
//					}
//				}
//				else
//				{
//					// Node has children
//					pLeafNodes[ index ] = 0;
//				}
//			}
//			else if ( nodeDepth == pMaxDepth )
//			{
//				// Check is node is empty
//				if ( ! node.hasBrick() )
//				{
//					// Empty node
//					pLeafNodes[ index ] = 1;
//				}
//				else
//				{
//					// Node has data
//					pLeafNodes[ index ] = 0;
//				}
//			}
//			else // ( localizationInfo.locDepth > pMaxDepth )
//			{
//				pLeafNodes[ index ] = 0;
//			}
//		}
//		else
//		{
//			// Uninitialized node
//			pLeafNodes[ index ] = 0;
//		}
//	}
//}

/******************************************************************************
 * Retrieve number of leaf nodes in tree based on a given max depth
 ******************************************************************************/
template< typename TDataStructure, typename TPageTable >
__global__
// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
void GVKernel_TrackLeafNodes( TDataStructure pDataStructure, TPageTable pPageTable, const unsigned int pNbNodes, const unsigned int pMaxDepth, unsigned int* pLeafNodes, float* pEmptyVolume )
{
	// Retrieve global index
	const uint index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

	// Check bounds
	if ( index < pNbNodes )
	{
		// Try to retrieve node from the node pool given its address
		GvStructure::GvNode node;
		pDataStructure.fetchNode( node, index );

		//---------------------------
		if ( node.childAddress == 0 )
		{
			// Node has data
			pLeafNodes[ index ] = 0;

			// Update volume
			pEmptyVolume[ index ] = 0.f;

			return;
		}
		//---------------------------

		// Retrieve its "node tile" address
		//const uint nodeTileAddress = index / 8/*NodeTileRes::getNumElements()*/;

		// Retrieve node depth
		const GvCore::GvLocalizationInfo localizationInfo = pPageTable.getLocalizationInfo( index );
		const unsigned int nodeDepth = localizationInfo.locDepth.get()/* + 1*/;
		
		// Check node depth
		if ( localizationInfo.locDepth.get() < pMaxDepth )
		{
			if ( node.childAddress != 0 )
			{
				// Check if node is a leaf
				if ( ! node.hasSubNodes() )
				{
					// Check if node is empty
					if ( ! node.hasBrick() )
					{
						// Empty node
						pLeafNodes[ index ] = 1;

						// Update volume
						//printf( "\n%u", nodeDepth );
						const float nodeSize = 1.f / static_cast< float >( 1 << nodeDepth );
						pEmptyVolume[ index ] = nodeSize * nodeSize * nodeSize;
					}
					else
					{
						// Node has data
						pLeafNodes[ index ] = 0;

						// Update volume
						pEmptyVolume[ index ] = 0.f;
					}
				}
				else
				{
					// Node has children
					pLeafNodes[ index ] = 0;

					// Update volume
					pEmptyVolume[ index ] = 0.f;
				}
			}
			else
			{
				// ...
				pLeafNodes[ index ] = 0;

				// Update volume
				pEmptyVolume[ index ] = 0.f;
			}
		}
		else if ( localizationInfo.locDepth.get() == pMaxDepth )
		{
			if ( node.childAddress != 0 )
			{
				// Check if node is empty
				if ( ! node.hasBrick() )
				{
					// Empty node
					pLeafNodes[ index ] = 1;

					// Update volume
					const float nodeSize = 1.f / static_cast< float >( 1 << nodeDepth );
					pEmptyVolume[ index ] = nodeSize * nodeSize * nodeSize;
				}
				else
				{
					// Node has data
					pLeafNodes[ index ] = 0;

					// Update volume
					pEmptyVolume[ index ] = 0.f;
				}
			}
			else
			{
				// ...
				pLeafNodes[ index ] = 0;

				// Update volume
				pEmptyVolume[ index ] = 0.f;
			}
		}
		else // ( localizationInfo.locDepth > pMaxDepth )
		{
			// Don't take node into account
			pLeafNodes[ index ] = 0;

			// Update volume
			pEmptyVolume[ index ] = 0.f;
		}
	}
}

///******************************************************************************
// * ...
// ******************************************************************************/
//template< typename TDataStructure, typename TPageTable >
//__global__ void GVKernel_TrackNodes( TDataStructure pDataStructure, TPageTable pPageTable, const unsigned int pNbNodes, const unsigned int pMaxDepth, unsigned int* pLeafNodes )
//{
//	// Retrieve global index
//	const uint index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
//
//	// Check bounds
//	if ( index < pNbNodes )
//	{
//		// Try to retrieve node from the node pool given its address
//		GvStructure::GvNode node;
//		pDataStructure.fetchNode( node, index );
//
//		// Check if node has been initialized
//		// - maybe its safer to check only for childAddress
//		//if ( node.isInitializated() )
//		if ( node.childAddress != 0 )
//		{
//			// Retrieve node depth
//			GvCore::GvLocalizationInfo localizationInfo = pPageTable.getLocalizationInfo( index );
//			
//			// Check if node is a leaf
//			if ( nodeDepth < pMaxDepth )
//			{
//				if ( ! node.hasSubNodes() )
//				{
//					pLeafNodes[ index ] = 1;
//				}
//				else
//				{
//					pLeafNodes[ index ] = 0;
//				}
//			}
//			else if ( nodeDepth == pMaxDepth )
//			{
//				pLeafNodes[ index ] = 1;
//			}
//			else // ( localizationInfo.locDepth > pMaxDepth )
//			{
//				pLeafNodes[ index ] = 0;
//			}
//		}
//		else
//		{
//			// Uninitialized node
//			pLeafNodes[ index ] = 0;
//		}
//	}
//}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TDataStructure, typename TPageTable >
__global__
// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
void GVKernel_TrackNodes( TDataStructure pDataStructure, TPageTable pPageTable, const unsigned int pNbNodes, const unsigned int pMaxDepth, unsigned int* pLeafNodes, float* pEmptyVolume )
{
	//const float pi = 3.141592f;

	// Retrieve global index
	const uint index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

	// Check bounds
	if ( index < pNbNodes )
	{
		// Try to retrieve node from the node pool given its address
		GvStructure::GvNode node;
		pDataStructure.fetchNode( node, index );

		// Retrieve its "node tile" address
		const uint nodeTileAddress = index / 8/*NodeTileRes::getNumElements()*/;

		// Retrieve node depth
		const GvCore::GvLocalizationInfo localizationInfo = pPageTable.getLocalizationInfo( nodeTileAddress );
		const unsigned int nodeDepth = localizationInfo.locDepth.get() + 1;

		// Check node depth
		if ( nodeDepth < pMaxDepth )
		{
			if ( node.childAddress != 0 )
			{
				// Check if node is a leaf
				if ( ! node.hasSubNodes() )
				{
					// Leaf node
					pLeafNodes[ index ] = 1;

					// Update volume
					const float nodeSize = 1.f / static_cast< float >( 1 << nodeDepth );
					pEmptyVolume[ index ] = nodeSize * nodeSize * nodeSize;
				}
				else
				{
					// Node has children
					pLeafNodes[ index ] = 0;

					// Update volume
					pEmptyVolume[ index ] = 0.f;
				}
			}
			else
			{
				// ...
				pLeafNodes[ index ] = 0;

				// Update volume
				pEmptyVolume[ index ] = 0.f;
			}
		}
		else if ( nodeDepth == pMaxDepth )
		{
			if ( node.childAddress != 0 )
			{
				// Leaf node
				pLeafNodes[ index ] = 1;

				// Update volume
				const float nodeSize = 1.f / static_cast< float >( 1 << nodeDepth );
				pEmptyVolume[ index ] = nodeSize * nodeSize * nodeSize;
			}
			else
			{
				// ...
				pLeafNodes[ index ] = 0;

				// Update volume
				pEmptyVolume[ index ] = 0.f;
			}
		}
		else // ( localizationInfo.locDepth > pMaxDepth )
		{
			// Don't take node into account
			pLeafNodes[ index ] = 0;

			// Update volume
			pEmptyVolume[ index ] = 0.f;
		}
	}
}
//---------------------------------------------------------------------------------------------------

} // namespace GvStructure

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvDataProductionManagerKernel.inl"

#endif // !_GV_DATA_PRODUCTION_MANAGER_KERNEL_H_

