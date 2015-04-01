
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
#include "GvRendering/GvRendererHelpersKernel.h"
#include "GvRendering/GvRendererContext.h"
#include "GvPerfMon/GvPerformanceMonitor.h"

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvRendering

{
/**
 * Fast absolute value for float3 using bit level arithmetic
 */
__forceinline__ __device__
float3 __absf( float3 f )
{
	f.x = __int_as_float( __float_as_int( f.x ) & 0x7fffffff );
	f.y = __int_as_float( __float_as_int( f.y ) & 0x7fffffff );
	f.z = __int_as_float( __float_as_int( f.z ) & 0x7fffffff );

	return f;
}

#ifdef CACHE_NODEVISITOR_FULL
__forceinline__ __device__
// Hash valid only if 2^(nodeDepth + 1) < CACHE_SIZE
uint computeHash( uint3 nodeCoordinates, uint nodeDepth ) {
	return ( nodeCoordinates.x + nodeCoordinates.y + nodeCoordinates.z ) % CACHE_SIZE;
	//if( nodeDepth <= CACHE_SIZE ) {
	//	uint pow = 1u << ( static_cast< int >( nodeDepth ) + 1 );
	//	return ( nodeCoordinates.x + nodeCoordinates.y + nodeCoordinates.z ) % pow + pow;
	//} else {
	//	return 0;
	//}
}


template< bool priorityOnBrick, class TVolTreeKernelType, class GPUCacheType >
__device__
__forceinline__ void GvNodeVisitorKernel
::visit( TVolTreeKernelType& pVolumeTree, GPUCacheType& pGpuCache, GvStructure::GvNode& pNode,
		const float3 pSamplePosTree, const float pConeAperture, float& pNodeSizeTree, float3& pSampleOffsetInNodeTree,
		GvSamplerKernel< TVolTreeKernelType >& pBrickSampler, bool& pRequestEmitted
		, Node *cache
		)
{
	// Useful variables initialization
	const uint maxDepth = 32u; // TODO 32 => constant ?
	const float scale = static_cast< float >( 1u << ( maxDepth - 1u ));
	uint3 initialSamplePosValue = make_uint3( pSamplePosTree * scale );

	// Size of node and voxel
	const float nodeRes = static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
	const float brickRes = static_cast< float >( TVolTreeKernelType::BrickResolution::maxRes );

	//uint optimalAncestorDepth = static_cast< uint >( - ( __logf( brickRes ) + __logf( pConeAperture )) * 1.f / __logf(nodeRes) - 1 - ANCESTOR_NUMBER + .5f );
	//uint nodeDepth = fminf( ANCESTOR_NUMBER, optimalAncestorDepth );
	//if( threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 ) {
	//	printf( "%f,%u ", pConeAperture, optimalDepth );
	//}
	uint nodeDepth = CACHE_DEPTH;
	pNodeSizeTree = nodeRes * __powf( 1.f / nodeRes, nodeDepth );

	// Hash
	uint bitsToShift = maxDepth - 1u - nodeDepth;
	uint3 nodeCoordinates; // Coordinates of the node where the information are.
	nodeCoordinates.x = initialSamplePosValue.x >> bitsToShift;
	nodeCoordinates.y = initialSamplePosValue.y >> bitsToShift;
	nodeCoordinates.z = initialSamplePosValue.z >> bitsToShift;
	uint hash = computeHash( nodeCoordinates, nodeDepth );

	float voxelSizeTree;

	uint brickChildAddressEnc;
	uint brickParentAddressEnc;

	float3 brickChildNormalizedOffset;
	float brickChildNormalizedScale = cache[hash].brickChildNormalizedScale;

	uint nodeTileAddress = cache[hash].nodeTileAddress;
	float3 nodePosTree = cache[hash].nodePosTree;
	//float3 nodePosTree2 = pNodeSizeTree * 1.f / nodeRes * make_float3( nodeCoordinates );

	float halfNode = pNodeSizeTree * 1.f / nodeRes * .5f;
	float3 posInNode = __absf( pSamplePosTree - ( nodePosTree + make_float3( halfNode )));

	if( nodeTileAddress != 0 && brickChildNormalizedScale != -1
			&& ( posInNode.x < halfNode )
			&& ( posInNode.y < halfNode )
			&& ( posInNode.z < halfNode ) ) {
		//&& nodePosTree2.x == nodePosTree.x
		//&& nodePosTree2.y == nodePosTree.y
		//&& nodePosTree2.z == nodePosTree.z ) {
		// If the point is in the cache, load the rest of the data
		brickChildAddressEnc = cache[hash].brickChildAddressEnc;
		brickParentAddressEnc = cache[hash].brickParentAddressEnc;
		brickChildNormalizedOffset = cache[hash].brickChildNormalizedOffset;
	} else {
		// Alas, the point is not in the cache. Restart from the root.
		pNodeSizeTree = nodeRes;
		nodeTileAddress = pVolumeTree._rootAddress;
		nodeDepth = 0;
		bitsToShift = maxDepth - 1u;

		brickChildAddressEnc  = 0u;
		brickParentAddressEnc = 0u;

		brickChildNormalizedOffset = make_float3( 0.0f );
		brickChildNormalizedScale  = 1.0f;
	}

	voxelSizeTree = pNodeSizeTree * 1.f / brickRes;

	//if( nodeDepth > 0 ) {
	//	printf("d:%i ", nodeDepth );
	//	//printf(" %f ", pNodeSizeTree );
	//}
	//if( nodeTileAddress != pVolumeTree._rootAddress ) {
	//	printf( "a:%x ", nodeTileAddress );
	//}

	// Traverse the data structure from root node
	// until a descent criterion is not fulfilled anymore.
	bool descentSizeCriteria;
	do {
		// [ 1 ] - Update size parameters
		// Current node size
		pNodeSizeTree *= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
		// Current voxel size
		voxelSizeTree *= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );

		// [ 2 ] - Update node info
		//
		// The goal is to fetch info of the current traversed node from the "node pool"
		uint3 nodeChildCoordinates;

		nodeChildCoordinates.x = initialSamplePosValue.x >> bitsToShift;
		nodeChildCoordinates.y = initialSamplePosValue.y >> bitsToShift;
		nodeChildCoordinates.z = initialSamplePosValue.z >> bitsToShift;
		--bitsToShift;

		nodePosTree = pNodeSizeTree * make_float3( nodeChildCoordinates );
		hash = computeHash( nodeChildCoordinates, nodeDepth );
		// Check if the node needs to be put in the cache.
		if( nodeDepth == CACHE_DEPTH
				//&& voxelSizeTree > pConeAperture * __powf( static_cast< float >( nodeRes ), 5 )
				&& cache[hash].nodeTileAddress == 0 ) {
			// Only one thread per warp is allowed to write (otherwise, atomic writes
			// take too much time)
			const unsigned int laneID = ( threadIdx.x + blockDim.x * threadIdx.y ) % 32;
			const unsigned int mask = 0xfffffffe << laneID ;
			if (( mask & __ballot( 1 )) == 0 ) {
				// Several warps may be writing at the same time => atomic write
				// If the blocs were smaller (32 threads), this atomic operation would be
				// useless.
				if( atomicCAS( &cache[hash].nodeTileAddress, 0u, nodeTileAddress ) == 0 ){
					cache[hash].nodePosTree = nodePosTree;

					cache[hash].brickChildAddressEnc = brickChildAddressEnc;
					cache[hash].brickParentAddressEnc = brickParentAddressEnc;
					cache[hash].brickChildNormalizedOffset = brickChildNormalizedOffset;
					cache[hash].brickChildNormalizedScale = brickChildNormalizedScale;
				}
			}
		}

		nodeChildCoordinates.x &= 0x1;
		nodeChildCoordinates.y &= 0x1;
		nodeChildCoordinates.z &= 0x1;

		const uint nodeChildAddressOffset = TVolTreeKernelType::NodeResolution::toFloat1( nodeChildCoordinates );
		const uint nodeAddress = nodeTileAddress + nodeChildAddressOffset;

		pVolumeTree.fetchNode( pNode, nodeAddress );

		// Update descent condition
		descentSizeCriteria = ( voxelSizeTree > pConeAperture ) && ( nodeDepth < k_maxVolTreeDepth );

		// Update brick info
		if ( brickChildAddressEnc ) {
			brickParentAddressEnc = brickChildAddressEnc;
			brickChildNormalizedScale  = 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );		// 0.5f;
			brickChildNormalizedOffset = brickChildNormalizedScale * make_float3( nodeChildCoordinates );
		} else {
			brickChildNormalizedScale  *= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );	// 0.5f;
			brickChildNormalizedOffset += brickChildNormalizedScale * make_float3( nodeChildCoordinates );
		}
		brickChildAddressEnc = pNode.hasBrick() ? pNode.getBrickAddressEncoded() : 0;

		// ---- Flag used data (the traversed one) ----
		// Set current node as "used"
		pGpuCache._nodeCacheManager.setElementUsage( nodeTileAddress );

		// Set current brick as "used"
		if ( pNode.hasBrick() ) {
			pGpuCache._brickCacheManager.setElementUsage( pNode.getBrickAddress() );
		}

		// ---- Emit requests if needed (node subdivision or brick loading/producing) ----
		// Process requests based on traversal strategy (priority on bricks or nodes)
		if ( priorityOnBrick ) {
			// Low resolution first
			if ( ( pNode.isBrick() && !pNode.hasBrick() ) || !( pNode.isInitializated() ) ) {
				pGpuCache.loadRequest( nodeAddress );
				pRequestEmitted = true;
			}
			else if ( !pNode.hasSubNodes() && descentSizeCriteria && !pNode.isTerminal() ) {
				pGpuCache.subDivRequest( nodeAddress );
				pRequestEmitted = true;
			}
		} else {	 // High resolution immediately
			if ( descentSizeCriteria && !pNode.isTerminal() ) {
				if ( ! pNode.hasSubNodes() ) {
					pGpuCache.subDivRequest( nodeAddress );
					pRequestEmitted = true;
				}
			} else if ( ( pNode.isBrick() && !pNode.hasBrick() ) || !( pNode.isInitializated() ) ) {
				pGpuCache.loadRequest( nodeAddress );
				pRequestEmitted = true;
			}
		}


		nodeTileAddress = pNode.getChildAddress().x;

		// Update octree depth
		++nodeDepth;

	} while ( descentSizeCriteria && pNode.hasSubNodes() ); // END of the data structure traversal

	// Compute sample offset in node tree
	pSampleOffsetInNodeTree = pSamplePosTree - nodePosTree;

	// Update brickSampler properties
	if ( pNode.isBrick() ) {
		pBrickSampler._nodeSizeTree = pNodeSizeTree;
		pBrickSampler._sampleOffsetInNodeTree = pSampleOffsetInNodeTree;
		pBrickSampler._scaleTree2BrickPool = pVolumeTree.brickSizeInCacheNormalized.x / pBrickSampler._nodeSizeTree;

		//uint3 tmpAddressUnenc = GvStructure::GvNode::unpackBrickAddress( brickParentAddressEnc ) - make_uint3(1, 1, 1);
		//uint tmpAddressEnc = GvStructure::GvNode:ackBrickAddress(tmpAddressUnenc);

		pBrickSampler._brickParentPosInPool = pVolumeTree.brickCacheResINV * make_float3( GvStructure::GvNode::unpackBrickAddress( brickParentAddressEnc))
			+ brickChildNormalizedOffset * pVolumeTree.brickSizeInCacheNormalized.x;

		if ( brickChildAddressEnc ) {
			// Should be mipmapping here, between level with the parent
			//pBrickSampler.mipMapOn = true; // "true" is not sufficient :  when no parent, program is very slow.
			//pBrickSampler._mipMapOn = ( brickParentAddressEnc == 0 ) ? false : true;
			pBrickSampler._mipMapOn = brickParentAddressEnc != 0;

			pBrickSampler._brickChildPosInPool = make_float3( GvStructure::GvNode::unpackBrickAddress( brickChildAddressEnc ) ) * pVolumeTree.brickCacheResINV;
		} else {
			// No mipmapping here
			pBrickSampler._mipMapOn = false;
			pBrickSampler._brickChildPosInPool  = pBrickSampler._brickParentPosInPool;
			pBrickSampler._scaleTree2BrickPool *= brickChildNormalizedScale;
		}
	}
}
#elif defined( ANCESTOR )
template< bool priorityOnBrick, class TVolTreeKernelType, class GPUCacheType >
__device__
__forceinline__ void GvNodeVisitorKernel
::visit( TVolTreeKernelType& pVolumeTree, GPUCacheType& pGpuCache, GvStructure::GvNode& pNode,
		const float3 pSamplePosTree, const float pConeAperture, float& pNodeSizeTree, float3& pSampleOffsetInNodeTree,
		GvSamplerKernel< TVolTreeKernelType >& pBrickSampler, bool& pRequestEmitted
		, AncestorInfo *__restrict__ ancestorInfo
		)
{
	// Number of node per level and voxel per brick
	const float nodeRes = static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
	const float brickRes = static_cast< float >( TVolTreeKernelType::BrickResolution::maxRes );

	// Useful variables initialization
#ifndef FLOAT_DESCENT
	const uint maxDepth = 32u; // TODO 32 => constant ?
	const float scale = static_cast< float >( 1u << ( maxDepth - 1u ));
	uint3 initialSamplePosValue = make_uint3( pSamplePosTree * scale );
#endif //FLOAT_DESCENT

	uint optimalAncestorDepth = static_cast< uint >( - ( __logf( brickRes ) + __logf( pConeAperture )) * __fdividef( 1.f, __logf(nodeRes)) - 1 - ANCESTOR_NUMBER + .5f );

	uint nodeDepth = ancestorInfo->nodeDepth;
#ifndef FLOAT_DESCENT
	uint bitsToShift = maxDepth - 1u - nodeDepth;
#endif // FLOAT_DESCENT


	uint brickChildAddressEnc;
	uint brickParentAddressEnc;

	float3 brickChildNormalizedOffset;
	float brickChildNormalizedScale;

	uint nodeTileAddress = ancestorInfo->nodeTileAddress;
	float3 nodePosTree = ancestorInfo->nodePosTree;
	//pNodeSizeTree = ancestorInfo->nodeSize;
	pNodeSizeTree = __powf( 1.f / nodeRes, nodeDepth ) * nodeRes;

	float halfNode = pNodeSizeTree * 1.f / nodeRes;
	float3 posInNode = __absf( pSamplePosTree - ( nodePosTree + make_float3( halfNode )));
	//float3 posInNode =  pSamplePosTree - ( nodePosTree + make_float3( halfNode ));
	//posInNode.x = fabsf( posInNode.x );
	//posInNode.y = fabsf( posInNode.y );
	//posInNode.z = fabsf( posInNode.z );

	if( nodeTileAddress != 0
			&& ( posInNode.x < halfNode )
			&& ( posInNode.y < halfNode )
			&& ( posInNode.z < halfNode ) ) {
		// If the point is in the cache, load the rest of the data
		brickChildAddressEnc = ancestorInfo->brickChildAddressEnc;
		brickParentAddressEnc = ancestorInfo->brickParentAddressEnc;
		brickChildNormalizedOffset = ancestorInfo->brickChildNormalizedOffset;
		brickChildNormalizedScale = ancestorInfo->brickChildNormalizedScale;

	} else {
		// Alas, the point is not in the cache. Restart from the root.
		pNodeSizeTree = nodeRes;
		nodeTileAddress = pVolumeTree._rootAddress;
		nodeDepth = 0;
#ifdef FLOAT_DESCENT
		nodePosTree = make_float3( 0.f );
#else
		bitsToShift = maxDepth - 1u;
#endif // FLOAT_DESCENT

		brickChildAddressEnc  = 0u;
		brickParentAddressEnc = 0u;

		brickChildNormalizedOffset = make_float3( 0.0f );
		brickChildNormalizedScale  = 1.0f;
	}

	// Traverse the data structure from root node
	// until a descent criterion is not fulfilled anymore.
	bool descentSizeCriteria;
	do {
		// [ 1 ] - Update size parameters
		// Current node size
		pNodeSizeTree *= 1.f / nodeRes;
		// Current voxel size
		float voxelSizeTree = pNodeSizeTree * 1.f / brickRes;

		// Update descent condition
		descentSizeCriteria = ( voxelSizeTree > pConeAperture ) && ( nodeDepth < k_maxVolTreeDepth );

		// [ 2 ] - Update node info
		//
		// The goal is to fetch info of the current traversed node from the "node pool"
#ifdef FLOAT_DESCENT
		//uint3 nodeChildCoordinates = make_uint3( __fdividef(( pSamplePosTree - nodePosTree ), pNodeSizeTree ));
		uint3 nodeChildCoordinates;
		nodeChildCoordinates.x = static_cast< uint >( __fdividef(( pSamplePosTree.x - nodePosTree.x), pNodeSizeTree ));
		nodeChildCoordinates.y = static_cast< uint >( __fdividef(( pSamplePosTree.y - nodePosTree.y), pNodeSizeTree ));
		nodeChildCoordinates.z = static_cast< uint >( __fdividef(( pSamplePosTree.z - nodePosTree.z), pNodeSizeTree ));
		nodePosTree = nodePosTree + pNodeSizeTree * make_float3( nodeChildCoordinates );

		uint nodeChildAddressOffset = TVolTreeKernelType::NodeResolution::toFloat1( nodeChildCoordinates );
		uint nodeAddress = nodeTileAddress + nodeChildAddressOffset;
#else // FLOAT_DESCENT
		uint3 nodeChildCoordinates;

		nodeChildCoordinates.x = initialSamplePosValue.x >> bitsToShift;
		nodeChildCoordinates.y = initialSamplePosValue.y >> bitsToShift;
		nodeChildCoordinates.z = initialSamplePosValue.z >> bitsToShift;
		--bitsToShift;

		nodePosTree = pNodeSizeTree * make_float3( nodeChildCoordinates );

		nodeChildCoordinates.x %= 2;
		nodeChildCoordinates.y %= 2;
		nodeChildCoordinates.z %= 2;

		const uint nodeChildAddressOffset = TVolTreeKernelType::NodeResolution::toFloat1( nodeChildCoordinates );
		const uint nodeAddress = nodeTileAddress + nodeChildAddressOffset;
#endif // FLOAT_DESCENT

		pVolumeTree.fetchNode( pNode, nodeAddress );

		// Update brick info
		if ( brickChildAddressEnc ) {
			brickParentAddressEnc = brickChildAddressEnc;
			brickChildNormalizedScale  = 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
			brickChildNormalizedOffset = brickChildNormalizedScale * make_float3( nodeChildCoordinates );
		} else {
			brickChildNormalizedScale  *= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
			brickChildNormalizedOffset += brickChildNormalizedScale * make_float3( nodeChildCoordinates );
		}
		brickChildAddressEnc = pNode.hasBrick() ? pNode.getBrickAddressEncoded() : 0;

		// ---- Flag used data (the traversed one) ----
		// Set current node as "used"
		pGpuCache._nodeCacheManager.setElementUsage( nodeTileAddress );

		// Set current brick as "used"
		if ( pNode.hasBrick() ) {
			pGpuCache._brickCacheManager.setElementUsage( pNode.getBrickAddress() );
		}

		// ---- Emit requests if needed (node subdivision or brick loading/producing) ----
		// Process requests based on traversal strategy (priority on bricks or nodes)
		if ( priorityOnBrick ) {
			// Low resolution first
			if ( ( pNode.isBrick() && !pNode.hasBrick() ) || !( pNode.isInitializated() ) ) {
				pGpuCache.loadRequest( nodeAddress );
				pRequestEmitted = true;
			}
			else if ( !pNode.hasSubNodes() && descentSizeCriteria && !pNode.isTerminal() ) {
				pGpuCache.subDivRequest( nodeAddress );
				pRequestEmitted = true;
			}
		} else {	 // High resolution immediately
			if ( descentSizeCriteria && !pNode.isTerminal() ) {
				if ( ! pNode.hasSubNodes() ) {
					pGpuCache.subDivRequest( nodeAddress );
					pRequestEmitted = true;
				}
			} else if ( ( pNode.isBrick() && !pNode.hasBrick() ) || !( pNode.isInitializated() ) ) {
				pGpuCache.loadRequest( nodeAddress );
				pRequestEmitted = true;
			}
		}

		nodeTileAddress = pNode.getChildAddress().x;

		// Update octree depth
		++nodeDepth;

		// Check if the node can be put in the cache
		if( descentSizeCriteria && pNode.hasSubNodes() && nodeDepth == optimalAncestorDepth ) {
			ancestorInfo->nodeDepth = nodeDepth;
			ancestorInfo->nodeTileAddress = nodeTileAddress;
			ancestorInfo->nodePosTree = nodePosTree;

			ancestorInfo->brickChildAddressEnc = brickChildAddressEnc;
			ancestorInfo->brickParentAddressEnc = brickParentAddressEnc;
			ancestorInfo->brickChildNormalizedOffset = brickChildNormalizedOffset;
			ancestorInfo->brickChildNormalizedScale = brickChildNormalizedScale;
		}

	} while ( descentSizeCriteria && pNode.hasSubNodes() ); // END of the data structure traversal

	// Compute sample offset in node tree
	pSampleOffsetInNodeTree = pSamplePosTree - nodePosTree;

	// Update brickSampler properties
	if ( pNode.isBrick() ) {
		pBrickSampler._nodeSizeTree = pNodeSizeTree;
		pBrickSampler._sampleOffsetInNodeTree = pSampleOffsetInNodeTree;
		pBrickSampler._scaleTree2BrickPool = __fdividef( pVolumeTree.brickSizeInCacheNormalized.x, pBrickSampler._nodeSizeTree);

		//uint3 tmpAddressUnenc = GvStructure::GvNode::unpackBrickAddress( brickParentAddressEnc ) - make_uint3(1, 1, 1);
		//uint tmpAddressEnc = GvStructure::GvNode:ackBrickAddress(tmpAddressUnenc);

		pBrickSampler._brickParentPosInPool = pVolumeTree.brickCacheResINV * make_float3( GvStructure::GvNode::unpackBrickAddress( brickParentAddressEnc))
			+ brickChildNormalizedOffset * pVolumeTree.brickSizeInCacheNormalized.x;

		if ( brickChildAddressEnc ) {
			// Should be mipmapping here, between level with the parent
			//pBrickSampler.mipMapOn = true; // "true" is not sufficient :  when no parent, program is very slow.
			pBrickSampler._mipMapOn = brickParentAddressEnc != 0;

			pBrickSampler._brickChildPosInPool = make_float3( GvStructure::GvNode::unpackBrickAddress( brickChildAddressEnc ) ) * pVolumeTree.brickCacheResINV;
		} else {
			// No mipmapping here
			pBrickSampler._mipMapOn = false;
			pBrickSampler._brickChildPosInPool  = pBrickSampler._brickParentPosInPool;
			pBrickSampler._scaleTree2BrickPool *= brickChildNormalizedScale;
		}
	}
}

#else

/******************************************************************************
 * Descent in data structure (in general octree) until max depth is reach or current traversed node has no subnodes,
 * or cone aperture is greater than voxel size.
 *
 * @param pVolumeTree the data structure
 * @param pGpuCache the cache
 * @param node a node that user has to provide. It will be filled with the final node of the descent
 * @param pSamplePosTree A given position in tree
 * @param pConeAperture A given cone aperture
 * @param pNodeSizeTree the returned node size
 * @param pSampleOffsetInNodeTree the returned sample offset in node tree
 * @param pBrickSampler The sampler object used to sample data in the data structure, it will be initialized after the descent
 * @param pRequestEmitted a returned flag to tell wheter or not a request has been emitted during descent
 ******************************************************************************/
template< bool priorityOnBrick, class TVolTreeKernelType, class GPUCacheType >
__device__
__forceinline__ void GvNodeVisitorKernel
::visit( TVolTreeKernelType& pVolumeTree, GPUCacheType& pGpuCache, GvStructure::GvNode& pNode,
		const float3 pSamplePosTree, const float pConeAperture, float& pNodeSizeTree, float3& pSampleOffsetInNodeTree,
		GvSamplerKernel< TVolTreeKernelType >& pBrickSampler, bool& pRequestEmitted
#ifdef CACHE_NODEVISITOR
		, Node *cache
#endif // CACHE_NODEVISITOR
#ifdef GRAND_FATHER
		, GrandFatherInfo *grandFatherInfo
#endif // GRAND_FATHER
		)
{
	// Useful variables initialization
	//uint nodeDepth = 0;
	//float3 nodePosTree = make_float3( 0.0f );
	//pNodeSizeTree = static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
	//float voxelSizeTree = pNodeSizeTree / static_cast< float >( TVolTreeKernelType::BrickResolution::maxRes );

	//uint brickChildAddressEnc  = 0u;
	//uint brickParentAddressEnc = 0u;

	//float3 brickChildNormalizedOffset = make_float3( 0.0f );
	//float brickChildNormalizedScale  = 1.0f;

	// Initialize the address of the first node in the "node pool".
	// While traversing the data structure, this address will be
	// updated to the one associated to the current traversed node.
	// It will be used to fetch info of the node stored in then "node pool".
	//uint nodeTileAddress = pVolumeTree._rootAddress;

	const float brickRes = static_cast< float >( TVolTreeKernelType::BrickResolution::maxRes );
#ifdef FLOAT_DESCENT
	float nodeSizeTreeInv = 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
#else
	const uint maxDepth = 32u; // TODO 32 => constant ?
	const float scale = static_cast< float >( 1u << ( maxDepth - 1u ));
	uint3 initialSamplePosValue = make_uint3( pSamplePosTree * scale );
#endif // FLOAT_DESCENT

#ifndef GRAND_FATHER
	pNodeSizeTree = static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
	float voxelSizeTree;// = pNodeSizeTree / static_cast< float >( TVolTreeKernelType::BrickResolution::maxRes );
	uint nodeTileAddress = pVolumeTree._rootAddress;
	uint nodeDepth = 0u;

	uint brickChildAddressEnc  = 0u;
	uint brickParentAddressEnc = 0u;

	float3 brickChildNormalizedOffset = make_float3( 0.0f );
	float brickChildNormalizedScale  = 1.0f;
#else
	// Load grand father info
	uint grandFather = grandFatherInfo->grandFather;
	uint father = !grandFather;
	float voxelSizeTree;
	uint nodeTileAddress = grandFatherInfo->nodeTileAddress[grandFather];
	uint nodeDepth = grandFatherInfo->nodeDepth[grandFather];

	uint brickChildAddressEnc;
	uint brickParentAddressEnc;

	float3 brickChildNormalizedOffset;
	float brickChildNormalizedScale;
#endif // GRAND_FATHER

#ifndef FLOAT_DESCENT
#ifdef GRAND_FATHER
	uint bitsToShift = maxDepth - 1u - nodeDepth;
#else
	uint bitsToShift = maxDepth - 1u;
#endif // GRAND_FATHER
#endif // FLOAT_DESCENT

	float3 nodePosTree;
#ifdef GRAND_FATHER

	nodePosTree = grandFatherInfo->nodePosTree[grandFather];

	// Size of node and voxel
	const float maxRes = static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
	pNodeSizeTree = __powf( 1.f / maxRes, nodeDepth - 1 );

	float halfNode = pNodeSizeTree * .5f;
	float3 posInNode = __absf( pSamplePosTree - ( nodePosTree + make_float3( halfNode )));

	if( nodeTileAddress == 0
			|| ( posInNode.x >= halfNode )
			|| ( posInNode.y >= halfNode )
			|| ( posInNode.z >= halfNode ) ) {
		// If the point is not in the grand father, restart from the root.
		pNodeSizeTree = static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
		nodeTileAddress = pVolumeTree._rootAddress;
		nodeDepth = 0;
		bitsToShift = maxDepth - 1u;

		brickChildAddressEnc  = 0u;
		brickParentAddressEnc = 0u;

		brickChildNormalizedOffset = make_float3( 0.0f );
		brickChildNormalizedScale  = 1.0f;
	} else {
		// Else, load the rest of the values

		brickChildAddressEnc = grandFatherInfo->brickChildAddressEnc[grandFather];
		brickParentAddressEnc = grandFatherInfo->brickParentAddressEnc[grandFather];
		brickChildNormalizedOffset = grandFatherInfo->brickChildNormalizedOffset[grandFather];
		brickChildNormalizedScale = grandFatherInfo->brickChildNormalizedScale[grandFather];
	}

	//voxelSizeTree = pNodeSizeTree * 1.f / static_cast< float >( TVolTreeKernelType::BrickResolution::maxRes );

	// Check that the grand father is sometime used.
	//if( nodeDepth > 5 ) {
	//	printf("d:%i ", nodeDepth );
	//	//printf(" %f ", pNodeSizeTree );
	//}
	//if( nodeTileAddress != pVolumeTree._rootAddress ) {
	//	printf( "a:%x ", nodeTileAddress );
	//}
#endif //GRAND_FATHER

	// Traverse the data structure from root node
	// until a descent criterion is not fulfilled anymore.
	bool descentSizeCriteria = true;
	do {
#ifdef GRAND_FATHER
		grandFatherInfo->brickChildAddressEnc[father] = brickChildAddressEnc;
		grandFatherInfo->brickParentAddressEnc[father] = brickParentAddressEnc;
		grandFatherInfo->brickChildNormalizedOffset[father] = brickChildNormalizedOffset;
		grandFatherInfo->brickChildNormalizedScale[father] = brickChildNormalizedScale;
		grandFatherInfo->nodeTileAddress[father] = nodeTileAddress;
		//grandFatherInfo->nodeSize[father] = pNodeSizeTree;
		grandFatherInfo->nodeDepth[father] = nodeDepth;
		grandFatherInfo->nodePosTree[father] = nodePosTree;
#endif // GRAND_FATHER

		// [ 1 ] - Update size parameters
		// Current node size
		pNodeSizeTree *= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
		// Current voxel size
		//voxelSizeTree *= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
		voxelSizeTree = pNodeSizeTree * 1.f / brickRes;
#ifdef FLOAT_DESCENT
		// Current node resolution (nb nodes in a dimension)
		nodeSizeTreeInv *= static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
#endif //FLOAT_DESCENT

		// [ 2 ] - Update node info
		//
		// The goal is to fetch info of the current traversed node from the "node pool"
#ifdef FLOAT_DESCENT

		uint3 nodeChildCoordinates = make_uint3( nodeSizeTreeInv * ( pSamplePosTree - nodePosTree ) );
		const uint nodeChildAddressOffset = TVolTreeKernelType::NodeResolution::toFloat1( nodeChildCoordinates );// & nodeChildAddressMask;
		const uint nodeAddress = nodeTileAddress + nodeChildAddressOffset;
		nodePosTree = nodePosTree + pNodeSizeTree * make_float3( nodeChildCoordinates );
#else // FLOAT_DESCENT
		uint3 nodeChildCoordinates;

		nodeChildCoordinates.x = initialSamplePosValue.x >> bitsToShift;
		nodeChildCoordinates.y = initialSamplePosValue.y >> bitsToShift;
		nodeChildCoordinates.z = initialSamplePosValue.z >> bitsToShift;
		--bitsToShift;

		nodePosTree = pNodeSizeTree * make_float3( nodeChildCoordinates );

		nodeChildCoordinates.x &= 0x1;
		nodeChildCoordinates.y &= 0x1;
		nodeChildCoordinates.z &= 0x1;

		const uint nodeChildAddressOffset = TVolTreeKernelType::NodeResolution::toFloat1( nodeChildCoordinates );
		const uint nodeAddress = nodeTileAddress + nodeChildAddressOffset;
#endif // FLOAT_DESCENT


#ifdef CACHE_NODEVISITOR_SIMPLE
		// Try to retrieve node from the node pool given its address
		//int hash = ( nodeAddress & 0xAAAAAAAA ) % CACHE_SIZE;
		//int hash = ( nodeAddress & 0x55555555 ) % CACHE_SIZE;
		int hash = nodeAddress % CACHE_SIZE;
		Node tmp = cache[hash];

		if(( tmp.address == nodeAddress ) && ( tmp.node.childAddress != 0 )) {
			// Value is already in cache
			pNode = cache[hash].node;

			// Update brick info
			if ( brickChildAddressEnc ) {
				brickParentAddressEnc = brickChildAddressEnc;
				brickChildNormalizedScale  = 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );		// 0.5f;
				brickChildNormalizedOffset = brickChildNormalizedScale * make_float3( nodeChildCoordinates );
			} else {
				brickChildNormalizedScale  *= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );	// 0.5f;
				brickChildNormalizedOffset += brickChildNormalizedScale * make_float3( nodeChildCoordinates );
			}
			brickChildAddressEnc = pNode.hasBrick() ? pNode.getBrickAddressEncoded() : 0;

			descentSizeCriteria = ( voxelSizeTree > pConeAperture ) && ( nodeDepth < k_maxVolTreeDepth );
			// Update octree depth
			++nodeDepth;

			// Check loaded value
			//GvStructure::GvNode nodeTmp;
			//pVolumeTree.fetchNode( nodeTmp, nodeAddress );
			//if ( pNode.childAddress != nodeTmp.childAddress
			//  || pNode.brickAddress != nodeTmp.brickAddress ) {
			//	printf("R-%x : %x!=%x, %x!=%x\n",nodeAddress, pNode.childAddress, nodeTmp.childAddress, pNode.brickAddress, nodeTmp.brickAddress);
			//}

			//pVolumeTree.fetchNode( pNode, nodeAddress );
		} else {
			// Value was not in cache
#endif // CACHE_NODEVISITOR_SIMPLE
			pVolumeTree.fetchNode( pNode, nodeAddress );

			// Update descent condition
			descentSizeCriteria = ( voxelSizeTree > pConeAperture ) && ( nodeDepth < k_maxVolTreeDepth );
			// Update octree depth
			++nodeDepth;
#ifdef CACHE_NODEVISITOR_SIMPLE
			// Put node in cache if needed
			if( descentSizeCriteria && pNode.hasSubNodes() && tmp.address == 0 ) {
				// Only one write per warp: search the active warp with the highest id.
				// If the blocs become bigger, it may be interesting to loop to ensure that all
				// node are written. With the current size of block (64), this doesn't happen
				// often enough to be worth it.
				const unsigned int laneID = ( threadIdx.x + blockDim.x * threadIdx.y ) % 32;
				const unsigned int mask = 0xfffffffe << laneID ;
				if (( mask & __ballot( 1 )) == 0 ) {
					// Several warps may be writing at the same time => atomic write
					// If the blocs were smaller (32 threads), this atomic operation would be
					// useless.
					if( atomicCAS( &cache[hash].address, (uint)(0), nodeAddress ) == 0 ){
						cache[hash].node.brickAddress = pNode.brickAddress;
						cache[hash].node.childAddress = pNode.childAddress;
					}
				}

				// Check what was written
				//if( cache[hash].address == nodeAddress
				//	&& (cache[hash].node.brickAddress != pNode.brickAddress
				//		|| cache[hash].node.childAddress != pNode.childAddress )) {
				//	printf("W-%x : %x!=%x, %x!=%x\n", nodeAddress, pNode.childAddress, cache[hash].node.childAddress, pNode.brickAddress, cache[hash].node.brickAddress);
				//}
			}
#endif // CACHE_NODEVISITOR_SIMPLE

			// Update brick info
			if ( brickChildAddressEnc ) {
				brickParentAddressEnc = brickChildAddressEnc;
				brickChildNormalizedScale  = 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );		// 0.5f;
				brickChildNormalizedOffset = brickChildNormalizedScale * make_float3( nodeChildCoordinates );
			} else {
				brickChildNormalizedScale  *= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );	// 0.5f;
				brickChildNormalizedOffset += brickChildNormalizedScale * make_float3( nodeChildCoordinates );
			}
			brickChildAddressEnc = pNode.hasBrick() ? pNode.getBrickAddressEncoded() : 0;

			// ---- Flag used data (the traversed one) ----
			// Set current node as "used"
			pGpuCache._nodeCacheManager.setElementUsage( nodeTileAddress );

			// Set current brick as "used"
			if ( pNode.hasBrick() ) {
				pGpuCache._brickCacheManager.setElementUsage( pNode.getBrickAddress() );
			}

			// ---- Emit requests if needed (node subdivision or brick loading/producing) ----
			// Process requests based on traversal strategy (priority on bricks or nodes)
			if ( priorityOnBrick ) {
				// Low resolution first
				if ( ( pNode.isBrick() && !pNode.hasBrick() ) || !( pNode.isInitializated() ) ) {
					pGpuCache.loadRequest( nodeAddress );
					pRequestEmitted = true;
				}
				else if ( !pNode.hasSubNodes() && descentSizeCriteria && !pNode.isTerminal() ) {
					pGpuCache.subDivRequest( nodeAddress );
					pRequestEmitted = true;
				}
			} else {	 // High resolution immediatly
				if ( descentSizeCriteria && !pNode.isTerminal() ) {
					if ( ! pNode.hasSubNodes() ) {
						pGpuCache.subDivRequest( nodeAddress );
						pRequestEmitted = true;
					}
				} else if ( ( pNode.isBrick() && !pNode.hasBrick() ) || !( pNode.isInitializated() ) ) {
					pGpuCache.loadRequest( nodeAddress );
					pRequestEmitted = true;
				}
			}
#ifdef CACHE_NODEVISITOR_SIMPLE
		}
#endif // CACHE_NODEVISITOR_SIMPLE


#ifdef GRAND_FATHER
		father = !father;
#endif // GRAND_FATHER
		nodeTileAddress = pNode.getChildAddress().x;
	} while ( descentSizeCriteria && pNode.hasSubNodes() ); // END of the data structure traversal

#ifdef GRAND_FATHER
	grandFatherInfo->grandFather = father;
#endif // GRAND_FATHER

	// Compute sample offset in node tree
	pSampleOffsetInNodeTree = pSamplePosTree - nodePosTree;

	// Update brickSampler properties
	if ( pNode.isBrick() ) {
		pBrickSampler._nodeSizeTree = pNodeSizeTree;
		pBrickSampler._sampleOffsetInNodeTree = pSampleOffsetInNodeTree;
		pBrickSampler._scaleTree2BrickPool = pVolumeTree.brickSizeInCacheNormalized.x / pBrickSampler._nodeSizeTree;

		//uint3 tmpAddressUnenc = GvStructure::GvNode::unpackBrickAddress( brickParentAddressEnc ) - make_uint3(1, 1, 1);
		//uint tmpAddressEnc = GvStructure::GvNode:ackBrickAddress(tmpAddressUnenc);

		pBrickSampler._brickParentPosInPool = pVolumeTree.brickCacheResINV * make_float3( GvStructure::GvNode::unpackBrickAddress( brickParentAddressEnc))
			+ brickChildNormalizedOffset * pVolumeTree.brickSizeInCacheNormalized.x;

		if ( brickChildAddressEnc ) {
			// Should be mipmapping here, between level with the parent
			//pBrickSampler.mipMapOn = true; // "true" is not sufficient :  when no parent, program is very slow.
			//pBrickSampler._mipMapOn = ( brickParentAddressEnc == 0 ) ? false : true;
			pBrickSampler._mipMapOn = brickParentAddressEnc != 0;

			pBrickSampler._brickChildPosInPool = make_float3( GvStructure::GvNode::unpackBrickAddress( brickChildAddressEnc ) ) * pVolumeTree.brickCacheResINV;
		} else {
			// No mipmapping here
			pBrickSampler._mipMapOn = false;
			pBrickSampler._brickChildPosInPool  = pBrickSampler._brickParentPosInPool;
			pBrickSampler._scaleTree2BrickPool *= brickChildNormalizedScale;
		}
	}
}
#endif // CACHE_NODEVISITOR_FULL

/******************************************************************************
 * Descent in volume tree until max depth is reach or current traversed node has no subnodes.
 * Perform a descent in a volume tree from a starting node tile address, until a max depth
 * Given a 3D sample position,
 *
 * @param pVolumeTree The volume tree on which descent in done
 * @param pMaxDepth Max depth of the descent
 * @param pSamplePos 3D sample position
 * @param pNodeTileAddress ...
 * @param pNode ...
 * @param pNodeSize ...
 * @param pNodePos ...
 * @param pNodeDepth ...
 * @param pBrickAddressEnc ...
 * @param pBrickPos ...
 * @param pBrickScale ...
 ******************************************************************************/
template< class VolumeTreeKernelType >
__device__
__forceinline__ void GvNodeVisitorKernel
::visit( VolumeTreeKernelType& pVolumeTree, uint pMaxDepth, float3 pSamplePos,
		 uint pNodeTileAddress, GvStructure::GvNode& pNode, float& pNodeSize, float3& pNodePos, uint& pNodeDepth,
		 uint& pBrickAddressEnc, float3& pBrickPos, float& pBrickScale )
{
	////descent////

	float nodeSizeInv = 1.0f;

	// WARNING uint nodeAddress;
	pBrickAddressEnc = 0;

	// Descent in volume tree until max depth is reach or current traversed node has no subnodes
	int i = 0;
	do
	{
		pNodeSize	*= 1.0f / (float)VolumeTreeKernelType::NodeResolution::maxRes;
		nodeSizeInv	*= (float)VolumeTreeKernelType::NodeResolution::maxRes;

		// Retrieve current voxel position
		uint3 curVoxel = make_uint3( nodeSizeInv * ( pSamplePos - pNodePos ) );
		uint curVoxelLinear = VolumeTreeKernelType::NodeResolution::toFloat1( curVoxel );

		float3 nodeposloc = make_float3( curVoxel ) * pNodeSize;
		pNodePos = pNodePos + nodeposloc;

		// Retrieve pNode info (child and data addresses) from pNodeTileAddress address and curVoxelLinear offset
		pVolumeTree.fetchNode( pNode, pNodeTileAddress, curVoxelLinear );

		if ( pNode.hasBrick() )
		{
			pBrickAddressEnc = pNode.getBrickAddressEncoded();
			pBrickPos = make_float3( 0.0f );
			pBrickScale = 1.0f;
		}
		else
		{
			pBrickScale = pBrickScale * 0.5f;
			pBrickPos += make_float3( curVoxel ) * pBrickScale;
		}

		pNodeTileAddress = pNode.getChildAddress().x;
		i++;
	}
	while ( ( i < pMaxDepth ) && pNode.hasSubNodes() );

	pNodeDepth = i;

	//i -= 1;		// <== TODO : don't seem to be used anymore, remove it
}

/******************************************************************************
 * Descent in data structure (in general octree) until max depth is reach or current traversed node has no subnodes,
 * or cone aperture is greater than voxel size.
 *
 * @param pVolumeTree the data structure
 * @param pGpuCache the cache
 * @param node a node that user has to provide. It will be filled with the final node of the descent
 * @param pSamplePosTree A given position in tree
 * @param pConeAperture A given cone aperture
 * @param pNodeSizeTree the returned node size
 * @param pSampleOffsetInNodeTree the returned sample offset in node tree
 * @param pBrickSampler The sampler object used to sample data in the data structure, it will be initialized after the descent
 * @param pRequestEmitted a returned flag to tell wheter or not a request has been emitted during descent
 ******************************************************************************/
template< class TVolTreeKernelType >
__device__
__forceinline__ void GvNodeVisitorKernel
::getNodeFather( TVolTreeKernelType& pVolumeTree, GvStructure::GvNode& pNode, const float3 pSamplePosTree, const uint pMaxNodeDepth )
{
	// Useful variables initialization
	uint nodeDepth = 0;
	float3 nodePosTree = make_float3( 0.0f );
	float nodeSizeTree = static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
	float nodeSizeTreeInv = 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );

	// Initialize the address of the first node in the "node pool".
	// While traversing the data structure, this address will be
	// updated to the one associated to the current traversed node.
	// It will be used to fetch info of the node stored in the "node pool".
	uint nodeTileAddress = pVolumeTree._rootAddress;

	// Traverse the data structure from root node
	// until a descent criterion is not fulfilled anymore.
	do
	{
		// [ 1 ] - Update size parameters
		nodeSizeTree		*= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );	// current node size
		nodeSizeTreeInv		*= static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );			// current node resolution (nb nodes in a dimension)

		// [ 2 ] - Update node info
		//
		// The goal is to fetch info of the current traversed node from the "node pool"
		uint3 nodeChildCoordinates = make_uint3( nodeSizeTreeInv * ( pSamplePosTree - nodePosTree ) );
		uint nodeChildAddressOffset = TVolTreeKernelType::NodeResolution::toFloat1( nodeChildCoordinates );// & nodeChildAddressMask;
		uint nodeAddress = nodeTileAddress + nodeChildAddressOffset;
		nodePosTree = nodePosTree + nodeSizeTree * make_float3( nodeChildCoordinates );
		// Try to retrieve node from the node pool given its address
		//pVolumeTree.fetchNode( pNode, nodeTileAddress, nodeChildAddressOffset );
		pVolumeTree.fetchNode( pNode, nodeAddress );

		nodeTileAddress = pNode.getChildAddress().x;

		// Update depth
		++nodeDepth;
	}
	while ( ( nodeDepth <= pMaxNodeDepth ) && pNode.hasSubNodes() );	// END of the data structure traversal
}

} // namespace GvRendering
