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
__forceinline__ void GvNodeVisitorKernel::visit(
	TVolTreeKernelType& pVolumeTree,
	GPUCacheType& pGpuCache,
	GvStructure::GvNode& pNode,
	const float3 pSamplePosTree,
	const float pConeAperture,
	float& pNodeSizeTree,
	float3& pSampleOffsetInNodeTree,
	GvSamplerKernel<TVolTreeKernelType>& pBrickSampler,
	bool& pRequestEmitted
) {
	// Useful variables initialization
	//Warning: works only with uniform subdivision (same resolution in each dimension of the nodes/bricks)
	uint nodeDepth = 0U;
	float3 nodePosTree = make_float3(0.f);
	pNodeSizeTree = 2.f;//TVolTreeKernelType::NodeResolution::maxRes;//The volume tree is centered on the center of the root node, each edge node of the first division beeing of length 1 //what about k_voxelSizeMultiplier ?;
	float nodeSizeTreeInv = 1.f/pNodeSizeTree;
	float voxelSizeTree = pNodeSizeTree/static_cast<float>(TVolTreeKernelType::BrickResolution::maxRes);

	uint brickChildAddressEnc  = 0U;
	uint brickParentAddressEnc = 0U;

	float3 brickChildNormalizedOffset 	= make_float3(0.f);
	float brickChildNormalizedScale  	= 1.f;

	// Initialize the address of the first node in the "node pool".
	// While traversing the data structure, this address will be
	// updated to the one associated to the current traversed node.
	// It will be used to fetch info of the node stored in then "node pool".
	uint nodeTileAddress = pVolumeTree._rootAddress;

	//A boolean to check if the final requested node's request was a brick request or not.
	bool brick_request;


	// Traverse the data structure from root node
	// until a descent criterion is not fulfilled anymore.
	bool descentSizeCriteria;
	do {
		// [ 1 ] - Update size parameters
		pNodeSizeTree		/= static_cast<float>(TVolTreeKernelType::NodeResolution::maxRes);	// current node size
		voxelSizeTree		/= static_cast<float>(TVolTreeKernelType::NodeResolution::maxRes);	// current voxel size
		nodeSizeTreeInv		*= static_cast<float>( TVolTreeKernelType::NodeResolution::maxRes);	// current node resolution (nb nodes in a dimension)

		// [ 2 ] - Update node info
		//
		// The goal is to fetch info of the current traversed node from the "node pool"
		const uint3 nodeChildCoordinates = make_uint3( nodeSizeTreeInv * ( pSamplePosTree - nodePosTree ) );
		const uint nodeChildAddressOffset = TVolTreeKernelType::NodeResolution::toFloat1( nodeChildCoordinates );// & nodeChildAddressMask;
		const uint nodeAddress = nodeTileAddress + nodeChildAddressOffset;
		nodePosTree = nodePosTree + pNodeSizeTree * make_float3( nodeChildCoordinates );
		// Try to retrieve node from the node pool given its address
		//pVolumeTree.fetchNode( pNode, nodeTileAddress, nodeChildAddressOffset );
		pVolumeTree.fetchNode( pNode, nodeAddress );

		// Update brick info
		if ( brickChildAddressEnc )
		{
			brickParentAddressEnc = brickChildAddressEnc;
			brickChildNormalizedScale  = 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );		// 0.5f;
			brickChildNormalizedOffset = brickChildNormalizedScale * make_float3( nodeChildCoordinates );
		}
		else
		{
			brickChildNormalizedScale  *= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );	// 0.5f;
			brickChildNormalizedOffset += brickChildNormalizedScale * make_float3( nodeChildCoordinates );
		}
		brickChildAddressEnc = pNode.hasBrick() ? pNode.getBrickAddressEncoded() : 0;

		// Update descent condition
		descentSizeCriteria = ( voxelSizeTree > pConeAperture ) && ( nodeDepth < k_maxVolTreeDepth );

		// ---- Flag used data (the traversed one) ----

		// Set current node as "used"
		pGpuCache._nodeCacheManager.setElementUsage(nodeTileAddress);
		// Set current brick as "used"
		if(pNode.hasBrick()) pGpuCache._brickCacheManager.setElementUsage(pNode.getBrickAddress());

		// ---- Emit requests if needed (node subdivision or brick loading/producing) ----

		// Process requests based on traversal strategy (priority on bricks or nodes)
		if ( priorityOnBrick )
		{
			// Low resolution first
			if ( ( pNode.isBrick() && !pNode.hasBrick() ) || !( pNode.isInitializated() ) )
			{
				pGpuCache.loadRequest( nodeAddress );
				brick_request = true;
				pRequestEmitted = true;
			}
			else if ( !pNode.hasSubNodes() && descentSizeCriteria && !pNode.isTerminal() )
			{
				pGpuCache.subDivRequest( nodeAddress );
				pRequestEmitted = true;
			}
		}
		else
		{	 // High resolution immediatly
			if ( descentSizeCriteria && !pNode.isTerminal() )
			{
				if ( ! pNode.hasSubNodes() )
				{
					pGpuCache.subDivRequest( nodeAddress );
					pRequestEmitted = true;
				}
			}
			else if ( ( pNode.isBrick() && !pNode.hasBrick() ) || !( pNode.isInitializated() ) )
			{
				pGpuCache.loadRequest( nodeAddress );
				brick_request = true;
				pRequestEmitted = true;
			}
		}

		nodeTileAddress = pNode.getChildAddress().x;

		// Update octree depth
		nodeDepth++;
	} while(descentSizeCriteria && pNode.hasSubNodes());	// END of the data structure traversal

	// Compute sample offset in node tree
	pSampleOffsetInNodeTree = pSamplePosTree - nodePosTree;

	// Update brickSampler properties
	//
	// The idea is to store useful variables that will ease the rendering process of this node :
	// - brickSampler is just a wrapper on the datapool to be able to fetch data inside
	// - given the previously found node, we store its associated brick address in cache to be able to fetch data in the datapool
	// - we can also store the brick address of the parent node to do linear interpolation of the two level of resolution
	// - for all of this, we store the bottom left position in cache of the associated bricks (note : brick address is a voxel index in the cache)
	if ( pNode.isBrick() )
	{
		pBrickSampler._nodeSizeTree = pNodeSizeTree;
		pBrickSampler._sampleOffsetInNodeTree = pSampleOffsetInNodeTree;
		pBrickSampler._scaleTree2BrickPool = pVolumeTree.brickSizeInCacheNormalized.x / pBrickSampler._nodeSizeTree;

		pBrickSampler._brickParentPosInPool = pVolumeTree.brickCacheResINV * make_float3( GvStructure::GvNode::unpackBrickAddress( brickParentAddressEnc ) )
			+ brickChildNormalizedOffset * pVolumeTree.brickSizeInCacheNormalized.x;

		if ( brickChildAddressEnc )
		{
			// Should be mipmapping here, betwwen level with the parent

			//pBrickSampler.mipMapOn = true; // "true" is not sufficient :  when no parent, program is very slow
			pBrickSampler._mipMapOn = ( brickParentAddressEnc == 0 ) ? false : true;

			pBrickSampler._brickChildPosInPool = make_float3( GvStructure::GvNode::unpackBrickAddress( brickChildAddressEnc ) ) * pVolumeTree.brickCacheResINV;
		}
		else
		{
			// No mipmapping here

			pBrickSampler._mipMapOn = false;
			pBrickSampler._brickChildPosInPool  = pBrickSampler._brickParentPosInPool;
			pBrickSampler._scaleTree2BrickPool *= brickChildNormalizedScale;
		}
	}

	//Check node's neighborhood to be sure to have a "regular volume-tree" in case of subdivision.
	if(pRequestEmitted && nodeDepth > 1U + cAllowedSubdivisionDifference) {
		//A neighbor is retrieved offsetting the requested position in each direction.
		uint dim;
		uint dir;
		for(dim=0U; dim<3U; dim++) for(dir=0U; dir<2U; dir++) {
			float sign = (dir == 0U ? -1.f : 1.f);
			float3 neighborPosTree = pSamplePosTree;

			//Need to check the boundaries depending on the direction.
			bool condition;
			switch(dim) {
				case 0:
					neighborPosTree.x += sign * pNodeSizeTree;
					condition = neighborPosTree.x > 0.f && neighborPosTree.x < 1.f;
					break;
				case 1:
					neighborPosTree.y += sign * pNodeSizeTree;
					condition = neighborPosTree.y > 0.f && neighborPosTree.y < 1.f;
					break;
				case 2:
					neighborPosTree.z += sign * pNodeSizeTree;
					condition = neighborPosTree.z > 0.f && neighborPosTree.z < 1.f;
			}

			//If the neighboor is inside the volume-tree, try to descend the structure until the node is a leaf or is deep enough regarding the original node (avoid a depth level difference of more than 1).
			//TODO: check more precisely about the subdivion (in case of nodes subdividing in more than 2 in each dimension or about the bricks dimensions in comparison with the nodes).
			if(condition) {
				uint neighborNodeDepth = 0U;
				float3 neighborNodePosTree = make_float3(0.f);
				float neighborNodeSizeTree = TVolTreeKernelType::NodeResolution::maxRes;
				float neighborNodeSizeTreeInv = 1.f/neighborNodeSizeTree;

				uint neighborNodeTileAddress = pVolumeTree._rootAddress;

				GvStructure::GvNode neighborNode;
				uint nodeAddress = 0U;

				//The descent.
				do {
					neighborNodeSizeTree /= static_cast<float>(TVolTreeKernelType::NodeResolution::maxRes);
					neighborNodeSizeTreeInv	*= static_cast<float>(TVolTreeKernelType::NodeResolution::maxRes);

					const uint3 nodeChildCoordinates = make_uint3(neighborNodeSizeTreeInv * (neighborPosTree - neighborNodePosTree));
					const uint nodeChildAddressOffset = TVolTreeKernelType::NodeResolution::toFloat1( nodeChildCoordinates);
					nodeAddress = neighborNodeTileAddress + nodeChildAddressOffset;

					pVolumeTree.fetchNode(neighborNode, nodeAddress);

					pGpuCache._nodeCacheManager.setElementUsage(neighborNodeTileAddress);
					if(neighborNode.hasBrick()) pGpuCache._brickCacheManager.setElementUsage(neighborNode.getBrickAddress());

					neighborNodeTileAddress = neighborNode.getChildAddress().x;

					neighborNodeDepth++;
				} while(neighborNodeDepth < nodeDepth - (cAllowedSubdivisionDifference - 1U) && neighborNode.hasSubNodes());

				//If the node and the neighbor node have a depth level difference, request a subdivision.
				if(neighborNodeDepth < nodeDepth - (cAllowedSubdivisionDifference - 1U)) {
					pGpuCache.subDivRequest(nodeAddress);
					//Debug variable to monitor the occurences of the regulararisation process.
					cRegularisationNb++;
				//And even if the nodes are on the same level, check if the neighbor needs it's brick.
				} else if(
					neighborNodeDepth == nodeDepth &&
					brick_request &&
					((neighborNode.isBrick() && !neighborNode.hasBrick()) || !neighborNode.isInitializated())
				) {
					pGpuCache.loadRequest(nodeAddress);
					//Debug variable to monitor the occurences of the regulararisation process.
					cRegularisationNb++;
				}
			}
		}
	}
}

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
__forceinline__ void GvNodeVisitorKernel::visit(
	VolumeTreeKernelType& pVolumeTree,
	uint pMaxDepth,
	float3 pSamplePos,
	uint pNodeTileAddress,
	GvStructure::GvNode& pNode,
	float& pNodeSize,
	float3& pNodePos,
	uint& pNodeDepth,
	uint& pBrickAddressEnc,
	float3& pBrickPos,
	float& pBrickScale
) {
	////descent////

	float nodeSizeInv = 1.0f;

	// WARNING uint nodeAddress;
	pBrickAddressEnc = 0;

	// Descent in volume tree until max depth is reach or current traversed node has no subnodes
	int i = 0;
	do {
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
	} while((i < pMaxDepth) && pNode.hasSubNodes());

	pNodeDepth = i;
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
 *
 * @return the depth of the node
 ******************************************************************************/
template< class TVolTreeKernelType >
__device__
__forceinline__ uint GvNodeVisitorKernel
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
		nodeDepth++;
	}
	while ( ( nodeDepth <= pMaxNodeDepth ) && pNode.hasSubNodes() );	// END of the data structure traversal

	return nodeDepth;
}

} // namespace GvRendering
