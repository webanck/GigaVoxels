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

using namespace GvRendering;
/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/



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
__forceinline__ void NodeVisitorKernel
::visitWithoutProxy( TVolTreeKernelType& pVolumeTree, GPUCacheType& pGpuCache, GvStructure::GvNode& pNode,
		 const float3 pSamplePosTree, const float pConeAperture, float& pNodeSizeTree, float3& pSampleOffsetInNodeTree,
		 GvRendering::GvSamplerKernel< TVolTreeKernelType >& pBrickSampler, bool& pRequestEmitted )
{
	// Useful variables initialization
	uint nodeDepth = 0;
	float3 nodePosTree = make_float3( 0.0f );
	pNodeSizeTree = static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
	float nodeSizeTreeInv = 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );
	float voxelSizeTree = pNodeSizeTree / static_cast< float >( TVolTreeKernelType::BrickResolution::maxRes );

	uint brickChildAddressEnc  = 0;
	uint brickParentAddressEnc = 0;

	float3 brickChildNormalizedOffset = make_float3( 0.0f );
	float brickChildNormalizedScale  = 1.0f;

	// Initialize the address of the first node in the "node pool".
	// While traversing the data structure, this address will be
	// updated to the one associated to the current traversed node.
	// It will be used to fetch info of the node stored in then "node pool".
	uint nodeTileAddress = pVolumeTree._rootAddress;

	// Traverse the data structure from root node
	// until a descent criterion is not fulfilled anymore.
	bool descentSizeCriteria;
	do
	{
		// [ 1 ] - Update size parameters
		pNodeSizeTree		*= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );	// current node size
		voxelSizeTree		*= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );	// current voxel size
		nodeSizeTreeInv		*= static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );			// current node resolution (nb nodes in a dimension)

		// [ 2 ] - Update node info
		//
		// The goal is to fetch info of the current traversed node from the "node pool"
		uint3 nodeChildCoordinates = make_uint3( nodeSizeTreeInv * ( pSamplePosTree - nodePosTree ) );
		uint nodeChildAddressOffset = TVolTreeKernelType::NodeResolution::toFloat1( nodeChildCoordinates );// & nodeChildAddressMask;
		uint nodeAddress = nodeTileAddress + nodeChildAddressOffset;
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

		// Update octree depth
		nodeDepth++;

		// ---- Flag used data (the traversed one) ----

		// Set current node as "used"
		pGpuCache._nodeCacheManager.setElementUsage( nodeTileAddress );

		// Set current brick as "used"
		if ( pNode.hasBrick() )
		{
			pGpuCache._brickCacheManager.setElementUsage( pNode.getBrickAddress() );
		}

		// ---- Emit requests if needed (node subdivision or brick loading/producing) ----

		// Process requests based on traversal strategy (priority on bricks or nodes)
		if ( priorityOnBrick )
		{
			// Low resolution first						  
			if ( ( pNode.isBrick() && !pNode.hasBrick() ) || !( pNode.isInitializated() ) )
			{
				pGpuCache.loadRequest( nodeAddress );
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
				pRequestEmitted = true;
			}
		}

		nodeTileAddress = pNode.getChildAddress().x;
	}
	while ( descentSizeCriteria && pNode.hasSubNodes() );	// END of the data structure traversal

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
}

inline __device__ float squareLength(float3 r)
{
    return dot(r, r);
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
template< bool priorityOnBrick, class TVolTreeKernelType, class GPUCacheType >
__device__
__forceinline__ void NodeVisitorKernel
::visitWithProxy( TVolTreeKernelType& pVolumeTree, GPUCacheType& pGpuCache, GvStructure::GvNode& pNode,
		 const float3 pSamplePosTree, const float pConeAperture, float& pNodeSizeTree, float3& pSampleOffsetInNodeTree,
		 GvRendering::GvSamplerKernel< TVolTreeKernelType >& pBrickSampler, bool& pRequestEmitted , 
		 uint& pNodeDepth , float3& pNodePosTree , uint& pNodeAddress, float& pNodeSizeTreeInv
#ifdef PROXY_PRINTOUT
		  , uint& skipped_levels
#endif
		 )//,uint& pBrickChildAddressEnc,uint& pBrickParentAddressEnc,float3& pBrickChildNormalizedOffset,float& pBrickChildNormalizedScale)
{
#ifdef PROXY_PRINTOUT
	skipped_levels = 0;
#endif
	
	pNodeSizeTree = (1.f/pNodeSizeTreeInv);
	
	//if (pNodeDepth>0)
	//	printf("hi with level %u on %u\n",pNodeDepth,cNumberOfAncestors );
	/*uint * pNodeDepth_parents = new uint[cAncestorLevel];
	float3  * pNodePosTree_parents = new float3[cAncestorLevel];
	float * pNodeSizeTreeInv_parents = new float[cAncestorLevel];
	uint * pNodeAddress_parents = new uint[cAncestorLevel];*/
	uint pNodeDepth_parents [5];
	float3  pNodePosTree_parents [5];
	float pNodeSizeTreeInv_parents [5];
	uint pNodeAddress_parents [5];


	//bool whereToWrite = 0;
	uint whereToWrite = 0;
	
	uint pBrickChildAddressEnc = 0;
	uint pBrickParentAddressEnc = 0;
	float3 pBrickChildNormalizedOffset = make_float3( 0.0f );
	float pBrickChildNormalizedScale = 1.0f;
	

	float pVoxelSizeTree;
	// Traverse the data structure from root node
	// until a descent criterion is not fulfilled anymore.
	//bool descentSizeCriteria;
	bool descentSizeCriteria2 ;
	//float pNodeSizeTree_parents[2];

	float halfNode = pNodeSizeTree/2.f;
	float3 posInNode = pSamplePosTree-(pNodePosTree+make_float3(halfNode));
	//if (  squareLength(pSamplePosTree-(pNodePosTree + make_float3(pNodeSizeTree ,pNodeSizeTree ,pNodeSizeTree ))) > (pNodeSizeTree)*(pNodeSizeTree)  ) 
	//if ( pNodeDepth == 0 || pSamplePosTree.x<=pNodePosTree.x|| pSamplePosTree.y<=pNodePosTree.y || pSamplePosTree.z<=pNodePosTree.z || pSamplePosTree.x>=pNodePosTree.x+pNodeSizeTree|| pSamplePosTree.y>=pNodePosTree.y+pNodeSizeTree || pSamplePosTree.z>=pNodePosTree.z+pNodeSizeTree)
	if ( pNodeDepth == 0 || (fabs(posInNode.x)>=halfNode) || (fabs(posInNode.y)>=halfNode) || (fabs(posInNode.z)>=halfNode) )
	{
		//printf("(%f,%f,%f) not in (%f,%f,%f) size %f\n",pSamplePosTree.x,pSamplePosTree.y,pSamplePosTree.z,pNodePosTree.x,pNodePosTree.y,pNodePosTree.z,pNodeSizeTree);
		//printf("skip with nodeDepth = %u\n",pNodeDepth);
		pNodeDepth = 0;
		pNodePosTree = make_float3( 0.0f );
		pNodeSizeTree = 1.f;
		pNodeSizeTreeInv = 1.f;
		pNodeAddress = pVolumeTree._rootAddress;
		

			
	pVoxelSizeTree = pNodeSizeTree / TVolTreeKernelType::BrickResolution::maxRes;
			

	

	// [ 2 ] - Update node info
	//
	// The goal is to fetch info of the current traversed node from the "node pool"
	
	//uint3 nodeChildCoordinates = make_uint3(0);// pNodeSizeTreeInv * ( pSamplePosTree - pNodePosTree ) );
	//uint nodeChildAddressOffset = 0;//TVolTreeKernelType::NodeResolution::toFloat1( nodeChildCoordinates );// & nodeChildAddressMask;
	//uint nodeAddress = pNodeAddress;//+ nodeChildAddressOffset;
	
	//pNodePosTree = pNodePosTree + pNodeSizeTree * make_float3( nodeChildCoordinates );

	// Try to retrieve node from the node pool given its address
	//pVolumeTree.fetchNode( pNode, nodeTileAddress, nodeChildAddressOffset );
	pVolumeTree.fetchNode( pNode, pNodeAddress );
	

	
	pBrickChildNormalizedScale  *= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );	// 0.5f;
	//pBrickChildNormalizedOffset += pBrickChildNormalizedScale * make_float3( 0 );
	
	
	pBrickChildAddressEnc = pNode.hasBrick() ? pNode.getBrickAddressEncoded() : 0;
	
	// Update descent condition
	descentSizeCriteria2 = ( pVoxelSizeTree > pConeAperture ) && ( pNodeDepth < k_maxVolTreeDepth );
	
	// Update octree depth
	
	// ---- Flag used data (the traversed one) ----

	// Set current node as "used"
	pGpuCache._nodeCacheManager.setElementUsage( pNodeAddress );

	// Set current brick as "used"
	if ( pNode.hasBrick() )
	{
		pGpuCache._brickCacheManager.setElementUsage( pNode.getBrickAddress() );
	}

	// ---- Emit requests if needed (node subdivision or brick loading/producing) ----

	// Process requests based on traversal strategy (priority on bricks or nodes)
	if ( priorityOnBrick )
	{
		// Low resolution first						  
		if ( ( pNode.isBrick() && !pNode.hasBrick() ) || !( pNode.isInitializated() ) )
		{
			pGpuCache.loadRequest( pNodeAddress );
			pRequestEmitted = true;
		}
		else if ( !pNode.hasSubNodes() && descentSizeCriteria2 && !pNode.isTerminal() )
		{
			pGpuCache.subDivRequest( pNodeAddress );
			pRequestEmitted = true;
		}
	}
	else
	{	 // High resolution immediatly
		if ( descentSizeCriteria2 && !pNode.isTerminal() )
		{
			if ( ! pNode.hasSubNodes() )
			{
				pGpuCache.subDivRequest( pNodeAddress );
				pRequestEmitted = true;
			}
		}
		else if ( ( pNode.isBrick() && !pNode.hasBrick() ) || !( pNode.isInitializated() ) )
		{
			pGpuCache.loadRequest( pNodeAddress );
			pRequestEmitted = true;
		}
	}
	

	} else {
	
		pVolumeTree.fetchNode( pNode, pNodeAddress );
		pVoxelSizeTree = pNodeSizeTree / TVolTreeKernelType::BrickResolution::maxRes;
		//pNodeDepth =  static_cast<uint>(log2f(pNodeSizeTreeInv)); 
		pGpuCache._nodeCacheManager.setElementUsage( pNodeAddress );
		descentSizeCriteria2 = true;
		// Set current brick as "used"
		if ( pNode.hasBrick() )
		{
			pGpuCache._brickCacheManager.setElementUsage( pNode.getBrickAddress() );
		}
#ifdef PROXY_PRINTOUT
		skipped_levels = pNodeDepth;
#endif
		

		
	}


	for (int k  = 1 ; k < cAncestorLevel; k++)
	{
		pNodeAddress_parents[k] = pNodeAddress;
		pNodeSizeTreeInv_parents[k] = pNodeSizeTreeInv;
		pNodeDepth_parents[k] = pNodeDepth ;
		pNodePosTree_parents[k] = pNodePosTree;
	}

			

	while ( descentSizeCriteria2 && pNode.hasSubNodes() )
	{
		
	
		
		
		pNodeAddress_parents[whereToWrite] = pNodeAddress;
		pNodeSizeTreeInv_parents[whereToWrite] = pNodeSizeTreeInv;
		pNodeDepth_parents[whereToWrite] = pNodeDepth ;
		pNodePosTree_parents[whereToWrite] = pNodePosTree;

		whereToWrite++;// =!whereToWrite;
		whereToWrite = whereToWrite==cAncestorLevel? 0 : whereToWrite;
		
		// Update descent condition
		//descentSizeCriteria = ( pVoxelSizeTree > pConeAperture ) && ( pNodeDepth < k_maxVolTreeDepth );
		pNodeDepth++;
		
		
		// [ 1 ] - Update size parameters
		pNodeSizeTree		/= static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );	// current node size
		pVoxelSizeTree		/= static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );	// current voxel size
		pNodeSizeTreeInv	*= static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );			// current node resolution (nb nodes in a dimension)
		
		
		descentSizeCriteria2 = ( pVoxelSizeTree > pConeAperture ) && ( pNodeDepth < k_maxVolTreeDepth );
		
		uint nodeTileAddress = pNode.getChildAddress().x;

		
		// [ 2 ] - Update node info
		//
		// The goal is to fetch info of the current traversed node from the "node pool"
		uint3 nodeChildCoordinates = make_uint3( pNodeSizeTreeInv * ( pSamplePosTree - pNodePosTree ) );
		uint nodeChildAddressOffset = TVolTreeKernelType::NodeResolution::toFloat1( nodeChildCoordinates );// & nodeChildAddressMask;
		pNodeAddress = nodeTileAddress + nodeChildAddressOffset;
		pNodePosTree = pNodePosTree + pNodeSizeTree * make_float3( nodeChildCoordinates );


		
		// Try to retrieve node from the node pool given its address
		//pVolumeTree.fetchNode( pNode, nodeTileAddress, nodeChildAddressOffset );
		pVolumeTree.fetchNode( pNode, pNodeAddress );
		
		// Update brick info
		if ( pBrickChildAddressEnc )
		{
			pBrickParentAddressEnc = pBrickChildAddressEnc;
			pBrickChildNormalizedScale  = 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );		// 0.5f;
			pBrickChildNormalizedOffset = pBrickChildNormalizedScale * make_float3( nodeChildCoordinates );
		}
		else
		{
			pBrickChildNormalizedScale  *= 1.0f / static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );	// 0.5f;
			pBrickChildNormalizedOffset += pBrickChildNormalizedScale * make_float3( nodeChildCoordinates );
		}
		pBrickChildAddressEnc = pNode.hasBrick() ? pNode.getBrickAddressEncoded() : 0;

		// Update octree depth
		//pNodeDepth++;

		// ---- Flag used data (the traversed one) ----

		// Set current node as "used"
		pGpuCache._nodeCacheManager.setElementUsage( nodeTileAddress );

		// Set current brick as "used"
		if ( pNode.hasBrick() )
		{
			pGpuCache._brickCacheManager.setElementUsage( pNode.getBrickAddress() );
		}

		// ---- Emit requests if needed (node subdivision or brick loading/producing) ----

		// Process requests based on traversal strategy (priority on bricks or nodes)
		if ( priorityOnBrick )
		{
			// Low resolution first						  
			if ( ( pNode.isBrick() && !pNode.hasBrick() ) || !( pNode.isInitializated() ) )
			{
				pGpuCache.loadRequest( pNodeAddress );
				pRequestEmitted = true;
			}
			else if ( !pNode.hasSubNodes() && descentSizeCriteria2 && !pNode.isTerminal() )
			{
				pGpuCache.subDivRequest( pNodeAddress );
				pRequestEmitted = true;
			}
		}
		else
		{	 // High resolution immediatly
			if ( descentSizeCriteria2 && !pNode.isTerminal() )
			{
				if ( ! pNode.hasSubNodes() )
				{
					pGpuCache.subDivRequest( pNodeAddress );
					pRequestEmitted = true;
				}
			}
			else if ( ( pNode.isBrick() && !pNode.hasBrick() ) || !( pNode.isInitializated() ) )
			{
				pGpuCache.loadRequest( pNodeAddress );
				pRequestEmitted = true;
			}
		}


	
		
		

	}
	// END of the data structure traversal

	//pNodeSizeTree		*=  static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );	// current node size
	//pVoxelSizeTree		*= static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );	// current voxel size
	//pNodeSizeTreeInv	/= static_cast< float >( TVolTreeKernelType::NodeResolution::maxRes );			// current node resolution (nb nodes in a dimension)



	// Compute sample offset in node tree
	pSampleOffsetInNodeTree = pSamplePosTree - pNodePosTree;

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

		pBrickSampler._brickParentPosInPool = pVolumeTree.brickCacheResINV * make_float3( GvStructure::GvNode::unpackBrickAddress( pBrickParentAddressEnc ) )
			+ pBrickChildNormalizedOffset * pVolumeTree.brickSizeInCacheNormalized.x;
		
		if ( pBrickChildAddressEnc )
		{
			// Should be mipmapping here, betwwen level with the parent

			//pBrickSampler.mipMapOn = true; // "true" is not sufficient :  when no parent, program is very slow
			pBrickSampler._mipMapOn = ( pBrickParentAddressEnc == 0 ) ? false : true;

			pBrickSampler._brickChildPosInPool = make_float3( GvStructure::GvNode::unpackBrickAddress( pBrickChildAddressEnc ) ) * pVolumeTree.brickCacheResINV;
		}
		else
		{
			// No mipmapping here

			pBrickSampler._mipMapOn = false;
			pBrickSampler._brickChildPosInPool  = pBrickSampler._brickParentPosInPool;
			pBrickSampler._scaleTree2BrickPool *= pBrickChildNormalizedScale;
		}
	}



		//pNodeDepth -=2;
		/*if (pNodeDepth_parents[whereToWrite] !=0 && pNodeDepth-pNodeDepth_parents[whereToWrite]==2 && k_maxVolTreeDepth==2)
			printf ("noeud de prof %u enregistre un Gp de prof %u \n",pNodeDepth,pNodeDepth_parents[whereToWrite]);*/
		pNodeDepth = pNodeDepth_parents[whereToWrite];
		
		if (pNodeDepth) 
		{
			pNodePosTree = pNodePosTree_parents[whereToWrite];
			pNodeSizeTreeInv = pNodeSizeTreeInv_parents[whereToWrite];
			pNodeAddress = pNodeAddress_parents[whereToWrite];
		}
	/*	
	delete[] pNodeDepth_parents;
	delete[] pNodePosTree_parents;
	delete[] pNodeSizeTreeInv_parents;
	delete[] pNodeAddress_parents;*/
	//pNodeSizeTree = pNodeSizeTree_parents[1];
	
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
__forceinline__ void NodeVisitorKernel
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
__forceinline__ void NodeVisitorKernel
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
}

