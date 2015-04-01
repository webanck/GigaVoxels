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

#ifndef _GV_NODE_VISITOR_KERNEL_H_
#define _GV_NODE_VISITOR_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/vector_types_ext.h"
#include "GvStructure/GvVolumeTreeKernel.h"
#include "GvStructure/GvNode.h"
#include "GvRendering/GvSamplerKernel.h"

// Cuda
#include <host_defines.h>
#include <vector_types.h>

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

namespace GvRendering
{
// Float descent
//#define FLOAT_DESCENT

// Cache type
//#define CACHE_NODEVISITOR_SIMPLE
//#define CACHE_NODEVISITOR_FULL
//#define GRAND_FATHER
//#define ANCESTOR

// Cache
#ifdef CACHE_NODEVISITOR_SIMPLE
struct Node {
	GvStructure::GvNode node;
	uint address;
};
#define CACHE_SIZE ( 6u * 1024u / sizeof( GvRendering::Node ))
#endif // CACHE_NODEVISITOR_SIMPLE

#ifdef CACHE_NODEVISITOR_FULL
#define CACHE_SIZE 128
#define CACHE_DEPTH 4
struct Node {
	uint nodeTileAddress;

	float3 nodePosTree;

	uint brickChildAddressEnc;
	uint brickParentAddressEnc;
	float3 brickChildNormalizedOffset;
	float brickChildNormalizedScale;
};
#endif // CACHE_NODEVISITOR_FULL

#if defined( CACHE_NODEVISITOR_SIMPLE ) && defined( CACHE_NODEVISITOR_FULL )
#error
#endif

#if defined( CACHE_NODEVISITOR_SIMPLE ) || defined( CACHE_NODEVISITOR_FULL )
#define CACHE_NODEVISITOR
//#define CACHE_SIZE ( 6u * 1024u / sizeof( GvRendering::Node ))
#endif

// Grand father
#ifdef GRAND_FATHER
#define NTHREADS 128
struct GrandFatherInfo {
	uint grandFather; // indicate whether the grand father is 0 or 1 in arrays

	uint nodeDepth[2];

	//float nodeSize[2];

	uint nodeTileAddress[2];

	float3 nodePosTree[2];

	uint brickChildAddressEnc[2];
	uint brickParentAddressEnc[2];
	float3 brickChildNormalizedOffset[2];
	float brickChildNormalizedScale[2];
};

#endif // GRAND_FATHER

#ifdef ANCESTOR
//#define NTHREADS 128
#define ANCESTOR_NUMBER 4 // we keep the nth ancestor
struct AncestorInfo {
	float3 nodePosTree;
	uint nodeTileAddress;

	uchar nodeDepth;

	uint brickChildAddressEnc;
	uint brickParentAddressEnc;
	float brickChildNormalizedScale;
	float3 brickChildNormalizedOffset;
};
#endif  // ANCESTOR

#if defined( GRAND_FATHER ) && defined( CACHE_NODEVISITOR_FULL )
#error ERROR
#endif

/**
 * @class GvNodeVisitorKernel
 *
 * @brief The GvNodeVisitorKernel class provides ...
 *
 * @ingroup GvRendering
 *
 * ...
 */
class GvNodeVisitorKernel
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
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
	 * @param pRequestEmitted a returned flag to tell whether or not a request has been emitted during descent
	 */
template< bool priorityOnBrick, class TVolTreeKernelType, class GPUCacheType >
__device__
static __forceinline__ void visit( TVolTreeKernelType& pVolumeTree, GPUCacheType& pGpuCache, GvStructure::GvNode& pNode,
			const float3 pSamplePosTree, const float pConeAperture, float& pNodeSizeTree, float3& pSampleOffsetInNodeTree,
			GvSamplerKernel< TVolTreeKernelType >& pBrickSampler, bool& pRequestEmitted
#ifdef CACHE_NODEVISITOR
			, Node *cache
#endif // CACHE_NODEVISITOR
#ifdef GRAND_FATHER
		, GrandFatherInfo *grandFatherInfo
#endif // GRAND_FATHER
#ifdef ANCESTOR
		, AncestorInfo *__restrict__ ancestorInfo
#endif // ANCESTOR
			);
	/**
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
	 */
	template< class VolumeTreeKernelType >
	__device__
	static __forceinline__ void visit( VolumeTreeKernelType& pVolumeTree, uint pMaxDepth, float3 pSamplePos,
										uint pNodeTileAddress, GvStructure::GvNode& pNode, float& pNodeSize, float3& pNodePos, uint& pNodeDepth,
										uint& pBrickAddressEnc, float3& pBrickPos, float& pBrickScale );

	/**
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
	 */
	template< class TVolTreeKernelType >
	__device__
	static __forceinline__ void getNodeFather( TVolTreeKernelType& pVolumeTree, GvStructure::GvNode& pNode, const float3 pSamplePosTree, const uint pMaxNodeDepth );

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

} // namespace GvRendering

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvNodeVisitorKernel.inl"

#endif // !_GV_NODE_VISITOR_KERNEL_H_
