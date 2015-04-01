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

#ifndef BVHTrianglesGPUCache_H
#define BVHTrianglesGPUCache_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <cutil.h>
#include <cutil_math.h>
#include <vector_types.h>

// Thrust
#include <thrust/device_vector.h>
#include <thrust/copy.h>

// GigaVoxels
#include <GvCore/GvIProvider.h>
#include <GvCore/RendererTypes.h>
#include <GvRendering/GvRendererHelpersKernel.h>
#include <GvCore/functional_ext.h>

// Project
#include "BvhTreeCache.hcu"
#include "BvhTreeCacheManager.h"

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

////////////////////////////////////////////////////////////////////////////////
//! Volume Tree Cache manager
////////////////////////////////////////////////////////////////////////////////

/** 
 * @struct BvhTreeCache
 *
 * @brief The BvhTreeCache struct provides ...
 *
 * @param BvhTreeType ...
 * @param ProducerType ...
 */
template< typename BvhTreeType, typename ProducerType >
class BvhTreeCache
{
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition for nodes cache manager
	 */
	typedef GPUCacheManager
	<
		GvCore::StaticRes3D< 2, 1, 1 >,
		GvCore::GvIProvider< 0, ProducerType >
	>
	NodesCacheManager;

	/**
	 * Type definition for bricks cache manager
	 */
	typedef GPUCacheManager
	<
		GvCore::StaticRes3D< BVH_DATA_PAGE_SIZE, 1, 1 >,
		GvCore::GvIProvider< 1, ProducerType >
	>
	BricksCacheManager;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Node cache manager
	 */
	NodesCacheManager* nodesCacheManager;
	
	/**
	 * Brick cache manager
	 */
	BricksCacheManager* bricksCacheManager;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param bvhTree BVH tree
	 * @param gpuprod producer
	 * @param voltreepoolres ...
	 * @param nodetileres nodetile resolution
	 * @param brickpoolres brick pool resolution
	 * @param brickRes brick resolution
	 */
	BvhTreeCache( BvhTreeType* bvhTree, ProducerType* gpuprod, uint3 voltreepoolres, uint3 nodetileres,	uint3 brickpoolres, uint3 brickRes );

	/**
	 * Pre-render pass
	 */
	void preRenderPass();
	
	/**
	 * Post-render pass
	 */
	uint handleRequests();

	/**
	 * Clear cache
	 */
	void clearCache();

#if USE_SYNTHETIC_INFO
	/**
	 * ...
	 */
	void clearSyntheticInfo()
	{
		bricksCacheManager->d_SyntheticCacheStateBufferArray->fill(0);
	}
#endif

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

	/**
	 * BVH tree (data structure)
	 */
	BvhTreeType* _bvhTree;

	/**
	 * Producer
	 */
	ProducerType* _bvhProducer;

	/**
	 * Node pool resolution
	 */
	uint3 _nodePoolRes;
	
	/**
	 * Brick pool resolution
	 */
	uint3 _brickPoolRes;

	/**
	 * Global buffer of requests for each node
	 */
	GvCore::Array3DGPULinear< uint >* d_UpdateBufferArray;	// unified path
	
	/**
	 * Buffer of requests containing only valid requests (hence the name compacted)
	 */
	thrust::device_vector< uint >* d_UpdateBufferCompactList;

	/**
	 * ...
	 */
	uint numNodeTilesNotInUse;
	
	/**
	 * ...
	 */
	uint numBricksNotInUse;

	/**
	 * ...
	 */
	uint totalNumBricksLoaded;
	
	// CUDPP
	/**
	 * ...
	 */
	size_t* d_numElementsPtr;
	
	// CUDPP
	/**
	 * ...
	 */
	CUDPPHandle scanplan;

	/******************************** METHODS *********************************/

	/**
	 * Update all needed symbols in constant memory
	 */
	void updateSymbols();

	/**
	 * Update time stamps
	 */
	void updateTimeStamps();

	/**
	 * Manage updates
	 *
	 * @return ...
	 */
	uint manageUpdates();

	/**
	 * Manage the node subdivision requests
	 *
	 * @param pNumUpdateElems number of elements to process
	 *
	 * @return ...
	 */
	uint manageSubDivisions( uint pNumUpdateElems );
	
	/**
	 * Manage the brick load/produce requests
	 *
	 * @param pNumUpdateElems number of elements to process
	 *
	 * @return ...
	 */
	uint manageDataLoadGPUProd( uint pNumUpdateElems );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "BvhTreeCache.inl"

////////////////////////////////////////////////////////////////////////////////
//! Volume Tree Cache manager
////////////////////////////////////////////////////////////////////////////////

/******************************************************************************
 * Clear cache
 ******************************************************************************/
template< typename BvhTreeType, typename ProducerType >
void BvhTreeCache< BvhTreeType, ProducerType >
::clearCache()
{
	//volTree->clear();
	//volTree->initCache(gpuProducer->getBVHTrianglesManager());

	//volTreeCacheManager->clearCache();
	//bricksCacheManager->clearCache();
}

#endif // !BVHTrianglesGPUCache_H
