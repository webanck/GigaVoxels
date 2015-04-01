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

#ifndef _GPU_TRIANGLE_PRODUCER_BVH_H_
#define _GPU_TRIANGLE_PRODUCER_BVH_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/Array3D.h>
//#include <GvCore/Array3DGPULinear.h>
#include <GvCore/GPUPool.h>
#include <GvCore/GvIProvider.h>
//#include <GvCore/IProviderKernel.h>
//#include <gigavoxels/cache/GPUCacheHelper.h>

// Project
#include "IBvhTreeProviderKernel.h"
#include "BvhTreeCacheHelper.h"
#include "BVHTrianglesManager.h"
#include "GPUTriangleProducerBVH.hcu"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

using namespace gigavoxels; // FIXME

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class GPUTriangleProducerBVH
 *
 * @brief The GPUTriangleProducerBVH class provides the mecanisms to produce data
 * following data requests emitted during N-tree traversal (rendering).
 *
 * It is the main user entry point to produce data from CPU, for instance,
 * loading data from disk or procedurally generating data.
 *
 * This class is implements the mandatory functions of the GvIProvider base class.
 */
template< typename MeshVertexTList, uint DataPageSize, typename BvhTreeType >
class GPUTriangleProducerBVH
:	public GvCore::GvIProvider< 0, GPUTriangleProducerBVH< MeshVertexTList, DataPageSize, BvhTreeType > >
,	public GvCore::GvIProvider< 1, GPUTriangleProducerBVH< MeshVertexTList, DataPageSize, BvhTreeType > >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition of the data pool
	 */
	typedef GvCore::GPUPoolHost< GvCore::Array3D, MeshVertexTList > VertexBufferPoolType;

	/**
	 * Type definition of the associated device-side object
	 */
	typedef GPUTriangleProducerBVHKernel< MeshVertexTList, DataPageSize > KernelProducerType;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Node pool
	 */
	GvCore::Array3D< VolTreeBVHNode >* nodesBufferArray;

	/**
	 * Data pool
	 */
	VertexBufferPoolType* vertexBufferPool;

	/**
	 * Device-side associated object
	 */
	KernelProducerType kernelProducer;

	/**
	 * Triangles manager used to load mesh files
	 */
	BVHTrianglesManager< MeshVertexTList, DataPageSize >* bvhTrianglesManager;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GPUTriangleProducerBVH( BvhTreeType* bvhTree, const std::string& baseFileName )
	{
		// Store data structure
		_bvhTree = bvhTree;

		// Triangles manager's initialization
		bvhTrianglesManager = new BVHTrianglesManager< MeshVertexTList, DataPageSize >();
#if 1
		//bvhTrianglesManager->loadPowerPlant( baseFileName );
		bvhTrianglesManager->loadMesh( baseFileName );
		//bvhTrianglesManager->saveRawMesh( baseFileName );
#else
		bvhTrianglesManager->loadRawMesh( baseFileName );
#endif
		bvhTrianglesManager->generateBuffers( 2 );
		nodesBufferArray = bvhTrianglesManager->getNodeBufferArray();
		vertexBufferPool = bvhTrianglesManager->getVertexBufferPool();

		// Producer's device_side associated object initialization
		kernelProducer.init( (VolTreeBVHNodeStorage*)nodesBufferArray->getGPUMappedPointer(), vertexBufferPool->getKernelPool() );
	}

	/**
	 * Get the triangles manager
	 *
	 * @return the triangles manager
	 */
	BVHTrianglesManager< MeshVertexTList, DataPageSize >* getBVHTrianglesManager()
	{
		return bvhTrianglesManager;
	}

	/**
	 * ...
	 */
	void renderGL()
	{
		bvhTrianglesManager->renderGL();
	}

	/**
	 * ...
	 */
	void renderFullGL()
	{
		bvhTrianglesManager->renderFullGL();
	}
	
	/**
	 * ...
	 */
	void renderDebugGL()
	{
		bvhTrianglesManager->renderDebugGL();
	}

	/**
	 * Produce node tiles
	 *
	 * @param numElems ...
	 * @param nodesAddressCompactList ...
	 * @param elemAddressCompactList ...
	 * @param gpuPool ...
	 * @param pageTable ...
	 */
	template < typename ElementRes, typename GPUPoolType, typename PageTableType >
	inline void produceData( uint numElems,
							thrust::device_vector< uint >* nodesAddressCompactList,
							thrust::device_vector< uint >* elemAddressCompactList,
							GPUPoolType gpuPool, PageTableType pageTable, Loki::Int2Type< 0 > );

	/**
	 * Produce bricks of data
	 *
	 * @param numElems ...
	 * @param nodesAddressCompactList ...
	 * @param elemAddressCompactList ...
	 * @param gpuPool ...
	 * @param pageTable ...
	 */
	template < typename ElementRes, typename GPUPoolType, typename PageTableType >
	inline void produceData( uint numElems,
							thrust::device_vector< uint >* nodesAddressCompactList,
							thrust::device_vector< uint >* elemAddressCompactList,
							GPUPoolType gpuPool, PageTableType pageTable, Loki::Int2Type< 1 > );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Data structure
	 */
	BvhTreeType *_bvhTree;
	
	/**
	 * Cache helper
	 */
	BvhTreeCacheHelper _cacheHelper;
};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GPUTriangleProducerBVH.inl"

#endif
