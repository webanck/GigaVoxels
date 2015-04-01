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
//#include <GvCore/IProviderKernel.h>
//#include <gigavoxels/cache/GPUCacheHelper.h>

#include <GvCore/GvProvider.h>

// Project
#include "IBvhTreeProviderKernel.h"
#include "BvhTreeCacheHelper.h"
#include "BVHTrianglesManager.h"
#include "GPUTriangleProducerBVHKernel.h"

// STL
#include <string>

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
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
class GPUTriangleProducerBVH : public GvCore::GvProvider< TDataStructureType, TDataProductionManager >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition of the data pool
	 */
	typedef typename TDataStructureType::DataTypeList DataTypeList;

	/**
	 * Type definition of the node pool type
	 */
	typedef typename TDataStructureType::NodePoolType NodePoolType;

	/**
	 * Type definition of the data pool type
	 */
	typedef typename TDataStructureType::DataPoolType DataPoolType;

	/**
	 * Nodes buffer type
	 */
	typedef GvCore::Array3D< VolTreeBVHNode > NodesBufferType;

	/**
	 * Type definition of the data pool
	 */
	typedef GvCore::GPUPoolHost< GvCore::Array3D, DataTypeList > DataBufferType;

	/**
	 * Type definition of the associated device-side object
	 */
	typedef GPUTriangleProducerBVHKernel< TDataStructureType, DataPageSize > KernelProducerType;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Node pool
	 */
	NodesBufferType* _nodesBuffer;

	/**
	 * Data pool
	 */
	DataBufferType* _dataBuffer;

	/**
	 * Device-side associated object
	 */
	KernelProducerType _kernelProducer;

	/**
	 * Triangles manager used to load mesh files
	 */
	BVHTrianglesManager< DataTypeList, DataPageSize >* _bvhTrianglesManager;

	/**
	 * Mesh/scene filename
	 */
	std::string _filename;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GPUTriangleProducerBVH();

	/**
	 * Destructor
	 */
	virtual ~GPUTriangleProducerBVH();

	/**
	 * Initialize
	 *
	 * @param pDataStructure data structure
	 * @param pDataProductionManager data production manager
	 */
	virtual void initialize( TDataStructureType* pDataStructure, TDataProductionManager* pDataProductionManager );

	/**
	 * Finalize
	 */
	virtual void finalize();

	/**
	 * Get the triangles manager
	 *
	 * @return the triangles manager
	 */
	BVHTrianglesManager< DataTypeList, DataPageSize >* getBVHTrianglesManager();

	/**
	 * This method is called by the cache manager when you have to produce data for a given pool.
	 *
	 * @param pNumElems the number of elements you have to produce.
	 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
	 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
	 */
	inline virtual void produceData( uint pNumElems,
										thrust::device_vector< uint >* pNodesAddressCompactList,
										thrust::device_vector< uint >* pElemAddressCompactList,
										Loki::Int2Type< 0 > );

	/**
	 * This method is called by the cache manager when you have to produce data for a given pool.
	 *
	 * @param pNumElems the number of elements you have to produce.
	 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
	 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
	 */
	inline virtual void produceData( uint pNumElems,
										thrust::device_vector< uint >* pNodesAddressCompactList,
										thrust::device_vector< uint >* pElemAddressCompactList,
										Loki::Int2Type< 1 > );

	/**
	 * ...
	 */
	void renderGL();

	/**
	 * ...
	 */
	void renderFullGL();
	
	/**
	 * ...
	 */
	void renderDebugGL();

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
	 * Cache helper
	 */
	BvhTreeCacheHelper _cacheHelper;

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GPUTriangleProducerBVH( const GPUTriangleProducerBVH& );

	/**
	 * Copy operator forbidden.
	 */
	GPUTriangleProducerBVH& operator=( const GPUTriangleProducerBVH& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GPUTriangleProducerBVH.inl"

#endif
