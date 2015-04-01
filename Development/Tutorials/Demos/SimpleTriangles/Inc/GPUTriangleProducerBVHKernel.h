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

#ifndef _GPU_TRIANGLE_PRODUCER_BVH_HCU_
#define _GPU_TRIANGLE_PRODUCER_BVH_HCU_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GPUPool.h>
//#include <GvCore/IntersectionTests.hcu>

//#include "BvhTree.hcu"

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

/** 
 * @class GPUTriangleProducerBVHKernel
 *
 * @brief The GPUTriangleProducerBVHKernel class provides ...
 *
 * ...
 *
 * @param TDataTypeList ...
 * @param TDataPageSize ...
 */
template< typename TDataStructureType, uint TDataPageSize >
class GPUTriangleProducerBVHKernel
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Defines the data type list
	 */
	typedef typename TDataStructureType::DataTypeList DataTypeList;

	///**
	// * Nodes buffer type
	// */
	//typedef GvCore::Array3D< VolTreeBVHNode > NodesBufferType;

	/**
	 * Type definition of the data pool
	 */
	typedef GvCore::GPUPoolKernel< GvCore::Array3DKernelLinear, DataTypeList > DataBufferKernelType;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Nodes buffer in host memory (mapped)
	 */
	/**
	 * Seems to be unused anymore because produceNodeTileData() seems to be unused anymore...
	 */
	VolTreeBVHNodeStorage* _nodesBufferKernel;

	/**
	 * Data pool (position and color)
	 */
	DataBufferKernelType _dataBufferKernel;

	/******************************** METHODS *********************************/

	/**
	 * Initialize
	 *
	 * @param h_nodesbufferarray node buffer
	 * @param h_vertexbufferpool data buffer
	 */
	inline void init( VolTreeBVHNodeStorage* h_nodesbufferarray, GvCore::GPUPoolKernel< GvCore::Array3DKernelLinear, DataTypeList > h_vertexbufferpool );

	/**
	 * Produce node tiles
	 *
	 * @param nodePool ...
	 * @param requestID ...
	 * @param processID ...
	 * @param pNewElemAddress ...
	 * @param parentLocInfo ...
	 *
	 * @return ...
	 */
	template< typename GPUPoolKernelType >
	__device__
	inline uint produceData( GPUPoolKernelType& nodePool, uint requestID, uint processID,
							uint3 pNewElemAddress, const VolTreeBVHNodeUser& node, Loki::Int2Type< 0 > );

	/**
	 * Produce bricks of data
	 *
	 * @param dataPool ...
	 * @param requestID ...
	 * @param processID ...
	 * @param pNewElemAddress ...
	 * @param parentLocInfo ...
	 *
	 * @return ...
	 */
	template< typename GPUPoolKernelType >
	__device__
	inline uint produceData( GPUPoolKernelType& dataPool, uint requestID, uint processID,
							uint3 pNewElemAddress, VolTreeBVHNodeUser& pNode, Loki::Int2Type< 1 > );

	/**
	 * Seems to be unused anymore...
	 */
	template< class GPUTreeBVHType >
	__device__
	inline uint produceNodeTileData( GPUTreeBVHType& gpuTreeBVH, uint requestID, uint processID, VolTreeBVHNodeUser& node, uint newNodeTileAddressNode );

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

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GPUTriangleProducerBVHKernel.inl"

#endif
