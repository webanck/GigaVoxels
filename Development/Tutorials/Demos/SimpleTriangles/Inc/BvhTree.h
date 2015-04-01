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

#ifndef _BVH_TREE_H_
#define _BVH_TREE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <vector_types.h>

// Cuda SDK
#include <helper_math.h>

// GigaVoxels
#include <GvCore/GPUPool.h>
#include <GvStructure/GvIDataStructure.h>

#include "RendererBVHTrianglesCommon.h"

// FIXME
#include <GvCore/RendererTypes.h>

#include "BVHTrianglesManager.h"
#include "BvhTreeKernel.h"

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
 * @struct BvhTree
 *
 * @brief The BvhTree struct provides interface to manage BVHs data structere
 *
 * @param DataTList Data type list provided by the user
 * (exemple with a normal and a color by voxel : typedef Loki::TL::MakeTypelist< uchar4, half4 >::Result DataType;)
 */
template< class DataTList >
struct BvhTree : public GvStructure::GvIDataStructure
{
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	/**
	 * Defines the data type list
	 */
	typedef DataTList DataTypeList;

	/**
	 * Type definition of the node pool type
	 */
	typedef GvCore::Array3DGPULinear< VolTreeBVHNodeUser > NodePoolType;

	/**
	 * Type definition of the data pool type
	 */
	typedef GvCore::GPUPoolHost< GvCore::Array3DGPULinear, DataTList > DataPoolType;

	/**
	 * Typedef of its associated device-side object
	 */
	typedef BvhTreeKernel< DataTList > BvhTreeKernelType;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Root node
	 *
	 * - seems to be unused anymore
	 */
	uint _rootNode;

	/**
	 * Node pool
	 */
	NodePoolType* _nodePool;
	
	/**
	 * Data pool
	 */
	DataPoolType* _dataPool;

	/**
	 * Associated device-side object
	 */
	BvhTreeKernelType _kernelObject;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param pNodePoolSize node pool size
	 * @param pVertexPoolSize data pool size
	 */
	BvhTree( uint pNodePoolSize, uint pVertexPoolSize );

	/**
	 * Destructor
	 */
	virtual ~BvhTree();

	/**
	 * Get the associated device-side object
	 */
	BvhTreeKernelType getKernelObject();

	/**
	 * CUDA initialization
	 */
	void cuda_Init();

	/**
	 * Initialize the cache
	 *
	 * @param pBvhTrianglesManager Helper class that store the node and data pools from a mesh
	 */
	void initCache( BVHTrianglesManager< DataTList, BVH_DATA_PAGE_SIZE >* pBvhTrianglesManager );

	/**
	 * Clear
	 */
	void clear();

	/**
	 * This method is called to serialize an object
	 *
	 * @param pStream the stream where to write
	 */
	virtual void write( std::ostream& pStream ) const;

	/**
	 * This method is called deserialize an object
	 *
	 * @param pStream the stream from which to read
	 */
	virtual void read( std::istream& pStream );

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

	/**
	 * Copy constructor forbidden.
	 */
	BvhTree( const BvhTree& );

	/**
	 * Copy operator forbidden.
	 */
	BvhTree& operator=( const BvhTree& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "BvhTree.inl"

#endif // !_BVH_TREE_H_
