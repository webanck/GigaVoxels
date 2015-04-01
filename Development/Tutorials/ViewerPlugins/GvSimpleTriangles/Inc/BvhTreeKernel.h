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

#ifndef _GPU_Tree_BVH_hcu_
#define _GPU_Tree_BVH_hcu_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/Array3DGPULinear.h>
#include <GvCore/GvCUDATexHelpers.h>

#include "BVHTriangles.hcu"

#include <cuda.h>
#include "RendererBVHTrianglesCommon.h"

// Cuda SDK
#include <helper_math.h>

//#include "CUDATexHelpers.h"
//#include "Array3DKernel.h"

#include <loki/Typelist.h>
#include <loki/HierarchyGenerators.h>
#include <loki/TypeManip.h>
#include <loki/NullType.h>

#include "GPUTreeBVHCommon.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * ...
 */
#define TEXDATAPOOL_BVHTRIANGLES 10

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

////////// DATA ARRAYS ///////////

/**
 * Node buffer
 *
 * Seems to be not used anymore
 */
texture< uint, 1, cudaReadModeElementType > volumeTreeBVHTexLinear;

// FIXME
//GPUPoolTextureReferences(TEXDATAPOOL_BVHTRIANGLES, 4, 1, BVHVertexPosType, cudaReadModeElementType);
//GPUPoolTextureReferences(TEXDATAPOOL_BVHTRIANGLES, 4, 1, BVHVertexColorType, cudaReadModeElementType);

/**
 * 1D texture used to store user data (color)
 */
GPUPoolTextureReferences( TEXDATAPOOL_BVHTRIANGLES, 4, 1, uchar4, cudaReadModeElementType );
/**
 * 1D texture used to store user data (position)
 */
GPUPoolTextureReferences( TEXDATAPOOL_BVHTRIANGLES, 4, 1, float4, cudaReadModeElementType );

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @struct BvhTreeKernel
 *
 * @brief The BvhTreeKernel struct provides ...
 *
 * @param TDataTypeList Data type list provided by the user
 * (exemple with a normal and a color by voxel : typedef Loki::TL::MakeTypelist< uchar4, half4 >::Result DataType;)
 *
 */
template< class TDataTypeList >
struct BvhTreeKernel
{
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition of the data pool
	 */
	typedef typename GvCore::GPUPool_KernelPoolFromHostPool< GvCore::Array3DGPULinear, TDataTypeList >::Result KernelPoolType;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Node pool
	 */
	GvCore::Array3DKernelLinear< VolTreeBVHNodeStorageUINT > _volumeTreeBVHArray;

	/**
	 * Data pool
	 */
	KernelPoolType _dataPool;

	/******************************** METHODS *********************************/

	/**
	 * ...
	 *
	 * @param ... ...
	 *
	 * @return ...
	 */
	template< int channel >
	__device__
	typename GvCore::DataChannelType< TDataTypeList, channel >::Result getVertexData( uint address );

	/**
	 * ...
	 *
	 * @param ... ...
	 * @param ... ...
	 */
	__device__
	inline void fetchBVHNode( VolTreeBVHNodeUser& resnode, uint address );

	/**
	 * ...
	 *
	 * @param ... ...
	 * @param ... ...
	 * @param ... ...
	 */
	__device__
	inline void parallelFetchBVHNode( uint Pid, VolTreeBVHNodeUser& resnode, uint address );

	/**
	 * ...
	 *
	 * @param ... ...
	 * @param ... ...
	 * @param ... ...
	 */
	__device__
	inline void parallelFetchBVHNodeTile( uint Pid, VolTreeBVHNodeUser* resnodetile, uint address );

	/**
	 * ...
	 *
	 * @param ... ...
	 * @param ... ...
	 */
	__device__
	inline void writeBVHNode( const VolTreeBVHNodeUser& node, uint address );

	/**
	 * ...
	 *
	 * @param ... ...
	 * @param ... ...
	 * @param ... ...
	 */
	__device__
	inline void parallelWriteBVHNode( uint Pid, const VolTreeBVHNodeStorage& node, uint address );

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

#include "BvhTreeKernel.inl"

#endif // !_GPU_Tree_BVH_hcu_
