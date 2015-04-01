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

#ifndef _GV_VOLUME_TREE_KERNEL_H_
#define _GV_VOLUME_TREE_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/GvCUDATexHelpers.h"
#include "GvCore/GPUPool.h"
#include "GvCore/GvLocalizationInfo.h"
#include "GvRendering/GvRendererHelpersKernel.h"
#include "GvStructure/GvNode.h"

// CUDA
#include <cuda_surface_types.h>
#include <surface_types.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

// Define pools names
#define TEXDATAPOOL 0
#define USE_LINEAR_VOLTREE_TEX 0

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

//----------------------------
// Node pool content declaration (i.e. volume tree children and volume tree data.)
//----------------------------

#if USE_LINEAR_VOLTREE_TEX

/**
 * Volume tree children are placed in a 1D texture
 */
texture< uint, cudaTextureType1D, cudaReadModeElementType > volumeTreeChildTexLinear; // linear texture

/**
 * Volume tree data are placed in a 1D texture
 */
texture< uint, cudaTextureType1D, cudaReadModeElementType > volumeTreeDataTexLinear; // linear texture

#else // USE_LINEAR_VOLTREE_TEX

/**
 * Volume tree children are placed in constant memory
 */
__constant__ GvCore::Array3DKernelLinear< uint > k_volumeTreeChildArray;

/**
 * Volume tree data are placed in constant memory
 */
__constant__ GvCore::Array3DKernelLinear< uint > k_volumeTreeDataArray;

#endif // USE_LINEAR_VOLTREE_TEX

//----------------------------
// Surfaces declaration.
// Surfaces are used to write into textures.
//----------------------------

//namespace GvStructure
//{
    /**
     * Surfaces declaration used to write data into cache.
	 *
	 * NOTE : there are only 8 surfaces available.
	 *
	 * It is a wrapper to declare surfaces :
	 * - surface< void, cudaSurfaceType3D > surfaceRef_0;
     */
//#if (__CUDA_ARCH__ >= 200)
	// => moved to GvCore/GPUPool.h
    //GPUPoolSurfaceReferences( 0 )
    //GPUPoolSurfaceReferences( 1 )
    //GPUPoolSurfaceReferences( 2 )
    //GPUPoolSurfaceReferences( 3 )
    //GPUPoolSurfaceReferences( 4 )
    //GPUPoolSurfaceReferences( 5 )
    //GPUPoolSurfaceReferences( 6 )
    //GPUPoolSurfaceReferences( 7 )
 //#endif
//}

//----------------------------
// 3D textures declaration
//----------------------------

/**
 * Char type.
 */
GPUPoolTextureReferences( TEXDATAPOOL, 4, 3, char, cudaReadModeNormalizedFloat );
GPUPoolTextureReferences( TEXDATAPOOL, 4, 3, char4, cudaReadModeNormalizedFloat );

/**
 * Unsigned char type.
 */
GPUPoolTextureReferences( TEXDATAPOOL, 4, 3, uchar, cudaReadModeNormalizedFloat );
GPUPoolTextureReferences( TEXDATAPOOL, 4, 3, uchar4, cudaReadModeNormalizedFloat );

/**
 * Short type.
 */
GPUPoolTextureReferences( TEXDATAPOOL, 4, 3, short, cudaReadModeNormalizedFloat );
GPUPoolTextureReferences( TEXDATAPOOL, 4, 3, short2, cudaReadModeNormalizedFloat );
GPUPoolTextureReferences( TEXDATAPOOL, 4, 3, short4, cudaReadModeNormalizedFloat );

/**
 * Unsigned short type.
 */
GPUPoolTextureReferences( TEXDATAPOOL, 4, 3, ushort, cudaReadModeNormalizedFloat );
GPUPoolTextureReferences( TEXDATAPOOL, 4, 3, ushort2, cudaReadModeNormalizedFloat );
GPUPoolTextureReferences( TEXDATAPOOL, 4, 3, ushort4, cudaReadModeNormalizedFloat );

/**
 * Float type.
 */
GPUPoolTextureReferences( TEXDATAPOOL, 4, 3, float, cudaReadModeElementType );
GPUPoolTextureReferences( TEXDATAPOOL, 4, 3, float2, cudaReadModeElementType );
GPUPoolTextureReferences( TEXDATAPOOL, 4, 3, float4, cudaReadModeElementType );

/**
 * Half type.
 * Note : a redirection is used to float4
 */
GPUPoolTextureRedirection( TEXDATAPOOL, 4, 3, half4, cudaReadModeElementType, float4 );

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvStructure
{

/** 
 * @struct VolumeTreeKernel
 *
 * @brief The VolumeTreeKernel struct provides the interface to a GigaVoxels
 * data structure on device (GPU).
 *
 * @ingroup GvStructure
 *
 * This is the device-side associated object to a GigaVoxels data structure.
 *
 * @todo: Rename VolumeTreeKernel as GvVolumeTreeKernel.
 */
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize >
struct VolumeTreeKernel
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	/**
	 * Enumeration to define the brick border size
	 */
	enum
	{
		brickBorderSize = BorderSize
	};

	/**
	 * Type definition of the node resolution
	 */
	typedef NodeTileRes NodeResolution;

	/**
	 * Type definition of the brick resolution
	 */
	typedef BrickRes BrickResolution;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Root node address
	 */
	uint _rootAddress;

	/**
	 * Size of a voxel in the cache (i.e. pool of bricks of voxels implemented as a 3D texture)
	 */
	float3 brickCacheResINV;

	/**
	 * Size of a brick of voxels in the cache (i.e. pool of bricks of voxels implemented as a 3D texture)
	 */
	float3 brickSizeInCacheNormalized;

	/******************************** METHODS *********************************/

	/** @name Sampling data
	 *
	 *  Methods to sample user data attributes in the data structure (i.e. color, normal, density, etc...)
	 */
	///@{

	/**
	 * Sample data in specified channel at a given position.
	 * 3D texture are used with hardware tri-linear interpolation.
	 *
	 * @param pBrickPos Brick position in the pool of bricks
	 * @param pPosInBrick Position in brick
	 *
	 * @return the sampled value
	 */
	template< int TChannel >
	__device__
	__forceinline__ float4 getSampleValueTriLinear( float3 pBrickPos, float3 pPosInBrick ) const;

	/**
	 * Sample data in specified channel at a given position.
	 * 3D texture are used with hardware tri-linear interpolation.
	 *
	 * @param mipMapInterpCoef mipmap interpolation coefficient
	 * @param brickChildPosInPool brick child position in pool
	 * @param brickParentPosInPool brick parent position in pool
	 * @param posInBrick position in brick
	 * @param coneAperture cone aperture
	 *
	 * @return the sampled value
	 */
	// QUESTION : le paramètre "coneAperture" ne semble pas utilisé ? A quoi sert-il (servait ou servira) ?
	template< int TChannel >
	__device__
	__forceinline__ float4 getSampleValueQuadriLinear( float mipMapInterpCoef, float3 brickChildPosInPool,
											  float3 brickParentPosInPool, float3 posInBrick, float coneAperture ) const;

	/**
	 * Sample data in specified channel at a given position.
	 * 3D texture are used with hardware tri-linear interpolation.
	 *
	 * @param mipMapInterpCoef mipmap interpolation coefficient
	 * @param brickChildPosInPool brick child position in pool
	 * @param brickParentPosInPool brick parent position in pool
	 * @param posInBrick position in brick
	 * @param coneAperture cone aperture
	 *
	 * @return the sampled value
	 */
	template< int TChannel >
	__device__
	__forceinline__ float4 getSampleValue( float3 brickChildPosInPool, float3 brickParentPosInPool,
								  float3 sampleOffsetInBrick, float coneAperture, bool mipMapOn, float mipMapInterpCoef ) const;

	///@}

	/**
	 * ...
	 *
	 * @param nodeTileAddress ...
	 * @param nodeOffset ...
	 *
	 * @return ...
	 */
	__device__
	__forceinline__ uint computenodeAddress( uint3 nodeTileAddress, uint3 nodeOffset ) const;

	/**
	 * ...
	 *
	 * @param nodeTileAddress ...
	 * @param nodeOffset ...
	 *
	 * @return ...
	 */
	__device__
	__forceinline__ uint3 computeNodeAddress( uint3 nodeTileAddress, uint3 nodeOffset ) const;

	/** @name Reading nodes information
	 *
	 *  Methods to read nodes information from the data structure (nodes address and its flags)
	 */
	///@{

	/**
	 * Retrieve node information (address + flags) from data structure
	 *
	 * @param resnode ...
	 * @param nodeTileAddress ...
	 * @param nodeOffset ...
	 */
	__device__
	__forceinline__ void fetchNode( GvNode& resnode, uint3 nodeTileAddress, uint3 nodeOffset ) const;

	/**
	 * Retrieve node information (address + flags) from data structure
	 *
	 * @param resnode ...
	 * @param nodeTileAddress ...
	 * @param nodeOffset ...
	 */
	__device__
	__forceinline__ void fetchNode( GvNode& resnode, uint nodeTileAddress, uint nodeOffset ) const;

	/**
	 * Retrieve node information (address + flags) from data structure
	 *
	 * @param resnode ...
	 * @param nodeAddress ...
	 */
	__device__
	__forceinline__ void fetchNode( GvNode& resnode, uint nodeAddress ) const;

	///**
	// * ...
	// *
	// * @param resnode ...
	// * @param nodeTileAddress ...
	// * @param nodeOffset ...
	// */
	//__device__
	//__forceinline__ void fetchNodeChild( GvNode& resnode, uint nodeTileAddress, uint nodeOffset );

	///**
	// * ...
	// *
	// * @param resnode ...
	// * @param nodeTileAddress ...
	// * @param nodeOffset ...
	// */
	//__device__
	//__forceinline__ void fetchNodeData( GvNode& resnode, uint nodeTileAddress, uint nodeOffset );

	///@}

	/** @name Writing nodes information
	 *
	 *  Methods to write nodes information into the data structure (nodes address and its flags)
	 */
	///@{

	/**
	 * Write node information (address + flags) in data structure
	 *
	 * @param node ...
	 * @param nodeTileAddress ...
	 * @param nodeOffset ...
	 */
	__device__
	__forceinline__ void setNode( GvNode node, uint3 nodeTileAddress, uint3 nodeOffset );

	/**
	 * Write node information (address + flags) in data structure
	 *
	 * @param node ...
	 * @param nodeTileAddress ...
	 * @param nodeOffset ...
	 */
	__device__
	__forceinline__ void setNode( GvNode node, uint nodeTileAddress, uint nodeOffset );

	/**
	 * Write node information (address + flags) in data structure
	 *
	 * @param node ...
	 * @param nodeAddress ...
	 */
	__device__
	__forceinline__ void setNode( GvNode node, uint nodeAddress );

	///@}

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

} //namespace GvStructure

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvVolumeTreeKernel.inl"

#endif // !_GV_VOLUME_TREE_KERNEL_H_
