/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2012 INRIA <http://www.inria.fr/>
 *
 * Authors : GigaVoxels Team
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

#ifndef _CUDA_TEX_HELPERS_H_
#define _CUDA_TEX_HELPERS_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Loki
#include <loki/TypeManip.h>

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

// Cuda
#include <host_defines.h>
#include <texture_fetch_functions.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

// Helpers to emulate pseudo-templated textures in cuda

/**
 * MACRO that transforms an argument to a string constant.
 *
 * @param x the argument to stringify.
 */
#define QUOTEME( x ) #x

/**
 * MACRO used to build a texture reference identifiant (used in CUDA functions where a texture reference is required).
 * Each argument is concatenated to create the full texture name.
 *
 * @param _TexName the pool (0 for data)
 * @param _ChannelNum the channel index (i.e. color, normal, etc...)
 * @param _TexDim the texture dimension (1, 2 or 3)
 * @param _TexType the texture type (i.e. uchar4, float4, etc...)
 * @param _TexReadMode the texture read mode
 */
#define CreateGPUPoolChannelTextureReferenceName( _TexName, _ChannelNum, _TexDim, _TexType, _TexReadMode ) \
	GPUPool_ ## _TexName ## _ ## _ChannelNum ## _ ## _TexDim ## D_ ## _TexType ## _ ## _TexReadMode

/**
 * MACRO used to build a texture reference name (a string constant).
 * Each argument is concatenated to create the full texture name.
 *
 * @param _TexName the pool (0 for data)
 * @param _ChannelNum the channel index (i.e. color, normal, etc...)
 * @param _TexDim the texture dimension (1, 2 or 3)
 * @param _TexType the texture type (i.e. uchar4, float4, etc...)
 * @param _TexReadMode the texture read mode
 */
#define CreateGPUPoolChannelTextureReferenceNameString( _TexName, _ChannelNum, _TexDim, _TexType, _TexReadMode ) \
	QUOTEME( GPUPool_ ## _TexName ## _ ## _ChannelNum ## _ ## _TexDim ## D_ ## _TexType ## _ ## _TexReadMode )

/**
 * MACRO used to declare a texture reference.
 *
 * @param _TexName the pool (0 for data)
 * @param _ChannelNum the channel index (i.e. color, normal, etc...)
 * @param _TexDim the texture dimension (1, 2 or 3)
 * @param _TexType the texture type (i.e. uchar4, float4, etc...)
 * @param _TexReadMode the texture read mode
 */
#define CreateGPUPoolChannelTextureReference( _TexName, _ChannelNum, _TexDim, _TexType, _TexReadMode ) \
	texture< _TexType, _TexDim, _TexReadMode > CreateGPUPoolChannelTextureReferenceName( _TexName, _ChannelNum, _TexDim, _TexType, _TexReadMode );

/**
 * MACRO used to build a surface reference identifiant (used in CUDA functions where a surface reference is required).
 * Each argument is concatenated to create the full surface name.
 *
 * @param _TexName the pool (0 for data)
 * @param _TexDim the surface dimension (1, 2 or 3)
 */
#define CreateGPUPoolChannelSurfaceReferenceName( _ChannelNum ) \
    surface_ ## _ChannelNum

#define CreateGPUPoolChannelSurfaceReferenceNameString( _ChannelNum ) \
    QUOTEME( surface_ ## _ChannelNum )

/**
 * MACRO used to declare a surface reference.
 *
 * @param _TexName the pool (0 for data)
 * @param _TexDim the surface dimension (1, 2 or 3)
 */
#define CreateGPUPoolChannelSurfaceReference( _ChannelNum ) \
    surface< void, cudaSurfaceType3D > CreateGPUPoolChannelSurfaceReferenceName( _ChannelNum );

/******************************************************************************
 ******************************************************************************
 ******************************************************************************/

/******************************************************************************
 * Helper function used to define texture fetch functions for 1D, 2D and 3D.
 * This generic function is empty.
 *
 * @param PoolName the pool (0 for data)
 * @param ChannelNum the channel index (i.e. color, normal, etc...)
 * @param TexDim the texture dimension (1, 2 or 3)
 * @param TexType the texture type (i.e. uchar4, float4, etc...)
 * @param TexReadMode the texture read mode
 * @param pos the sample position
 * @param res the resulting fetched data
 ******************************************************************************/
template< int PoolName, int ChannelNum, int TexDim, typename TexType, int TexReadMode, typename SamplePosType, typename SampleValType >
__device__ __forceinline__ void CUDATexHelpers_SampleTexRef( Loki::Int2Type< PoolName >, Loki::Int2Type< ChannelNum >, Loki::Int2Type< TexDim >, TexType, Loki::Int2Type< TexReadMode >, const SamplePosType& pos, SampleValType& res )
{
}

/******************************************************************************
 * MACRO used to define texture fetch functions for 1D, 2D and 3D.
 * These are specialization of the previous generic functions.
 *
 * @param _PoolName the pool (0 for data)
 * @param _ChannelNum the channel index (i.e. color, normal, etc...)
 * @param _TexDim the texture dimension (1, 2 or 3)
 * @param _TexType the texture type (i.e. uchar4, float4, etc...)
 * @param _TexReadMode the texture read mode
 ******************************************************************************/
#define CreateGPUPoolChannelSamplingFunctions( _PoolName, _ChannelNum, _TexDim, _TexType, _TexReadMode ) \
template< typename SamplePosType > \
__device__ __forceinline__ void CUDATexHelpers_SampleTexRef( Loki::Int2Type< _PoolName >, Loki::Int2Type< _ChannelNum >, Loki::Int2Type< 3 >, _TexType, Loki::Int2Type< _TexReadMode >, const SamplePosType& pos, float4& res ) { \
	res = make_float4( tex3D( CreateGPUPoolChannelTextureReferenceName( _PoolName, _ChannelNum, _TexDim, _TexType, _TexReadMode ), pos.x, pos.y, pos.z ) ); \
} \
template< typename SamplePosType > \
__device__ __forceinline__ void CUDATexHelpers_SampleTexRef( Loki::Int2Type< _PoolName >, Loki::Int2Type< _ChannelNum >, Loki::Int2Type< 2 >, _TexType, Loki::Int2Type< _TexReadMode >, const SamplePosType& pos, float4& res ) { \
	res = make_float4( tex2D( CreateGPUPoolChannelTextureReferenceName( _PoolName, _ChannelNum, _TexDim, _TexType, _TexReadMode ), pos.x, pos.y ) ); \
} \
template< typename SamplePosType > \
__device__ __forceinline__ void CUDATexHelpers_SampleTexRef( Loki::Int2Type< _PoolName >, Loki::Int2Type< _ChannelNum >, Loki::Int2Type< 1 >, _TexType, Loki::Int2Type< _TexReadMode >, const SamplePosType& pos, float4& res ) { \
	res = make_float4( tex1D( CreateGPUPoolChannelTextureReferenceName( _PoolName, _ChannelNum, _TexDim, _TexType, _TexReadMode ), pos.x ) ); \
}

/******************************************************************************
 * MACRO used to define texture fetch functions for 1D, 2D and 3D when a conversion to a destination type is requested.
 * These are specialization of the previous generic functions.
 *
 * @param _PoolName the pool (0 for data)
 * @param _ChannelNum the channel index (i.e. color, normal, etc...)
 * @param _TexDim the texture dimension (1, 2 or 3)
 * @param _TexType the texture type (i.e. uchar4, float4, etc...)
 * @param _TexReadMode the texture read mode
 * @param _TexTypeDst the texture destination type (i.e. uchar4, float4, etc...)
 ******************************************************************************/
#define CreateGPUPoolChannelSamplingRedirections( _PoolName, _ChannelNum, _TexDim, _TexType, _TexReadMode, _TexTypeDst ) \
template< typename SamplePosType > \
__device__ __forceinline__ void CUDATexHelpers_SampleTexRef( Loki::Int2Type< _PoolName >, Loki::Int2Type< _ChannelNum >, Loki::Int2Type< 3 >, _TexType, Loki::Int2Type< _TexReadMode >, const SamplePosType& pos, float4& res ) { \
	CUDATexHelpers_SampleTexRef( Loki::Int2Type< _PoolName >(), Loki::Int2Type< _ChannelNum >(), Loki::Int2Type< 3 >(), _TexTypeDst(), Loki::Int2Type< _TexReadMode >(), pos, res ); \
} \
template< typename SamplePosType > \
__device__ __forceinline__ void CUDATexHelpers_SampleTexRef( Loki::Int2Type< _PoolName >, Loki::Int2Type< _ChannelNum >, Loki::Int2Type< 2 >, _TexType, Loki::Int2Type< _TexReadMode> , const SamplePosType& pos, float4& res ) { \
	CUDATexHelpers_SampleTexRef( Loki::Int2Type< _PoolName >(), Loki::Int2Type< _ChannelNum >(), Loki::Int2Type< 2 >(), _TexTypeDst(), Loki::Int2Type< _TexReadMode >(), pos, res ); \
} \
template< typename SamplePosType > \
__device__ __forceinline__ void CUDATexHelpers_SampleTexRef( Loki::Int2Type< _PoolName >, Loki::Int2Type< _ChannelNum >, Loki::Int2Type< 1 >, _TexType, Loki::Int2Type< _TexReadMode >, const SamplePosType& pos, float4& res ) { \
	CUDATexHelpers_SampleTexRef( Loki::Int2Type< _PoolName >(), Loki::Int2Type< _ChannelNum >(), Loki::Int2Type< 1 >(), _TexTypeDst(), Loki::Int2Type< _TexReadMode >(), pos, res ); \
}

/******************************************************************************
 ******************************************************************************
 ******************************************************************************/

/******************************************************************************
 * Helper function used to bind a texture reference to a CUDA array.
 * This generic function is empty.
 *
 * @param PoolName the pool (0 for data)
 * @param ChannelNum the channel index (i.e. color, normal, etc...)
 * @param TexDim the texture dimension (1, 2 or 3)
 * @param TexType the texture type (i.e. uchar4, float4, etc...)
 * @param TexReadMode the texture read mode
 * @param GPUArrayType ...
  ******************************************************************************/
template< int PoolName, int ChannelNum, int TexDim, typename TexType, int TexReadMode, typename GPUArrayType >
inline void CUDATexHelpers_BindGPUArrayToTexRef( Loki::Int2Type< PoolName >, Loki::Int2Type< ChannelNum >, Loki::Int2Type< TexDim >, TexType, Loki::Int2Type< TexReadMode >,
												 GPUArrayType* gpuArray, bool normalizedAccess, cudaTextureFilterMode filterMode, cudaTextureAddressMode addressMode ){ };

/******************************************************************************
 * MACRO used to bind a texture reference to a CUDA array.
 * These are specialization of the previous generic functions.
 *
 * @param _PoolName the pool (0 for data)
 * @param _ChannelNum the channel index (i.e. color, normal, etc...)
 * @param _TexDim the texture dimension (1, 2 or 3)
 * @param _TexType the texture type (i.e. uchar4, float4, etc...)
 * @param _TexReadMode the texture read mode
 ******************************************************************************/
#define CreateGPUPoolChannelBindFunction( _PoolName, _ChannelNum, _TexDim, _TexType, _TexReadMode ) \
template< typename GPUArrayType > \
inline void CUDATexHelpers_BindGPUArrayToTexRef( Loki::Int2Type< _PoolName >, Loki::Int2Type< _ChannelNum >, Loki::Int2Type< _TexDim >, _TexType, Loki::Int2Type< _TexReadMode >, \
		GPUArrayType* gpuArray, bool normalizedAccess, cudaTextureFilterMode filterMode, cudaTextureAddressMode addressMode ) { \
    gpuArray->bindToTextureReference( &CreateGPUPoolChannelTextureReferenceName( _PoolName, _ChannelNum, _TexDim, _TexType, _TexReadMode ), \
            CreateGPUPoolChannelTextureReferenceNameString( _PoolName, _ChannelNum, _TexDim, _TexType, _TexReadMode ), \
			normalizedAccess, filterMode, addressMode );\
}

/******************************************************************************
 * MACRO used to bind a surface reference to a CUDA array.
 *
 * @param _ChannelNum the channel index (i.e. color, normal, etc...)
 ******************************************************************************/
#define CreateGPUPoolChannelBindSurfaceFunction( _ChannelNum ) \
template< typename GPUArrayType > \
inline void CUDATexHelpers_BindGPUArrayToSurfRef( Loki::Int2Type< _ChannelNum >, GPUArrayType* gpuArray ) { \
    gpuArray->bindToSurfaceReference( &CreateGPUPoolChannelSurfaceReferenceName( _ChannelNum ), \
            CreateGPUPoolChannelSurfaceReferenceNameString( _ChannelNum ) );\
}

/******************************************************************************
 * MACRO used to bind a texture reference to a CUDA array when a conversion to a destination type is requested.
 * These are specialization of the previous generic functions.
 *
 * @param _PoolName the pool (0 for data)
 * @param _ChannelNum the channel index (i.e. color, normal, etc...)
 * @param _TexDim the texture dimension (1, 2 or 3)
 * @param _TexType the texture type (i.e. uchar4, float4, etc...)
 * @param _TexReadMode the texture read mode
 * @param _TexTypeDst the texture destination type (i.e. uchar4, float4, etc...)
 ******************************************************************************/
#define CreateGPUPoolChannelBindRedirection( _PoolName, _ChannelNum, _TexDim, _TexType, _TexReadMode, _TexTypeDst ) \
template< typename GPUArrayType > \
inline void CUDATexHelpers_BindGPUArrayToTexRef( Loki::Int2Type< _PoolName >, Loki::Int2Type< _ChannelNum >, Loki::Int2Type< _TexDim >, _TexType, Loki::Int2Type< _TexReadMode >, \
	GPUArrayType* gpuArray, bool normalizedAccess, cudaTextureFilterMode filterMode, cudaTextureAddressMode addressMode ) { \
    gpuArray->bindToTextureReference( &CreateGPUPoolChannelTextureReferenceName( _PoolName, _ChannelNum, _TexDim, _TexTypeDst, _TexReadMode ), \
            CreateGPUPoolChannelTextureReferenceNameString( _PoolName, _ChannelNum, _TexDim, _TexTypeDst, _TexReadMode ), \
            normalizedAccess, filterMode, addressMode );\
}

/******************************************************************************
 ******************************************************************************
 ******************************************************************************/

/**
 * MACRO used to define texture functions for binding and sampling
 * when a conversion to a destination type is requested.
 */
#define DeclareGPUPoolChannelRedirection( _PoolName, _ChannelNum, _TexDim, _TexType, _TexReadMode, _TexTypeDst ) \
	CreateGPUPoolChannelSamplingRedirections( _PoolName, _ChannelNum, _TexDim, _TexType, _TexReadMode, _TexTypeDst ) \
	CreateGPUPoolChannelBindRedirection( _PoolName, _ChannelNum, _TexDim, _TexType, _TexReadMode, _TexTypeDst )

/**
 * MACRO used to declare a texture reference object
 * and define its associated functions for binding and sampling.
 */
#define DeclareGPUPoolChannel( _PoolName, _ChannelNum, _TexDim, _TexType, _TexReadMode ) \
	CreateGPUPoolChannelTextureReference( _PoolName, _ChannelNum, _TexDim, _TexType, _TexReadMode ) \
	CreateGPUPoolChannelSamplingFunctions( _PoolName, _ChannelNum, _TexDim, _TexType, _TexReadMode ) \
	CreateGPUPoolChannelBindFunction( _PoolName, _ChannelNum, _TexDim, _TexType, _TexReadMode )

/**
 * MACRO used to declare texture reference objects
 * and define their associated functions for binding and sampling
 * when a conversion to a destination type is requested.
 *
 * NOTE : HARD-CODED => it is done to only access the first 4 user defined channels (i.e color, normal, etc...)
 */
#define GPUPoolTextureRedirection( _PoolName, _NumChannels, _TexDim, _TexType, _TexReadMode, _TexTypeDst ) \
	DeclareGPUPoolChannelRedirection( _PoolName, 0, _TexDim, _TexType, _TexReadMode, _TexTypeDst ) \
	DeclareGPUPoolChannelRedirection( _PoolName, 1, _TexDim, _TexType, _TexReadMode, _TexTypeDst ) \
	DeclareGPUPoolChannelRedirection( _PoolName, 2, _TexDim, _TexType, _TexReadMode, _TexTypeDst ) \
	DeclareGPUPoolChannelRedirection( _PoolName, 3, _TexDim, _TexType, _TexReadMode, _TexTypeDst )

/**
 * MACRO used to declare texture reference objects
 * and define their associated functions for binding and sampling.
 *
 * NOTE : HARD-CODED => it is done to only access the first 4 user defined channels (i.e color, normal, etc...)
 */
#define GPUPoolTextureReferences( _PoolName, _NumChannels, _TexDim, _TexType, _TexReadMode ) \
	DeclareGPUPoolChannel( _PoolName, 0, _TexDim, _TexType, _TexReadMode) \
	DeclareGPUPoolChannel( _PoolName, 1, _TexDim, _TexType, _TexReadMode ) \
	DeclareGPUPoolChannel( _PoolName, 2, _TexDim, _TexType, _TexReadMode ) \
	DeclareGPUPoolChannel( _PoolName, 3, _TexDim, _TexType, _TexReadMode )

/**
 * MACRO used to declare surface reference objects
 * and define their associated functions for binding and sampling.
 *
 * NOTE : HARD-CODED => it is done to only access the first 8 user defined channels (i.e color, normal, etc...)
 */
#define GPUPoolSurfaceReferences( _ChannelNum ) \
    DeclareSurfaceGPUPoolChannel( _ChannelNum )

/**
 * MACRO used to declare a texture reference object
 * and define its associated functions for binding and sampling.
 */
#define DeclareSurfaceGPUPoolChannel( _ChannelNum ) \
    CreateGPUPoolChannelSurfaceReference( _ChannelNum ) \
    CreateGPUPoolChannelBindSurfaceFunction( _ChannelNum )

/******************************************************************************
 ******************************************************************************
 ******************************************************************************/

/******************************************************************************
 * Texture fecth function.
 *
 * @param _PoolName the pool (0 for data)
 * @param _ChannelNum the channel index (i.e. color, normal, etc...)
 * @param _TexDim the texture dimension (1, 2 or 3)
 * @param _TexType the texture type (i.e. uchar4, float4, etc...)
 * @param _TexReadMode the texture read mode
 * @param _pos the sample position
 * @param _res the resulting fetched data
 ******************************************************************************/
#define gpuPoolTexFetch( _PoolName, _ChannelNum, _TexDim, _TexType, _TexReadMode, _pos, _res )\
	CUDATexHelpers_SampleTexRef( Loki::Int2Type< _PoolName >(), Loki::Int2Type< _ChannelNum >(), Loki::Int2Type< _TexDim >(), _TexType(), Loki::Int2Type< _TexReadMode >(), _pos, _res );

/******************************************************************************
 * Texture binding function.
 *
 * @param _PoolName the pool (0 for data)
 * @param _ChannelNum the channel index (i.e. color, normal, etc...)
 * @param _TexDim the texture dimension (1, 2 or 3)
 * @param _TexType the texture type (i.e. uchar4, float4, etc...)
 * @param _TexReadMode the texture read mode
 * @param _gpuArray ...
 * @param _normalizedAccess ...
 * @param _filterMode ...
 * @param _addressMode ...
 ******************************************************************************/
#define gpuPoolBindToTexRef( _PoolName, _ChannelNum, _TexDim, _TexType, _TexReadMode, _gpuArray, _normalizedAccess, _filterMode, _addressMode )\
	CUDATexHelpers_BindGPUArrayToTexRef( Loki::Int2Type< _PoolName >(), Loki::Int2Type< _ChannelNum >(), Loki::Int2Type< _TexDim >(), _TexType(), Loki::Int2Type< _TexReadMode >(), \
			_gpuArray, _normalizedAccess, _filterMode, _addressMode );

#endif
