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

#ifndef _GV_RENDERER_HELPERS_KERNEL_H_
#define _GV_RENDERER_HELPERS_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

// Cuda SDK
#include <helper_math.h>

//#include "GvVolumeTreeKernel.h"
#include "GvRendering/GvRendererContext.h"

// Cuda
#include <vector_types.h>
#include <device_functions.h>
#include <cuda_texture_types.h>
#include <cuda_surface_types.h>
#include <surface_functions.h>

// GigaVoxels
#include "GvCore/vector_types_ext.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

namespace GvRendering
{

__device__ uint cRegularisationNb;

/**
 * Texture references used to read input color/depth buffers from graphics library (i.e. OpenGL)
 */
texture< uchar4, cudaTextureType2D, cudaReadModeElementType > _inputColorTexture;
texture< float, cudaTextureType2D, cudaReadModeElementType > _inputDepthTexture;

/**
 * Surface references used to read input/output color/depth buffers from graphics library (i.e. OpenGL)
 */
surface< void, cudaSurfaceType2D > _colorSurface;
surface< void, cudaSurfaceType2D > _depthSurface;

/******************************************************************************
 * Intersect box
 ******************************************************************************/
#if 0
	// intersect ray with a box
	// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
	__device__
	__forceinline__ int intersectBox(Ray r, const float3 boxmin, const float3 boxmax, float &tnear, float &tfar)
	{
		// compute intersection of ray with all six bbox planes
		float3 invR = make_float3(1.0f) / r.dir;
		float3 tbot = invR * (boxmin - r.start);
		float3 ttop = invR * (boxmax - r.start);

		// re-order intersections to find smallest and largest on each axis
		float3 tmin = fminf(ttop, tbot);
		float3 tmax = fmaxf(ttop, tbot);

		// find the largest tmin and the smallest tmax
		float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
		float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

		tnear = largest_tmin;
		tfar = smallest_tmax;

		return smallest_tmax > largest_tmin;
	}
#else
	// Optimizing ray tracing for CUDA: https://wiki.tkk.fi/download/attachments/40023967/gpgpu.pdf
	__device__
	__forceinline__ int intersectBox( const float3 rayStart, const float3 rayDir, const float3 boxmin, const float3 boxmax, float& tmin, float& tmax )
	{
		float3 inv_dir;
		//float3 orig_inv_dir;

		inv_dir.x = 1.0f / rayDir.x;
		inv_dir.y = 1.0f / rayDir.y;
		inv_dir.z = 1.0f / rayDir.z;

		/*orig_inv_dir.x = rayStart.x * inv_dir.x;
		orig_inv_dir.y = rayStart.y * inv_dir.y;
		orig_inv_dir.z = rayStart.z * inv_dir.z;*/

		float t0, t1;

		/*t0 = boxmin.x * inv_dir.x - orig_inv_dir.x;
		t1 = boxmax.x * inv_dir.x - orig_inv_dir.x;*/
		t0 = ( boxmin.x - rayStart.x ) * inv_dir.x;
		t1 = ( boxmax.x - rayStart.x ) * inv_dir.x;
		tmin = max( tmin, min( t0, t1 ) );
		tmax = min( tmax, max( t0, t1 ) );

		/*t0 = boxmin.y * inv_dir.y - orig_inv_dir.y;
		t1 = boxmax.y * inv_dir.y - orig_inv_dir.y;*/
		t0 = ( boxmin.y - rayStart.y ) * inv_dir.y;
		t1 = ( boxmax.y - rayStart.y ) * inv_dir.y;
		tmin = max( tmin, min( t0, t1 ) );
		tmax = min( tmax, max( t0, t1 ) );

		/*t0 = boxmin.z * inv_dir.z - orig_inv_dir.z;
		t1 = boxmax.z * inv_dir.z - orig_inv_dir.z;*/
		t0 = ( boxmin.z - rayStart.z ) * inv_dir.z;
		t1 = ( boxmax.z - rayStart.z ) * inv_dir.z;
		tmin = max( tmin, min( t0, t1 ) );
		tmax = min( tmax, max( t0, t1 ) );

		return tmin < tmax;
	}
#endif

	//__device__
	//__forceinline__ float getPixelSizeAtDist(const float dist)
	//{
	//	// overestimate to avoid aliasing
	//	float scaleFactor = 1.333f;

	//	return k_renderViewContext.pixelSize.x * dist * scaleFactor * k_renderViewContext.frustumNearINV;
	//}

	/******************************************************************************
	 * Helper function that compute the mip-mapping coefficient
	 * to use during interpolation of two levels of resolution.
	 *
	 * @param pConeAperture cone aperture
	 * @param pNodeSize node size
	 ******************************************************************************/
	template< typename TNodeRes, typename TBrickRes >
	__device__
	__forceinline__ float getMipMapInterpCoef( const float pConeAperture, const float pNodeSize )
	{
		// Compute ratio between node size and current cone aperture
		float voxelSizeInv = static_cast< float >( TBrickRes::maxRes ) / pNodeSize;
		float x = pConeAperture * voxelSizeInv;

		// NOTE : log2( 1 / voxelSize ) = -log2( voxelSize ) ==> indicates the depth

		// Handle different cases according to node resolution
		if ( TNodeRes::x == 2 && TNodeRes::y == 2 && TNodeRes::z == 2 )
		{
			return __log2f( x );
		}
		else
		{
			// This case appears in the Menger Sponge tutorial,
			// where 3x3 node tiles are used instead of octrees.
			//
			// TO DO : validate this with Fabrice
			return __logf( x ) / __logf( static_cast< float >( TNodeRes::x ) );
		}
	}

	/******************************************************************************
	 * Helper function to get ray length in node
	 *
	 * @param sampleOffsetInNodeTree sample offset in node
	 * @param nodeSizeTree size of node
	 * @param rayDirTree ray direction
	 *
	 * @return ...
	 ******************************************************************************/
	__device__
	__forceinline__ float getRayLengthInNode( const float3 sampleOffsetInNodeTree, const float nodeSizeTree, const float3 rayDirTree )
	{
		float3 directions = stepZero( rayDirTree ); // To precompute somewhere

		float3 planes = directions * nodeSizeTree;

		float3 distToBorder = ( planes - sampleOffsetInNodeTree ) / rayDirTree;

		return fminf( distToBorder.x, fminf( distToBorder.y, distToBorder.z ) );
	}

	///******************************************************************************
	// * Retrieve the depth value from the input depth buffer
	// * at a given pixel position.
	// *
	// * Trick to prevent type conversions when copying to PBO,
	// * convert DEPTH24_STENCIL8 into float32
	// *
	// * @param pPixelCoords pixel coordinates
	// *
	// * @return the depth value at given pixel
	// ******************************************************************************/
	//__device__
	//__forceinline__ float getFrameDepthIn( const uint2 pPixelCoords )
	//{
	//	uint tmpival;

	//	//tmpival= *((uint*)k_renderViewContext.inFrameDepth.getPointer(pPixelCoords));
	//	//tmpival= __cc_float_as_int( k_renderViewContext.inFrameDepth.get(pPixelCoords) );	// Todo : check depth test still working
	//	tmpival = __float_as_int( k_renderViewContext.inFrameDepth.get( pPixelCoords ) );	// Todo : check depth test still working
	//	tmpival = ( tmpival & 0xFFFFFF00 ) >> 8;

	//	return static_cast< float >( tmpival ) / 16777215.0f;	// 16777215 <=> (2 power 24) - 1
	//}

	///******************************************************************************
	// * ...
	// *
	// * @param ...
	// * @param ...
	// ******************************************************************************/
	//// ATTENTION
	//// il faut faut gérer le stencil buffer, refaire un getDepth pour savoir ce qu'il y avant et faire un OU logique avec les 24 bits du stencil !!
	//__device__
	//__forceinline__ void setFrameDepthOut( const uint2 pixelCoords, const float d )
	//{
	//#if 1
	//	uint tmpival;
	//	tmpival = (uint)floorf(d * 16777215.0f);
	//	tmpival = (tmpival << 8);

	//	k_renderViewContext.inFrameDepth.set(pixelCoords, __int_as_float(tmpival));
	//#else
	//	k_renderViewContext.inFrameDepth.set(pixelCoords, d);
	//#endif
	//}

	/******************************************************************************
	 * Get the color at given pixel from input color buffer
	 *
	 * @param pPixel pixel coordinates
	 *
	 * @return the pixel color
	 ******************************************************************************/
	__device__
	__forceinline__ uchar4 getInputColor( const uint2 pixelCoords )
	{
		uchar4 ret;

		switch ( k_renderViewContext._graphicsResourceAccess[ GvGraphicsInteroperabiltyHandler::eColorInput ] )
		{
			case GvGraphicsResource::ePointer:
				{
					int offset = pixelCoords.x + pixelCoords.y * k_renderViewContext.frameSize.x;
					ret = static_cast< uchar4* >( k_renderViewContext._graphicsResources[ GvGraphicsInteroperabiltyHandler::eColorInput ] )[ offset ];
					break;
				}
			case GvGraphicsResource::eTexture:
				{
					ret = tex2D( _inputColorTexture, k_renderViewContext._inputColorTextureOffset + pixelCoords.x, pixelCoords.y );
					break;
				}
			case GvGraphicsResource::eSurface:
				{
					// Note : cudaBoundaryModeTrap means that out-of-range accesses cause the kernel execution to fail.
					// Possible values can be : cudaBoundaryModeTrap, cudaBoundaryModeClamp or cudaBoundaryModeZero.
					ret = surf2Dread< uchar4 >( _colorSurface, pixelCoords.x * sizeof( uchar4 ), pixelCoords.y, cudaBoundaryModeTrap );
					break;
				}
			default:
				{
					ret = k_renderViewContext._clearColor;
					break;
				}
		}

		return ret;
	}

	/******************************************************************************
	 * Set the color at given pixel into output color buffer
	 *
	 * @param pPixel pixel coordinates
	 * @param pColor color
	 ******************************************************************************/
	__device__
	__forceinline__ void setOutputColor( const uint2 pixelCoords, uchar4 color )
	{
		switch ( k_renderViewContext._graphicsResourceAccess[ GvGraphicsInteroperabiltyHandler::eColorOutput ] )
		{
			case GvGraphicsResource::ePointer:
				{
					int offset = pixelCoords.x + pixelCoords.y * k_renderViewContext.frameSize.x;
					static_cast< uchar4* >( k_renderViewContext._graphicsResources[ GvGraphicsInteroperabiltyHandler::eColorOutput ] )[ offset ] = color;
					break;
				}

			case GvGraphicsResource::eSurface:
				{
					// Note : cudaBoundaryModeTrap means that out-of-range accesses cause the kernel execution to fail.
					// Possible values can be : cudaBoundaryModeTrap, cudaBoundaryModeClamp or cudaBoundaryModeZero.
					surf2Dwrite( color, _colorSurface, pixelCoords.x * sizeof( uchar4 ), pixelCoords.y, cudaBoundaryModeTrap );
					break;
				}

			default:
				{
					break;
				}
		}
	}

	/******************************************************************************
	 * Get the depth at given pixel from input depth buffer
	 *
	 * @param pPixel pixel coordinates
	 *
	 * @return the pixel depth
	 ******************************************************************************/
	__device__
	__forceinline__ float getInputDepth( const uint2 pixelCoords )
	{
		float tmpfval;

		// Read depth from Z-buffer
		switch ( k_renderViewContext._graphicsResourceAccess[ GvGraphicsInteroperabiltyHandler::eDepthInput ] )
		{
			case GvGraphicsResource::ePointer:
				{
					int offset = pixelCoords.x + pixelCoords.y * k_renderViewContext.frameSize.x;
					tmpfval = static_cast< float* >( k_renderViewContext._graphicsResources[ GvGraphicsInteroperabiltyHandler::eDepthInput ] )[ offset ];
					break;
				}

			case GvGraphicsResource::eTexture:
				{
					tmpfval = tex2D( _inputDepthTexture, k_renderViewContext._inputDepthTextureOffset + pixelCoords.x, pixelCoords.y );
					break;
				}

			case GvGraphicsResource::eSurface:
				{
					// Note : cudaBoundaryModeTrap means that out-of-range accesses cause the kernel execution to fail.
					// Possible values can be : cudaBoundaryModeTrap, cudaBoundaryModeClamp or cudaBoundaryModeZero.
					surf2Dread< float >( &tmpfval, _depthSurface, pixelCoords.x * sizeof( float ), pixelCoords.y, cudaBoundaryModeTrap );
					break;
				}

			default:
				{
					// TO DO : clean code
					//tmpfval = k_renderViewContext._clearDepth;
					//{
					//	uint tmpival = static_cast< uint >( floorf( k_renderViewContext._clearDepth * 16777215.0f ) );
					//	tmpival = tmpival << 8;
					//	float Zdepth = __int_as_float( tmpival );
					//	tmpfval = Zdepth;
					//}

					return k_renderViewContext._clearDepth;		// TO DO : ?
				}
		}

		// Decode depth from Z-buffer
		uint tmpival = __float_as_int( tmpfval );
		tmpival = ( tmpival & 0xFFFFFF00 ) >> 8;

		return __saturatef( static_cast< float >( tmpival ) / 16777215.0f );
	}

	/******************************************************************************
	 * Set the depth at given pixel into output depth buffer
	 *
	 * @param pPixel pixel coordinates
	 * @param pDepth depth
	 ******************************************************************************/
	__device__
	__forceinline__ void setOutputDepth( const uint2 pixelCoords, float depth )
	{
		// Encode depth to Z-buffer
		uint tmpival = static_cast< uint >( floorf( depth * 16777215.0f ) );
		tmpival = tmpival << 8;
		float Zdepth = __int_as_float( tmpival );

		// Write depth to Z-buffer
		switch ( k_renderViewContext._graphicsResourceAccess[ GvGraphicsInteroperabiltyHandler::eDepthOutput ] )
		{
			case GvGraphicsResource::ePointer:
				{
					int offset = pixelCoords.x + pixelCoords.y * k_renderViewContext.frameSize.x;
					static_cast< float* >( k_renderViewContext._graphicsResources[ GvGraphicsInteroperabiltyHandler::eDepthOutput ] )[ offset ] = Zdepth;
				}
				break;

			case GvGraphicsResource::eSurface:
				// Note : cudaBoundaryModeTrap means that out-of-range accesses cause the kernel execution to fail.
				// Possible values can be : cudaBoundaryModeTrap, cudaBoundaryModeClamp or cudaBoundaryModeZero.
				surf2Dwrite( depth, _depthSurface, pixelCoords.x * sizeof( float ), pixelCoords.y, cudaBoundaryModeTrap );
				break;

			default:
				break;
		}
	}

	/******************************************************************************
	 * ...
	 *
	 * @param ...
	 *
	 * @return ...
	 ******************************************************************************/
	__host__ __device__
	inline uint interleaveBits32( uint2 input )
	{
		uint res;

		input.x = (input.x | (input.x <<  8)) & 0x00FF00FF;
		input.x = (input.x | (input.x <<  4)) & 0x0F0F0F0F;
		input.x = (input.x | (input.x <<  2)) & 0x33333333;
		input.x = (input.x | (input.x <<  1)) & 0x55555555;

		input.y = (input.y | (input.y <<  8)) & 0x00FF00FF;
		input.y = (input.y | (input.y <<  4)) & 0x0F0F0F0F;
		input.y = (input.y | (input.y <<  2)) & 0x33333333;
		input.y = (input.y | (input.y <<  1)) & 0x55555555;

		res= input.x | (input.y << 1);

		return res;
	}

	/******************************************************************************
	 * Helper function used to deinterleave an unsigned 32 bits integer
	 *
	 * @param pInput the input value
	 * @param pResult the deinterleaved output value
	 ******************************************************************************/
	__host__ __device__
	inline void deinterleaveBits32( const uint pInput, uint2& pResult )
	{
		pResult.x = pInput & 0x55555555;
		pResult.y = ( pInput >> 1 ) & 0x55555555;

		pResult.x = ( pResult.x | pResult.x>>1 ) & 0x33333333;
		pResult.x = ( pResult.x | pResult.x>>2 ) & 0x0F0F0F0F;
		pResult.x = ( pResult.x | pResult.x>>4 ) & 0x00FF00FF;
		pResult.x = ( pResult.x | pResult.x>>8 ) & 0x0000FFFF;

		pResult.y = ( pResult.y | pResult.y>>1 ) & 0x33333333;
		pResult.y = ( pResult.y | pResult.y>>2 ) & 0x0F0F0F0F;
		pResult.y = ( pResult.y | pResult.y>>4 ) & 0x00FF00FF;
		pResult.y = ( pResult.y | pResult.y>>8 ) & 0x0000FFFF;
	}

	/******************************************************************************
	 * ...
	 *
	 * @param ...
	 *
	 * @return ...
	 ******************************************************************************/
	__host__ __device__
	inline uint interleaveBits32( uint3 input )
	{
		uint res;

		input.x = (input.x | (input.x << 16)) & 0x030000FF;
		input.x = (input.x | (input.x <<  8)) & 0x0300F00F;
		input.x = (input.x | (input.x <<  4)) & 0x030C30C3;
		input.x = (input.x | (input.x <<  2)) & 0x09249249;

		input.y = (input.y | (input.y << 16)) & 0x030000FF;
		input.y = (input.y | (input.y <<  8)) & 0x0300F00F;
		input.y = (input.y | (input.y <<  4)) & 0x030C30C3;
		input.y = (input.y | (input.y <<  2)) & 0x09249249;

		input.z = (input.z | (input.z << 16)) & 0x030000FF;
		input.z = (input.z | (input.z <<  8)) & 0x0300F00F;
		input.z = (input.z | (input.z <<  4)) & 0x030C30C3;
		input.z = (input.z | (input.z <<  2)) & 0x09249249;

		res = input.x | (input.y << 1) | (input.z << 2);

		return res;
	}

} //namespace GvRendering

#endif
