///*
// * GigaVoxels is a ray-guided streaming library used for efficient
// * 3D real-time rendering of highly detailed volumetric scenes.
// *
// * Copyright (C) 2011-2012 INRIA <http://www.inria.fr/>
// *
// * Authors : GigaVoxels Team
// *
// * This program is free software: you can redistribute it and/or modify
// * it under the terms of the GNU General Public License as published by
// * the Free Software Foundation, either version 3 of the License, or
// * (at your option) any later version.
// *
// * This program is distributed in the hope that it will be useful,
// * but WITHOUT ANY WARRANTY; without even the implied warranty of
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// * GNU General Public License for more details.
// *
// * You should have received a copy of the GNU General Public License
// * along with this program.  If not, see <http://www.gnu.org/licenses/>.
// */
//
///** 
// * @version 1.0
// */
//
//#ifndef _GV_GRAPHICS_INTEROPERABILTY_HANDLER_KERNEL_
//#define _GV_GRAPHICS_INTEROPERABILTY_HANDLER_KERNEL_
//
///******************************************************************************
// ******************************* INCLUDE SECTION ******************************
// ******************************************************************************/
//
//// GigaVoxels
//#include "GvCore/GvCoreConfig.h"
////#include "GvRendering/GvRendererHelpersKernel.h"
//
//// Cuda
//#include <host_defines.h>
//#include <vector_types.h>
//#include <texture_types.h>
//#include <surface_types.h>
//#include <device_functions.h>
//#include <cuda_texture_types.h>
//#include <cuda_surface_types.h>
//#include <texture_fetch_functions.h>
//#include <surface_functions.h>
//
///******************************************************************************
// ************************* DEFINE AND CONSTANT SECTION ************************
// ******************************************************************************/
//
///******************************************************************************
// ***************************** TYPE DEFINITION ********************************
// ******************************************************************************/
//
//namespace GvRendering
//{
//
///**
// * Texture references used to read input color/depth buffers from graphics library (i.e. OpenGL)
// */
//texture< uchar4, cudaTextureType2D, cudaReadModeElementType > _inputColorTexture;
//texture< float, cudaTextureType2D, cudaReadModeElementType > _inputDepthTexture;
//
///**
// * Surface references used to read input/output color/depth buffers from graphics library (i.e. OpenGL)
// */
//surface< void, cudaSurfaceType2D > _colorSurface;
//surface< void, cudaSurfaceType2D > _depthSurface;
//
//}
//
///******************************************************************************
// ******************************** CLASS USED **********************************
// ******************************************************************************/
//
///******************************************************************************
// ****************************** CLASS DEFINITION ******************************
// ******************************************************************************/
//
//namespace GvRendering
//{
//
///** 
// * @class GvGraphicsInteroperabiltyHandlerKernel
// *
// * @brief The GvGraphicsInteroperabiltyHandlerKernel class provides methods
// * to read/ write color and depth buffers.
// *
// * ...
// */
////class GvGraphicsInteroperabiltyHandlerKernel
////{
//
//	/**************************************************************************
//	 ***************************** PUBLIC SECTION *****************************
//	 **************************************************************************/
//
////public:
//
//	/****************************** INNER TYPES *******************************/
//
//	/******************************* ATTRIBUTES *******************************/
//
//	/******************************** METHODS *********************************/
//
//	/**
//	 * Get the color at given pixel from input color buffer
//	 *
//	 * @param pPixel pixel coordinates
//	 *
//	 * @return the pixel color
//	 */
//	__device__
//	/*static*/ __forceinline__ uchar4 getInputColor( const uint2 pPixel );
//
//	/**
//	 * Set the color at given pixel into output color buffer
//	 *
//	 * @param pPixel pixel coordinates
//	 * @param pColor color
//	 */
//	__device__
//	/*static*/ __forceinline__ void setOutputColor( const uint2 pPixel, uchar4 pColor );
//
//	/**
//	 * Get the depth at given pixel from input depth buffer
//	 *
//	 * @param pPixel pixel coordinates
//	 *
//	 * @return the pixel depth
//	 */
//	__device__
//	/*static*/ __forceinline__ float getInputDepth( const uint2 pPixel );
//
//	/**
//	 * Set the depth at given pixel into output depth buffer
//	 *
//	 * @param pPixel pixel coordinates
//	 * @param pDepth depth
//	 */
//	__device__
//	/*static*/ __forceinline__ void setOutputDepth( const uint2 pPixel, float pDepth );
//
//	/**************************************************************************
//	 **************************** PROTECTED SECTION ***************************
//	 **************************************************************************/
//
////protected:
//
//	/****************************** INNER TYPES *******************************/
//
//	/******************************* ATTRIBUTES *******************************/
//
//	/******************************** METHODS *********************************/
//
//	/**************************************************************************
//	 ***************************** PRIVATE SECTION ****************************
//	 **************************************************************************/
//
////private:
//
//	/****************************** INNER TYPES *******************************/
//
//	/******************************* ATTRIBUTES *******************************/
//
//	/******************************** METHODS *********************************/
//
////};
//
//}
//
///**************************************************************************
// ***************************** INLINE SECTION *****************************
// **************************************************************************/
//
//#include "GvGraphicsInteroperabiltyHandlerKernel.inl"
//
//#endif // !_GV_GRAPHICS_INTEROPERABILTY_HANDLER_KERNEL_
