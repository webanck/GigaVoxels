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

#ifndef _GV_RENDERER_CONTEXT_H_
#define _GV_RENDERER_CONTEXT_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvRendering/GvGraphicsInteroperabiltyHandler.h"
#include "GvRendering/GvGraphicsResource.h"

// Cuda
#include <host_defines.h>

// Cutil
//#include <helper_math.h>

// GigaVoxels
#include "GvCore/Array3DKernelLinear.h"
#include "GvCore/vector_types_ext.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Opacity step
 */
#define cOpacityStep 0.99f

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvRendering
{

/** 
 * @struct GvRendererContext
 *
 * @brief The GvRendererContext struct provides access to useful variables
 * from rendering context (view matrix, model matrix, etc...)
 *
 * TO DO : analyse memory alignement of data in this structure (ex : float3).
 *
 * @ingroup GvRenderer
 */
struct GvRendererContext
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/******************************* ATTRIBUTES *******************************/

	// TO DO
	// [ 1 ]
	// - Maybe store all constants that could changed in a frame in this monolythic structure to avoid multiple copy of constant at each frame
	// - cause actually, many graphics cards have only one copy engine, so it will be serialized.
	// [ 2 ]
	// - check the data alignment : maybe using float3 or single 32-bits, could misaligned memory access pattern, I forgot ?
	
	/**
	 * View matrix
	 */
	float4x4 viewMatrix;

	/**
	 * Inverted view matrix
	 */
	float4x4 invViewMatrix;

	/**
	 * Model matrix
	 */
	float4x4 modelMatrix;

	/**
	 * Inverted model matrix
	 */
	float4x4 invModelMatrix;

	/**
	 * Distance to the near depth clipping plane
	 */
	float frustumNear;

	/**
	 * Distance to the near depth clipping plane
	 */
	float frustumNearINV;

	/**
	 * Distance to the far depth clipping plane
	 */
	float frustumFar;
	
	/**
	 * Specify the coordinate for the right vertical clipping plane 
	 */
	float frustumRight;

	/**
	 * Specify the coordinate for the top horizontal clipping plane
	 */
	float frustumTop;

	float frustumC; //cf: http://www.opengl.org/sdk/docs/man/xhtml/glFrustum.xml
	float frustumD;

	/**
	 * Pixel size
	 */
	float2 pixelSize;

	/**
	 * Frame size (viewport dimension)
	 */
	uint2 frameSize;

	/**
	 * Camera position (in world coordinate system)
	 */
	float3 viewCenterWP;
	// TEST
	float3 viewCenterTP;

	/**
	 * Camera's vector from eye to [left, bottom, -near] clip plane position
	 * (in world coordinate system)
	 *
	 * This vector is used during ray casting as base direction
	 * from which camera to pixel ray directions are computed.
	 */
	float3 viewPlaneDirWP;
	// TEST
	float3 viewPlaneDirTP;

	/**
	 * Camera's X axis (in world coordinate system)
	 */
	float3 viewPlaneXAxisWP;
	// TEST
	float3 viewPlaneXAxisTP;

	/**
	 * Camera's Y axis (in world coordinate system)
	 */
	float3 viewPlaneYAxisWP;
	// TEST
	float3 viewPlaneYAxisTP;
	
	/**
	 * Projected 2D Bounding Box of the GigaVoxels 3D BBox.
	 * It holds its bottom left corner and its size.
	 * ( bottomLeftCorner.x, bottomLeftCorner.y, frameSize.x, frameSize.y )
	 */
	uint4 _projectedBBox;
	
	/**
	 * Color and depth graphics resources
	 */
	void* _graphicsResources[ GvGraphicsInteroperabiltyHandler::eNbGraphicsResourceDeviceSlots ];
	GvGraphicsResource::MappedAddressType _graphicsResourceAccess[ GvGraphicsInteroperabiltyHandler::eNbGraphicsResourceDeviceSlots ];
	unsigned int _inputColorTextureOffset;
	unsigned int _inputDepthTextureOffset;

	// TO DO : à deplacer en dehors du context ?
	/**
	 * Specify clear values for the color buffers
	 */
	uchar4 _clearColor;

	// TO DO : à deplacer en dehors du context ?
	/**
	 * Specify the clear value for the depth buffer
	 */
	float _clearDepth;

	/******************************** METHODS *********************************/

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

	/******************************** METHODS *********************************/

};

} // namespace GvRendering

/**************************************************************************
 **************************** CONSTANTS SECTION ***************************
 **************************************************************************/

/**
 * Render view context
 *
 * It provides access to useful variables from rendering context (view matrix, model matrix, etc...)
 * As a CUDA constant, all values will be available in KERNEL and DEVICE code.
 */
__constant__ GvRendering::GvRendererContext k_renderViewContext;

/**
 * Max volume tree depth
 */
__constant__ uint k_maxVolTreeDepth;

/**
 * Current time
 */
__constant__ uint k_currentTime;

namespace GvRendering
{

	/**
	 * Voxel size multiplier
	 */
	__constant__ float k_voxelSizeMultiplier;

}

#endif
