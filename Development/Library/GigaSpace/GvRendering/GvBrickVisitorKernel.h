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

#ifndef _GV_BRICK_VISITOR_KERNEL_H_
#define _GV_BRICK_VISITOR_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/vector_types_ext.h"
#include "GvRendering/GvSamplerKernel.h"

// Cuda
#include <host_defines.h>
#include <vector_types.h>

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

namespace GvRendering
{

/**
 * @class GvBrickVisitorKernel
 *
 * @brief The GvBrickVisitorKernel class provides ...
 *
 * @ingroup GvRendering
 *
 * ...
 */
class GvBrickVisitorKernel
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * DEVICE function
	 *
	 * This is the function where shading is done (ray marching along ray and data sampling).
	 * Shading is done with cone-tracing (LOD is selected by comparing cone aperture versus voxel size).
	 *
	 * @param pVolumeTree data structure
	 * @param pSampleShader shader
	 * @param pGpuCache cache (not used for the moment...)
	 * @param pRayStartTree camera position in Tree coordinate system
	 * @param pRayDirTree ray direction in Tree coordinate system
	 * @param pTTree the distance from the eye to current position along the ray
	 * @param pRayLengthInNodeTree the distance along the ray from start to end of the brick, according to ray direction
	 * @param pBrickSampler The object in charge of sampling data (i.e. texture fetches)
	 * @param pModifInfoWriten (not used for the moment...)
	 *
	 * @return the distance where ray-marching has stopped
	 */
	template< bool TFastUpdateMode, bool TPriorityOnBrick, class TVolumeTreeKernelType, class TSampleShaderType, class TGPUCacheType >
	__device__
	static float visit( TVolumeTreeKernelType& pVolumeTree, TSampleShaderType& pSampleShader,
						TGPUCacheType& pGpuCache, const float3 pRayStartTree, const float3 pRayDirTree, const float pTTree,
						const float pRayLengthInNodeTree, GvSamplerKernel< TVolumeTreeKernelType >& pBrickSampler, bool& pModifInfoWriten );

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

} // namespace GvRendering

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvBrickVisitorKernel.inl"

#endif // !_GV_BRICK_VISITOR_KERNEL_H_
