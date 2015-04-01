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

#ifndef _GV_RENDERER_CUDA_KERNEL_H_
#define _GV_RENDERER_CUDA_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <helper_math.h>

// Gigavoxels
#include "GvCore/GPUPool.h"
#include "GvCore/RendererTypes.h"
#include "GvRendering/GvRendererHelpersKernel.h"
#include "GvRendering/GvSamplerKernel.h"
#include "GvRendering/GvNodeVisitorKernel.h"
#include "GvRendering/GvBrickVisitorKernel.h"
#include "GvRendering/GvRendererContext.h"
#include "GvPerfMon/GvPerformanceMonitor.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

namespace GvRendering
{ 

	/**
	 * Model matrix (CUDA constant memory)
	 */
	__constant__ float4x4 k_modelMatrix;

	/**
	 * Inverse model matrix (CUDA constant memory)
	 */
	__constant__ float4x4 k_modelMatrixInv;

}

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/
 
namespace GvRendering
{

	/**
	 * CUDA kernel
	 * This is the main GigaVoxels KERNEL
	 * It is in charge of casting rays and found the color and depth values at pixels.
	 *
	 * @param pVolumeTree data structure
	 * @param pCache cache
	 */
	template
	<
		class TBlockResolution, bool TFastUpdateMode, bool TPriorityOnBrick, 
		class TSampleShaderType, class TVolTreeKernelType, class TCacheType
	>
	__global__
	void RenderKernelSimple( TVolTreeKernelType pVolumeTree, TCacheType pCache );

	// FIXME: Move this to another place
	/**
	 * CUDA kernel ...
	 *
	 * @param syntheticBuffer ...
	 * @param totalNumElems ...
	 */
	__global__
	void SyntheticInfo_Render( uchar4 *syntheticBuffer, uint totalNumElems );


} // namespace GvRendering

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvRendering
{

/** 
 * @class GvRendererKernel
 *
 * @brief The GvRendererKernel class provides ...
 *
 * ...
 */
class GvRendererKernel
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/
	
	/**
	 * This function is used to :
	 * - traverse the data structure (and emit requests if necessary)
	 * - render bricks
	 *
	 * @param pDataStructure data structure
	 * @param pShader shader
	 * @param pCache cache
	 * @param pPixelCoords pixel coordinates in window
	 * @param pRayStartTree ray start point
	 * @param pRayDirTree ray direction
	 * @param ptMaxTree max distance from camera found after box intersection test and comparing with input z (from the scene)
	 * @param ptTree the distance from camera found after box intersection test and comparing with input z (from the scene)
	 */
	template< bool TFastUpdateMode, bool TPriorityOnBrick, class VolTreeKernelType, class SampleShaderType, class TCacheType >
	__device__
	static __forceinline__ void render( VolTreeKernelType& pDataStructure, SampleShaderType& pShader, TCacheType& pCache,
										uint2 pPixelCoords, const float3 pRayStartTree, const float3 pRayDirTree, const float ptMaxTree, float& ptTree );
	
	/**
	 * Initialize the pixel coordinates.
	 *
	 * @param Pid the input thread identifiant
	 * @param blockPos the computed block position
	 * @param pixelCoords the computed pixel coordinates
	 */
	template< class TBlockResolution >
	__device__
	static __forceinline__ void initPixelCoords( const uint Pid, /*uint2& blockPos,*/ uint2& pixelCoords );
	
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

#include "GvRendererCUDAKernel.inl"

#endif // !_GV_RENDERER_CUDA_KERNEL_H_
