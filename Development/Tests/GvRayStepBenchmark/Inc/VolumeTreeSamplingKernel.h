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

#ifndef _VOLUME_TREE_SAMPLING_H_
#define _VOLUME_TREE_SAMPLING_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvRendering/GvIRenderShader.h>
#include <GvRendering/GvVolumeTreeSamplerKernel.h>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/**
 * Renderer's parameters
 */
__constant__ bool cHasRendererParametersActivated;
__constant__ float cRendererRayStep;

/******************************************************************************
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
 ******************************************************************************/
template< bool TFastUpdateMode, bool TPriorityOnBrick, class TVolumeTreeKernelType, class TSampleShaderType, class TGPUCacheType >
__device__
float rendererBrickSampling( TVolumeTreeKernelType& pVolumeTree, TSampleShaderType& pSampleShader,
	TGPUCacheType& pGpuCache, const float3 pRayStartTree, const float3 pRayDirTree, const float pTTree,
	const float pRayLengthInNodeTree, GvRendering::GvVolumeTreeSamplerKernel< TVolumeTreeKernelType >& pBrickSampler, bool& pModifInfoWriten )
{
	// Current position in tree space
	float3 samplePosTree = pRayStartTree + pTTree * pRayDirTree;

	// Local distance
	float dt = 0.0f;

	// Step
	float rayStep = 0.0f;

	if ( ! cHasRendererParametersActivated )
	{
		// Traverse the brick
		while ( dt <= pRayLengthInNodeTree && !pSampleShader.stopCriterion( samplePosTree ) )
		{
			// Update global distance
			float fullT = pTTree + dt;

			// Get the cone aperture at the given distance
			float coneAperture = pSampleShader.getConeAperture( fullT );
		
			// Update sampler mipmap parameters
			if ( ! pBrickSampler.updateMipMapParameters( coneAperture ) )
			{
				break;
			}
		
			// Move sampler position
			pBrickSampler.moveSampleOffsetInNodeTree( rayStep * pRayDirTree );
		
			// Update position
			samplePosTree = pRayStartTree + fullT * pRayDirTree;
		
			// Compute next step
			rayStep = max( coneAperture, pBrickSampler._nodeSizeTree * ( 0.66f / static_cast< float>( TVolumeTreeKernelType::BrickResolution::x ) ) );
							
			// Shading (+ adaptative step)
			pSampleShader.run( pBrickSampler, samplePosTree, pRayDirTree, rayStep, coneAperture );

			// Update local distance
			dt += rayStep;
		}
	}
	else
	{
		// Traverse the brick
		while ( dt <= pRayLengthInNodeTree && !pSampleShader.stopCriterion( samplePosTree ) )
		{
			// Update global distance
			float fullT = pTTree + dt;

			// Get the cone aperture at the given distance
			float coneAperture = pSampleShader.getConeAperture( fullT );
		
			// Update sampler mipmap parameters
			if ( ! pBrickSampler.updateMipMapParameters( coneAperture ) )
			{
				break;
			}
		
			// Move sampler position
			pBrickSampler.moveSampleOffsetInNodeTree( rayStep * pRayDirTree );
		
			// Update position
			samplePosTree = pRayStartTree + fullT * pRayDirTree;
		
			// Compute next step
			//rayStep = max( coneAperture, pBrickSampler._nodeSizeTree * ( 0.66f / static_cast< float>( TVolumeTreeKernelType::BrickResolution::x ) ) );
			//
			// TEST
			// Modify renderer's ray step to check if slices disappear
			// [ 1 ] - try constant ray step and check LOD 0
			rayStep = cRendererRayStep;
			// [ 2 ] - try constant ray step and check Level 0 - BUT take min with node size to handle more LODs
			//rayStep = min( cRendererRayStep, pBrickSampler._nodeSizeTree * ( 0.66f / static_cast< float>( TVolumeTreeKernelType::BrickResolution::x ) ) );
				
			// Shading (+ adaptative step)
			pSampleShader.run( pBrickSampler, samplePosTree, pRayDirTree, rayStep, coneAperture );

			// Update local distance
			dt += rayStep;
		}
	}

	return dt;
}

#endif // !_VOLUME_TREE_SAMPLING_H_
