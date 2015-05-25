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

#ifndef _SHADER_KERNEL_H_
#define _SHADER_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvRendering/GvIRenderShader.h>
#include <GvUtils/GvCommonShaderKernel.h>

// Project
#include "PluginConfig.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Position of the light source in volume tree referential.
 */
__constant__ float3 cLightPositionTree;
/**
* Position of the light source in the scene referential.
*/
__constant__ float3 cLightPositionScene;

/**
 * Shader material property (according to opacity)
 */
__constant__ float cShaderMaterialProperty;

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
 * @class ShaderLoadKernel
 *
 * @brief The ShaderLoadKernel struct provides...
 *
 * ...
 */
class ShaderKernel
:	public GvUtils::GvCommonShaderKernel
,	public GvRendering::GvIRenderShader< ShaderKernel >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * ...
	 *
	 * @param materialColor ...
	 * @param normalVec ...
	 * @param lightVec ...
	 * @param eyeVec ...
	 * @param ambientTerm ...
	 * @param diffuseTerm ...
	 * @param specularTerm ...
	 *
	 * @return The RGB color values of the sample.
	 */
	__device__
	inline float3 shadePointLight(
		float3 materialColor,
		float3 normalVec,
		float3 lightVec,
		float3 eyeVec,
		float3 ambientTerm,
		float3 diffuseTerm,
		float3 specularTerm
	);

	/**
	 * Sets the _accColor value of the shader corresponding to the color of the voxel/sample.
	 *
	 * @param brickSampler ...
	 * @param pGpuCache The GPU cache required to access volume tree data.
	 * @param pRequestEmitted A boolean to set to true if a cache request has been done.
	 * @param samplePosScene The position of the voxel/sample in the (TODO:scene or graph?).
	 * @param rayDir The direction of the launched cone.
	 * @param rayStep ...
	 * @param coneAperture The aperture of the cone at the encounter with the voxel/sample.
	 */
	template <typename TSamplerType, class TGPUCacheType>
	__device__
	inline void runImpl(
		const TSamplerType& pBrickSampler,
		TGPUCacheType& pGpuCache,
		bool& pRequestEmitted,
		const float3 pSamplePosScene,
		const float3 pRayDir,
		float& pRayStep,
		const float pConeAperture
	);

	/**
	* Traces (using a secondary shader) a ray/cone from a voxel/sample to light to integrate light absorbtion.
	*
	* @param pBrickSampler ...
	* @param pGpuCache ...
	* @param pRequestEmitted ...
	* @param pSamplePosTree ...
	* @param pScreenConeAperture ...
	* @return The remaining light intensity (1.0 for full light, 0.0 for full shadow).
	*/
	template <typename TSamplerType, class TGPUCacheType>
	__device__
	float4 marchShadowRay(
		const TSamplerType& pBrickSampler,
		TGPUCacheType& pGpuCache,
		bool& pRequestEmitted,
		const float3 pSamplePosTree,
		const float pScreenConeAperture
	) const;

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

class ShadowRayShaderKernel:
	public GvUtils::GvCommonShaderKernel,
	public GvRendering::GvIRenderShader<ShadowRayShaderKernel>
{
public:

	float _lightDistanceTree;
	float _lightSize;
	float _sampleSize;

	__device__ ShadowRayShaderKernel(/*const float3 pLightPosition, */const float pLightDistanceTree, const float pLightSize, const float pSampleSize);


	/**
	* Sets the _accColor value of the shader corresponding to the color of the voxel/sample.
	*
	* @param brickSampler ...
	* @param pGpuCache The GPU cache required to access volume tree data.
	* @param pRequestEmitted A boolean to set to true if a cache request has been done.
	* @param samplePosScene The position of the voxel/sample in the (TODO:scene or graph?).
	* @param rayDir The direction of the launched cone.
	* @param rayStep ...
	* @param coneAperture The aperture of the cone at the encounter with the voxel/sample.
	*/
	template <typename TSamplerType, class TGPUCacheType>
	__device__
	inline void runImpl(
		const TSamplerType& pBrickSampler,
		TGPUCacheType& pGpuCache,
		bool& pRequestEmitted,
		const float3 pSamplePosScene,
		const float3 pRayDir,
		float& pRayStep,
		const float pConeAperture
	);

	/**
	* This method returns the cone aperture for a given distance.
	*
	* @param pTTree the current distance along the ray's direction.
	*
	* @return the cone aperture
	*/
	__device__
	inline float getConeApertureImpl(const float pTTree) const;

	/**
	* This method is called before each sampling to check whether or not the ray should stop.
	*
	* @param pRayPosTree the current ray's position in volume tree referential.
	*
	* @return true if you want to stop the ray, false otherwise.
	*/
	__device__
	__forceinline__ bool stopCriterionImpl(const float3& pRayPosTree) const;
};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "ShaderKernel.inl"

#endif // !_SHADER_KERNEL_H_