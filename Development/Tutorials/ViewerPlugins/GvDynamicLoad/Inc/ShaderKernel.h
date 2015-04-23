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
 * ...
 */
__constant__ float3 cLightPosition;

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
template <typename TProducerType, typename TDataStructureType, typename TCacheType>
class ShaderKernel:
	public GvUtils::GvCommonShaderKernel,
	public GvRendering::GvIRenderShader< ShaderKernel<TProducerType, TDataStructureType, TCacheType> >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:
	/******************************* TYPES ************************************/
	typedef TDataStructureType DataStructureType;
	typedef TCacheType CacheType;

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
	 * ...
	 *
	 * @param brickSampler ...
	 * @param samplePosScene ...
	 * @param rayDir ...
	 * @param rayStep ...
	 * @param coneAperture ...
	 */
	template <typename TSamplerType, class TGPUCacheType>
	__device__
	inline void runImpl(
		const TSamplerType& pBrickSampler,
		TGPUCacheType& pGpuCache,
		const float3 pSamplePosScene,
		const float3 pRayDir,
		float& pRayStep,
		const float pConeAperture
	);

	/**
	* Traces a ray/cone from a voxel sample to light to integrate light absorbtion.
	*
	* @param pBrickSampler ...
	* @param pGpuCache ...
	* @param pSamplePosScene ...
	* @param pRayStep ...
	* @param pScreenConeAperture ...
	* @return The remaining light intensity (1.0 for full light, 0.0 for full shadow).
	*/
	template <typename TSamplerType, class TGPUCacheType>
	__device__
	float marchShadowRay(
		const TSamplerType& pBrickSampler,
		TGPUCacheType& pGpuCache,
		const float3 pSamplePosScene,
		float& pRayStep,
		const float pScreenConeAperture
	);

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
	/**
	* ...
	*
	* @param brickSampler ...
	* @param samplePosScene ...
	* @param rayDir ...
	* @param rayStep ...
	* @param coneAperture ...
	*/
	template <typename TSamplerType, class TGPUCacheType>
	__device__
	void runImpl(
		const TSamplerType& pBrickSampler,
		TGPUCacheType& pGpuCache,
		const float3 pSamplePosScene,
		const float3 pRayDir,
		float& pRayStep,
		const float pConeAperture
	);
};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "ShaderKernel.inl"

#endif // !_SHADER_KERNEL_H_
