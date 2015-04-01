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

#ifndef _SHADER_KERNEL_H_
#define _SHADER_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvRendering/GvIRenderShader.h>
#include <GvUtils/GvCommonShaderKernel.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Light position
 */
__constant__ float3 cLightPosition;

/**
 * Depth of field parameters
 */
__constant__ float3 cDofParameters;

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
 * @struct ShaderKernel
 *
 * @brief The ShaderKernel struct provides the way to shade the data structure.
 *
 * It is used in conjonction with the base class GvIRenderShader to implement the shader functions.
 */
class ShaderKernel
:	public GvUtils::GvCommonShaderKernel
,	public GvRendering::GvIRenderShader< ShaderKernel >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Do the shading equation at a givent point
	 *
	 * @param pMaterialColor material color
	 * @param pNormalVec normal
	 * @param pLightVec light vector
	 * @param pEyeVec eye vector
	 * @param pAmbientTerm ambiant term
	 * @param pDiffuseTerm diffuse term
	 * @param pSpecularTerm specular term
	 *
	 * @return the computed color
	 */
	__device__
	inline float3 shadePointLight( const float3 pMaterialColor, const float3 pNormalVec, const float3 pLightVec, const float3 pEyeVec,
								const float3 pAmbientTerm, const float3 pDiffuseTerm, const float3 pSpecularTerm );

	/**
	 * This method returns the cone aperture for a given distance.
	 *
	 * @param pTTree the current distance along the ray's direction.
	 *
	 * @return the cone aperture
	 */
	__device__
	inline float getConeApertureImpl( const float pTTree ) const;

	/**
	 * This method is called for each sample. For example, shading or secondary rays
	 * should be done here.
	 *
	 * @param pBrickSampler brick sampler
	 * @param pSamplePosScene position of the sample in the scene
	 * @param pRayDir ray direction
	 * @param pRayStep ray step
	 * @param pConeAperture cone aperture
	 */
	template< typename TSamplerType >
	__device__
	inline void runImpl( const TSamplerType& pBrickSampler, const float3 pSamplePosScene,
						const float3 pRayDir, float& pRayStep, const float pConeAperture );

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

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "ShaderKernel.inl"

#endif // !_SHADER_KERNEL_H_
