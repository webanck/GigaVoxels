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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
//#include "GvRendering/GvRendererContext.h"

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

namespace GvUtils
{

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
	static __forceinline__ float3 ambientLightingModel( const float3 pMaterialColor, const float3 pAmbientTerm )
	{
		// Ambient
		return pMaterialColor * pAmbientTerm;
	}

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
	static __forceinline__ float3 ambientAndDiffuseLightingModel( const float3 pMaterialColor, const float3 pNormalVec, const float3 pLightVec,
								const float3 pAmbientTerm, const float3 pDiffuseTerm )
	{
		// Ambient
		float3 final_color = pMaterialColor * pAmbientTerm;

		// Diffuse
		//float lightDist = length(pLightVec);
		float3 lightVecNorm = ( pLightVec );
		const float lambertTerm = dot( pNormalVec, lightVecNorm );
		if ( lambertTerm > 0.0f )
		{
			// Diffuse
			final_color += pMaterialColor * pDiffuseTerm * lambertTerm;
		}

		return final_color;
	}

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
	static __forceinline__ float3 ADSLightingModel( const float3 pMaterialColor, const float3 pNormalVec, const float3 pLightVec, const float3 pEyeVec,
								const float3 pAmbientTerm, const float3 pDiffuseTerm, const float3 pSpecularTerm )
	{
		// Ambient
		float3 final_color = pMaterialColor * pAmbientTerm;

		// Diffuse
		//float lightDist=length(pLightVec);
		float3 lightVecNorm = ( pLightVec );
		float lambertTerm = ( dot( pNormalVec, lightVecNorm ) );
		if ( lambertTerm > 0.0f )
		{
			// Diffuse
			final_color += pMaterialColor * pDiffuseTerm * lambertTerm;

			// Specular
			float3 halfVec = normalize( lightVecNorm + pEyeVec );//*0.5f;
			float specular = __powf( max( dot( pNormalVec, halfVec ), 0.0f ), 64.0f );
			final_color += make_float3( specular ) * pSpecularTerm;
		}

		return final_color;
	}

/******************************************************************************
 * This method is called for each sample. For example, shading or secondary rays
 * should be done here.
 *
 * @param pBrickSampler brick sampler
 * @param pSamplePosScene position of the sample in the scene
 * @param pRayDir ray direction
 * @param pRayStep ray step
 * @param pConeAperture cone aperture
 ******************************************************************************/
template< typename SamplerType >
__device__
__forceinline__ void GvCommonShaderKernel::runImpl( const SamplerType& brickSampler, const float3 samplePosScene,
										const float3 rayDir, float& rayStep, const float coneAperture )
{
}

} // namespace GvUtils
