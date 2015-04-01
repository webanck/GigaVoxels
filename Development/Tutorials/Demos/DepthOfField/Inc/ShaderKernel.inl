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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvRendering/GvRendererContext.h>

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
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
 ******************************************************************************/
__device__
inline float3 ShaderKernel::shadePointLight( const float3 pMaterialColor, const float3 pNormalVec, const float3 pLightVec, const float3 pEyeVec,
											const float3 pAmbientTerm, const float3 pDiffuseTerm, const float3 pSpecularTerm )
{
	float3 final_color = pMaterialColor * pAmbientTerm;

	//float lightDist = length( pLightVec );
	float3 lightVecNorm = ( pLightVec );
	const float lambertTerm = ( dot( pNormalVec, lightVecNorm ) );

	if ( lambertTerm > 0.0f )
	{
		// Diffuse
		final_color += pMaterialColor * pDiffuseTerm * lambertTerm;

		// Specular term
		//const float3 halfVec = normalize( lightVecNorm + pEyeVec );//*0.5f;
		//const float specular = __powf( max( dot( pNormalVec, halfVec ), 0.0f ), 64.0f );
		//final_color += make_float3( specular ) * pSpecularTerm;
	}

	return final_color;
}

/******************************************************************************
 * This method returns the cone aperture for a given distance.
 *
 * @param pTTree the current distance along the ray's direction.
 *
 * @return the cone aperture
 ******************************************************************************/
__device__
inline float ShaderKernel::getConeApertureImpl( const float pTTree ) const
{
	// New values
	float aperture = cDofParameters.x;//userParam.y / 10.0f;
	float focalLength = cDofParameters.y;//userParam.z / 200.0f;
	float planeInFocus = cDofParameters.z;//userParam.x / 20.0f;

	// Compute Circle of Confusion (CoC)
	///float R= fabs( aperture * (focalLength*(dist-planeInFocus))/(dist*(planeInFocus-focalLength))   );
	//float R=(aperture*focalLength)/(planeInFocus-focalLength) * abs(dist-planeInFocus)/dist;
	//float CoC = aperture * focalLength / (planeInFocus - focalLength) * fabsf(planeInFocus - pTTree) / pTTree;
	float CoC = fabsf( aperture * ( focalLength * ( planeInFocus - pTTree ) ) / ( pTTree * ( planeInFocus - focalLength ) ) );
	//float CoC = aperture * focalLength / (planeInFocus - focalLength) * fabsf(planeInFocus - pTTree) / pTTree;

	//return k_pixelSize.x*R *k_frustumNearINV;

	float scaleFactor = 1.0f + CoC;

	// It is an estimation of the size of a voxel at given distance from the camera.
	// It is based on THALES theorem. Its computation is rotation invariant.
	return k_renderViewContext.pixelSize.x * pTTree * scaleFactor * k_renderViewContext.frustumNearINV;
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
template< typename TSamplerType >
__device__
inline void ShaderKernel::runImpl( const TSamplerType& pBrickSampler, const float3 pSamplePosScene,
									const float3 pRayDir, float& pRayStep, const float pConeAperture )
{
	float4 col = pBrickSampler.template getValue< 0 >( pConeAperture );

	if ( col.w > 0.0f )
	{
		float3 grad = make_float3( 0.0f );

		float gradStep = pRayStep * 0.25f;

		float4 v0, v1;

		v0 = pBrickSampler.template getValue< 0 >( pConeAperture, make_float3( gradStep, 0.0f, 0.0f) );
		v1 = pBrickSampler.template getValue< 0 >( pConeAperture, make_float3(-gradStep, 0.0f, 0.0f) );
		grad.x = v0.w - v1.w;

		v0 = pBrickSampler.template getValue< 0 >( pConeAperture, make_float3(0.0f,  gradStep, 0.0f) );
		v1 = pBrickSampler.template getValue< 0 >( pConeAperture, make_float3(0.0f, -gradStep, 0.0f) );
		grad.y = v0.w - v1.w;

		v0 = pBrickSampler.template getValue< 0 >( pConeAperture, make_float3(0.0f, 0.0f,  gradStep) );
		v1 = pBrickSampler.template getValue< 0 >( pConeAperture, make_float3(0.0f, 0.0f, -gradStep) );
		grad.z = v0.w - v1.w;

		if ( length( grad ) > 0.0f )
		{
			grad =- grad;
			grad = normalize( grad );

			float vis = 1.0f;

			float3 lightVec = normalize( cLightPosition - pSamplePosScene );
			float3 viewVec = (-1.0f * pRayDir);

			float3 rgb;
			rgb.x = col.x;
			rgb.y = col.y;
			rgb.z = col.z;
			rgb = shadePointLight( rgb, grad, lightVec, viewVec, make_float3( 0.2f * vis ), make_float3( 1.0f * (0.3f + vis*0.7f) ), make_float3(0.9f) );
			//col.x = rgb.x;
			//col.y = rgb.y;
			//col.z = rgb.z;

			// Due to alpha pre-multiplication
			//
			// Note : inspecting device SASS code, assembly langage seemed to compute (1.f / color.w) each time, that's why we use a constant and multiply
			const float alphaPremultiplyConstant = 1.f / col.w;
			col.x = rgb.x * alphaPremultiplyConstant;
			col.y = rgb.y * alphaPremultiplyConstant;
			col.z = rgb.z * alphaPremultiplyConstant;
		}
		else
		{
			//col = make_float4( 0.0f );

			// Problem !
			// In this case, the normal generated by the gradient is null...
			// That generates visual artefacts...
			//col = make_float4( 0.0f );
			//color = make_float4( 1.0, 0.f, 0.f, 1.0f );
			// Ambient : no shading
			//float3 final_color = materialColor * ambientTerm;
			float vis = 1.0f;
			float3 rgb;
			rgb.x = col.x; rgb.y = col.y; rgb.z = col.z;
			float3 final_color = rgb * make_float3( 0.2f * vis );

			// Due to alpha pre-multiplication
			//
			// Note : inspecting device SASS code, assembly langage seemed to compute (1.f / color.w) each time, that's why we use a constant and multiply
			const float alphaPremultiplyConstant = 1.f / col.w;
			col.x = final_color.x * alphaPremultiplyConstant;
			col.y = final_color.y * alphaPremultiplyConstant;
			col.z = final_color.z * alphaPremultiplyConstant;
		}

		// -- [ Opacity correction ] --
		// The standard equation :
		//		_accColor = _accColor + ( 1.0f - _accColor.w ) * color;
		// must take alpha correction into account
		// NOTE : if ( color.w == 0 ) then alphaCorrection equals 0.f
		const float alphaCorrection = ( 1.0f -_accColor.w ) * ( 1.0f - __powf( 1.0f - col.w, pRayStep * 512.f ) );

		// Accumulate the color
		_accColor.x += alphaCorrection * col.x;
		_accColor.y += alphaCorrection * col.y;
		_accColor.z += alphaCorrection * col.z;
		_accColor.w += alphaCorrection;
	}
}
