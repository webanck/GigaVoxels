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
inline void ShaderKernel::runImpl( const SamplerType& brickSampler, const float3 samplePosScene, const float3 rayDir, float& rayStep, const float coneAperture )
{
	float4 color = make_float4( 0.0f );
	float4 data = brickSampler.template getValue< 0 >( coneAperture );

	if ( data.w > 0.0f )
	{
		float3 transferFunction[] =
		{
			{0.f, 0.f, 1.f},
			{0.12f, 0.39f, 0.10f},
			{0.58f, 0.29f, 0.f}
		};

		int transferFunctionSize = sizeof( transferFunction ) / sizeof( transferFunction[ 0 ] );

		float trCoef = samplePosScene.y * (float)transferFunctionSize;

		int iCoef = (int)floorf(trCoef);
		float fCoef = trCoef - (float)iCoef;

		if (iCoef < transferFunctionSize - 1)
		{
			color.x = lerp(transferFunction[ iCoef ].x, transferFunction[ iCoef + 1 ].x, fCoef );
			color.y = lerp(transferFunction[ iCoef ].y, transferFunction[ iCoef + 1 ].y, fCoef );
			color.z = lerp(transferFunction[ iCoef ].z, transferFunction[ iCoef + 1 ].z, fCoef );
		}
		else
		{
			color.x = transferFunction[ transferFunctionSize - 1 ].x;
			color.y = transferFunction[ transferFunctionSize - 1 ].y;
			color.z = transferFunction[ transferFunctionSize - 1 ].z;
		}
		//float n = clamp(data.w * 2.f, 0.f, 1.f);
		//const float3 color0 = make_float3(0.12f, 0.39f, 0.10f);
		//const float3 color1 = make_float3(0.58f, 0.29f, 0.f);

		//color.x = lerp(color0.x, color1.x, n);
		//color.y = lerp(color0.y, color1.y, n);
		//color.z = lerp(color0.z, color1.z, n);
		color.w = 1.f;

		float3 grad = make_float3( 0.0f );

		float gradStep = rayStep * 0.25f;

		float4 v0, v1;

		v0 = brickSampler.template getValue<0>(coneAperture, make_float3( gradStep, 0.0f, 0.0f));
		v1 = brickSampler.template getValue<0>(coneAperture, make_float3(-gradStep, 0.0f, 0.0f));
		grad.x=v0.w-v1.w;

		v0 = brickSampler.template getValue<0>(coneAperture, make_float3(0.0f,  gradStep, 0.0f));
		v1 = brickSampler.template getValue<0>(coneAperture, make_float3(0.0f, -gradStep, 0.0f));
		grad.y=v0.w-v1.w;

		v0 = brickSampler.template getValue<0>(coneAperture, make_float3(0.0f, 0.0f,  gradStep));
		v1 = brickSampler.template getValue<0>(coneAperture, make_float3(0.0f, 0.0f, -gradStep));
		grad.z=v0.w-v1.w;

		if ( length( grad ) > 0.0f )
		{
			grad=-grad;
			grad=normalize(grad);

			float vis=1.0f;

			float3 lightVec = normalize( make_float3( 0.f, 10.f, 0.f ) - samplePosScene );
			float3 viewVec = -1.0f * rayDir;

			float3 rgb;
			rgb.x=color.x; rgb.y=color.y; rgb.z=color.z;
			rgb=shadePointLight(rgb, grad, lightVec, viewVec, make_float3( 0.2f * vis ), make_float3(1.0f*(0.3f+vis*0.7f)), make_float3(0.9f) );
			//color.x = rgb.x; color.y = rgb.y; color.z = rgb.z;

			// Due to alpha pre-multiplication
			//
			// Note : inspecting device SASS code, assembly langage seemed to compute (1.f / color.w) each time, that's why we use a constant and multiply
			const float alphaPremultiplyConstant = 1.f / color.w;
			color.x = rgb.x * alphaPremultiplyConstant;
			color.y = rgb.y * alphaPremultiplyConstant;
			color.z = rgb.z * alphaPremultiplyConstant;
		}
		else
		{
			// Problem !
			// In this case, the normal generated by the gradient is null...
			// That generates visual artefacts...
			//col = make_float4( 0.0f );
			//color = make_float4( 1.0, 0.f, 0.f, 1.0f );
			// Ambient : no shading
			//float3 final_color = materialColor * ambientTerm;
			float vis = 1.0f;
			float3 rgb;
			rgb.x = color.x; rgb.y = color.y; rgb.z = color.z;
			float3 final_color = rgb * make_float3( 0.2f * vis );

			// Due to alpha pre-multiplication
			//
			// Note : inspecting device SASS code, assembly langage seemed to compute (1.f / color.w) each time, that's why we use a constant and multiply
			const float alphaPremultiplyConstant = 1.f / color.w;
			color.x = final_color.x * alphaPremultiplyConstant;
			color.y = final_color.y * alphaPremultiplyConstant;
			color.z = final_color.z * alphaPremultiplyConstant;
		}

		// -- [ Opacity correction ] --
		// The standard equation :
		//		_accColor = _accColor + ( 1.0f - _accColor.w ) * color;
		// must take alpha correction into account
		// NOTE : if ( color.w == 0 ) then alphaCorrection equals 0.f
		const float alphaCorrection = ( 1.0f -_accColor.w ) * ( 1.0f - __powf( 1.0f - color.w, rayStep * 512.f ) );

		// Accumulate the color
		_accColor.x += alphaCorrection * color.x;
		_accColor.y += alphaCorrection * color.y;
		_accColor.z += alphaCorrection * color.z;
		_accColor.w += alphaCorrection;
	}

	//// alpha pre-multiplication
	//color.x = color.x * color.w;
	//color.y = color.y * color.w;
	//color.z = color.z * color.w;

	//// accumulate the color
	//_accColor = _accColor + (1.0f - _accColor.w) * color;
}

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
inline float3 ShaderKernel::shadePointLight( float3 materialColor, float3 normalVec, float3 lightVec, float3 eyeVec, float3 ambientTerm, float3 diffuseTerm, float3 specularTerm )
{
	float3 final_color = materialColor * ambientTerm;

	//float lightDist=length(lightVec);
	float3 lightVecNorm= lightVec;
	float lambertTerm = dot( normalVec, lightVecNorm );

	if ( lambertTerm > 0.0f )
	{
		//Diffuse
		final_color += materialColor*diffuseTerm*lambertTerm ;

		float3 halfVec = normalize( lightVecNorm + eyeVec );//*0.5f;
		float specular = __powf( max( dot( normalVec, halfVec), 0.0f ), 64.0f );

		//Specular
		//final_color += make_float3(specular)*specularTerm;
	}

	return final_color;
}
