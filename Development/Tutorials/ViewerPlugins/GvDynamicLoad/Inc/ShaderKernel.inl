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
#include <GvRendering/GvRendererContext.h>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
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
 * @return ...
 ******************************************************************************/
__device__
inline float3 ShaderKernel::shadePointLight( float3 materialColor, float3 normalVec, float3 lightVec, float3 eyeVec,
	float3 ambientTerm, float3 diffuseTerm, float3 specularTerm )
{
	float3 final_color = materialColor * ambientTerm;

	//float lightDist=length(lightVec);
	float3 lightVecNorm = (lightVec);
	float lambertTerm = (dot( normalVec, lightVecNorm ));

	if ( lambertTerm > 0.0f )
	{
		//Diffuse
		final_color += materialColor * diffuseTerm * lambertTerm ;

		float3 halfVec = normalize( lightVecNorm + eyeVec );//*0.5f;
		float specular = __powf( max( dot( normalVec, halfVec ), 0.0f ), 64.0f );

		//Specular
		//final_color += make_float3(specular)*specularTerm;
	}

	return final_color;
}

/******************************************************************************
 * ...
 *
 * @param brickSampler ...
 * @param samplePosScene ...
 * @param rayDir ...
 * @param rayStep ...
 * @param coneAperture ...
 ******************************************************************************/
template< typename BrickSamplerType >
__device__
inline void ShaderKernel::runImpl(const BrickSamplerType& brickSampler, const float3 samplePosScene, const float3 rayDir, float& rayStep, const float coneAperture) {
	//Retrieve material color from voxel's attached data.
	const float4 material_color = brickSampler.template getValue<0>(coneAperture);
	const float alpha = material_color.w;

	//Process only visible voxels.
	if(alpha > 0.f) {
		//Retrieve normal from voxel's attached data.
		const float4 normal  = brickSampler.template getValue<1>(coneAperture);
		const float3 normal3 = make_float3(normal.x, normal.y, normal.z);

		//Process only data with non null normal.
		if(length(normal3) > 0.f) {
			const float3 normalVec 	= normalize(normal3);
			const float3 color 		= make_float3(material_color.x, material_color.y, material_color.z);
			const float3 lightVec 	= normalize(cLightPosition - samplePosScene);
			const float3 viewVec 	= -1.f * rayDir;
			const float3 ambient	= make_float3(0.2f);
			const float3 diffuse	= make_float3(1.f);
			const float3 specular	= make_float3(0.9f);

			//Common shading.
			const float3 shaded_color = shadePointLight(color, normalVec, lightVec, viewVec, ambient, diffuse, specular);
			// -- [ Opacity correction ] --
			// The standard equation :
			//		_accColor = _accColor + ( 1.0f - _accColor.w ) * color;
			// must take alpha correction into account
			// NOTE : if ( color.w == 0 ) then alphaCorrection equals 0.f
			const float alpha_correction = (1.f - _accColor.w) * (1.f - __powf(1.f - alpha, rayStep * 512.f));//pourquoi 512??
			const float3 corrected_color = shaded_color / alpha * alpha_correction;
			_accColor.x += corrected_color.x;
			_accColor.y += corrected_color.y;
			_accColor.z += corrected_color.z;
			_accColor.w += alpha_correction;


			//Printing only normals.
			// if(length(normalVec) > 0) {
			// 	const float3 normalized = normalize(normalVec);
			// 	_accColor = make_float4(
			// 		(normalized.x + 1.f)/2.f,
			// 		(normalized.y + 1.f)/2.f,
			// 		(normalized.z + 1.f)/2.f,
			// 		1.f
			// 	);
			// }

		}
	}
}
