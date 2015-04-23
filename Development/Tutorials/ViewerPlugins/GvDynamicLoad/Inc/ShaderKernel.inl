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

#include <GvRendering/GvRendererHelpersKernel.h>
#include <GvRendering/GvNodeVisitorKernel.h>
#include <GvRendering/GvBrickVisitorKernel.h>
#include <GvRendering/GvSamplerKernel.h>

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
template <typename TProducerType, typename TDataStructureType, typename TCacheType>
__device__
inline float3 ShaderKernel<TProducerType, TDataStructureType, TCacheType>::shadePointLight(
	float3 materialColor,
	float3 normalVec,
	float3 lightVec,
	float3 eyeVec,
	float3 ambientTerm,
	float3 diffuseTerm,
	float3 specularTerm
) {
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
template <typename TProducerType, typename TDataStructureType, typename TCacheType>
template <typename TSamplerType, typename TGPUCacheType>
__device__
inline void ShaderKernel<TProducerType, TDataStructureType, TCacheType>::runImpl(
	const TSamplerType& pBrickSampler,
	TGPUCacheType& pGpuCache,
	const float3 pSamplePosScene,
	const float3 pRayDir,
	float& pRayStep,
	const float pConeAperture
) {
	//Retrieve material color from voxel's attached data.
	const float4 material_color = pBrickSampler.template getValue<0>(pConeAperture);
	const float alpha = material_color.w;

	//Process only visible voxels.
	if(alpha > 0.f) {
		//Retrieve normal from voxel's attached data.
		const float4 normal  = pBrickSampler.template getValue<1>(pConeAperture);
		const float3 normal3 = make_float3(normal.x, normal.y, normal.z);

		//Process only data with non null normal.
		if(length(normal3) > 0.f) {
			const float3 normalVec 	= normalize(normal3);
			const float3 color 		= make_float3(material_color.x, material_color.y, material_color.z);
			const float3 lightVec 	= normalize(cLightPosition - pSamplePosScene);
			const float3 viewVec 	= -1.f * pRayDir;
			const float3 ambient	= make_float3(0.2f);
			const float3 diffuse	= make_float3(1.f);
			const float3 specular	= make_float3(0.9f);

			//Shadows.
			const float light_intensity = marchShadowRay(pBrickSampler, pGpuCache, pSamplePosScene, pRayStep, pConeAperture);
			//Common shading.
			const float3 shaded_color = shadePointLight(color, normalVec, lightVec, viewVec, ambient, diffuse, specular) * light_intensity;
			// -- [ Opacity correction ] --
			// The standard equation :
			//		_accColor = _accColor + ( 1.0f - _accColor.w ) * color;
			// must take alpha correction into account
			// NOTE : if ( color.w == 0 ) then alphaCorrection equals 0.f
			const float alpha_correction = (1.f - _accColor.w) * (1.f - __powf(1.f - alpha, pRayStep * 512.f));//pourquoi 512??
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

template <typename TProducerType, typename TDataStructureType, class TCacheType>
template <typename TSamplerType, class TGPUCacheType>
__device__
float ShaderKernel<TProducerType, TDataStructureType, TCacheType>::marchShadowRay(
	const TSamplerType& pBrickSampler,
	TGPUCacheType& pGpuCache,
	const float3 pSamplePosScene,
	float& pRayStep,
	const float pScreenConeAperture
) {

	const float3 lightVec 		= pSamplePosScene - cLightPosition;
	const float3 lightDirection = normalize(lightVec);
	const float  lightDistance 	= length(lightVec);
	const float  lightDiameter	= 0.f; //a parameter to be: light tweaking if not point light
	// const float sampleDiameter = ???? => retrieved during structure traversal

	float coneAperture = pScreenConeAperture; //The cone aperture is initialized with the cone aperture which yielded the current sample.
	float marched_length = 0.f;
	// float light_intensity = 1.f; //We are considering a light intensity which can only decrease.
	//TODO: consider only the distance to the light which is actually INSIDE the voxels geometry (outside empty).
	ShadowRayShaderKernel shader; // The shader used for shadow ray marching.

	while(marched_length < lightDistance && shader.getColor().w < 1.f) {

		// Declare an empty node of the data structure.
		// It will be filled during the traversal according to current sample position "samplePosTree".
		GvStructure::GvNode node;

		// [ 1 ] - Descent the data structure (in general an octree)
		// until max depth is reach or current traversed node has no subnodes,
		// or cone aperture is greater than voxel size.
		float sampleDiameter = 0.f;
		float3 sampleOffsetInNodeTree = make_float3(0.f);
		//The new brick sampler will be filled by the node visitor.
		TSamplerType new_brickSampler;
		new_brickSampler._volumeTree = pBrickSampler._volumeTree;
		bool modifInfoWriten = false;

		// // bool TPriorityOnBrick = true; //TPriorityOnBrick = ? => bool(true)
		const float3 samplePosTree = pSamplePosScene;//samplePosTree != samplePosScene ?
		const float const_coneAperture = coneAperture;
		const float3 pRayDirTree = lightDirection; //pRayDirTree = ?

		GvRendering::GvNodeVisitorKernel::visit<
			true
		>(
			*(new_brickSampler._volumeTree),
			pGpuCache,
			node,
			samplePosTree,
			const_coneAperture,
			sampleDiameter,
			sampleOffsetInNodeTree,
			new_brickSampler,
			modifInfoWriten
		);

		const float rayLengthInNodeTree = GvRendering::getRayLengthInNode(sampleOffsetInNodeTree, sampleDiameter, pRayDirTree);

		//Different cases regarding the retrieved node: a brick or not.
		if(!node.isBrick()) {
			marched_length += rayLengthInNodeTree;
		} else {
			// Where the "shading is done": just accumulating alpha from brick samples.
			const float3 pRayStartTree = pSamplePosScene; //pRayStartTree = ?
			const float ptTree = marched_length; //ptTree = ?
			const float rayLengthInBrick = GvRendering::GvBrickVisitorKernel::visit<
				true,
				true
			>(
				*(new_brickSampler._volumeTree),
				shader,
				pGpuCache,
				pRayStartTree,
				pRayDirTree,
				ptTree,
				rayLengthInNodeTree,
				new_brickSampler,
				modifInfoWriten
			);
			marched_length += rayLengthInBrick;
			// marched_length += coneAperture;
		}

		//Update the cone aperture (thales theorem) depending on the position between the sample and the light and the diameters of the light and the sample.
		coneAperture = marched_length * (lightDiameter - sampleDiameter) / lightDistance + sampleDiameter;
	}

	return (shader.getColor().w < 1.f ? 1.f - shader.getColor().w : 0.f);
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
template <typename TSamplerType, class TGPUCacheType>
__device__
inline void ShadowRayShaderKernel::runImpl(
	const TSamplerType& pBrickSampler,
	TGPUCacheType& pGpuCache,
	const float3 pSamplePosScene,
	const float3 pRayDir,
	float& pRayStep,
	const float pConeAperture
) {
	_accColor.x = 1.f;
	_accColor.y = 1.f;
	_accColor.z = 1.f;
	_accColor.w += 0.25f;
}
