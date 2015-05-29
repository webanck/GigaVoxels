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

#define ALPHA_MULTIPLIER 100.f

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

__device__
inline float3 ShaderKernel::shadePointLight(
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
		final_color += make_float3(specular)*specularTerm;
	}

	// return clamp(final_color, 0.f, 1.f);
	return final_color;
}

template <typename TSamplerType, typename TGPUCacheType>
__device__
inline void ShaderKernel::runImpl(
	const TSamplerType& pBrickSampler,
	TGPUCacheType& pGpuCache,
	bool& pRequestEmitted,
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
			const float3 lightVec 	= normalize(cLightPositionTree - pSamplePosScene);
			const float3 viewVec 	= -1.f * pRayDir;
			const float3 ambient	= make_float3(0.2f);
			const float3 diffuse	= make_float3(1.f);
			const float3 specular	= make_float3(0.9f);
			const float voxelSize 	= pBrickSampler._nodeSizeTree/BrickRes::getFloat3().x;


			//Shadows.
			const float4 light_intensity = marchShadowRay(pBrickSampler, pGpuCache, pRequestEmitted, pSamplePosScene, pConeAperture);
			// //Common shading.
			const float3 shaded_color = shadePointLight(color, normalVec, lightVec, viewVec, ambient, diffuse, specular);


			// // -- [ Opacity correction ] --
			// // The standard equation :
			// //		_accColor = _accColor + ( 1.0f - _accColor.w ) * color;
			// // must take alpha correction into account
			// // NOTE : if ( color.w == 0 ) then alphaCorrection equals 0.f
			const float alpha_correction = (1.f - _accColor.w) * (1.f - __powf(1.f - alpha, pRayStep * cShaderMaterialProperty));
			// const float alpha_correction = voxelSize * ALPHA_MULTIPLIER;
			// // _accColor.w = 1.f; //mat objects

			//0)Original.
			const float3 corrected_color = shaded_color / alpha * alpha_correction;
			_accColor.x += corrected_color.x;
			_accColor.y += corrected_color.y;
			_accColor.z += corrected_color.z;
			_accColor.w += alpha_correction;

			// // 1) Shadows only.
			// _accColor.x += color.x / alpha * alpha_correction * light_intensity.x * light_intensity.w;
			// _accColor.y += color.y / alpha * alpha_correction * light_intensity.y * light_intensity.w;
			// _accColor.z += color.z / alpha * alpha_correction * light_intensity.z * light_intensity.w;
			// _accColor.w += alpha_correction;

			// //2) Shaded and shadowed.
			// _accColor.x += shaded_color.x / alpha * alpha_correction * light_intensity.x * light_intensity.w;
			// _accColor.y += shaded_color.y / alpha * alpha_correction * light_intensity.y * light_intensity.w;
			// _accColor.z += shaded_color.z / alpha * alpha_correction * light_intensity.z * light_intensity.w;
			// _accColor.w += alpha_correction;

			// _accColor.w = 1.f; //mat objects


			// // Printing only normals.
			// if(length(normalVec) > 0) {
			// 	const float3 normalized = normalize(normalVec);
			// 	_accColor = make_float4(
			// 		(normalized.x + 1.f)/2.f,
			// 		(normalized.y + 1.f)/2.f,
			// 		(normalized.z + 1.f)/2.f,
			// 		1.f
			// 	);
			// }

			// //Printing only sample position.
			// _accColor = make_float4(
			// 	pSamplePosScene.x,
			// 	pSamplePosScene.y,
			// 	pSamplePosScene.z,
			// 	1.f
			// );

			// //Printing only light direction.
			// _accColor = make_float4(
			// 	(lightVec.x + 1.f)/2.f,
			// 	(lightVec.y + 1.f)/2.f,
			// 	(lightVec.z + 1.f)/2.f,
			// 	1.f
			// );


			// //Printing only view direction.
			// _accColor = make_float4(
			// 	(viewVec.x + 1.f)/2.f,
			// 	(viewVec.y + 1.f)/2.f,
			// 	(viewVec.z + 1.f)/2.f,
			// 	1.f
			// );

			//Printing only if a request has been added to the cache.
			// _accColor = make_float4(
			// 	pRequestEmitted,
			// 	pRequestEmitted,
			// 	pRequestEmitted,
			// 	1.f
			// );

			// //Printing only node size.
			// _accColor = make_float4(
			// 	pBrickSampler._nodeSizeTree/0.5f,
			// 	pBrickSampler._nodeSizeTree/0.5f,
			// 	pBrickSampler._nodeSizeTree/0.5f,
			// 	1.f
			// );

			// //Printing only cone aperture.
			// _accColor = make_float4(
			// 	pConeAperture/0.01f,
			// 	pConeAperture/0.01f,
			// 	pConeAperture/0.01f,
			// 	1.f
			// );

			//debug
			// _accColor = make_float4(normal3, 1.f);

		}
	}
}


template <typename TSamplerType, class TGPUCacheType>
__device__
float4 ShaderKernel::marchShadowRay(
	const TSamplerType& pBrickSampler,
	TGPUCacheType& pGpuCache,
	bool& pRequestEmitted,
	const float3 pSamplePosTree,
	const float pScreenConeAperture
) const {
	const bool priorityOnBrick 		= false;
	const float brickDivisions 		= BrickRes::getFloat3().x;
	const float3 firstSamplePosTree = pSamplePosTree;
	const float firstSampleDiameter = pBrickSampler._nodeSizeTree/brickDivisions;
	const float3 lightVec 			= cLightPositionTree - firstSamplePosTree;
	const float3 lightDirection 	= normalize(lightVec);
	const float lightDistance 		= length(lightVec); //TODO: consider only the distance to the light which is actually INSIDE the voxels geometry (the remaining distance is empty).
	const float lightDiameter		= 0.0001f; //a parameter to be: light tweaking if not point light
	// float coneAperture 			= 1.33f * pBrickSampler._nodeSizeTree;
	float coneAperture 				= 1.33f * firstSampleDiameter;
	// const float starting_marched_length = 2.f * firstSampleDiameter;
	const float starting_marched_length = coneAperture;



	float marched_length = starting_marched_length;
	ShadowRayShaderKernel shader(lightDistance, lightDiameter, firstSampleDiameter); // The shader used for shadow ray marching.
	float3 samplePosTree = firstSamplePosTree + lightDirection * marched_length;
	const uint maxLoops = 100;
	uint i = 0;
	//Ray/cone marching from just after the first sample to the light and accumulating alpha in the shader.
	while(marched_length < lightDistance && !shader.stopCriterionImpl(samplePosTree) && i++ < maxLoops) {
		//Position of the next sample will give the node and offset in it's brick.
		float3 samplePosTree = firstSamplePosTree + lightDirection * marched_length;

		//The next node will be filled by the node visitor.
		GvStructure::GvNode node;
		float sampleDiameter = 0.f;
		float3 sampleOffsetInNodeTree = make_float3(0.f);
		//The new brick sampler will be filled by the node visitor.
		TSamplerType new_brickSampler;
		new_brickSampler._volumeTree = pBrickSampler._volumeTree;

		//Node visitor call.
		GvRendering::GvNodeVisitorKernel::visit<
			priorityOnBrick
		>(
			*(new_brickSampler._volumeTree),
			pGpuCache,
			node,
			samplePosTree,
			coneAperture,
			sampleDiameter,
			sampleOffsetInNodeTree,
			new_brickSampler,
			pRequestEmitted
		);
		//if(pRequestEmitted) break;

		const float rayLengthInNodeTree = GvRendering::getRayLengthInNode(sampleOffsetInNodeTree, sampleDiameter, lightDirection);

		//Different cases regarding the retrieved node: with a brick or not.
		if(!node.isBrick() /*|| node.isTerminal() || !node.hasSubNodes()*/) {
			marched_length += rayLengthInNodeTree;
			marched_length += shader.getConeApertureImpl(marched_length);
		} else {
			// Where the "shading is done": just accumulating alpha from brick samples.
			const float maxRayLengthInBrick = maxcc(0.f, mincc(rayLengthInNodeTree, lightDistance - marched_length));
			const float rayLengthInBrick = GvRendering::GvBrickVisitorKernel::visit<
				true,
				priorityOnBrick
			>(
				*(new_brickSampler._volumeTree),
				shader,
				pGpuCache,
				firstSamplePosTree,
				lightDirection,
				marched_length,
				maxRayLengthInBrick,
				new_brickSampler,
				pRequestEmitted
			);
			marched_length += rayLengthInBrick;
		}

		//Update the cone aperture (thales theorem) depending on the position between the sample and the light and the diameters of the light and the sample.
		coneAperture = shader.getConeApertureImpl(marched_length);
	}

	// //Debug printing only the marched distance ratio.
	// const float marched_ratio = mincc(marched_length/lightDistance, 1.f);
	// _accColor = make_float4(
	// 	marched_ratio,
	// 	marched_ratio,
	// 	marched_ratio,
	// 	1.f
	// );

	// // Debug printing
	// _accColor = make_float4(
	// 	i >= maxLoops,
	// 	maxcc(mincc(marched_length/lightDistance, 1.f), shader.getColor().w >= 1.f),
	// 	pRequestEmitted,
	// 	1.f
	// );
	// // if(marched_length == starting_marched_length)
	// if(i >= maxLoops || marched_length == starting_marched_length)
	// 	_accColor.x = 1.f;
	//
	// if(shader.getColor().w >= 1.f && marched_length >= lightDistance)
	// 	_accColor.y = 1.f;
	//
	// if(pRequestEmitted)
	// 	_accColor.z = 1.f;

	// //Debug printing only light direction.
	// _accColor = make_float4(
	// 	(lightDirection.x + 1.f)/2.f,
	// 	(lightDirection.y + 1.f)/2.f,
	// 	(lightDirection.z + 1.f)/2.f,
	// 	1.f
	// );

	// //Debug printing only light distance.
	// _accColor = make_float4(
	// 	lightDistance,
	// 	lightDistance,
	// 	lightDistance,
	// 	1.f
	// );

	return make_float4(
		1.f - shader.getColor().x,
		1.f - shader.getColor().y,
		1.f - shader.getColor().z,
		1.f - shader.getColor().w
	);
}







__device__
ShadowRayShaderKernel::ShadowRayShaderKernel(/*const float3 pLightPosition, */const float pLightDistanceTree, const float pLightSize, const float pSampleSize):
	_lightDistanceTree(pLightDistanceTree),
	_lightSize(pLightSize),
	_sampleSize(pSampleSize)
{
	_accColor = make_float4(0.f);
	// _accColor.w = -1.f;
}

template <typename TSamplerType, class TGPUCacheType>
__device__
inline void ShadowRayShaderKernel::runImpl(
	const TSamplerType& pBrickSampler,
	TGPUCacheType& pGpuCache,
	bool& pRequestEmitted,
	const float3 pSamplePosScene,
	const float3 pRayDir,
	float& pRayStep,
	const float pConeAperture
) {
	//Retrieve material color from voxel's attached data.
	const float4 material_color = pBrickSampler.template getValue<0>(pConeAperture);
	const float alpha = material_color.w;
	const float voxelSize = pBrickSampler._nodeSizeTree/BrickRes::getFloat3().x;

	const float alpha_correction = (1.f - _accColor.w) * (1.f - __powf(1.f - alpha, pRayStep * cShaderMaterialProperty));

	//Process only visible voxels.
	if(alpha > 0.f) {
		//Smooth shadows.
		// _accColor.w += (1.f - _accColor.w) * alpha * voxelSize;

		//Accumulating the color absorbtion in RGB components, accounting for the alpha.

		// //1) The integration ins't linear so not the same regarding the voxelSize.
		// _accColor.x += (1.f - _accColor.x) * (1.f - material_color.x) * alpha * voxelSize;
		// _accColor.y += (1.f - _accColor.y) * (1.f - material_color.y) * alpha * voxelSize;
		// _accColor.z += (1.f - _accColor.z) * (1.f - material_color.z) * alpha * voxelSize;
		// // _accColor.w += (1.f - _accColor.w) * alpha * voxelSize;
		// _accColor.w += alpha * voxelSize;

		// //2) It should be better here but transparent effects observed, because of thin shell?
		// _accColor.x += (1.f - material_color.x) * alpha * voxelSize * ALPHA_MULTIPLIER;
		// _accColor.y += (1.f - material_color.y) * alpha * voxelSize * ALPHA_MULTIPLIER;
		// _accColor.z += (1.f - material_color.z) * alpha * voxelSize * ALPHA_MULTIPLIER;
		// _accColor.w += alpha * voxelSize;

		// //3) Transparent effects observed too but not so similar appearance between different levels of details.
		// _accColor.x += (1.f - material_color.x) / alpha * alpha_correction * voxelSize;
		// _accColor.y += (1.f - material_color.y) / alpha * alpha_correction * voxelSize;
		// _accColor.z += (1.f - material_color.z) / alpha * alpha_correction * voxelSize;
		// _accColor.w += alpha_correction * voxelSize;

		//4)
		_accColor.x += (1.f - material_color.x) / alpha * alpha_correction;
		_accColor.y += (1.f - material_color.y) / alpha * alpha_correction;
		_accColor.z += (1.f - material_color.z) / alpha * alpha_correction;
		_accColor.w += alpha_correction;


		//Avoid out of range values for next occurences.
		clamp(_accColor, 0.f, 1.f);
	}
}

__device__
inline float ShadowRayShaderKernel::getConeApertureImpl(const float tTree) const {
	//Compute the cone aperture (thales theorem) depending on the position between the sample and the light and the diameters of the light and the sample.

	// // Overestimate to avoid aliasing
	// const float scaleFactor = 1.333f;

	//Three different cases: light diameter bigger/smaller/same than the sample diameter.
	if(_lightSize > _sampleSize) return tTree/_lightDistanceTree*(_lightSize - _sampleSize) + _sampleSize;
	else if(_lightSize < _sampleSize) return (_lightDistanceTree - tTree)/_lightDistanceTree*(_sampleSize - _lightSize) + _lightSize;
	else return _lightSize;
}

__device__
__forceinline__ bool ShadowRayShaderKernel::stopCriterionImpl(const float3& pRayPosTree) const {
	return
		_accColor.w >= 1.f ||
		pRayPosTree.x < 0.f ||
		pRayPosTree.y < 0.f ||
		pRayPosTree.z < 0.f ||
		pRayPosTree.x > 1.f ||
		pRayPosTree.y > 1.f ||
		pRayPosTree.z > 1.f
	;
}
