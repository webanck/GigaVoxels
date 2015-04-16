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

//Globals for pipeline access from CUDA shader.
__device__ typename ShaderType::KernelType::DataStructureType * G_D_DATA_STRUCTURE;
__device__ typename ShaderType::KernelType::DataStructureType::VolTreeKernelType * G_D_DATA_STRUCTURE_KERNEL;
__device__ typename ShaderType::KernelType::CacheType * G_D_CACHE;

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
inline float3 ShaderKernel<TProducerType, TDataStructureType, TCacheType>::shadePointLight( float3 materialColor, float3 normalVec, float3 lightVec, float3 eyeVec, float3 ambientTerm, float3 diffuseTerm, float3 specularTerm ) {

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
template <typename BrickSamplerType>
__device__
inline void ShaderKernel<TProducerType, TDataStructureType, TCacheType>::runImpl(const BrickSamplerType& brickSampler, const float3 samplePosScene, const float3 rayDir, float& rayStep, const float coneAperture) {
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

			//Shadows.
			const float light_intensity = marchShadowRay<BrickSamplerType>(brickSampler, samplePosScene, rayStep, coneAperture);
			// const float light_intensity = 1.f;
			//Common shading.
			const float3 shaded_color = shadePointLight(color, normalVec, lightVec, viewVec, ambient, diffuse, specular) * light_intensity;
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

template <typename TProducerType, typename TDataStructureType, typename TCacheType>
template <typename BrickSamplerType>
__device__
float ShaderKernel<TProducerType, TDataStructureType, TCacheType>::marchShadowRay(const BrickSamplerType& brickSampler, const float3 samplePosScene, float& rayStep, const float screenConeAperture) {

	// The shader used for shadow ray marching.
	//ShadowRayShaderKernel<BrickSamplerType> shadowsShader;

	const float3 lightVec 		= samplePosScene - cLightPosition;
	const float3 lightDirection = normalize(lightVec);
	const float  lightDistance 	= length(lightVec);
	const float  lightDiameter	= 0.f; //a parameter to be: light tweaking if not point light
	// const float sampleDiameter = ???? => retrieved during structure traversal

	float coneAperture = screenConeAperture; //The cone aperture is initialized with the cone aperture which yielded the current sample.
	float marched_length = 0.f;
	// float light_intensity = 1.f; //We are considering a light intensity which can only decrease.
	//TODO: consider only the distance to the light which is actually INSIDE the voxels geometry (outside empty).
	ShadowRayShaderKernel * shader = new ShadowRayShaderKernel();
	while(marched_length < lightDistance && /*_shadowsShader*/shader->getColor().w < 1.f) {

		// Declare an empty node of the data structure.
		// It will be filled during the traversal according to current sample position "samplePosTree".
		GvStructure::GvNode node;

		// [ 1 ] - Descent the data structure (in general an octree)
		// until max depth is reach or current traversed node has no subnodes,
		// or cone aperture is greater than voxel size.
		float sampleDiameter = 0.f;
		float3 sampleOffsetInNodeTree = make_float3(0.f);
		//GvRendering::GvSamplerKernel<typename TDataStructureType::VolTreeKernelType> new_brickSampler;
		BrickSamplerType new_brickSampler;
		bool modifInfoWriten = false;
		//
		//
		// // bool TPriorityOnBrick = true; //TPriorityOnBrick = ? => bool(true)
		//typename TDataStructureType::VolTreeKernelType pDataStructure = _dataStructure->volumeTreeKernel();//pDataStructure = ?
		//TCacheType * pCache = _cache;//pCache = ?
		const float3 samplePosTree = samplePosScene;//samplePosTree != samplePosScene ?
		const float const_coneAperture = coneAperture;
		const float3 pRayDirTree = lightDirection; //pRayDirTree = ?

		typename TCacheType::DataProductionManagerKernelType new_cache = G_D_CACHE->getKernelObject();
		GvRendering::GvNodeVisitorKernel::visit<
			true,
			TDataStructureType::VolTreeKernelType,
			//TDataStructureType,
			typename TCacheType::DataProductionManagerKernelType
		>(
			//pDataStructure,
			//*pCache,
			*G_D_DATA_STRUCTURE_KERNEL,
			//(G_D_CACHE->getKernelObject()),
			new_cache,
			node,
			samplePosTree,
			const_coneAperture,
			sampleDiameter,
			sampleOffsetInNodeTree,
			new_brickSampler,
			//brickSampler,
			modifInfoWriten
		);
	// (
	// 	DataStructureType,
	// 	DataProductionManagerType,
	// 	GvStructure::GvNode,
	// 	const float3,
	// 	float,
	// 	float,
	// 	float3,
	// 	GvRendering::GvSamplerKernel<GvStructure::VolumeTreeKernel<
	// 		DataType,
	// 		NodeRes,
	// 		BrickRes,
	// 		1U>
	// 	>,
	// 	__nv_bool
	// )
	// 	(
	// 	TVolTreeKernelType &  	pVolumeTree,
	// 	GPUCacheType &  	pGpuCache,
	// 	GvStructure::GvNode &  	pNode,
	// 	const float3  	pSamplePosTree,
	// 	const float  	pConeAperture,
	// 	float &  	pNodeSizeTree,
	// 	float3 &  	pSampleOffsetInNodeTree,
	// 	GvSamplerKernel< TVolTreeKernelType > &  	pBrickSampler,
	// 	bool &  	pRequestEmitted
	// )

		const float rayLengthInNodeTree = GvRendering::getRayLengthInNode(sampleOffsetInNodeTree, sampleDiameter, pRayDirTree);

		//Different cases regarding the retrieved node: a brick or not.
		if(!node.isBrick()) {
			marched_length += rayLengthInNodeTree;
		} else {
			// Where the "shading is done": just accumulating alpha from brick samples.
			//bool TFastUpdateMode = true; //TFastUpdateMode = ?

			//GvUtils::GvSimpleHostShader<ShadowRayShaderKernel<BrickSamplerType> > pShader(); //pShader = ? => shadowsShader
			//shader
			const float3 pRayStartTree = samplePosScene; //pRayStartTree = ?
			const float ptTree = marched_length; //ptTree = ?
			//TODO: pourquoi que 2 argumpents aux templates?? template<bool TFastUpdateMode, bool TPriorityOnBrick, class TVolumeTreeKernelType , class TSampleShaderType , class TGPUCacheType >
			const float rayLengthInBrick = GvRendering::GvBrickVisitorKernel::visit<
				true,
				true,
				TDataStructureType::VolTreeKernelType,
				ShadowsShaderType,
				typename TCacheType::DataProductionManagerKernelType
			>(
				/**pDataStructure,
				*_shadowsShader,
				*pCache,*/
				*G_D_DATA_STRUCTURE_KERNEL,
				*shader,
				//*G_D_CACHE,
				new_cache,
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
		// marched_length += rayLengthInNodeTree;
		//Update the cone aperture (thales theorem) depending on the position between the sample and the light and the diameters of the light and the sample.
		coneAperture = marched_length * (lightDiameter - sampleDiameter) / lightDistance + sampleDiameter;
	}

	//return (_shadowsShader->getColor().w < 1.f ? 1.f - _shadowsShader->getColor().w : 0.f);
	float result = (shader->getColor().w < 1.f ? 1.f - shader->getColor().w : 0.f);
	delete shader;
	return result;

	// return 0.75f;

	// // Ray marching.
	// // Step with "ptTree" along ray from start to stop bounds.
	// int numLoop = 0;
	// while
	// 	( ptTree < ptMaxTree
	// 	&& numLoop < 5000	// TO DO : remove this hard-coded value or let only for DEBUG
	// 	&& !pShader.stopCriterion( samplePosTree ) )
	// {
	// 	//float3 samplePosTree = pRayStartTree + ptTree * pRayDirTree;
	// 	const float coneAperture = pShader.getConeAperture( ptTree );
	//
	// 	// Declare an empty node of the data structure.
	// 	// It will be filled during the traversal according to cuurent sample position "samplePosTree"
	// 	GvStructure::GvNode node;
	//
	// 	// [ 1 ]- Descent the data structure (in general an octree)
	// 	// until max depth is reach or current traversed node has no subnodes,
	// 	// or cone aperture is greater than voxel size.
	// 	float nodeSizeTree;
	// 	float3 sampleOffsetInNodeTree;
	// 	bool modifInfoWriten = false;
	// 	GvNodeVisitorKernel::visit< TPriorityOnBrick >
	// 						( pDataStructure, pCache, node, samplePosTree, coneAperture,
	// 						nodeSizeTree, sampleOffsetInNodeTree, brickSampler, modifInfoWriten );
	//
	// 	const float rayLengthInNodeTree = getRayLengthInNode( sampleOffsetInNodeTree, nodeSizeTree, pRayDirTree );
	//
	// 	// [ 2 ] - If node is a brick, renderer it.
	// 	if ( node.isBrick() )	// todo : check if it should be hasBrick() instead !??????????
	// 	{
	// 		// PASCAL
	// 		// This is used to stop the ray with a z-depth value smaller than the remaining brick ray length
	// 		//
	// 		// QUESTION : pas forcément, si objet qui cache est transparent !??
	// 		// => non, comme d'hab en OpenGL => dessiner d'abord les objets opaques
	// 		const float rayLengthInBrick = mincc( rayLengthInNodeTree, ptMaxTree - ptTree );	// ==> attention, ( ptMaxTree - ptTree < 0 ) ?? ==> non, à casue du test du WHILE !! c'est OK !!
	// 																							// MAIS possible en cas d'erreur sur "float" !!!!!
	//
	// 		// This is where shader program occurs
			// float dt = GvBrickVisitorKernel::visit< TFastUpdateMode, TPriorityOnBrick >
			// 								( pDataStructure, pShader, pCache, pRayStartTree, pRayDirTree,
			// 								//ptTree, rayLengthInNodeTree, brickSampler, modifInfoWriten );
			// 								ptTree, rayLengthInBrick, brickSampler, modifInfoWriten );
	//
	// 		ptTree += dt;
	// 	}
	// 	else
	// 	{
	// 		ptTree += rayLengthInNodeTree;// + coneAperture;
	// 		ptTree += pShader.getConeAperture( ptTree );
	// 	}
	//
	// 	samplePosTree = pRayStartTree + ptTree * pRayDirTree;
	//
	// 	// Update internal counter
	// 	numLoop++;
	// } // while

	//return 1.f;
}

template <typename TProducerType, typename TDataStructureType, typename TCacheType>
__host__
void ShaderKernel<TProducerType, TDataStructureType, TCacheType>::initialize(PipelineType * pPipeline) {
	// _dataStructure = pPipeline->editDataStructure();
	// _cache = pPipeline->editCache();
	// _shadowsShader = new ShadowRayShaderKernel();
	// _pipeline = pPipeline;

	cudaMalloc((void **)&G_D_DATA_STRUCTURE_KERNEL, sizeof(TDataStructureType *));
	cudaMalloc((void **)&G_D_DATA_STRUCTURE_KERNEL, sizeof(typename TDataStructureType::VolTreeKernelType *));
	cudaMalloc((void **)&G_D_CACHE, sizeof(TCacheType *));
	// cudaMalloc((void **)&G_D_CACHE_KERNEL, sizeof(typename TCacheType::DataProductionManagerKernelType *));

	TDataStructureType * data_structure = pPipeline->editDataStructure();
	typename TDataStructureType::VolTreeKernelType * data_structure_kernel = &(pPipeline->editDataStructure()->volumeTreeKernel);
	TCacheType * cache = pPipeline->editCache();
	// typename TCacheType::DataProductionManagerKernelType * cache_kernel = &(pPipeline->editCache()->getKernelObject());

	cudaMemcpy(G_D_DATA_STRUCTURE, &data_structure, sizeof(TDataStructureType *), cudaMemcpyHostToDevice);
	cudaMemcpy(G_D_DATA_STRUCTURE_KERNEL, &data_structure_kernel, sizeof(typename TDataStructureType::VolTreeKernelType *), cudaMemcpyHostToDevice);
	cudaMemcpy(G_D_CACHE, &cache, sizeof(TCacheType *), cudaMemcpyHostToDevice);
	// cudaMemcpy(G_D_CACHE_KERNEL, &cache_kernel, sizeof(typename TCacheType::DataProductionManagerKernelType *), cudaMemcpyHostToDevice);
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
template <typename BrickSamplerType>
__device__
void ShadowRayShaderKernel::runImpl(const BrickSamplerType& brickSampler, const float3 samplePosScene, const float3 rayDir, float& rayStep, const float coneAperture) {
	// _accColor.x = 0.5f;
	// _accColor.y = 0.5f;
	// _accColor.z = 0.5f;
	// _accColor.w = 0.5f;
}
