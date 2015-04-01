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
inline void ShaderKernel::runImpl( const SamplerType& brickSampler, const float3 samplePosScene,
										const float3 rayDir, float& rayStep, const float coneAperture )
{
	// Retrieve first channel element : color
	float4 color = brickSampler.template getValue< 0 >( coneAperture );

	// Test opacity
	if ( color.w > 0.0f )
	{
		// Retrieve second channel element : approxiamted stocked normal -> to be replaced by the good one.
		const float4 normal = brickSampler.template getValue< 1 >( coneAperture );


		//Compute the normal (between the 6 faces possible): the voxel cube face hit by the ray.
		float3 normalVec;
		//Also need the brick's voxel size.
		const float voxelDemiSize = 0.025f;

		//Voxel corners:
		const float3 c000 = make_float3(samplePosScene.x - voxelDemiSize, samplePosScene.y - voxelDemiSize, samplePosScene.z - voxelDemiSize);
		const float3 c001 = make_float3(samplePosScene.x - voxelDemiSize, samplePosScene.y - voxelDemiSize, samplePosScene.z + voxelDemiSize);
		const float3 c010 = make_float3(samplePosScene.x - voxelDemiSize, samplePosScene.y + voxelDemiSize, samplePosScene.z - voxelDemiSize);
		const float3 c011 = make_float3(samplePosScene.x - voxelDemiSize, samplePosScene.y + voxelDemiSize, samplePosScene.z + voxelDemiSize);
		const float3 c100 = make_float3(samplePosScene.x + voxelDemiSize, samplePosScene.y - voxelDemiSize, samplePosScene.z - voxelDemiSize);
		const float3 c101 = make_float3(samplePosScene.x + voxelDemiSize, samplePosScene.y - voxelDemiSize, samplePosScene.z + voxelDemiSize);
		const float3 c110 = make_float3(samplePosScene.x + voxelDemiSize, samplePosScene.y + voxelDemiSize, samplePosScene.z - voxelDemiSize);
		const float3 c111 = make_float3(samplePosScene.x + voxelDemiSize, samplePosScene.y + voxelDemiSize, samplePosScene.z + voxelDemiSize);

		//Light direction.
		const float3 lightDir = make_float3(samplePosScene.x - cLightPosition.x, samplePosScene.y - cLightPosition.y, samplePosScene.z - cLightPosition.z);
		//A light inside the voxel is re-considered as outside.
		float3 lightPosition = cLightPosition;
		if(
			cLightPosition.x >= c000.x && cLightPosition.x <= c100.x &&
			cLightPosition.y >= c000.y && cLightPosition.y <= c010.y &&
			cLightPosition.z >= c000.z && cLightPosition.z <= c001.z
		) lightPosition += lightDir * 4.f * voxelDemiSize;

		//Depending of the area where the light is, we can choose one normal between the 6.
		// if( //On the front face of the first axis.
		// 	lightPosition.x <= c000.x &&
		// 	lightPosition.y >= c000.y && lightPosition.y <= c010.y &&
		// 	lightPosition.z >= c000.z && lightPosition.z <= c001.z
		// ) normalVec = make_float3(-1.f, 0.f, 0.f);
		// else if( //On the back face of the first axis.
		// 	lightPosition.x >= c100.x &&
		// 	lightPosition.y >= c000.y && lightPosition.y <= c010.y &&
		// 	lightPosition.z >= c000.z && lightPosition.z <= c001.z
		// ) normalVec = make_float3(1.f, 0.f, 0.f);
		// else if( //On the front face of the second axis.
		// 	lightPosition.x >= c000.x && lightPosition.x <= c100.x &&
		// 	lightPosition.y <= c000.y &&
		// 	lightPosition.z >= c000.z && lightPosition.z <= c001.z
		// ) normalVec = make_float3(0.f, -1.f, 0.f);
		// else if( //On the back face of the second axis.
		// 	lightPosition.x >= c000.x && lightPosition.x <= c100.x &&
		// 	lightPosition.y >= c010.y &&
		// 	lightPosition.z >= c000.z && lightPosition.z <= c001.z
		// ) normalVec = make_float3(0.f, 1.f, 0.f);
		// else if( //On the front face of the third axis.
		// 	lightPosition.x >= c000.x && lightPosition.x <= c100.x &&
		// 	lightPosition.y >= c000.y && lightPosition.y <= c010.y &&
		// 	lightPosition.z <= c000.z
		// ) normalVec = make_float3(0.f, 0.f, -1.f);
		// else if( //On the back face of the third axis.
		// 	lightPosition.x >= c000.x && lightPosition.x <= c100.x &&
		// 	lightPosition.y >= c000.y && lightPosition.y <= c010.y &&
		// 	lightPosition.z >= c001.z
		// ) normalVec = make_float3(0.f, 0.f, 1.f);
		// else { //default case: no easy area -> all areas should be covered and this case removed when finished
		// 	normalVec = normalize( make_float3( normal.x, normal.y, normal.z ) );
		// }

		normalVec = normalize( make_float3( normal.x, normal.y, normal.z ) );
		if(lightPosition.x < c000.x) {
			if(lightPosition.y < c000.y) {
				if(lightPosition.z < c000.z) { //Coin c000.
					normalVec = cornerNormalSelection(lightPosition, rayDir, c000);
				} else if(lightPosition.z > c001.z) { //Coin c001.
					// float3 tmp = cornerNormalSelection(lightPosition, rayDir, c001);
					// normalVec = make_float3(tmp.x, tmp.y, -tmp.z);
				} else { //Edge (c000, c001).
					const float2 lightToCorner_xy = make_float2(-lightPosition.x, -lightPosition.y);
					const float2 rayDir_xy = make_float2(rayDir.x, rayDir.y);
					const float orientedArea_xy = dot(lightToCorner_xy, rayDir_xy);

					if(orientedArea_xy > 0.f) normalVec = make_float3(1.f, 0.f, 0.f);
					else if(orientedArea_xy < 0.f) normalVec = make_float3(0.f, 1.f, 0.f);
					else normalVec = make_float3(0.5f, 0.5f, 0.f);
				}
			} else if(lightPosition.y > c010.y) {
				if(lightPosition.z < c000.z) {

				} else if(lightPosition.z > c001.z) {

				} else {

				}
			} else {
				if(lightPosition.z < c000.z) {

				} else if(lightPosition.z > c001.z) {

				} else {

				}
			}
		} else if(lightPosition.x > c100.x) {
			if(lightPosition.y < c000.y) {
				if(lightPosition.z < c000.z) {

				} else if(lightPosition.z > c001.z) {

				} else {

				}
			} else if(lightPosition.y > c010.y) {
				if(lightPosition.z < c000.z) {

				} else if(lightPosition.z > c001.z) { //Coin c111.
					// normalVec = cornerNormalSelection(lightPosition, rayDir, c111);
				} else {

				}
			} else {
				if(lightPosition.z < c000.z) {

				} else if(lightPosition.z > c001.z) {

				} else {

				}
			}
		} else {
			if(lightPosition.y < c000.y) {
				if(lightPosition.z < c000.z) {

				} else if(lightPosition.z > c001.z) {

				} else {

				}
			} else if(lightPosition.y > c010.y) {
				if(lightPosition.z < c000.z) {

				} else if(lightPosition.z > c001.z) {

				} else {

				}
			} else {
				if(lightPosition.z < c000.z) {

				} else if(lightPosition.z > c001.z) {

				} else {

				}
			}
		}


		// Lambertian lighting
		const float3 lightVec = normalize( make_float3(-rayDir.x, -rayDir.y, -rayDir.z)); // could be normalized on HOST
		// const float3 lightVec = normalize( cLightPosition); // could be normalized on HOST
		const float3 rgb = make_float3(color.x, color.y, color.z) * min(1.f, max(0.0f, dot(normalVec, lightVec)));// / min(1.f, length(make_float3(samplePosScene.x - cLightPosition.x, samplePosScene.y - cLightPosition.y, samplePosScene.z - cLightPosition.z))));

		// Due to alpha pre-multiplication
		//
		// Note : inspecting device SASS code, assembly langage seemed to compute (1.f / color.w) each time, that's why we use a constant and multiply
		const float alphaPremultiplyConstant = 1.f / color.w;
		color.x = rgb.x * alphaPremultiplyConstant;
		color.y = rgb.y * alphaPremultiplyConstant;
		color.z = rgb.z * alphaPremultiplyConstant;

		// -- [ Opacity correction ] --
		// The standard equation :
		//		_accColor = _accColor + ( 1.0f - _accColor.w ) * color;
		// must take alpha correction into account
		// NOTE : if ( color.w == 0 ) then alphaCorrection equals 0.f
		const float alphaCorrection = ( 1.0f -_accColor.w ) * ( 1.0f - __powf( 1.0f - color.w, rayStep * cShaderMaterialProperty ) );

		// Accumulate the color
		_accColor.x += alphaCorrection * color.x;
		_accColor.y += alphaCorrection * color.y;
		_accColor.z += alphaCorrection * color.z;
		_accColor.w += alphaCorrection;
	}
}

__device__
float3 ShaderKernel::cornerNormalSelection(const float3 lightPosition, const float3 rayDir, const float3 corner) {
	float3 normal = make_float3(0.f, 0.f, 0.f);

	const float2 lightToCorner_xy = make_float2(corner.x - lightPosition.x, corner.y - lightPosition.y);
	const float2 rayDir_xy = make_float2(rayDir.x, rayDir.y);
	const float orientedArea_xy = dot(lightToCorner_xy, rayDir_xy);

	if(orientedArea_xy < 0) { //1x0y?z
		const float2 lightToCorner_xz = make_float2(corner.x - lightPosition.x, corner.z - lightPosition.z);
		const float2 rayDir_xz = make_float2(rayDir.x, rayDir.z);
		const float orientedArea_xz = dot(lightToCorner_xz, rayDir_xz);

		if(orientedArea_xz < 0) { //1x0y0z
			normal.x = +1.f;
		} else if(orientedArea_xz > 0) { //0x0y1z
			normal.z = +1.f;
		} else { //1x0y1z
			normal.x = +1.f;
			normal.y = +1.f;
		}
	} else if(orientedArea_xy > 0) { //0x1y?z
		const float2 lightToCorner_yz = make_float2(corner.y - lightPosition.y, corner.z - lightPosition.z);
		const float2 rayDir_yz = make_float2(rayDir.y, rayDir.z);
		const float orientedArea_yz = dot(lightToCorner_yz, rayDir_yz);

		if(orientedArea_yz < 0) { //0x1y0z
			normal.y = +1.f;
		} else if(orientedArea_yz > 0) { //0x0y1z
			normal.z = +1.f;
		} else { //0x1y1z
			normal.y = +1.f;
			normal.z = +1.f;
		}
	} else { //1x1y?z
		const float2 lightToCorner_xz = make_float2(corner.x - lightPosition.x, corner.z - lightPosition.z);
		const float2 rayDir_xz = make_float2(rayDir.x, rayDir.z);
		const float orientedArea_xz = dot(lightToCorner_xz, rayDir_xz);

		if(orientedArea_xz < 0) { //1x1y0z
			normal.x = +1.f;
			normal.y = +1.f;
		} else if(orientedArea_xz > 0) { //0x0y1z
			normal.z = +1.f;
		} else { //1x1y1z
			normal.x = +1.f;
			normal.y = +1.f;
			normal.z = +1.f;
		}
	}

	return normalize(normal);
}
