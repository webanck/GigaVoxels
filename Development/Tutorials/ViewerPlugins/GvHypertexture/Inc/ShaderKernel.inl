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

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * This method is called just before the cast of a ray. Use it to initialize any data
 *  you may need. You may also want to modify the initial distance along the ray (tTree).
 *
 * @param pRayStartTree the starting position of the ray in octree's space.
 * @param pRayDirTree the direction of the ray in octree's space.
 * @param pTTree the distance along the ray's direction we start from.
 ******************************************************************************/
__device__
inline void ShaderKernel::preShadeImpl( const float3& rayStartTree, const float3& rayDirTree, float& tTree )
{
	_accColor = make_float4( 0.f );
}

/******************************************************************************
 * This method is called after the ray stopped or left the bounding
 * volume. You may want to do some post-treatment of the color.
 ******************************************************************************/
__device__
inline void ShaderKernel::postShadeImpl()
{
	if ( _accColor.w >= opacityThreshold )
	{
		_accColor.w = 1.f;
	}
}

/******************************************************************************
 * This method returns the cone aperture for a given distance.
 *
 * @param pTTree the current distance along the ray's direction.
 *
 * @return the cone aperture
 ******************************************************************************/
__device__
inline float ShaderKernel::getConeApertureImpl( const float tTree ) const
{
	// Overestimate to avoid aliasing
	const float scaleFactor = 1.333f;

	return k_renderViewContext.pixelSize.x * tTree * scaleFactor * k_renderViewContext.frustumNearINV;
}

/******************************************************************************
 * This method returns the final rgba color that will be written to the color buffer.
 *
 * @return the final rgba color.
 ******************************************************************************/
__device__
inline float4 ShaderKernel::getColorImpl() const
{
	return _accColor;
}

/******************************************************************************
 * This method is called before each sampling to check whether or not the ray should stop.
 *
 * @param pRayPosInWorld the current ray's position in world space.
 *
 * @return true if you want to continue the ray. false otherwise.
 ******************************************************************************/
__device__
inline bool ShaderKernel::stopCriterionImpl( const float3& rayPosInWorld ) const
{
	return ( _accColor.w >= opacityThreshold );
}

/******************************************************************************
 * This method is called to know if we should stop at the current octree's level.
 *
 * @param pVoxelSize the voxel's size in the current octree level.
 *
 * @return false if you want to stop at the current octree's level. true otherwise.
 ******************************************************************************/
__device__
inline bool ShaderKernel::descentCriterionImpl( const float voxelSize ) const
{
	return true;
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
inline void ShaderKernel::runImpl( const SamplerType& brickSampler, const float3 samplePosScene,
										const float3 rayDir, float& rayStep, const float coneAperture )
{
	// Retrieve first channel element : color
	float4 color = brickSampler.template getValue< 0 >( coneAperture );

	// Test opacity
	if ( color.w > 0.0f )
	{
		// Retrieve second channel element : normal
		const float4 normal = brickSampler.template getValue< 1 >( coneAperture );

	//	//------------------------------------------------------------------------------------------------
	//	float3 grad = make_float3( 0.0f );

	//	float gradStep = rayStep * 0.25f;

	//	float4 v0, v1;

	//	v0 = brickSampler.template getValue< 0 >( coneAperture, make_float3( gradStep, 0.0f, 0.0f) );
	//	v1 = brickSampler.template getValue< 0 >( coneAperture, make_float3(-gradStep, 0.0f, 0.0f) );
	//	grad.x = v0.w - v1.w;

	//	v0 = brickSampler.template getValue< 0 >( coneAperture, make_float3(0.0f,  gradStep, 0.0f) );
	//	v1 = brickSampler.template getValue< 0 >( coneAperture, make_float3(0.0f, -gradStep, 0.0f) );
	//	grad.y = v0.w - v1.w;

	//	v0 = brickSampler.template getValue< 0 >( coneAperture, make_float3(0.0f, 0.0f,  gradStep) );
	//	v1 = brickSampler.template getValue< 0 >( coneAperture, make_float3(0.0f, 0.0f, -gradStep) );
	//	grad.z = v0.w - v1.w;

	//	//if ( length( grad ) > 0.0f )
	////	{
	//		grad =- grad;
	//		//grad = normalize( grad );

	//		float3 normal = grad;
	//		
	//	//}
	//	//------------------------------------------------------------------------------------------------
		
		// Lambertian lighting
		const float3 normalVec = normalize( make_float3( normal.x, normal.y, normal.z ) );
		const float3 lightVec = normalize( cLightPosition );
		const float3 rgb = make_float3( color.x, color.y, color.z ) * max( 0.0f, dot( normalVec, lightVec ) );

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
		const float alphaCorrection = ( 1.0f -_accColor.w ) * ( 1.0f - __powf( 1.0f - color.w, rayStep * 512.f ) );

		// Accumulate the color
		_accColor.x += alphaCorrection * color.x;
		_accColor.y += alphaCorrection * color.y;
		_accColor.z += alphaCorrection * color.z;
		_accColor.w += alphaCorrection;
	}
}
