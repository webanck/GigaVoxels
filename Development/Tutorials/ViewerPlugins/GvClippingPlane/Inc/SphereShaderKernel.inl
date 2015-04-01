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
inline void SphereShaderKernel::preShadeImpl( const float3& rayStartTree, const float3& rayDirTree, float& tTree )
{
	_accColor = make_float4( 0.f );
}

/******************************************************************************
 * This method is called after the ray stopped or left the bounding
 * volume. You may want to do some post-treatment of the color.
 ******************************************************************************/
__device__
inline void SphereShaderKernel::postShadeImpl()
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
inline float SphereShaderKernel::getConeApertureImpl( const float tTree ) const
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
inline float4 SphereShaderKernel::getColorImpl() const
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
inline bool SphereShaderKernel::stopCriterionImpl( const float3& rayPosInWorld ) const
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
inline bool SphereShaderKernel::descentCriterionImpl( const float voxelSize ) const
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
inline void SphereShaderKernel::runImpl( const SamplerType& brickSampler, const float3 samplePosScene,
										const float3 rayDir, float& rayStep, const float coneAperture )
{
	// Retrieve first channel element : color
	float4 color = brickSampler.template getValue< 0 >( coneAperture );

	// Test opacity
	if ( color.w > 0.0f )
	{
		float3 rgb;

		// Clipping plane
		//const bool testRegion = halfClipSpaceTest( samplePosScene );
		const bool testRegion = halfClipSpaceTest( samplePosScene - 0.5f );
		const float opacityStepReference = 32.0f;
		if ( testRegion )
		{
			color.w = 0.01f;

			color.x = 0.0f;
			color.y = 0.0f;
			color.z = 1.0f;

			rgb = make_float3( color.x, color.y, color.z );
		}
		else
		{
			// Retrieve second channel element : normal
			float4 normal = brickSampler.template getValue< 1 >( coneAperture );

			float3 normalVec = normalize( make_float3( normal.x, normal.y, normal.z ) );
			float3 lightVec = normalize( cLightPosition );

			// Lambertian lighting
			rgb = make_float3( color.x, color.y, color.z ) * max( 0.0f, dot( normalVec, lightVec ) );
		}
				
		// Due to alpha pre-multiplication
		color.x = rgb.x / color.w;
		color.y = rgb.y / color.w;
		color.z = rgb.z / color.w;
		
		// -- [ Opacity correction ] --
		// The standard equation :
		//		_accColor = _accColor + ( 1.0f - _accColor.w ) * color;
		// must take alpha correction into account
		// NOTE : if ( color.w == 0 ) then alphaCorrection equals 0.f
		//float alphaCorrection = ( 1.0f -_accColor.w ) * ( 1.0f - powf( 1.0f - color.w, rayStep * 512.f ) );
		float alphaCorrection = ( 1.0f -_accColor.w ) * ( 1.0f - powf( 1.0f - color.w, rayStep * opacityStepReference ) );

		// Accumulate the color
		_accColor.x += alphaCorrection * color.x;
		_accColor.y += alphaCorrection * color.y;
		_accColor.z += alphaCorrection * color.z;
		_accColor.w += alphaCorrection;
	}	
}

/******************************************************************************
 * Helper class to test if a point is inside the unit sphere centered in [0,0,0]
 *
 * @param pPoint the point to test
 *
 * @return flag to tell wheter or not the point is insied the sphere
 ******************************************************************************/
__device__
inline bool SphereShaderKernel::halfClipSpaceTest( const float3 pPoint )
{
	const float3 clipPlaneNormal = make_float3( cClippingPlane.x, cClippingPlane.y, cClippingPlane.z );
	const float clipPlaneDistance = cClippingPlane.w;
	const float epsillon = 0.001f;

	float distance = dot( pPoint, clipPlaneNormal ) - clipPlaneDistance;

	return ( distance > epsillon );
}
