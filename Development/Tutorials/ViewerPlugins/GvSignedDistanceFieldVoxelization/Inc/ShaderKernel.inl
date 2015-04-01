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
****************************** INLINE DEFINITION *****************************
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
	// We retrieve levelres ( size of a voxel )
	const float levelResInv = brickSampler._nodeSizeTree / 8.0f;

	// We retrieve potential value and normal 
	const float dist = brickSampler.template getValue< 0 >( coneAperture ).x;
	
	// Computing w from distance function : the voxel is approximated to be a sphere, and in funtion 
	// of the intersection between the the sphere and the mesh we compute w.
	float4 color = make_float4( 1.0f, 1.0f, 1.0f, 1.0f ) * clamp( ( levelResInv / 2.0f - dist ) / ( levelResInv ), 0.0f, 1.0f ) ;

	if ( color.w > 0.0f )
	{
		// We retrieve potential value and normal 
		const float4 normal = make_float4( brickSampler.template getValue< 1 >( coneAperture ).x, 
										 brickSampler.template getValue< 2 >( coneAperture ).x, 
										 brickSampler.template getValue< 3 >( coneAperture ).x,  0.0f );

		// Lambertian lighting
		const float3 normalVec = normalize( make_float3( normal.x, normal.y, normal.z ) );
		const float3 lightVec = normalize( cLightPosition );
		float3 rgb = make_float3( color.x, color.y, color.z ) * max( 0.0f, dot( normalVec, lightVec ) );

		// Due to alpha pre-multiplication
		//
		// Note : inspecting device SASS code, assembly langage seemed to compute (1.f / color.w) each time, that's why we use a constant and multiply
		const float alphaPremultiplyConstant = 1.f / color.w;
		color.x = rgb.x * alphaPremultiplyConstant;
		color.y = rgb.y * alphaPremultiplyConstant;
		color.z = rgb.z * alphaPremultiplyConstant;

		// -- [ Opacity correction ] --
		//
		// The standard equation :
		//		_accColor = _accColor + ( 1.0f - _accColor.w ) * color;
		// must take alpha correction into account
		//
		// NOTE : if ( color.w == 0 ) then alphaCorrection equals 0.f
		const float alphaCorrection = ( 1.0f -_accColor.w ) * ( 1.0f - __powf( 1.0f - color.w, rayStep * 512.f ) );

		// Accumulate the color
		_accColor.x += alphaCorrection * color.x;
		_accColor.y += alphaCorrection * color.y;
		_accColor.z += alphaCorrection * color.z;
		_accColor.w += alphaCorrection;
	}
}
