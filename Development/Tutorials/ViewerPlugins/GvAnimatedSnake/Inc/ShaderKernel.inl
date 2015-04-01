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
 * ...
 *
 * @param pSamplePosScene position of the sample in the scene
 *
 * @return ...
 ******************************************************************************/
__device__
inline float getOpacity( float3 pPosition )
{
	// TO DO
	// - use constant for value "1.f/16.f"
	//const float step = 1.f / 16.f;

	// LookUp function
	// - answer 1 if the sample is in the snake, 0 otherwise
	for ( int k = 0; k < 20; k++ )
	{
		if (
			// Check X position
			static_cast< float >( ( cDirections[ k ].x ) / 16.f ) <= pPosition.x && 
			static_cast< float >( ( cDirections[ k ].x ) / 16.f ) + 0.0625f >= pPosition.x &&
			// Check Y position
			static_cast< float >( ( cDirections[ k ].y ) / 16.f ) <= pPosition.y &&
			static_cast< float >( ( cDirections[ k ].y ) / 16.f ) + 0.0625f >= pPosition.y  &&
			// Check Z position
			static_cast< float >( ( cDirections[ k ].z ) / 16.f ) <= pPosition.z &&
			static_cast< float >( ( cDirections[ k ].z ) / 16.f ) + 0.0625f >= pPosition.z )
		{
			// Sample is in the snake
			return 1.f;
		}

	}

	// Sample is out of the snake
	return 0.f;
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

	// Fetch data from transfer function
	//float colorCyclingIndex = ( 0.5f * ( cosf( 2.f * 3.141592f / 256.f * k_currentTime * samplePosScene.x ) + 1.f ) );
	//float temp = 0.5f/*half amplitude*/ * ( 0.5f * ( sinf( 2.f * 3.141592f / 128.f * ( 0.5f * k_currentTime ) * samplePosScene.y / 10.f + 13564.25f ) + 1.f ) );

	//float colorCyclingIndex = ( 0.5f * ( cosf( 2.f * 3.141592f / 256.f * k_currentTime * samplePosScene.x * temp / 10.f + 64.18f ) + 1.f ) );
	
	color.w = getOpacity( samplePosScene );
	//color.w = ( length(make_float3(0.5f)-samplePosScene)) * &(sinf((2.f * 3.141592f / 128.f) *k_currentTime));
	//color = tex1D( colorCyclingLookupTable, colorCyclingIndex );
	
	/*float4 animatedColor = tex1D( colorCyclingLookupTable, colorCyclingIndex );
	color.x = animatedColor.x;
	color.y = animatedColor.y;
	color.z = animatedColor.z;*/

	// Test opacity
	if ( color.w > 0.0f )
	{
		// Lambertian lighting
		//float3 rgb = make_float3( color.x, color.y, color.z );

		// Due to alpha pre-multiplication
		//color.x = rgb.x ;/// color.w;
		//color.y = rgb.y ;/// color.w;
		//color.z = rgb.z ;/// color.w;
		
		// -- [ Opacity correction ] --
		// The standard equation :
		//		_accColor = _accColor + ( 1.0f - _accColor.w ) * color;
		// must take alpha correction into account
		// NOTE : if ( color.w == 0 ) then alphaCorrection equals 0.f
		//const float alphaCorrection = ( 1.0f -_accColor.w ) * ( 1.0f - __powf( 1.0f - color.w, rayStep * cShaderMaterialProperty ) );

		// Accumulate the color
		_accColor.x = color.x;
		_accColor.y = color.y;
		_accColor.z = color.z;
		_accColor.w = 1;// alphaCorrection;
	}	
}
