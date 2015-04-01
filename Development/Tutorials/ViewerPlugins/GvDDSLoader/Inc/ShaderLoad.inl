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
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Get the associated device-side object
 *
 * @return the associated device-side object
 ******************************************************************************/
inline ShaderLoadKernel ShaderLoad::getKernelObject()
{
	return kernelObject;
}

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
inline float3 ShaderLoadKernel::shadePointLight( float3 materialColor, float3 normalVec, float3 lightVec, float3 eyeVec,
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
 * @param rayStartTree ...
 * @param rayDirTree ...
 * @param tTree ...
 ******************************************************************************/
__device__
inline void ShaderLoadKernel::preShadeImpl( const float3& rayStartTree, const float3& rayDirTree, float& tTree )
{
	_accColor = make_float4(0.f);
}

/******************************************************************************
 * ...
 ******************************************************************************/
__device__
inline void ShaderLoadKernel::postShadeImpl()
{
	if ( _accColor.w >= cOpacityStep )
	{
		_accColor.w = 1.f;
	}
}

/******************************************************************************
 * ...
 *
 * @param tTree ...
 *
 * @return ...
 ******************************************************************************/
__device__
inline float ShaderLoadKernel::getConeApertureImpl( const float tTree ) const
{
	// overestimate to avoid aliasing
	const float scaleFactor = 1.333f;

	return k_renderViewContext.pixelSize.x * tTree * scaleFactor * k_renderViewContext.frustumNearINV;
}

/******************************************************************************
 * ...
 *
 * @return ...
 ******************************************************************************/
__device__
inline float4 ShaderLoadKernel::getColorImpl() const
{
	return _accColor;
}

/******************************************************************************
 * ...
 *
 * @param rayPosInWorld ...
 *
 * @return ...
 ******************************************************************************/
__device__
inline bool ShaderLoadKernel::stopCriterionImpl( const float3& rayPosInWorld ) const
{
	return ( _accColor.w >= cOpacityStep );
}

/******************************************************************************
 * ...
 *
 * @param voxelSize ...
 *
 * @return ...
 ******************************************************************************/
__device__
inline bool ShaderLoadKernel::descentCriterionImpl( const float voxelSize ) const
{
	return true;
}

/******************************************************************************
 * Map a distance to a color (with a transfer function)
 *
 * @param dist the distance to map
 *
 * @return The color corresponding to the distance
 ******************************************************************************/
__device__
inline float4 densityToColor( float pValue )
{
	return tex1D( transerFunctionTexture, pValue );
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
inline void ShaderLoadKernel::runImpl( const BrickSamplerType& brickSampler, const float3 samplePosScene,
					const float3 rayDir, float& rayStep, const float coneAperture )
{
	// Sample data
	float4 color = brickSampler.template getValue< 0 >( coneAperture );

	// Threshold data
	if ( color.w < cThreshold )
	{
		color.w = 0.f;
	}

	// Use transfer function to color density
	color = densityToColor( color.w );

	// Apply shading
	if ( color.w > 0.0f )
	{
		float3 grad = make_float3(0.0f);

		float gradStep = rayStep * cGradientStep;

		float4 v0, v1;

		v0 = brickSampler.template getValue< 0 >( coneAperture, make_float3( gradStep, 0.0f, 0.0f) );
		v1 = brickSampler.template getValue< 0 >( coneAperture, make_float3(-gradStep, 0.0f, 0.0f) );
		grad.x = v0.w - v1.w;

		v0 = brickSampler.template getValue< 0 >( coneAperture, make_float3(0.0f,  gradStep, 0.0f) );
		v1 = brickSampler.template getValue< 0 >( coneAperture, make_float3(0.0f, -gradStep, 0.0f) );
		grad.y = v0.w - v1.w;

		v0 = brickSampler.template getValue< 0 >( coneAperture, make_float3(0.0f, 0.0f,  gradStep) );
		v1 = brickSampler.template getValue< 0 >( coneAperture, make_float3(0.0f, 0.0f, -gradStep) );
		grad.z = v0.w - v1.w;

		if ( length( grad ) > 0.0f )
		{
			grad =- grad;
			grad = normalize( grad );

			float vis = 1.0f;

			float3 lightVec = normalize( cLightPosition - samplePosScene );
			float3 viewVec = (-1.0f * rayDir);

			float3 rgb;
			rgb.x = color.x; rgb.y = color.y; rgb.z = color.z;
			rgb = shadePointLight( rgb, grad, lightVec, viewVec, make_float3( 0.2f * vis ), make_float3( 1.0f * (0.3f + vis*0.7f) ), make_float3(0.9f) );
			color.x = rgb.x; color.y = rgb.y; color.z = rgb.z;

			// Due to alpha pre-multiplication
			color.x = color.x / color.w;
			color.y = color.y / color.w;
			color.z = color.z / color.w;
		}
		else
		{
			//color = make_float4( 0.0f );
			// TO DO
			// Replace by "no shading" with only ambient term
			float3 rgb;
			float vis = 1.0f;
			rgb.x = color.x; rgb.y = color.y; rgb.z = color.z;
			float3 final_color = rgb * make_float3( 0.2f * vis );

			// Due to alpha pre-multiplication
			color.x = final_color.x / color.w;
			color.y = final_color.y / color.w;
			color.z = final_color.z / color.w;
		}
		
		// -- [ Opacity correction ] --
		// The standard equation :
		//		_accColor = _accColor + ( 1.0f - _accColor.w ) * color;
		// must take alpha correction into account
		// NOTE : if ( color.w == 0 ) then alphaCorrection equals 0.f
		float alphaCorrection = ( 1.0f -_accColor.w ) * ( 1.0f - powf( 1.0f - color.w, rayStep * cFullOpacityDistance ) );

		// Accumulate the color
		_accColor.x += alphaCorrection * color.x;
		_accColor.y += alphaCorrection * color.y;
		_accColor.z += alphaCorrection * color.z;
		_accColor.w += alphaCorrection;
	}
}
