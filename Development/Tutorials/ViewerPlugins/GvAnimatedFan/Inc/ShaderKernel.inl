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
	const float lambertTerm = (dot( normalVec, lightVecNorm ));

	if ( lambertTerm > 0.0f )
	{
		// Diffuse component
		final_color += materialColor * diffuseTerm * lambertTerm ;

		// Specular component
		//const float3 halfVec = normalize( lightVecNorm + eyeVec );//*0.5f;
		//const float specular = __powf( max( dot( normalVec, halfVec ), 0.0f ), 64.0f );
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
inline void ShaderKernel::runImpl( const BrickSamplerType& brickSampler, const float3 samplePosScene,
					const float3 rayDir, float& rayStep, const float coneAperture )
{
	float4 col = brickSampler.template getValue< 0 >( coneAperture );
	float4 normal = brickSampler.template getValue< 1 >( coneAperture );

	float3 testPos = make_float3( samplePosScene.x - 0.5f, samplePosScene.y - 0.5f, 0.f );
	bool visible = false;
	float teta;
	float delta = static_cast< float >( k_currentTime ) / 32.f; 
	for ( int k = 0; k < 5; k++ )
	{
		teta = delta+ k*2.f*3.14159f/5.f;
		testPos.x = (samplePosScene.x -0.5f) * cosf(teta) - (samplePosScene.y-0.5f) * sinf(teta);
		testPos.y = (samplePosScene.x-0.5f) * sin(teta) + (samplePosScene.y -0.5f)* cos(teta);
				
		float3 line_vector = make_float3( -0.46786f + 0.10762f , 0.04484f-0.06428f , 0.f);
		float3 z_vector = make_float3( 0.f , 0.f , -1.f);
		float3 normal_in = cross(line_vector,z_vector);
		float3 point_vector = make_float3(testPos.x + 0.10762f , testPos.y  - 0.06428f , 0.f);
		float left = dot(normal_in,point_vector);
		line_vector = make_float3( -0.46487f + 0.10314f , -0.07025f+0.06876f , 0.f);
		z_vector = make_float3( 0.f , 0.f , 1.f);
		normal_in = cross(line_vector,z_vector);
		point_vector = make_float3(testPos.x  + 0.10314f , testPos.y + 0.06876f , 0.f);
		float right = dot(normal_in,point_vector);

		line_vector = make_float3( -0.10762f + 0.10314f , 0.06428f+0.06876f , 0.f);
		z_vector = make_float3( 0.f , 0.f , -1.f);
		normal_in = cross(line_vector,z_vector);
		point_vector = make_float3(testPos.x  + 0.10762f , testPos.y  - 0.06428f , 0.f);
		float up = dot(normal_in,point_vector);

		if ( (left>=0.f && right>=0.f && up >=0.f)   )    
		{
			visible = true;
			break;
		} 
	} 

	if (visible || length(testPos) <= 0.11061f )//|| ( samplePosScene.z<0.75f) || (samplePosScene.z>0.875f) || samplePosScene.x<0.0625f ||samplePosScene.y>0.9375f) 
	{
	}
	else
	{
		col.w =0;
	}

	//printf("%f %f %f \n",normal.x,normal.y,normal.z);
	if ( col.w > 0.0f )
	{
		
		/*float3 grad = make_float3(0.0f);

		float gradStep = rayStep * 0.25f;

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
		*/
		float3 grad = make_float3(normal.x ,normal.y,normal.z);
	
		if ( length( grad ) > 0.0f )
		{
			//grad =- grad;
			grad = normalize( grad );

			const float vis = 1.0f;

			const float3 lightVec = normalize( cLightPosition - samplePosScene );
			const float3 viewVec = (-1.0f * rayDir);

			float3 rgb;
			rgb.x = col.x;
			rgb.y = col.y;
			rgb.z = col.z;
			rgb = shadePointLight( rgb, grad, lightVec, viewVec, make_float3( 0.5f * vis ), make_float3( 1.0f * (0.3f + vis*0.7f) ), make_float3(0.9f) );
		
			// Due to alpha pre-multiplication
			//
			// Note : inspecting device SASS code, assembly langage seemed to compute (1.f / color.w) each time, that's why we use a constant and multiply
			const float alphaPremultiplyConstant = 1.f / col.w;
			col.x = rgb.x * alphaPremultiplyConstant;
			col.y = rgb.y * alphaPremultiplyConstant;
			col.z = rgb.z * alphaPremultiplyConstant;

			/*float eps =1.f/255.f;
			col.x = (col.x > 153.f/255.f -eps && col.x < 153.f/255.f +eps)? col.x : 1.f;
			col.y = (col.y > 153.f/255.f -eps && col.y < 153.f/255.f +eps)? col.y : 0.f;
			col.z = (col.z > 153.f/255.f -eps && col.z < 153.f/255.f +eps)? col.z : 0.f;*/
		}
		else
		{
			//col = make_float4( 0.0f );

			// Problem !
			// In this case, the normal generated by the gradient is null...
			// That generates visual artefacts...
			//col = make_float4( 0.0f );
			//color = make_float4( 1.0, 0.f, 0.f, 1.0f );
			// Ambient : no shading
			//float3 final_color = materialColor * ambientTerm;
			const float vis = 1.0f;
			float3 rgb;
			rgb.x = col.x; rgb.y = col.y; rgb.z = col.z;
			const float3 final_color = rgb * make_float3( 0.2f * vis );
			//final_color = make_float3( 1.0, 0.f, 0.f );
			
			// Due to alpha pre-multiplication
			//
			// Note : inspecting device SASS code, assembly langage seemed to compute (1.f / color.w) each time, that's why we use a constant and multiply
			const float alphaPremultiplyConstant = 1.f / col.w;
			col.x = final_color.x * alphaPremultiplyConstant;
			col.y = final_color.y * alphaPremultiplyConstant;
			col.z = final_color.z * alphaPremultiplyConstant;
		}
	
		// -- [ Opacity correction ] --
		// The standard equation :
		//		_accColor = _accColor + ( 1.0f - _accColor.w ) * color;
		// must take alpha correction into account
		// NOTE : if ( color.w == 0 ) then alphaCorrection equals 0.f
		const float alphaCorrection = ( 1.0f -_accColor.w ) * ( 1.0f - __powf( 1.0f - col.w, rayStep * 512.f ) );

		// Accumulate the color
		_accColor.x += alphaCorrection * col.x;
		_accColor.y += alphaCorrection * col.y;
		_accColor.z += alphaCorrection * col.z;
		_accColor.w += alphaCorrection;
	}
}
