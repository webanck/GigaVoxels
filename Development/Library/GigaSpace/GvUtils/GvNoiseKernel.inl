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
//#include "GvUtils/GvNoise.h"

// Cuda
#include <math_functions.h>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvUtils
{
/******************************************************************************
 * Compute the Perlin noise given a 3D position
 *
 * @param x x coordinate position
 * @param y y coordinate position
 * @param z z coordinate position
 *
 * @return the noise at given position
 ******************************************************************************/
__device__
__forceinline__ float GvNoiseKernel::getValue( float x, float y, float z )
{
	int X = int( floorf( x )) & 255,
		Y = int( floorf( y )) & 255,
		Z = int( floorf( z )) & 255;

	x -= floorf( x );
	y -= floorf( y );
	z -= floorf( z );

	float u = fade( x );
	float v = fade( y );
	float w = fade( z );

	int A = gs_permutationTable[X]+Y, AA = gs_permutationTable[A]+Z, AB = gs_permutationTable[A+1]+Z,
		B = gs_permutationTable[X+1]+Y, BA = gs_permutationTable[B]+Z, BB = gs_permutationTable[B+1]+Z;

	return lerp( 
			lerp( 
				lerp( grad( gs_permutationTable[AA], x, y, z ),
					  grad( gs_permutationTable[BA], x - 1, y, z ), 
					  u ),
				lerp( grad( gs_permutationTable[AB], x, y - 1, z ),
					  grad( gs_permutationTable[BB], x - 1, y - 1, z ), 
					  u ),
			   	v ),
			lerp( 
				lerp( grad( gs_permutationTable[AA + 1], x, y, z - 1 ),
					  grad( gs_permutationTable[BA + 1], x - 1, y, z - 1 ), 
					  u ),
			    lerp( grad( gs_permutationTable[AB + 1], x, y - 1, z - 1 ),
					  grad( gs_permutationTable[BB + 1], x - 1, y - 1, z - 1 ), 
					  u ), 
			    v ),
			w );
}

/******************************************************************************
 * Compute the Perlin noise given a 3D position
 *
 * @param pPoint 3D position
 *
 * @return the noise at given position
 ******************************************************************************/
__device__
__forceinline__ float GvNoiseKernel::getValue( float3 pPoint )
{
	return getValue( pPoint.x, pPoint.y, pPoint.z );
}

/******************************************************************************
 * Fade function
 *
 * @param t parameter
 *
 * @return ...
 ******************************************************************************/
__device__
__forceinline__ float GvNoiseKernel::fade( float t )
{
	return t * t * t * ( t * ( t * 6.f - 15.f ) + 10.f );
}

/******************************************************************************
 * Grad function
 *
 * @param hash hash
 * @param x x
 * @param y y
 * @param z z
 *
 * @return ...
 ******************************************************************************/
__device__
__forceinline__ float GvNoiseKernel::grad( int hash, float x, float y, float z )
{
	// Compute conditions prior to doing branching
	// (help the compiler parallelize computation).
	const int h = hash & 15;
	bool c1 = h < 8;
	bool c2 = h < 4;
	bool c3 = h == 12 || h == 14;
	bool c4 = (h & 1) == 0;
	bool c5 = (h & 2) == 0;
	const float u = c1 ? x : y;
	const float v = c2 ? y : c3 ? x : z;
	return (c4 ? u : -u) + ( c5 ? v : -v);

	// Clean version
	//const int h = hash & 15;
	//const float u = h < 8 ? x : y;
	//const float v = h < 4 ? y : h == 12 || h == 14 ? x : z;
	//return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

/******************************************************************************
 * Compute the Perlin noise given a 3D position using preinitialized textures.
 *
 * @param x x coordinate position
 * @param y y coordinate position
 * @param z z coordinate position
 *
 * @return the noise at given position
 ******************************************************************************/
__device__
__forceinline__ float GvNoiseKernel::getValueT( const float x, const float y, const float z )
{
	return GvNoiseKernel::getValueT( make_float3( x, y, z ) );
}

/******************************************************************************
 * Compute the Perlin noise given a 3D position using preinitialized textures.
 *
 * @param pPoint 3D position
 *
 * @return the noise at given position
 ******************************************************************************/
__device__
__forceinline__ float GvNoiseKernel::getValueT( const float3 point )
{
	float3 P;
	// The next 6 lines are equivalents (but faster) to:
	// P = static_cast< float >( static_cast< int >( floor( point )) % 256.f ) / 256.f
	// The division by 256 is necessary because P will be used to access a denormalized 
	// 	texture.
	P.x = floorf( point.x ) * ( 1.f / 256.f );
	P.y = floorf( point.y ) * ( 1.f / 256.f );
	P.z = floorf( point.z ) * ( 1.f / 256.f );
	P.x -= floorf( P.x );
	P.y -= floorf( P.y );
	P.z -= floorf( P.z );

	float3 p = point;
	p.x -= floorf( point.x );
	p.y -= floorf( point.y );
	p.z -= floorf( point.z );

	float3 f;
	f.x = fade( p.x );
	f.y = fade( p.y );
	f.z = fade( p.z );

	// Get the 4 needed values from the texture in a single fetch.
	const float4 perm = permSampleT( P.x, P.y ) + P.z;
	const float AA = perm.x;
	const float AB = perm.y;
	const float BA = perm.z;
	const float BB = perm.w;

	return lerp(
				lerp(
					lerp( gradT( AA, p ),
						  gradT( BA, p + make_float3( -1.f, 0.f, 0.f )),
						  f.x ),
					lerp( gradT( AB, p + make_float3( 0.f, -1.f, 0.f )),
						  gradT( BB, p + make_float3( -1.f, -1.f, 0.f )),
						  f.x ),
					f.y ),
				lerp(
					// Normally we should add 1 but AA, BA, AB and BB are still normalized, 
					// 	so we only add 1/256.
					lerp( gradT( AA + 1.f / 256.f, p + make_float3( 0.f, 0.f, -1.f )),
						  gradT( BA + 1.f / 256.f, p + make_float3( -1.f, 0.f, -1.f )),
						  f.x ),
					lerp( gradT( AB + 1.f / 256.f, p + make_float3( 0.f, -1.f, -1.f )),
						  gradT( BB + 1.f / 256.f, p + make_float3( -1.f, -1.f, -1.f )),
						  f.x ),
					f.y ),
				f.z );
}

/******************************************************************************
 * Take a sample in the permutation table.
 ******************************************************************************/
__device__
__forceinline__ float4 GvNoiseKernel::permSampleT( float x, float y )
{
	// x and y are already divided by 256.
	// The result is immediately divided by 256, we need it this way to prepare the 
	// fetch in the next texture (grad).
	return tex2D( gs_permutationTableTexture, x, y ) * 255.f / 256.f;
}

/******************************************************************************
 * Grad function using textures
 *
 * @param hash hash
 * @param p p
 *
 * @return ...
 ******************************************************************************/
__device__
__forceinline__ float GvNoiseKernel::gradT( float hash, float3 p )
{
	// The lookup table is filled with numbers such as there is no need to denormalized the result.
	float3 g = make_float3( tex1D( gs_gradientTexture, hash ));
	return dot( g, p );
}

} // namespace GvUtils
