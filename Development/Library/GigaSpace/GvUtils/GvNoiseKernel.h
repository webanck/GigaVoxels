/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2015 INRIA <http://www.inria.fr/>
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

#ifndef _GV_NOISE_KERNEL_H_
#define _GV_NOISE_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

// Cuda
#include <host_defines.h>
#include <vector_types.h>
#include <cuda_texture_types.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvUtils
{
	
/**
 * Lookup table
 */
texture< uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat > gs_permutationTableTexture;

/**
 * Gradient table
 */
texture< char4, cudaTextureType1D, cudaReadModeNormalizedFloat > gs_gradientTexture;

/**
 * ...
 */
__constant__ int gs_permutationTable[ 512 ] =
{
	151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
	140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
	247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
	57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
	74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
	60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
	65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
	200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
	52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
	207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
	119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
	129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
	218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
	81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
	184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,205, 93,
	222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180,
	// Repeat
	151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233,  7,225,
	140, 36,103, 30, 69,142,  8, 99, 37,240, 21, 10, 23,190,  6,148,
	247,120,234, 75,  0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
	57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
	74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
	60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
	65, 25, 63,161,  1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
	200,196,135,130,116,188,159, 86,164,100,109,198,173,186,  3, 64,
	52,217,226,250,124,123,  5,202, 38,147,118,126,255, 82, 85,212,
	207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
	119,248,152,  2, 44,154,163, 70,221,153,101,155,167, 43,172,  9,
	129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
	218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
	81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
	184, 84,204,176,115,121, 50, 45,127,  4,150,254,138,236,205, 93,
	222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180,
};

/**
 * @class GvNoise
 *
 * @brief The GvNoise class provides basic interface to create noise
 *
 * It is based on Perlin noise that can be used to generate procedural textures,
 * enhance realism for for visual effects and/or add details.
 */
class GvNoiseKernel
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Compute the Perlin noise given a 3D position using device arrays
	 *
	 * @param x x coordinate position
	 * @param y y coordinate position
	 * @param z z coordinate position
	 *
	 * @return the noise at given position
	 */
	__device__
	static __forceinline__ float getValue( float x, float y, float z );

	/**
	 * Compute the Perlin noise given a 3D position using device arrays
	 *
	 * @param pPoint 3D position
	 *
	 * @return the noise at given position
	 */
	__device__
	static __forceinline__ float getValue( float3 pPoint );

	/**
	 * Compute the Perlin noise given a 3D position using preinitialized textures.
	 *
	 * @param x x coordinate position
	 * @param y y coordinate position
	 * @param z z coordinate position
	 *
	 * @return the noise at given position
	 */
	__device__
	static __forceinline__ float getValueT( const float x, const float y, const float z );

	/**
	 * Compute the Perlin noise given a 3D position using preinitialized textures.
	 *
	 * @param pPoint 3D position
	 *
	 * @return the noise at given position
	 */
	__device__
	static __forceinline__ float getValueT( const float3 pPoint );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * ...
	 *
	 * @param g ...
	 * @param x ...
	 * @param y ...
	 * @param z ...
	 *
	 * @return ...
	 */
	__device__
	static __forceinline__ float dotN( float3 g, float x, float y, float z )
	{
		return g.x * x + g.y * y + g.z * z;
	}

	/**
	 * Fade function
	 *
	 * @param t parameter
	 *
	 * @return ...
	 */
	__device__
	static __forceinline__ float fade( float t );

	/**
	 * Grad function
	 *
	 * @param hash hash
	 * @param x x
	 * @param y y
	 * @param z z
	 *
	 * @return ...
	 */
	__device__
	static __forceinline__ float grad( int hash, float x, float y, float z );

	/**
	 * Take a sample in the permutation table.
	 */
	__device__
	static __forceinline__ float4 permSampleT( float x, float y );

	/**
	 * Grad function
	 *
	 * @param hash hash
	 * @param p p
	 *
	 * @return ...
	 */
	__device__
	static __forceinline__ float gradT( float hash, float3 p );

};

} // namespace GvUtils

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvNoiseKernel.inl"

#endif // !_GV_NOISE_KERNEL_H_
