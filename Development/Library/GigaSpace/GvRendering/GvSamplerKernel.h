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

#ifndef _GV_SAMPLER_KERNEL_H_
#define _GV_SAMPLER_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

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

namespace GvRendering
{

/** 
 * @struct GvSamplerKernel
 *
 * @brief The GvSamplerKernel struct provides features
 * to sample data in a data stucture.
 *
 * The rendering stage is done brick by brick along a ray.
 * The sampler is used to store useful current parameters needed to fecth data from data pool.
 *
 * @param VolumeTreeKernelType the data structure to sample data into.
 */
template< typename VolumeTreeKernelType >
struct GvSamplerKernel
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Data structure.
	 * It is used to sample data into (the data pool is store inside the data structure)
	 */
	VolumeTreeKernelType* _volumeTree;

	/**
	 * Position of the brick in data pool (i.e. in 3D texture space)
	 */
	float3 _brickChildPosInPool;

	/**
	 * Position of the parent brick in data pool (i.e. in 3D texture space)
	 */
	float3 _brickParentPosInPool;
		
	/**
	 * Sample offset in the node
	 */
	float3 _sampleOffsetInNodeTree;
		
	/**
	 * Node/brick size
	 */
	float _nodeSizeTree;

	/**
	 * Ray length in node starting from sampleFirstOffsetInNodeTree
	 */
	float _rayLengthInNodeTree;	// not used anymore ?

	/**
	 * Flag telling wheter or not mipmapping is activated to render/visit the current brick
	 */
	bool _mipMapOn;

	/**
	 * If mipmapping is activated, this represents the coefficient to blend between child and parent brick
	 *
	 * note : 0.0f -> child, 1.0f -> parent
	 */
	float _mipMapInterpCoef;

	/**
	 * Coefficient used to transform/scale tree space to brick pool space
	 */
	float _scaleTree2BrickPool;

	/******************************** METHODS *********************************/

	/**
	 * Sample data at given cone aperture
	 *
	 * @param coneAperture the cone aperture
	 *
	 * @return the sampled value
	 */
	template< int channel >
	__device__
	__forceinline__ float4 getValue( const float coneAperture ) const;

	/**
	 * Sample data at given cone aperture and offset in tree
	 *
	 * @param coneAperture the cone aperture
	 * @param offsetTree the offset in the tree
	 *
	 * @return the sampled value
	 */
	template< int channel >
	__device__
	__forceinline__ float4 getValue( const float coneAperture, const float3 offsetTree ) const;

	/**
	 * Move sample offset in node tree
	 *
	 * @param offsetTree offset in tree
	 */
	__device__
	__forceinline__ void moveSampleOffsetInNodeTree( const float3 offsetTree );

	/**
	 * Update MipMap parameters given cone aperture
	 *
	 * @param coneAperture the cone aperture
	 *
	 * @return It returns false if coneAperture > voxelSize in parent brick
	 */
	__device__
	__forceinline__ bool updateMipMapParameters( const float coneAperture );

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

};

}

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvSamplerKernel.inl"

#endif // !_GV_SAMPLER_KERNEL_H_
