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

#ifndef _SPHERESHADERKERNEL_HCU_
#define _SPHERESHADERKERNEL_HCU_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Gigavoxels
#include <GvRenderer/GvIRenderShader.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Light position
 */
__constant__ float3 cLightPosition;

/**
 * Stop if we reached our maximum opacity
 */
__device__ static const float opacityThreshold = 0.99f;

/**
 * Clipping plane
 */
__constant__ float4 cClippingPlane;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @struct SphereShaderKernel
 *
 * @brief The SphereShaderKernel struct provides the way to shade the data structure.
 *
 * It is used in conjonction with the base class GvIRenderShader to implement the shader functions.
 */
struct SphereShaderKernel : public GvRenderer::GvIRenderShader< SphereShaderKernel >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * This method is called just before the cast of a ray. Use it to initialize any data
	 *  you may need. You may also want to modify the initial distance along the ray (tTree).
	 *
	 * @param pRayStartTree the starting position of the ray in octree's space.
	 * @param pRayDirTree the direction of the ray in octree's space.
	 * @param pTTree the distance along the ray's direction we start from.
	 */
	__device__
	inline void preShadeImpl( const float3& pRayStartTree, const float3& pRayDirTree, float& pTTree );

	/**
	 * This method is called after the ray stopped or left the bounding
	 * volume. You may want to do some post-treatment of the color.
	 */
	__device__
	inline void postShadeImpl();

	/**
	 * This method returns the cone aperture for a given distance.
	 *
	 * @param pTTree the current distance along the ray's direction.
	 *
	 * @return the cone aperture
	 */
	__device__
	inline float getConeApertureImpl( const float pTTree ) const;

	/**
	 * This method returns the final rgba color that will be written to the color buffer.
	 *
	 * @return the final rgba color.
	 */
	__device__
	inline float4 getColorImpl() const;

	/**
	 * This method is called before each sampling to check whether or not the ray should stop.
	 *
	 * @param pRayPosInWorld the current ray's position in world space.
	 *
	 * @return true if you want to continue the ray. false otherwise.
	 */
	__device__
	inline bool stopCriterionImpl( const float3& pRayPosInWorld ) const;

	/**
	 * This method is called to know if we should stop at the current octree's level.
	 *
	 * @param pVoxelSize the voxel's size in the current octree level.
	 *
	 * @return false if you want to stop at the current octree's level. true otherwise.
	 */
	__device__
	inline bool descentCriterionImpl( const float pVoxelSize ) const;

	/**
	 * This method is called for each sample. For example, shading or secondary rays
	 * should be done here.
	 *
	 * @param pBrickSampler brick sampler
	 * @param pSamplePosScene position of the sample in the scene
	 * @param pRayDir ray direction
	 * @param pRayStep ray step
	 * @param pConeAperture cone aperture
	 */
	template< typename TSamplerType >
	__device__
	inline void runImpl( const TSamplerType& pBrickSampler, const float3 pSamplePosScene,
						const float3 pRayDir, float& pRayStep, const float pConeAperture );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Helper class to test if a point is inside the unit sphere centered in [0,0,0]
	 *
	 * @param pPoint the point to test
	 *
	 * @return flag to tell wheter or not the point is insied the sphere
	 */
	__device__
	static inline bool halfClipSpaceTest( const float3 pPoint );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Accumulated color during ray casting
	 */
	float4 _accColor;

	/******************************** METHODS *********************************/

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "SphereShaderKernel.inl"

#endif // !_SPHERESHADERKERNEL_HCU_
