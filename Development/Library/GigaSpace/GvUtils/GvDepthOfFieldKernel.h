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

#ifndef _GV_DEPTH_OF_FIELD_KERNEL_H_
#define _GV_DEPTH_OF_FIELD_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

// Cuda
#include <host_defines.h>

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
 * @class GvDepthOfFieldKernel
 *
 * @brief The GvDepthOfFieldKernel class provides basic interface to create depth of field
 *
 * It is based on CoC computation, the circle of confusion
 * (depth of field is used to enhance realism for visual effects)
 */
class GvDepthOfFieldKernel
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Get the CoC (circle of confusion) for the world-space distance
	 * from the camera-object distance calculated from camera parameters
	 *
	 * Object distance can be calculated from the z values in the z-buffer:
	 * objectdistance = -zfar * znear / (z * (zfar - znear) - zfar)
	 *
	 * @param pAperture camera lens aperture
	 * @param pFocalLength camera focal length
	 * @param pPlaneInFocus distance from the lens to the plane in focus
	 * @param pObjectDistance object distance from the lens
	 *
	 * @return the circle of confusion
	 */
	__device__
	static __forceinline__ float getCoC( const float pAperture, const float pFocalLength, const float pPlaneInFocus, const float pObjectDistance );

	/**
	 * Get the CoC (circle of confusion) calculated from the z-buffer values,
	 * with the camera parameters lumped into scale and bias terms :
	 * CoC = abs( z * CoCScale + CoCBias )
	 *
	 * @param pAperture camera lens aperture
	 * @param pFocalLength camera focal length
	 * @param pPlaneInFocus distance from the lens to the plane in focus
	 * @param pZNear camera z-near plane distance
	 * @param pZFar camera z-far plane distance
	 * @param pObjectDistance object distance from the lens
	 *
	 * @return the circle of confusion
	 */
	__device__
	static __forceinline__ float getCoC( const float pAperture, const float pFocalLength, const float pPlaneInFocus, const float pZNear, const float pZFar, const float pObjectDistance );
	
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
	 * Compute the scale term of the CoC (circle of confusion) given camera parameters
	 *
	 * @param pAperture camera lens aperture
	 * @param pFocalLength camera focal length
	 * @param pPlaneInFocus distance from the lens to the plane in focus
	 * @param pZNear camera z-near plane distance
	 * @param pZFar camera z-far plane distance
	 *
	 * @return the the scale term of the circle of confusion
	 */
	__device__
	static __forceinline__ float getCoCScale( const float pAperture, const float pFocalLength, const float pPlaneInFocus, const float pZNear, const float pZFar );

	/**
	 * Compute the bias term of the CoC (circle of confusion) given camera parameters
	 *
	 * @param pAperture camera lens aperture
	 * @param pFocalLength camera focal length
	 * @param pPlaneInFocus distance from the lens to the plane in focus
	 * @param pZNear camera z-near plane distance
	 *
	 * @return the the bias term of the circle of confusion
	 */
	__device__
	static __forceinline__ float getCoCBias( const float pAperture, const float pFocalLength, const float pPlaneInFocus, const float pZNear );

};

} // namespace GvUtils

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvDepthOfFieldKernel.inl"

#endif // !_GV_DEPTH_OF_FIELD_KERNEL_H_
