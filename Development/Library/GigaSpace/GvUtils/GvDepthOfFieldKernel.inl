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

// Cuda
#include <math_functions.h>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvUtils
{

/******************************************************************************
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
 ******************************************************************************/
__device__
__forceinline__ float GvDepthOfFieldKernel::getCoC( const float pAperture, const float pFocalLength, const float pPlaneInFocus, const float pObjectDistance )
{
	return fabsf( pAperture * ( pFocalLength * ( pObjectDistance - pPlaneInFocus ) ) / ( pObjectDistance * ( pPlaneInFocus - pFocalLength ) ) );
}

/******************************************************************************
 * Get the CoC (circle of confusion) calculated from the z-buffer values,
 * with the camera parameters lumped into scale and bias terms :
 * CoC = abs( z * CoCScale + CoCBias )
 *
 * @param pAperture camera lens aperture
 * @param pFocalLength camera focal length
 * @param pPlaneInFocus distance from the lens to the plane in focus
 * @param pZNear camera z-near plane distance
 * @param pZFar camera z-far plane distance
 * @param pZ object z-buffer value
 *
 * @return the circle of confusion
 ******************************************************************************/
__device__
__forceinline__ float GvDepthOfFieldKernel::getCoC( const float pAperture, const float pFocalLength, const float pPlaneInFocus, const float pZNear, const float pZFar, const float pZ )
{
	return fabsf( pZ * getCoCScale( pAperture, pFocalLength, pPlaneInFocus, pZNear, pZFar ) + getCoCBias( pAperture, pFocalLength, pPlaneInFocus, pZNear ) );
}

/******************************************************************************
 * Compute the scale term of the CoC (circle of confusion) given camera parameters
 *
 * @param pAperture camera lens aperture
 * @param pFocalLength camera focal length
 * @param pPlaneInFocus distance from the lens to the plane in focus
 * @param pZNear camera z-near plane distance
 * @param pZFar camera z-far plane distance
 *
 * @return the the scale term of the circle of confusion
 ******************************************************************************/
__device__
__forceinline__ float GvDepthOfFieldKernel::getCoCScale( const float pAperture, const float pFocalLength, const float pPlaneInFocus, const float pZNear, const float pZFar )
{
	return ( pAperture * pFocalLength * pPlaneInFocus * ( pZFar - pZNear ) ) / ( ( pPlaneInFocus - pFocalLength ) * pZNear * pZFar );
}

/******************************************************************************
 * Compute the bias term of the CoC (circle of confusion) given camera parameters
 *
 * @param pAperture camera lens aperture
 * @param pFocalLength camera focal length
 * @param pPlaneInFocus distance from the lens to the plane in focus
 * @param pZNear camera z-near plane distance
 *
 * @return the the bias term of the circle of confusion
 ******************************************************************************/
__device__
__forceinline__ float GvDepthOfFieldKernel::getCoCBias( const float pAperture, const float pFocalLength, const float pPlaneInFocus, const float pZNear )
{
	return ( pAperture * pFocalLength * ( pZNear - pPlaneInFocus ) ) / ( ( pPlaneInFocus * pFocalLength ) * pZNear );	// Question : last term should be ( pPlaneInFocus - pFocalLength ) ?
}

} // namespace GvUtils
