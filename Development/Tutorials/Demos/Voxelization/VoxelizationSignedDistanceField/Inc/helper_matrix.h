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

#ifndef _HELPER_MATRIX_H_
#define _HELPER_MATRIX_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/vector_types_ext.h>

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

/**
 * @class MatrixHelper
 *
 * @brief The MatrixHelper class provides helper functions to manipulate matrices.
 *
 * ...
 *
 */
class MatrixHelper
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * Define a viewing transformation
	 *
	 * @param eyeX specifies the position of the eye point
	 * @param eyeY specifies the position of the eye point
	 * @param eyeZ specifies the position of the eye point
	 * @param centerX specifies the position of the reference point
	 * @param centerY specifies the position of the reference point
	 * @param centerZ specifies the position of the reference point
	 * @param upX specifies the direction of the up vector
	 * @param upY specifies the direction of the up vector
	 * @param upZ specifies the direction of the up vector
	 *
	 * @return the resulting viewing transformation
	 */
	static inline float4x4 lookAt( float eyeX, float eyeY, float eyeZ,
								float centerX, float centerY, float centerZ,
								float upX, float upY, float upZ );

	/**
	 * Define a translation matrix
	 *
	 * @param x specify the x, y, and z coordinates of a translation vector
	 * @param y specify the x, y, and z coordinates of a translation vector
	 * @param z specify the x, y, and z coordinates of a translation vector
	 *
	 * @return the resulting translation matrix
	 */
	static inline float4x4 translate( float x, float y, float z);

	/**
	 * Multiply two matrices
	 *
	 * @param a first matrix
	 * @param b second matrix
	 *
	 * @return the resulting matrix
	 */
	static inline float4x4 mul( const float4x4& a, const float4x4& b );

	/**
	 * Define a transformation that produces a parallel projection (i.e. orthographic)
	 *
	 * @param left specify the coordinates for the left and right vertical clipping planes
	 * @param right specify the coordinates for the left and right vertical clipping planes
	 * @param bottom specify the coordinates for the bottom and top horizontal clipping planes
	 * @param top specify the coordinates for the bottom and top horizontal clipping planes
	 * @param nearVal specify the distances to the nearer and farther depth clipping planes. These values are negative if the plane is to be behind the viewer.
	 * @param farVal specify the distances to the nearer and farther depth clipping planes. These values are negative if the plane is to be behind the viewer.
	 *
	 * @return the resulting orthographic matrix
	 */
	static inline float4x4 ortho( float left, float right, float bottom, float top,	float nearVal, float farVal );

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

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "helper_matrix.inl"

#endif // !_HELPER_MATRIX_H_
