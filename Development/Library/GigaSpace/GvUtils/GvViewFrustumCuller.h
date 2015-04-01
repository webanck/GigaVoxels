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

#ifndef _GV_VIEW_FRUSTUM_CULLER_H_
#define _GV_VIEW_FRUSTUM_CULLER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/vector_types_ext.h"

// OpenGL
#include <GL/glew.h>

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
 * @class GvViewFrustumCuller
 *
 * @brief The GvViewFrustumCuller class provides interface
 * to view frustum culling features.
 *
 * ...
 *
 * @todo use http://www.iquilezles.org/www/articles/frustumcorrect/frustumcorrect.htm
 * http://www.iquilezles.org/www/articles/frustum/frustum.htm
 */
class GIGASPACE_EXPORT GvViewFrustumCuller
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition of a plane
	 */
	typedef float4 GvPlane;

	/**
	 * Enumeration of the viewing frustum planes
	 */
	enum ViewingFrustumPlane
	{
		eNear,
		eFar,
		eLeft,
		eRight,
		eBottom,
		eTop,
		eNbViewingFrustumPlanes
	};

	/**
	 * Enumeration of the viewing frustum planes
	 */
	enum IntersectionType
	{
		eOutside,
		eInside,
		eIntersecting,
		eNbIntersectionTypes
	};
	
	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvViewFrustumCuller();

	/**
	 * Destructor
	 */
	 virtual ~GvViewFrustumCuller();

	 /**
	  * Frustum / Box intersection
	  */
	 int frustumBoxIntersect();

	  /**
	  * Plane / AABB intersection
	  */
	 int planeAABBIntersect();

	 /**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * List of the 6 viewing frustum planes
	 * (near, far, bottom, top, left and right)
	 */
	GvPlane _planes[ GvViewFrustumCuller::eNbViewingFrustumPlanes ];
	
	/******************************** METHODS *********************************/

	/**
	  * Fast extraction of viewing frustum planes from the Model-View-Projection matrix
	  *
	  * @param pMatrix ...
	  * @param pNormalize ...
	  */
	 void extractViewingFrustumPlanes( const float4x4& pMatrix, bool pNormalize );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} // namespace GvUtils

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

//#include "GvViewFrustumCuller.inl"

#endif
