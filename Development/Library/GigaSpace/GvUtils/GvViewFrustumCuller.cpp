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

#include "GvUtils/GvViewFrustumCuller.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// STL
#include <limits>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvUtils;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvViewFrustumCuller::GvViewFrustumCuller()
{
}

/******************************************************************************
 * Desconstructor
 ******************************************************************************/
GvViewFrustumCuller::~GvViewFrustumCuller()
{
}

/******************************************************************************
 * Fast extraction of viewing frustum planes from the Model-View-Projection matrix
 *
 * @param pMatrix ...
 * @param pNormalize ...
 ******************************************************************************/
void GvViewFrustumCuller::extractViewingFrustumPlanes( const float4x4& pMatrix, bool pNormalize )
{
	// Matrices are stored in column-major order :
	//
	// 0  4  8 12
	// 1  5  9 13
	// 2  6 10 14
	// 3  7 11 15

	// Left clipping plane
	_planes[ eLeft ].x = pMatrix._array[ 3 ] + pMatrix._array[ 0 ];
	_planes[ eLeft ].y = pMatrix._array[ 7 ] + pMatrix._array[ 4 ];
	_planes[ eLeft ].z = pMatrix._array[ 11 ] + pMatrix._array[ 8 ];
	_planes[ eLeft ].w = pMatrix._array[ 15 ] + pMatrix._array[ 12 ];

	// Right clipping plane
	_planes[ eRight ].x = pMatrix._array[ 3 ] - pMatrix._array[ 0 ];
	_planes[ eRight ].y = pMatrix._array[ 7 ] - pMatrix._array[ 4 ];
	_planes[ eRight ].z = pMatrix._array[ 11 ] - pMatrix._array[ 8 ];
	_planes[ eRight ].w = pMatrix._array[ 15 ] - pMatrix._array[ 12 ];

	// Bottom clipping plane
	_planes[ eBottom ].x = pMatrix._array[ 3 ] + pMatrix._array[ 1 ];
	_planes[ eBottom ].y = pMatrix._array[ 7 ] + pMatrix._array[ 5 ];
	_planes[ eBottom ].z = pMatrix._array[ 11 ] + pMatrix._array[ 9 ];
	_planes[ eBottom ].w = pMatrix._array[ 15 ] + pMatrix._array[ 13 ];

	// Top clipping plane
	_planes[ eTop ].x = pMatrix._array[ 3 ] - pMatrix._array[ 1 ];
	_planes[ eTop ].y = pMatrix._array[ 7 ] - pMatrix._array[ 5 ];
	_planes[ eTop ].z = pMatrix._array[ 11 ]- pMatrix._array[ 9 ];
	_planes[ eTop ].w = pMatrix._array[ 15 ] - pMatrix._array[ 13 ];

	// Near clipping plane
	_planes[ eNear ].x = pMatrix._array[ 3 ] + pMatrix._array[ 2 ];
	_planes[ eNear ].y = pMatrix._array[ 7 ] + pMatrix._array[ 6 ];
	_planes[ eNear ].z = pMatrix._array[ 11 ] + pMatrix._array[ 10 ];
	_planes[ eNear ].w = pMatrix._array[ 15 ] + pMatrix._array[ 14 ];

	// Far clipping plane
	_planes[ eFar ].x = pMatrix._array[ 3 ] - pMatrix._array[ 2 ];
	_planes[ eFar ].y = pMatrix._array[ 7 ] - pMatrix._array[ 6 ];
	_planes[ eFar ].z = pMatrix._array[ 11 ] - pMatrix._array[ 10 ];
	_planes[ eFar ].w = pMatrix._array[ 15 ] - pMatrix._array[ 14 ];

	// Normalize the plane equations, if requested
	if ( pNormalize )
	{
		_planes[ eLeft ] = normalize( _planes[ eLeft ] );
		_planes[ eRight ] = normalize( _planes[ eRight ] );
		_planes[ eBottom ] = normalize( _planes[ eBottom ] );
		_planes[ eTop ] = normalize( _planes[ eTop ] );
		_planes[ eNear ] = normalize( _planes[ eNear ] );
		_planes[ eFar ] = normalize( _planes[ eFar ] );
	}
}

/******************************************************************************
 * Frustum / Box intersection
 ******************************************************************************/
int GvViewFrustumCuller::frustumBoxIntersect()
{
	bool intersecting = false;
	int result;

	// Iterate through viewing frustum planes
	for ( int i = 0; i < eNbViewingFrustumPlanes; i++ )
	{
		result = planeAABBIntersect();

		if ( result == eOutside )
		{
			return eOutside;
		}
		else if ( result == eIntersecting )
		{
			intersecting = true;
		}
	}

	if ( intersecting )
	{
		return eIntersecting;
	}
	else
	{
		return eInside;
	}
}

/******************************************************************************
 * Plane / AABB intersection
 ******************************************************************************/
int GvViewFrustumCuller::planeAABBIntersect()
{
	return 0;
}
