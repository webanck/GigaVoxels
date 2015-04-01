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

#include "GvUtils/GvProxyGeometry.h"

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
GvProxyGeometry::GvProxyGeometry()
{
	_geometry[ 0 ][ 0 ] = 0.f;
	_geometry[ 0 ][ 1 ] = 0.f;
	_geometry[ 0 ][ 2 ] = 0.f;

	_geometry[ 1 ][ 0 ] = 1.f;
	_geometry[ 1 ][ 1 ] = 0.f;
	_geometry[ 1 ][ 2 ] = 0.f;

	_geometry[ 2 ][ 0 ] = 1.f;
	_geometry[ 2 ][ 1 ] = 0.f;
	_geometry[ 2 ][ 2 ] = 1.f;

	_geometry[ 3 ][ 0 ] = 0.f;
	_geometry[ 3 ][ 1 ] = 0.f;
	_geometry[ 3 ][ 2 ] = 1.f;

	_geometry[ 4 ][ 0 ] = 0.f;
	_geometry[ 4 ][ 1 ] = 1.f;
	_geometry[ 4 ][ 2 ] = 0.f;

	_geometry[ 5 ][ 0 ] = 1.f;
	_geometry[ 5 ][ 1 ] = 1.f;
	_geometry[ 5 ][ 2 ] = 0.f;

	_geometry[ 6 ][ 0 ] = 1.f;
	_geometry[ 6 ][ 1 ] = 1.f;
	_geometry[ 6 ][ 2 ] = 1.f;

	_geometry[ 7 ][ 0 ] = 0.f;
	_geometry[ 7 ][ 1 ] = 1.f;
	_geometry[ 7 ][ 2 ] = 1.f;
}

/******************************************************************************
 * Desconstructor
 ******************************************************************************/
GvProxyGeometry::~GvProxyGeometry()
{
}

/******************************************************************************
 * 
 ******************************************************************************/
static void multMatrixVecd( const double pMatrix[ 16 ], const double pIn[ 4 ],
		      double pOut[ 4 ] )
{
	for ( int i = 0; i < 4; i++ )
	{
		pOut[ i ] = 
			pIn[ 0 ] * pMatrix[ 0 * 4 + i ] +
			pIn[ 1 ] * pMatrix[ 1 * 4 + i ] +
			pIn[ 2 ] * pMatrix[ 2 * 4 + i ] +
			pIn[ 3 ] * pMatrix[ 3 * 4 + i ];
	}
}

/******************************************************************************
 * Kinf of gluProject() method — Map object coordinates to window coordinates
 ******************************************************************************/
bool project(	double pObjx, double pObjy, double pObjz, 
				const double pModelMatrix[ 16 ], const double pProjMatrix[ 16 ], const double pViewport[ 4 ],
				double* pWinx, double* pWiny, double* pWinz )
{
	double in[ 4 ];
	double out[ 4 ];

	in[ 0 ] = pObjx;
	in[ 1 ] = pObjy;
	in[ 2 ] = pObjz;
	in[ 3 ] = 1.0;
	
	multMatrixVecd( pModelMatrix, in, out );
	multMatrixVecd( pProjMatrix, out, in );
	if ( in[ 3 ] == 0.0 )
	{
		return false;
	}
	in[ 0 ] /= in[ 3 ];
	in[ 1 ] /= in[ 3 ];
	in[ 2 ] /= in[ 3 ];

	/* Map x, y and z to range 0-1 */
	in[ 0 ] = in[ 0 ] * 0.5 + 0.5;
	in[ 1 ] = in[ 1 ] * 0.5 + 0.5;
	in[ 2 ] = in[ 2 ] * 0.5 + 0.5;

	/* Map x,y to viewport */
	in[ 0 ] = in[ 0 ] * pViewport[ 2 ] + pViewport[ 0 ];
	in[ 1 ] = in[ 1 ] * pViewport[ 3 ] + pViewport[ 1 ];

	*pWinx = in[ 0 ];
	*pWiny = in[ 1 ];
	*pWinz = in[ 2 ];

	return true;
}

/******************************************************************************
 * Compute the AABB of a 3D model projected on screen
 ******************************************************************************/
void GvProxyGeometry::getProjectedAABB( float& pMinX, float& pMinY, float& pMaxX, float& pMaxY ) const
{
	//float projectedGeometry[ 8 ][ 3 ];

	//pMinX = std::numeric_limits< float >::max();
	//pMaxX = std::numeric_limits< float >::min();
	//pMinY = std::numeric_limits< float >::max();
	//pMaxY = std::numeric_limits< float >::min();
	//for ( int i = 0; i < 8; i++ )
	//{
	//	// Replace following code by gluProject code
	//	//camera()->getProjectedCoordinatesOf( _geometry[ i ], projectedGeometry[ i ] );
	//	project( _geometry[ i ][ 0 ], _geometry[ i ][ 1 ], _geometry[ i ][ 2 ], );

	//	if ( projectedGeometry[ i ][ 0 ] < pMinX )
	//	{
	//		pMinX = projectedGeometry[ i ][ 0 ];
	//	}
	//	
	//	if ( projectedGeometry[ i ][ 0 ] > pMaxX )
	//	{
	//		pMaxX = projectedGeometry[ i ][ 0 ];
	//	}

	//	if ( projectedGeometry[ i ][ 1 ] < pMinY )
	//	{
	//		pMinY = projectedGeometry[ i ][ 1 ];
	//	}
	//	
	//	if ( projectedGeometry[ i ][ 1 ] > pMaxY )
	//	{
	//		pMaxY = projectedGeometry[ i ][ 1 ];
	//	}
	//}
}

/******************************************************************************
 * Compute the AABB of a 3D model projected on screen
 ******************************************************************************/
void GvProxyGeometry::getProjected2DBBox( const GLdouble* pModel, const GLdouble* pProj,const GLint* pView, GLdouble& pMinX, GLdouble& pMinY, GLdouble& pMaxX, GLdouble& pMaxY )
{
	GLdouble winX;
	GLdouble winY;
	GLdouble winZ;

	GLint width = pView[ 2 ] - pView[ 0 ];
	GLint height = pView[ 3 ] - pView[ 1 ];

	pMinX = std::numeric_limits< GLdouble >::max();
	pMaxX = std::numeric_limits< GLdouble >::min();
	pMinY = std::numeric_limits< GLdouble >::max();
	pMaxY = std::numeric_limits< GLdouble >::min();
	for ( int i = 0; i < 8; i++ )
	{
		// Replace following code by gluProject code to speed computation.
		GLint isOK = gluProject( _geometry[ i ][ 0 ], _geometry[ i ][ 1 ], _geometry[ i ][ 2 ], pModel, pProj, pView, &winX, &winY, &winZ );
		if ( ! isOK )
		{
			int toto = 0;
			toto++;
		}

		if ( winX < pMinX )
		{
			pMinX = winX;
		}
		
		if ( winX > pMaxX )
		{
			pMaxX = winX;
		}

		if ( winY < pMinY )
		{
			pMinY = winY;
		}
		
		if ( winY > pMaxY )
		{
			pMaxY = winY;
		}
	}

	// WARNING :
	// The x and y coordinates of the returned Vec are expressed in pixel, (0,0) being the upper left corner of the window.
	// The z coordinate ranges between 0.0 (near plane) and 1.0 (excluded, far plane). See the gluProject man page for details.
	//GLdouble tmpMinY = pMinY;
	//pMinY =  pMaxY;
	//pMaxY = height - tmpMinY;

	// Clamp against viewport borders
	if ( pMinX < pView[ 0 ] )
	{
		pMinX = pView[ 0 ];
	}
	if ( pMaxX < pView[ 0 ] )
	{
		pMaxX = pView[ 0 ];
	}

	if ( pMaxX > width )
	{
		pMaxX = width;
	}
	if ( pMinX > width )
	{
		pMinX = width;
	}

	if ( pMinY < pView[ 1 ] )
	{
		pMinY = pView[ 1 ];
	}
	if ( pMaxY < pView[ 1 ] )
	{
		pMaxY = pView[ 1 ];
	}

	if ( pMaxY > height )
	{
		pMaxY = height;
	}
	if ( pMinY > height )
	{
		pMinY = height;
	}
}
