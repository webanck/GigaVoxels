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

#include "SampleViewer.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GsGraphics/GsGraphicsCore.h>

// QGLViewer
#include <QGLViewer/manipulatedFrame.h>

// System
#include <cstdio>
#include <cstdlib>

//--------------------------------------------------------
// STL
#include <algorithm>

#include <cassert>

#include <QGLViewer/constraint.h>

// GigaVoxels
#include <GvCore/vector_types_ext.h>
//--------------------------------------------------------

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

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
SampleViewer::SampleViewer()
:	QGLViewer()
,	_clippingPlane( NULL )
,	_sampleCore( NULL )
,	_light1( NULL )
{
	_sampleCore = new SampleCore();
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleViewer::~SampleViewer()
{
	delete _sampleCore;
}

/******************************************************************************
 * Initialize the viewer
 ******************************************************************************/
void SampleViewer::init()
{
	// GLEW initialization
	GLenum error = glewInit();
	if ( error != GLEW_OK )
	{
		// Problem : glewInit failed
		fprintf( stderr, "Error: %s\n", glewGetErrorString( error ) );

		// Exit program
		exit( 1 );
	}

	// LOG associated Graphics Core library properties/capabilities (i.e. OpenGL)
	GsGraphics::GsGraphicsCore::printInfo();

	// GigaVoxels pipeline initialization
	_sampleCore->init();

	// Read QGLViewer XML settings file if any
	restoreStateFromFile();

	// Light initialization
	_light1 = new qglviewer::ManipulatedFrame();
	_light1->setPosition( 1.f, 1.f, 1.f );

	glEnable( GL_LIGHT1 );

	const GLfloat ambient[]  = { 0.2f, 0.2f, 2.0f, 1.0f };
	const GLfloat diffuse[]  = { 0.8f, 0.8f, 1.0f, 1.0f };
	const GLfloat specular[] = { 0.0f, 0.0f, 1.0f, 1.0f };

	glLightfv( GL_LIGHT1, GL_AMBIENT,  ambient );
	glLightfv( GL_LIGHT1, GL_SPECULAR, specular );
	glLightfv( GL_LIGHT1, GL_DIFFUSE,  diffuse );

	// Clipping Plane
	_clippingPlane = new qglviewer::ManipulatedFrame();
	// GigaVoxels model matrix : no optimal...
	//_clippingPlane->setPosition( -0.5f, -0.5f, -0.5f );	// GigaVoxels
	//_clippingPlane->translate( 0.5f, 0.5f, 0.5f );	// center of GigaVoxels BBox
	//_clippingPlane->setTranslation( -0.5f, -0.5f, -0.5f );	// GigaVoxels
	_clippingPlane->setTranslation( -0.5f + 0.5f, -0.5f + 0.5f, -0.5f + 0.5f );	// GigaVoxels
	//-------------------------- TEST
	qglviewer::LocalConstraint* localConstraint = new qglviewer::LocalConstraint();
	localConstraint->setTranslationConstraintType( qglviewer::AxisPlaneConstraint::AXIS );
	qglviewer::Vec dir( 0.0, 0.0, 1.0 );
	localConstraint->setTranslationConstraintDirection( dir );
	_clippingPlane->setConstraint( localConstraint );
	//--------------------------
		
	// Viewer initialization
	setMouseTracking( true );
	setAnimationPeriod( 0 );
	startAnimation();
}

//---------------------------------------------------------------------------------

// OutVD > 0 means ray is back-facing the plane
// returns false if there is no intersection because ray is perpedicular to plane
bool ray_to_plane( const float3& pRayOrig, const float3& pRayDir, const float4& pPlane, float* pOutT, float* pOutVD )
{
	*pOutVD = pPlane.x * pRayDir.x + pPlane.y * pRayDir.y + pPlane.z * pRayDir.z;
	
	if ( *pOutVD == 0.0f )
	{
		return false;
	}
	
	*pOutT = - ( pPlane.x * pRayOrig.x + pPlane.y * pRayOrig.y + pPlane.z * pRayOrig.z + pPlane.w ) / *pOutVD;

	return true;
}

// Maximum out_point_count == 6, so out_points must point to 6-element array.
// out_point_count == 0 mean no intersection.
// out_points are not sorted.
void calc_plane_aabb_intersection_points( const float4 &plane,
    const float3 &aabb_min, const float3 &aabb_max,
    float3 *out_points, unsigned &out_point_count )
{
    out_point_count = 0;
    float vd, t;

    // Test edges along X axis, pointing right.
    float3 dir = make_float3(aabb_max.x - aabb_min.x, 0.f, 0.f);
    float3 orig = aabb_min;
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;
    orig = make_float3(aabb_min.x, aabb_max.y, aabb_min.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;
    orig = make_float3(aabb_min.x, aabb_min.y, aabb_max.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;
    orig = make_float3(aabb_min.x, aabb_max.y, aabb_max.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;

    // Test edges along Y axis, pointing up.
    dir = make_float3(0.f, aabb_max.y - aabb_min.y, 0.f);
    orig = make_float3(aabb_min.x, aabb_min.y, aabb_min.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;
    orig = make_float3(aabb_max.x, aabb_min.y, aabb_min.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;
    orig = make_float3(aabb_min.x, aabb_min.y, aabb_max.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;
    orig = make_float3(aabb_max.x, aabb_min.y, aabb_max.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;

    // Test edges along Z axis, pointing forward.
    dir = make_float3(0.f, 0.f, aabb_max.z - aabb_min.z);
    orig = make_float3(aabb_min.x, aabb_min.y, aabb_min.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;
    orig = make_float3(aabb_max.x, aabb_min.y, aabb_min.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;
    orig = make_float3(aabb_min.x, aabb_max.y, aabb_min.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;
    orig = make_float3(aabb_max.x, aabb_max.y, aabb_min.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;

	assert( out_point_count <= 6 );
}

struct Comp3DVectorfunctor
{
	bool operator()( const float3& pPointA, const float3& pPointB )
	{
		float3 v = cross( pPointA - _origin, pPointB - _origin );

		return dot( v, _planeNormal ) < 0.0f;
	}

	float3 _planeNormal;
	float3 _origin;
};

/******************************************************************************
 * ...
 ******************************************************************************/
void sort_points( float3* pPoints, unsigned pPoint_count, const float4& pPlane )
{
    if ( pPoint_count == 0 )
	{
		return;
	}

    const float3 plane_normal = make_float3( pPlane.x, pPlane.y, pPlane.z );
    const float3 origin = pPoints[ 0 ];

	Comp3DVectorfunctor comp3DVector;
	comp3DVector._planeNormal = plane_normal;
	comp3DVector._origin = origin;

    std::sort( pPoints, pPoints + pPoint_count, comp3DVector );
}

/******************************************************************************
 * Used to display the N-tree
 *
 * @param p1 Position
 * @param p2 Position
 ******************************************************************************/
void drawCube( const float3& p1, const float3& p2 )
{
	glBegin(GL_QUADS);
	// Front Face
	glVertex3f(p1.x, p1.y, p2.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p2.z);	// Top Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p2.z);	// Top Right Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p2.z);	// Bottom Right Of The Texture and Quad

	// Back Face
	glVertex3f(p1.x, p1.y, p1.z);	// Bottom Right Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p1.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p1.z);	// Top Left Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p1.z);	// Top Right Of The Texture and Quad

	// Top Face
	glVertex3f(p1.x, p2.y, p1.z);	// Top Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p1.z);	// Top Right Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p2.z);	// Bottom Right Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p2.z);	// Bottom Left Of The Texture and Quad

	// Bottom Face
	glVertex3f(p1.x, p1.y, p1.z);	// Top Right Of The Texture and Quad
	glVertex3f(p1.x, p1.y, p2.z);	// Bottom Right Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p2.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p1.z);	// Top Left Of The Texture and Quad

	// Right face
	glVertex3f(p2.x, p1.y, p1.z);	// Bottom Right Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p2.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p2.z);	// Top Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p1.z);	// Top Right Of The Texture and Quad

	// Left Face
	glVertex3f(p1.x, p1.y, p1.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p1.z);	// Top Left Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p2.z);	// Top Right Of The Texture and Quad
	glVertex3f(p1.x, p1.y, p2.z);	// Bottom Right Of The Texture and Quad

	glEnd();
}
//---------------------------------------------------------------------------------

/******************************************************************************
 * Draw function called each frame
 ******************************************************************************/
void SampleViewer::draw()
{
	glClearColor( 0.0f, 0.1f, 0.3f, 0.0f );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	glEnable( GL_DEPTH_TEST );
	glDisable( GL_LIGHTING );

	float pos[ 4 ] = { 1.0f, 1.0f, 1.0f, 1.0f };
	_light1->getPosition( pos[ 0 ], pos[ 1 ], pos[ 2 ] );

	glLightfv( GL_LIGHT1, GL_POSITION, pos );
	glEnable( GL_LIGHT1 ); // must be enabled for drawLight()

	if (_light1->grabsMouse())
	{
		drawLight( GL_LIGHT1, 1.2f );
	}
	else
	{
		drawLight( GL_LIGHT1 );
	}

	glDisable( GL_LIGHT1 );
	glDisable( GL_DEPTH_TEST );

	//---------------------------------------------------------------------
	glEnable( GL_DEPTH_TEST );

	// Enable plane clipping
//	glEnable( GL_CLIP_PLANE0 );

	//glPushMatrix();
	//glMultMatrixd( _clippingPlane->matrix() );
	////glMultMatrixd( _clippingPlane->matrix() );
	////glMultMatrixd( _clippingPlane->worldMatrix() );
	//// Since the Clipping Plane equation is multiplied by the current modelView, we can define a 
	//// constant equation (plane normal along Z and passing by the origin) since we are here in the
	//// manipulatedFrame coordinates system (we glMultMatrixd() with the manipulatedFrame matrix()).
	////static const GLdouble equation[] = { 0.0, 0.0, 1.0, 0.0 };
	////glClipPlane( GL_CLIP_PLANE0, equation );
	//bool clippingPlaneUsed = false;
	//if ( _clippingPlane->grabsMouse() )
	//{
	//	clippingPlaneUsed = true;
	//}
	//else
	//{
	//	//
	//}
	//// Draws a 3D arrow along the positive Z axis.
	//glColor3f( 0.2f, 0.8f, 0.5f );
	////glPushMatrix();
	////glTranslatef( -0.5f, -0.5f, -0.5f );
	//	drawArrow( 0.4f, 0.015f );
	////glPopMatrix();
	//// Draw a plane representation: Its normal...
	//// ...and a quad (with a slightly shifted z so that it is not clipped).
	//if ( clippingPlaneUsed )
	//{
	//	//glColor3f( 0.0f, 1.0f, 0.0f );
	//	glColor4f( 0.0f, 1.0f, 0.0f, 0.2f );
	//}
	//else
	//{
	//	//glColor3f( 0.8f, 0.8f, 0.8f );
	//	glColor4f( 0.8f, 0.8f, 0.8f, 0.2f );
	//}
	//// Activate blending
	//glEnable( GL_BLEND );
	//glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	////glPushMatrix();
	////glTranslatef( -0.5f, -0.5f, -0.5f );
	//	glBegin( GL_QUADS );
	//		/*glVertex3f( -1.0, -1.0, 0.001f );
	//		glVertex3f( -1.0,  1.0, 0.001f );
	//		glVertex3f(  1.0,  1.0, 0.001f );
	//		glVertex3f(  1.0, -1.0, 0.001f );*/
	//		glVertex3f( -1.0, -1.0, 0.000001f );
	//		glVertex3f( -1.0,  1.0, 0.000001f );
	//		glVertex3f(  1.0,  1.0, 0.000001f );
	//		glVertex3f(  1.0, -1.0, 0.000001f );
	//	glEnd();
	////glPopMatrix();
	//glPopMatrix();

	//----------------------------------------
	// Compute in World Space
	//const GLdouble* clippingPlaneMatrix = _clippingPlane->matrix();
	const GLdouble* clippingPlaneMatrix = _clippingPlane->worldMatrix();
	float4x4 myClippingPlaneMatrix;
	for ( int i = 0; i < 16; i++ )
	{
		myClippingPlaneMatrix._array[ i ] = static_cast< float >( clippingPlaneMatrix[ i ] );
	}
	float4x4 inverseTransposeClippingPlaneMatrix = transpose( inverse( myClippingPlaneMatrix ) );
	float3 planeNormal = mulRot( myClippingPlaneMatrix, make_float3( 0.0f, 0.0f, 1.0f ) );
	float3 planeOrigin = mul( transpose( myClippingPlaneMatrix ), make_float3( 0.0f, 0.0f, 0.0f ) );
	float d = -dot( planeNormal, planeOrigin );
	float4 plane = make_float4( planeNormal.x, planeNormal.y, planeNormal.z, d );
	////--------------------------
	//std::cout << "\nCLIP_FRAME [ " << myClippingPlaneMatrix._array[ 12 ] << " ; " << myClippingPlaneMatrix._array[ 13 ] << " ; " << myClippingPlaneMatrix._array[ 14 ] << " ]" << std::endl;
	//std::cout << "NORMAL [ " << planeNormal.x << " ; " << planeNormal.y << " ; " << planeNormal.z << " ]" << std::endl;
	//std::cout << "ORIGIN [ " << planeOrigin.x << " ; " << planeOrigin.y << " ; " << planeOrigin.z << " ]" << std::endl;
	//std::cout << "PLANE [ " << plane.x << " ; " << plane.y << " ; " << plane.z << " ; " << plane.w << " ]" << std::endl;
	////--------------------------
	float3 aabb_min = make_float3( 0.0f, 0.0f, 0.0f );	// no world transform for GigaVoxels => it has been commented...
	float3 aabb_max = make_float3( 1.0f, 1.0f, 1.0f );
	//--------------------
	aabb_min += make_float3( -0.5f, -0.5f, -0.5f );	// no world transform for GigaVoxels => it has been commented...
	aabb_max += make_float3( -0.5f, -0.5f, -0.5f );
	//--------------------
	float3 out_points[ 6 ];
	unsigned out_point_count = 0;
	calc_plane_aabb_intersection_points( plane, aabb_min, aabb_max, out_points, out_point_count );
	sort_points( out_points, out_point_count, plane );

	glLineWidth( 3.f );

	glPushMatrix();
	//glTranslatef( -0.5f, -0.5f, -0.5f );
		glBegin( GL_LINE_LOOP );
			glColor3f( 1.0f, 0.0f, 0.0f );
			for ( unsigned int i = 0; i < out_point_count; i++ )
			{
				glVertex3f( out_points[ i ].x, out_points[ i ].y, out_points[ i ].z );
			}
		glEnd();
	glPopMatrix();

	glLineWidth( 1.f );

	//----------------------------------------------------------------------------
	glPushMatrix();
	glMultMatrixd( _clippingPlane->matrix() );

	bool clippingPlaneUsed = false;
	if ( _clippingPlane->grabsMouse() )
	{
		clippingPlaneUsed = true;
	}
	else
	{
		//
	}
	// Draws a 3D arrow along the positive Z axis.
	glColor3f( 0.2f, 0.8f, 0.5f );
	//glPushMatrix();
	//glTranslatef( -0.5f, -0.5f, -0.5f );	// center of bbox
		drawArrow( 0.4f, 0.015f );
	//glPopMatrix();
	
	glPopMatrix();

	// Draw a plane representation: Its normal...
	// ...and a quad (with a slightly shifted z so that it is not clipped).
	if ( clippingPlaneUsed )
	{
		//glColor3f( 0.0f, 1.0f, 0.0f );
		glColor4f( 0.0f, 1.0f, 0.0f, 0.2f );
	}
	else
	{
		//glColor3f( 0.8f, 0.8f, 0.8f );
		glColor4f( 0.8f, 0.8f, 0.8f, 0.2f );
	}
	glEnable( GL_BLEND );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	glPushMatrix();
		glBegin( GL_POLYGON );
			//glColor4f( 0.0f, 1.0f, 0.0f, 0.2f );
			for ( unsigned int i = 0; i < out_point_count; i++ )
			{
				glVertex3f( out_points[ i ].x, out_points[ i ].y, out_points[ i ].z );
			}
		glEnd();
	glPopMatrix();
	glDisable( GL_BLEND );
	//----------------------------------------------------------------------------

	//----------------------------------------

	//glDisable( GL_BLEND );

	// Enable plane clipping
	//glDisable( GL_CLIP_PLANE0 );

	glDisable( GL_DEPTH_TEST );
	//---------------------------------------------------------------------

	// Update DEVICE memory with light position
	float3 lightPos = make_float3( pos[ 0 ], pos[ 1 ], pos[ 2 ] );
	_sampleCore->setLightPosition( pos[ 0 ], pos[ 1 ], pos[ 2 ] );

	// ------------------------------------------
	glPushMatrix();
	glTranslatef( -0.5f, -0.5f, -0.5f );
		glPolygonMode( GL_FRONT_AND_BACK , GL_LINE );
		glColor3f( 1.f, 1.f, 1.f );
		drawCube( make_float3( 0.f, 0.f, 0.f ), make_float3( 1.f, 1.f, 1.f ) );
		glPolygonMode( GL_FRONT_AND_BACK , GL_FILL);
	glPopMatrix();

	// ------------------------------------------

	// Ask rendition of the GigaVoxels pipeline
	_sampleCore->setClippingPlaneGeometry( out_points, out_point_count );
	_sampleCore->setClippingPlaneMatrix( _clippingPlane->matrix()/*_clippingPlane->matrix()*/ );
	_sampleCore->draw();
}

/******************************************************************************
 * Resize GL event handler
 *
 * @param width the new width
 * @param height the new height
 ******************************************************************************/
void SampleViewer::resizeGL( int width, int height )
{
	QGLViewer::resizeGL( width, height );
	_sampleCore->resize( width, height );
}

/******************************************************************************
 * Get the viewer size hint
 *
 * @return the viewer size hint
 ******************************************************************************/
QSize SampleViewer::sizeHint() const
{
	return QSize( 512, 512 );
}

/******************************************************************************
 * Key press event handler
 *
 * @param e the event
 ******************************************************************************/
void SampleViewer::keyPressEvent( QKeyEvent* e )
{
	QGLViewer::keyPressEvent( e );

	switch ( e->key() )
	{
	case Qt::Key_Plus:
		_sampleCore->incMaxVolTreeDepth();
		break;

	case Qt::Key_Minus:
		_sampleCore->decMaxVolTreeDepth();
		break;

	case Qt::Key_C:
		_sampleCore->clearCache();
		break;

	case Qt::Key_D:
		_sampleCore->toggleDynamicUpdate();
		break;

	case Qt::Key_I:
		_sampleCore->togglePerfmonDisplay( 1 );
		break;

	case Qt::Key_T:
		_sampleCore->toggleDisplayOctree();
		break;

	case Qt::Key_U:
		_sampleCore->togglePerfmonDisplay( 2 );
		break;
	}
}

///******************************************************************************
// * Implementation of the MouseGrabber main method.
// * 
// * The ManipulatedFrame grabsMouse() when the mouse is within a 10 pixels region around its Camera::projectedCoordinatesOf() position().
// *
// * @param pX ...
// * @param pY ...
// * @param pCamera ...
// ******************************************************************************/
//void ClippingPlaneManipulator::checkIfGrabsMouse( int pX, int pY, const qglviewer::Camera* const pCamera )
//{
//	const int thresold = 10;
//	const qglviewer::Vec proj = pCamera->projectedCoordinatesOf( position() );
//	setGrabsMouse( keepsGrabbingMouse_ || ( ( fabs( pX - proj.x ) < thresold ) && ( fabs( pY - proj.y ) < thresold ) ) );
//}
