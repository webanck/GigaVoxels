///*
// * GigaVoxels is a ray-guided streaming library used for efficient
// * 3D real-time rendering of highly detailed volumetric scenes.
// *
// * Copyright (C) 2011-2012 INRIA <http://www.inria.fr/>
// *
// * Authors : GigaVoxels Team
// *
// * This program is free software: you can redistribute it and/or modify
// * it under the terms of the GNU General Public License as published by
// * the Free Software Foundation, either version 3 of the License, or
// * (at your option) any later version.
// *
// * This program is distributed in the hope that it will be useful,
// * but WITHOUT ANY WARRANTY; without even the implied warranty of
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// * GNU General Public License for more details.
// *
// * You should have received a copy of the GNU General Public License
// * along with this program.  If not, see <http://www.gnu.org/licenses/>.
// */
//
///** 
// * @version 1.0
// */
//
//#include "SampleViewer.h"
//
///******************************************************************************
// ******************************* INCLUDE SECTION ******************************
// ******************************************************************************/
//
//// Cuda
//#include <cutil.h>
//
//// QGLViewer
//#include <QGLViewer/manipulatedFrame.h>
//
///******************************************************************************
// ****************************** NAMESPACE SECTION *****************************
// ******************************************************************************/
//
///******************************************************************************
// ************************* DEFINE AND CONSTANT SECTION ************************
// ******************************************************************************/
//
///******************************************************************************
// ***************************** TYPE DEFINITION ********************************
// ******************************************************************************/
//
///******************************************************************************
// ***************************** METHOD DEFINITION ******************************
// ******************************************************************************/
//
///******************************************************************************
// * Constructor
// ******************************************************************************/
//SampleViewer::SampleViewer()
//:	QGLViewer()
//,	_clippingPlane( NULL )
//,	_sampleCore( NULL )
//,	_light1( NULL )
//{
//	_sampleCore = new SampleCore();
//}
//
///******************************************************************************
// * Destructor
// ******************************************************************************/
//SampleViewer::~SampleViewer()
//{
//	delete _sampleCore;
//}
//
///******************************************************************************
// * Initialize the viewer
// ******************************************************************************/
//void SampleViewer::init()
//{
//	if ( glewInit() != GLEW_OK )
//	{
//		exit( 1 );
//	}
//
//	// GigaVoxels pipeline initialization
//	_sampleCore->init();
//
//	// Read QGLViewer XML settings file if any
//	restoreStateFromFile();
//
//	// Light initialization
//	_light1 = new qglviewer::ManipulatedFrame();
//	_light1->setPosition( 1.f, 1.f, 1.f );
//
//	glEnable( GL_LIGHT1 );
//
//	const GLfloat ambient[]  = { 0.2f, 0.2f, 2.0f, 1.0f };
//	const GLfloat diffuse[]  = { 0.8f, 0.8f, 1.0f, 1.0f };
//	const GLfloat specular[] = { 0.0f, 0.0f, 1.0f, 1.0f };
//
//	glLightfv( GL_LIGHT1, GL_AMBIENT,  ambient );
//	glLightfv( GL_LIGHT1, GL_SPECULAR, specular );
//	glLightfv( GL_LIGHT1, GL_DIFFUSE,  diffuse );
//
//	// Clipping Plane
//	_clippingPlane = new qglviewer::ManipulatedFrame();
//	_clippingPlane->setPosition( 1.5f, 1.5f, 1.5f );
//	
//	// Viewer initialization
//	setMouseTracking( true );
//	setAnimationPeriod( 0 );
//	startAnimation();
//}
//
///******************************************************************************
// * Draw function called each frame
// ******************************************************************************/
//void SampleViewer::draw()
//{
//	glClearColor( 0.0f, 0.1f, 0.3f, 0.0f );
//	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
//
//	glEnable( GL_DEPTH_TEST );
//	glDisable( GL_LIGHTING );
//
//	float pos[ 4 ] = { 1.0f, 1.0f, 1.0f, 1.0f };
//	_light1->getPosition( pos[ 0 ], pos[ 1 ], pos[ 2 ] );
//
//	glLightfv( GL_LIGHT1, GL_POSITION, pos );
//	glEnable( GL_LIGHT1 ); // must be enabled for drawLight()
//
//	if (_light1->grabsMouse())
//	{
//		drawLight( GL_LIGHT1, 1.2f );
//	}
//	else
//	{
//		drawLight( GL_LIGHT1 );
//	}
//
//	glDisable( GL_LIGHT1 );
//	glDisable( GL_DEPTH_TEST );
//
//	//---------------------------------------------------------------------
//	glEnable( GL_DEPTH_TEST );
//	glPushMatrix();
//	glMultMatrixd( _clippingPlane->matrix() );
//	// Since the Clipping Plane equation is multiplied by the current modelView, we can define a 
//	// constant equation (plane normal along Z and passing by the origin) since we are here in the
//	// manipulatedFrame coordinates system (we glMultMatrixd() with the manipulatedFrame matrix()).
//	//static const GLdouble equation[] = { 0.0, 0.0, 1.0, 0.0 };
//	//glClipPlane(GL_CLIP_PLANE0, equation);
//
//	bool clippingPlaneUsed = false;
//	if ( _clippingPlane->grabsMouse() )
//	{
//		clippingPlaneUsed = true;
//	}
//	else
//	{
//		//
//	}
//
//	// Draws a 3D arrow along the positive Z axis.
//	glColor3f( 0.2f, 0.8f, 0.5f );
//	drawArrow( 0.4f, 0.015f );
//	// Draw a plane representation: Its normal...
//	// ...and a quad (with a slightly shifted z so that it is not clipped).
//	if ( clippingPlaneUsed )
//	{
//		glColor3f( 0.0f, 1.0f, 0.0f );
//	}
//	else
//	{
//		glColor3f( 0.8f, 0.8f, 0.8f );
//	}
//	glBegin(GL_QUADS);
//	glVertex3f(-1.0, -1.0, 0.001f);
//	glVertex3f(-1.0,  1.0, 0.001f);
//	glVertex3f( 1.0,  1.0, 0.001f);
//	glVertex3f( 1.0, -1.0, 0.001f);
//	glEnd();
//	glPopMatrix();
//
//	glDisable( GL_DEPTH_TEST );
//	//---------------------------------------------------------------------
//
//	// Update DEVICE memory with light position
//	float3 lightPos = make_float3( pos[ 0 ], pos[ 1 ], pos[ 2 ] );
//	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( "lightPosition", &lightPos,	sizeof( lightPos ), 0, cudaMemcpyHostToDevice ) );
//
//	// Ask rendition of the GigaVoxels pipeline
//	_sampleCore->draw();
//}
//
///******************************************************************************
// * Resize GL event handler
// *
// * @param width the new width
// * @param height the new height
// ******************************************************************************/
//void SampleViewer::resizeGL( int width, int height )
//{
//	QGLViewer::resizeGL( width, height );
//	_sampleCore->resize( width, height );
//}
//
///******************************************************************************
// * Get the viewer size hint
// *
// * @return the viewer size hint
// ******************************************************************************/
//QSize SampleViewer::sizeHint() const
//{
//	return QSize( 512, 512 );
//}
//
///******************************************************************************
// * Key press event handler
// *
// * @param e the event
// ******************************************************************************/
//void SampleViewer::keyPressEvent( QKeyEvent* e )
//{
//	QGLViewer::keyPressEvent( e );
//
//	switch ( e->key() )
//	{
//	case Qt::Key_Plus:
//		_sampleCore->incMaxVolTreeDepth();
//		break;
//
//	case Qt::Key_Minus:
//		_sampleCore->decMaxVolTreeDepth();
//		break;
//
//	case Qt::Key_C:
//		_sampleCore->clearCache();
//		break;
//
//	case Qt::Key_D:
//		_sampleCore->toggleDynamicUpdate();
//		break;
//
//	case Qt::Key_I:
//		_sampleCore->togglePerfmonDisplay( 1 );
//		break;
//
//	case Qt::Key_T:
//		_sampleCore->toggleDisplayOctree();
//		break;
//
//	case Qt::Key_U:
//		_sampleCore->togglePerfmonDisplay( 2 );
//		break;
//	}
//}
