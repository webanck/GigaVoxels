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

// System
#include <cstdio>
#include <cstdlib>

//------------------------
// TEST
#include <GL/freeglut.h>
//------------------------

// QGLViewer
#include <QGLViewer/manipulatedFrame.h>

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

	// Viewer initialization
	setMouseTracking( true );
	setAnimationPeriod( 0 );
	startAnimation();
}

/******************************************************************************
 * Draw function called each frame
 ******************************************************************************/
void SampleViewer::draw()
{
	// Clear default frame buffer
	//glClearColor( 0.0f, 0.1f, 0.3f, 0.0f );					=> already done by SampleViewr::setBackgroundColor()
	//glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );		=> already done in QGLViewr::preDraw() method

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

//	//------------------------
//	// Draw the user OpenGL scene
//	//glutSolidTorus( double rint, double rext, int ns, int nr );
//	// Activate blending
//	//glCullFace( GL_FRONT );
////	glEnable( GL_BLEND );
////	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
////	double rint = 0.3;
////	double rext = 1.5;
////	int ns = 100;
////	int nr  = 100;
//	//glColor4f( 0.f, 1.f, 0.f, 0.5f );
//	//glutSolidTorus( rint, rext, ns, nr );
//	glBegin( GL_QUADS );
//		glColor4f( 0.0f, 1.0f, 0.0f, 0.5f );
//		glVertex3f( -1.f, -1.f, 0.f );
//		glVertex3f( 1.f, -1.f, 0.f );
//		glVertex3f( 1.f, 1.f, 0.f );
//		glVertex3f( -1.f, 1.f, 0.f );
//	glEnd();
//	//------------------------

	glDisable( GL_LIGHT1 );
	glDisable( GL_DEPTH_TEST );

	// Update DEVICE memory with light position
	float3 lightPos = make_float3( pos[ 0 ], pos[ 1 ], pos[ 2 ] );
	_sampleCore->setLightPosition( pos[ 0 ], pos[ 1 ], pos[ 2 ] );

	// Ask rendition of the GigaVoxels pipeline
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
