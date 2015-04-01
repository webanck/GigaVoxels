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

#include "SampleViewer.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GvError.h>

// System
#include <cstdio>
#include <cstdlib>

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
:	QGLViewer( QGLFormat( QGL::DoubleBuffer | QGL::DepthBuffer | QGL::Rgba | QGL::AlphaChannel | QGL::StencilBuffer /*| QGL::DirectRendering*/ ) )
{
	mSampleCore = new SampleCore();
}

/******************************************************************************
 * Constructor
 ******************************************************************************/
SampleViewer::SampleViewer( QGLFormat& format )
:	QGLViewer( format )
{
	mSampleCore = new SampleCore();
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleViewer::~SampleViewer()
{
	delete mSampleCore;
}

/******************************************************************************
 * ...
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
	GV_CHECK_GL_ERROR();

	mSampleCore->init( this );
	GV_CHECK_GL_ERROR();

	restoreStateFromFile();

	mLight1 = new qglviewer::ManipulatedFrame();
	mLight1->setPosition( 1.0f, 1.0f, 1.0f );
	GV_CHECK_GL_ERROR();

	glEnable( GL_LIGHT1 );

	const GLfloat ambient[]  = { 0.2f, 0.2f, 2.0f, 1.0f };
	const GLfloat diffuse[]  = { 0.8f, 0.8f, 1.0f, 1.0f };
	const GLfloat specular[] = { 0.0f, 0.0f, 1.0f, 1.0f };
	GV_CHECK_GL_ERROR();

	glLightfv( GL_LIGHT1, GL_AMBIENT,  ambient );
	glLightfv( GL_LIGHT1, GL_SPECULAR, specular );
	glLightfv( GL_LIGHT1, GL_DIFFUSE,  diffuse );
	GV_CHECK_GL_ERROR();

	setMouseTracking( true );
	setAnimationPeriod( 0 );
	startAnimation();

	// Adapt scene radius
	setSceneRadius( 500.0 );

	GV_CHECK_GL_ERROR();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleViewer::draw()
{
	GV_CHECK_GL_ERROR();

	glClearColor( 0.0f, 0.1f, 0.3f, 0.0f );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	GV_CHECK_GL_ERROR();

	glEnable( GL_DEPTH_TEST );
	glDisable( GL_LIGHTING );
	GV_CHECK_GL_ERROR();

	float pos[ 4 ] = { 1.0f, 1.0f, 1.0f, 1.0f };
	mLight1->getPosition( pos[ 0 ], pos[ 1 ], pos[ 2 ] );

	float3 lightPos = make_float3( pos[ 0 ], pos[ 1 ], pos[ 2 ] );
	qglviewer::Vec light = qglviewer::Vec( lightPos.x, lightPos.y, lightPos.z );
	// Return the Camera frame coordinates of a point defined in world coordinates
	qglviewer::Vec lightV = camera()->cameraCoordinatesOf( light );
	// Returns the world coordinates of the point whose position is defined in the Camera coordinate system
	qglviewer::Vec wlight = camera()->worldCoordinatesOf( lightV );
	mSampleCore->setLightPosition( lightV.x, lightV.y, lightV.z );
	mSampleCore->setWorldLight( wlight.x, wlight.y, wlight.z );

	GV_CHECK_GL_ERROR();
	//Computing the view and projection matrices for brick production

	// Render GigaVoxels
	mSampleCore->draw();

	glLightfv( GL_LIGHT1, GL_POSITION, pos );
	glEnable( GL_LIGHT1 ); // must be enabled for drawLight()

	if ( mLight1->grabsMouse() )
	{
		drawLight( GL_LIGHT1, 1.2f );
	}
	else
	{
		drawLight( GL_LIGHT1 );
	}

	glDisable( GL_LIGHT1 );
	glDisable( GL_DEPTH_TEST );
}

/******************************************************************************
 * ...
 *
 * @param width ...
 * @param height ...
 ******************************************************************************/
void SampleViewer::resizeGL( int width, int height )
{
	QGLViewer::resizeGL( width, height );
	mSampleCore->resize( width, height );
}

/******************************************************************************
 * ...
 *
 * @return
 ******************************************************************************/
QSize SampleViewer::sizeHint() const
{
	return QSize( 512, 512 );
}

/******************************************************************************
 * ...
 *
 * @param e ...
 ******************************************************************************/
void SampleViewer::keyPressEvent( QKeyEvent* e )
{
	QGLViewer::keyPressEvent( e );

	switch ( e->key() )
	{
	case Qt::Key_Plus:
		mSampleCore->incMaxVolTreeDepth();
		break;

	case Qt::Key_Minus:
		mSampleCore->decMaxVolTreeDepth();
		break;

	case Qt::Key_C:
		mSampleCore->clearCache();
		break;

	case Qt::Key_D:
		mSampleCore->toggleDynamicUpdate();
		break;

	case Qt::Key_I:
		mSampleCore->togglePerfmonDisplay(1);
		break;

	case Qt::Key_T:
		mSampleCore->toggleDisplayOctree();
		break;

	case Qt::Key_U:
		mSampleCore->togglePerfmonDisplay(2);
		break;
	}
}
