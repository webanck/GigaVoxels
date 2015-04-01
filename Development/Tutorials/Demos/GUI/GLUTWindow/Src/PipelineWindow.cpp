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

#include "PipelineWindow.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GsGraphics/GsGraphicsCore.h>

// System
#include <cstdio>
#include <cstdlib>

// System
#include <cassert>

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
PipelineWindow::PipelineWindow()
{
	mPipeline = new Pipeline();
	assert( mPipeline != NULL );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
PipelineWindow::~PipelineWindow()
{
	delete mPipeline;
	mPipeline = NULL;
}

/******************************************************************************
 * Initialize
 *
 * @return Flag to tell wheter or not it succeded
 ******************************************************************************/
bool PipelineWindow::initialize()
{
	assert( mPipeline != NULL );

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

	// GigaVoxels pipeline
	mPipeline->init();

//	mLight1 = new qglviewer::ManipulatedFrame();
//	mLight1->setPosition(1.0f, 1.0f, 1.0f);

//	glEnable(GL_LIGHT1);

	const GLfloat ambient[]  = { 0.2f, 0.2f, 2.f, 1.f };
	const GLfloat diffuse[]  = { 0.8f, 0.8f, 1.f, 1.f };
	const GLfloat specular[] = { 0.f , 0.f , 1.f, 1.f };

	glLightfv( GL_LIGHT1, GL_AMBIENT,  ambient );
	glLightfv( GL_LIGHT1, GL_SPECULAR, specular );
	glLightfv( GL_LIGHT1, GL_DIFFUSE,  diffuse );

	// Viewer settings :
	glClearColor( 51.0f / 255.0f, 51.0f / 255.0f, 51.0f / 255.0f, 0.0f );
	// Update GigaVoxels clear color
	mPipeline->setClearColor( 51, 51, 51, 255 );

	return true;
}

/******************************************************************************
 * Finalize
 *
 * @return Flag to tell wheter or not it succeded
 ******************************************************************************/
bool PipelineWindow::finalize()
{
	return true;
}

/******************************************************************************
 * Display callback
 ******************************************************************************/
void PipelineWindow::onDisplayFuncExecuted()
{
	assert( mPipeline != NULL );

	// Clear buffers
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	glDisable( GL_DEPTH_TEST );
	glDisable( GL_LIGHTING );

	// CUDA : send light position to device memory
	float pos[ 4 ] = { 1.0f, 1.0f, 1.0f, 1.0f };
	float3 lightPosition = make_float3( pos[ 0 ], pos[ 1 ], pos[ 2 ] );
	mPipeline->setLightPosition( pos[ 0 ], pos[ 1 ], pos[ 2 ] );

	// Call GigaVoxels pipeline
	mPipeline->draw();
}

/******************************************************************************
 * Reshape callback
 *
 * @param pWidth The new window width in pixels
 * @param pHeight The new window height in pixels
 ******************************************************************************/
void PipelineWindow::onReshapeFuncExecuted( int width, int height )
{
	assert( mPipeline != NULL );
	mPipeline->resize( width, height );
}

/******************************************************************************
 * Keyboard callback
 *
 * @param pKey ASCII character of the pressed key
 * @param pX Mouse location in window relative coordinates when the key was pressed
 * @param pY Mouse location in window relative coordinates when the key was pressed
 ******************************************************************************/
void PipelineWindow::onKeyboardFuncExecuted( unsigned char pKey, int pX, int pY )
{
	assert( mPipeline != NULL );

	switch( pKey )
	{
		case 27 :	// escape
			exit( 0 );
			break;

		case 43 :	// +
			mPipeline->incMaxVolTreeDepth();
			break;

		case 45 :	// -
			mPipeline->decMaxVolTreeDepth();
			break;

		case 99 :	// c
			mPipeline->clearCache();
			break;

		case 100 :	// d
			mPipeline->toggleDynamicUpdate();
			break;

		case 105 :	// i
			mPipeline->togglePerfmonDisplay( 1 );
			break;

		case 116 :	// t
			mPipeline->toggleDisplayOctree();
			break;

		case 117 :	// u
			mPipeline->togglePerfmonDisplay( 2 );
			break;
	}
}

/******************************************************************************
 * Mouse callback
 *
 * @param pButton The button parameter is one of left, middle or right.
 * @param pState The state parameter indicates whether the callback was due to a release or press respectively.
 * @param pX Mouse location in window relative coordinates when the mouse button state changed
 * @param pY Mouse location in window relative coordinates when the mouse button state changed
 ******************************************************************************/
void PipelineWindow::onMouseFuncExecuted( int pButton, int pState, int pX, int pY )
{
	assert( mPipeline != NULL );
}

/******************************************************************************
 * Idle callback
 ******************************************************************************/
void PipelineWindow::onIdleFuncExecuted()
{
	assert( mPipeline != NULL );
}
