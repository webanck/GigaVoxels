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

// freeglut
#include <GL/glew.h>
#include <GL/freeglut.h>

// Project
#include "PipelineWindow.h"

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * The GigaVoxels pipeline window interface
 */
static PipelineWindow* gigaVoxelsPipelineWindow = NULL;

/**
 * Red diffuse light
 */
GLfloat light_diffuse[] = { 1.f, 0.f, 0.f, 1.f };

/**
 * Infinite light location
 */
GLfloat light_position[] = { 1.f, 1.f, 1.f, 0.f };

/**
 * Normals for the 6 faces of a cube
 */
GLfloat normals[ 6 ][ 3 ] =
{
	{ -1.f, 0.f, 0.f }, { 0.f, 1.f, 0.f }, { 1.f, 0.f, 0.f },
	{ 0.f, -1.f, 0.f }, { 0.f, 0.f, 1.f }, { 0.f, 0.f, -1.f }
};

/**
 * Vertex indices for the 6 faces of a cube
 */
GLint faces[ 6 ][ 4 ] =
{
	{ 0, 1, 2, 3 }, { 3, 2, 6, 7 }, { 7, 6, 5, 4 },
	{ 4, 5, 1, 0 }, { 5, 6, 2, 1 }, { 7, 4, 0, 3 }
};

/**
 * Will be filled in with X,Y,Z vertexes
 */
GLfloat vertices[ 8 ][ 3 ];

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Draw a box
 ******************************************************************************/
void drawBox( void )
{
	for ( int i = 0; i < 6; i++ )
	{
		glBegin( GL_QUADS );
			glNormal3fv( &normals[ i ][ 0 ]);
			glVertex3fv( &vertices[ faces[ i ][ 0 ] ][ 0 ]);
			glVertex3fv( &vertices[ faces[ i ][ 1 ] ][ 0 ]);
			glVertex3fv( &vertices[ faces[ i ][ 2 ] ][ 0 ]);
			glVertex3fv( &vertices[ faces[ i ][ 3 ] ][ 0 ]);
		glEnd();
	}
}

/******************************************************************************
 * Display callback
 ******************************************************************************/
void display( void )
{	
	// Clear buffers
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	// Draw scene
	drawBox();

	// GigaVoxels
	if ( gigaVoxelsPipelineWindow != NULL )
	{
		gigaVoxelsPipelineWindow->onDisplayFuncExecuted();
	}

	// Swap buffers
	glutSwapBuffers();
}

/******************************************************************************
 * Reshape callback
 *
 * @param pWidth The new window width in pixels
 * @param pHeight The new window height in pixels
 ******************************************************************************/
void reshape( int pWidth, int pHeight )
{
	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width)
	if ( pHeight == 0 )
	{
		pHeight = 1;
	}

	// Update viewport
	glViewport( 0, 0, pWidth, pHeight );

	// GigaVoxels
	gigaVoxelsPipelineWindow->onReshapeFuncExecuted( pWidth, pHeight );
}

/******************************************************************************
 * Keyboard callback
 *
 * @param pKey ASCII character of the pressed key
 * @param pX Mouse location in window relative coordinates when the key was pressed
 * @param pY Mouse location in window relative coordinates when the key was pressed
 ******************************************************************************/
void keyboard( unsigned char pKey, int pX, int pY )
{
	// GigaVoxels
	gigaVoxelsPipelineWindow->onKeyboardFuncExecuted( pKey, pX, pY );
}

/******************************************************************************
 * Mouse callback
 *
 * @param pButton The button parameter is one of left, middle or right.
 * @param pState The state parameter indicates whether the callback was due to a release or press respectively.
 * @param pX Mouse location in window relative coordinates when the mouse button state changed
 * @param pY Mouse location in window relative coordinates when the mouse button state changed
 ******************************************************************************/
void mouse( int pButton, int pState, int pX, int pY )
{
	// GigaVoxels
	gigaVoxelsPipelineWindow->onMouseFuncExecuted( pButton, pState, pX, pY );
}

/******************************************************************************
 * Idle callback
 ******************************************************************************/
void idle( void )
{
	// Marks the current window as needing to be redisplayed. 
	glutPostRedisplay();

	//// GigaVoxels
	//gigaVoxelsPipelineWindow->onIdleFuncExecuted();
}

/******************************************************************************
 * Reshape callback
 *
 * @param pWidth The new window width in pixels
 * @param pHeight The new window height in pixels
 ******************************************************************************/
void initialize( void )
{
	// Setup cube vertex data
	vertices[ 0 ][ 0 ] = vertices[ 1 ][ 0 ] = vertices[ 2 ][ 0 ] = vertices[ 3 ][ 0 ] = -1;
	vertices[ 4 ][ 0 ] = vertices[ 5 ][ 0 ] = vertices[ 6 ][ 0 ] = vertices[ 7 ][ 0 ] = 1;
	vertices[ 0 ][ 1 ] = vertices[ 1 ][ 1 ] = vertices[ 4 ][ 1 ] = vertices[ 5 ][ 1 ] = -1;
	vertices[ 2 ][ 1 ] = vertices[ 3 ][ 1 ] = vertices[ 6 ][ 1 ] = vertices[ 7 ][ 1 ] = 1;
	vertices[ 0 ][ 2 ] = vertices[ 3 ][ 2 ] = vertices[ 4 ][ 2 ] = vertices[ 7 ][ 2 ] = 1;
	vertices[ 1 ][ 2 ] = vertices[ 2 ][ 2 ] = vertices[ 5 ][ 2 ] = vertices[ 6 ][ 2 ] = -1;

	// Enable a single OpenGL light
	glLightfv( GL_LIGHT0, GL_DIFFUSE, light_diffuse );
	glLightfv( GL_LIGHT0, GL_POSITION, light_position );
	glEnable( GL_LIGHT0 );
	glEnable( GL_LIGHTING );

	// Use depth buffering for hidden surface elimination
	glEnable( GL_DEPTH_TEST );

	// Setup the view of the cube
	glMatrixMode( GL_PROJECTION );
	gluPerspective( 40.0,	// field of view in degree
					1.0,	// aspect ratio
					1.0,	// Z near
					10.0 );	// Z far
	glMatrixMode( GL_MODELVIEW );
	gluLookAt( 0.0, 0.0, 5.0,  // eye is at (0,0,5)
			   0.0, 0.0, 0.0,  // center is at (0,0,0)
			   0.0, 1.0, 0.);  // up is in positive Y direction

	// Adjust cube position to be asthetic angle
	glTranslatef( 0.f, 0.f, -1.f );
	glRotatef( 60.f, 1.f, 0.f, 0.f );
	glRotatef( -20.f, 0.f, 0.f, 1.f );

	// GigaVoxels
	if ( gigaVoxelsPipelineWindow != NULL )
	{
		gigaVoxelsPipelineWindow->initialize();
	}
}

/******************************************************************************
 * Main entry program
 *
 * @param pArgc Number of arguments
 * @param pArgv List of arguments
 *
 * @return ...
 ******************************************************************************/
int	main( int pArgc, char** pArgv )
{
	// GLUT initialization
	glutInit( &pArgc, pArgv );
	glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA | GLUT_ALPHA | GLUT_DEPTH | GLUT_STENCIL );
    
	// GLUT window creation
	int x = 0;
	int y = 0;
	glutInitWindowPosition( x, y );
	int width = 512;
	int height = 512;
	glutInitWindowSize( width, height );
	glutCreateWindow( "GigaVoxels interoperability with GLUT window (red 3D lighted cube)" );

	// GigaVoxels pipeline's creation
	gigaVoxelsPipelineWindow = new PipelineWindow();

	// Initialization
	initialize();

	// TODO : change this
	// HACK : GigaVoxels need to initialized OPENGL frame buffers and CUDA resources
	//        that depends on the size of the window 
	gigaVoxelsPipelineWindow->onReshapeFuncExecuted( width, height );

	// GLUT callback registration
	glutDisplayFunc( display );
	glutReshapeFunc( reshape );
	glutKeyboardFunc( keyboard );
	glutIdleFunc( idle );

	// Main event loop
	glutMainLoop();

	// CUDA tip: clean up to ensure correct profiling
	cudaError_t error = cudaDeviceReset();

	return 0;	// ANSI C requires main to return int
}
