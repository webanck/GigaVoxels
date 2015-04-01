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

// System
#include <cstdio>
#include <cstdlib>

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
,	_sampleCore( NULL )
,	_light1( NULL )
,	_lightManipulation( false )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleViewer::~SampleViewer()
{
	delete _sampleCore;
	delete _light1;
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

	// Initialize the GigaVoxels pipeline
	_sampleCore = new SampleCore();
	_sampleCore->init();

	// Read QGLViewer XML settings file if any
	restoreStateFromFile();

	// Viewer settings :
	// - sets the backgroundColor() of the viewer and calls qglClearColor()
	setBackgroundColor( QColor( 51, 51, 51 ) );
	// Update GigaVoxels clear color
	_sampleCore->setClearColor( 51, 51, 51, 255 );

	// Light initialization
	_light1 = new qglviewer::ManipulatedFrame();
	_light1->setPosition( 0.75f, 0.75f, 0.75f );
	glEnable( GL_LIGHT1 );
	const GLfloat ambient[]  = { 0.2f, 0.2f, 2.0f, 1.0f };
	const GLfloat diffuse[]  = { 0.8f, 0.8f, 1.0f, 1.0f };
	const GLfloat specular[] = { 0.0f, 0.0f, 1.0f, 1.0f };
	glLightfv( GL_LIGHT1, GL_AMBIENT,  ambient );
	glLightfv( GL_LIGHT1, GL_SPECULAR, specular );
	glLightfv( GL_LIGHT1, GL_DIFFUSE,  diffuse );
	glDisable( GL_LIGHT1 );
	glDisable( GL_LIGHTING );
	// Update GigaVoxels light position
	_sampleCore->setLightPosition( 0.75f, 0.75f, 0.75f );

	// Viewer initialization
	setMouseTracking( false );
	// QGLViewer uses a timer to redarw scene, this enables the maximum refreshing rate.
	setAnimationPeriod( 0 );
	startAnimation();

	//** Setups connection
	QObject::connect( _light1, SIGNAL( manipulated() ), this, SLOT( onLightFrameManipulated() ) );

	// Opens help window
	help();
}

/******************************************************************************
 * Draw function called each frame
 ******************************************************************************/
void SampleViewer::draw()
{
	// Clear default frame buffer
	// glClearColor( 0.0f, 0.1f, 0.3f, 0.0f );					// already done by SampleViewr::setBackgroundColor()
	// glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );	// already done in QGLViewr::preDraw() method

	// Render the GigaVoxels scene
	_sampleCore->draw();

	// Draw light if its manipulation is activated
	if ( _lightManipulation )
	{
		float pos[ 4 ] = { 1.0f, 1.0f, 1.0f, 1.0f };
		_light1->getPosition( pos[ 0 ], pos[ 1 ], pos[ 2 ] );

		glEnable( GL_LIGHT1 ); // must be enabled for drawLight()
		glLightfv( GL_LIGHT1, GL_POSITION, pos );
		glEnable( GL_DEPTH_TEST );
		if ( _light1->grabsMouse() )
		{
			drawLight( GL_LIGHT1, 1.2f );
		}
		else
		{
			drawLight( GL_LIGHT1 );
		}
		glDisable( GL_DEPTH_TEST );
		glDisable( GL_LIGHT1 );
	}
}

/******************************************************************************
 * Returns the QString displayed in the help() window main tab
 *
 * @return the help text
 ******************************************************************************/
QString SampleViewer::helpString() const
{
	QString text( "<h2>Simple Sphere Demo</h2><br>" );

	text += "<h3>Description</h3>";
	text += "This example illustrates the procedural geometry generation of a sphere on device (GPU).<br>";

	text += "<h3>Custom Keyboard Interaction</h3>";
	text += "Light.<br>";

	text += "<h3>Help</h3>";
	text += "Use the mouse to move the camera around the object. ";
	text += "You can respectively revolve around, zoom and translate with the three mouse buttons. ";
	text += "Left and middle buttons pressed together rotate around the camera view direction axis<br><br>";
	text += "Pressing <b>Alt</b> and one of the function keys (<b>F1</b>..<b>F12</b>) defines a camera keyFrame. ";
	text += "Simply press the function key again to restore it. Several keyFrames define a ";
	text += "camera path. Paths are saved when you quit the application and restored at next start.<br><br>";
	text += "Press <b>F</b> to display the frame rate, <b>A</b> for the world axis, ";
	text += "<b>Alt+Return</b> for full screen mode and <b>Control+S</b> to save a snapshot. ";
	text += "See the <b>Keyboard</b> tab in this window for a complete shortcut list.<br><br>";
	text += "Double clicks automates single click actions: A left button double click aligns the closer axis with the camera (if close enough). ";
	text += "A middle button double click fits the zoom of the camera and the right button re-centers the scene.<br><br>";
	text += "A left button double click while holding right button pressed defines the camera <i>Revolve Around Point</i>. ";
	text += "See the <b>Mouse</b> tab and the documentation web pages for details.<br><br>";

	text += "Press <b>Escape</b> to exit the viewer.";

	return text;
}

/******************************************************************************
 * Resize GL event handler
 *
 * @param width the new width
 * @param height the new height
 ******************************************************************************/
void SampleViewer::resizeGL( int pWidth, int pHeight )
{
	// Handle QGLViewer resize
	QGLViewer::resizeGL( pWidth, pHeight );

	// Handle GigaVoxels resize
	_sampleCore->resize( pWidth, pHeight );
}

/******************************************************************************
 * Get the viewer size hint
 *
 * @return the viewer size hint
 ******************************************************************************/
QSize SampleViewer::sizeHint() const
{
	// Default size
	return QSize( 512, 512 );
}

/******************************************************************************
 * Key press event handler
 *
 * @param pEvent the event
 ******************************************************************************/
void SampleViewer::keyPressEvent( QKeyEvent* pEvent )
{
	switch ( pEvent->key() )
	{
		case Qt::Key_Plus:
			_sampleCore->incMaxVolTreeDepth();
			break;

		case Qt::Key_Minus:
			_sampleCore->decMaxVolTreeDepth();
			break;

		case Qt::Key_C:
			// Tell GigaVoxels to clear its cache
			_sampleCore->clearCache();
			break;

		case Qt::Key_D:
			_sampleCore->toggleDynamicUpdate();
			break;

		case Qt::Key_I:
			// Toggle GigaVoxels performance monitor mechanism (if it has been activated during GigaVoxels compilation)
			_sampleCore->togglePerfmonDisplay( 1 );
			break;

		case Qt::Key_T:
			// Toggle the display of the GigaVoxels space partitioning structure
			_sampleCore->toggleDisplayOctree();
			break;

		case Qt::Key_U:
			// Toggle GigaVoxels performance monitor mechanism (if it has been activated during GigaVoxels compilation)
			_sampleCore->togglePerfmonDisplay( 2 );
			break;

		case Qt::Key_L:
			// Toggle light manipulation mechanism
			setLightManipulation( ! getLightManipulation() );
			break;

		case Qt::Key_R:
			// Reset light position
			_light1->setPosition( 0.75f, 0.75f, 0.75f );
			// Update GigaVoxels light position
			_sampleCore->setLightPosition( 0.75f, 0.75f, 0.75f );
			break;

		default:
			QGLViewer::keyPressEvent( pEvent );
			break;
	}
}

/******************************************************************************
 * Get the flag to tell wheter or not light manipulation is activated
 *
 * @return the light manipulation flag
 ******************************************************************************/
bool SampleViewer::getLightManipulation() const
{
	return _lightManipulation;
}

/******************************************************************************
 * Set the flag to tell wheter or not light manipulation is activated
 *
 * @param pFlag the light manipulation flag
 ******************************************************************************/
void SampleViewer::setLightManipulation( bool pFlag )
{
	_lightManipulation = pFlag;

	// Modify mouse tracking state to enable real-time light manipulation
	setMouseTracking( pFlag );
}

/******************************************************************************
 * Slot called when the light ManipulatedFrame has been modified
 ******************************************************************************/
void SampleViewer::onLightFrameManipulated()
{
	if ( _sampleCore != NULL )
	{
		float pos[ 4 ] = { 1.f, 1.f, 1.f, 1.f };
		_light1->getPosition( pos[ 0 ], pos[ 1 ], pos[ 2 ] );

		// Update GigaVoxels light position
		_sampleCore->setLightPosition( pos[ 0 ], pos[ 1 ], pos[ 2 ] );
	}
}
