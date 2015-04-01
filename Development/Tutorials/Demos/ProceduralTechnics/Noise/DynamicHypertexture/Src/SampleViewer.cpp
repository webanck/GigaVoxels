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

// GL
#include <GL/freeglut.h>

// Qtfe
#include "Qtfe.h"

// STL
#include <iostream>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>

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
:	_sampleCore( NULL )
,	_controlLight( false )
,	_moveLight( false )
,	_transferFunctionEditor( NULL )
{
	// Light parameters initialization
	for ( int i = 0; i < 7; i++ )
	{
		_light[ i ] = 0.f;
	}
	
	// Window title
	setWindowTitle( tr( "Noise on device : Amplified Surface example" ) );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleViewer::~SampleViewer()
{
	// Destroy GigaVoxels pipeline objects
	delete _sampleCore;

	// Destroy Qtfe editor because is has no parent widget
	delete _transferFunctionEditor;
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

	// Noise parameters
	_noiseParameters[0] = NOISE_FIRST_FREQUENCY_LOWER_BOUND;  // The first frequency of the noise
	_noiseParameters[1] = NOISE_SHELL_WIDTH_UPPER_BOUND;   // Unused for now

	// Initialize the noise parameters
	_sampleCore->setNoiseParameters( _noiseParameters[0], _noiseParameters[1] );

	// Initialize the transfer function editor
	_transferFunctionEditor = new Qtfe( NULL );

	// Modify transfer function window flags to always stay on top
	Qt::WindowFlags windowFlags = _transferFunctionEditor->windowFlags();
	windowFlags |= Qt::WindowStaysOnTopHint;
#ifndef WIN32
	windowFlags |= Qt::X11BypassWindowManagerHint;
#endif
	_transferFunctionEditor->setWindowFlags( windowFlags );

	// Do connection(s)
	connect( _transferFunctionEditor, SIGNAL( functionChanged() ), SLOT( onFunctionChanged() ) );
	
	// Try to load a transfer function from file
	QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	QString filename = dataRepository + QDir::separator() + QString( "TransferFunctions" ) + QDir::separator() + QString( "TransferFunction_Qtfe_01.xml" );
	QFileInfo fileInfo( filename );
	if ( ( ! fileInfo.isFile() ) || ( ! fileInfo.isReadable() ) )
	{
		// Idea
		// Maybe use Qt function : bool QFileInfo::permission ( QFile::Permissions permissions ) const

		// TO DO
		// Handle error : free memory and exit
		// ...
		std::cout << "ERROR. Check filename : " << filename.toLatin1().constData() << std::endl;
	}
	bool hasTransferFunctionBeenLoaded = _transferFunctionEditor->load( filename );
	if ( ! hasTransferFunctionBeenLoaded )
	{
		// LOG
		QString logMessage = tr( "Transfer function has not been loaded..." );
		std::cout << logMessage.toLatin1().constData() << std::endl;
		
		// Initialize a default transfer function
		// 4 channels [R,G,B,A] bound to 1 output
		_transferFunctionEditor->addChannels( 4 );
		_transferFunctionEditor->addOutputs( 1 );
		_transferFunctionEditor->bindChannelToOutputR( 0, 0 );
		_transferFunctionEditor->bindChannelToOutputG( 1, 0 );
		_transferFunctionEditor->bindChannelToOutputB( 2, 0 );
		_transferFunctionEditor->bindChannelToOutputA( 3, 0 );

		// Tell GigaVoxels that transfer function has been modified
		onFunctionChanged();

		// LOG
		logMessage = tr( "A default one has been created." );
		std::cout << logMessage.toLatin1().constData() << std::endl;
	}

	// Show the transfer function editor
	_transferFunctionEditor->resize( 367, 546 );
	_transferFunctionEditor->show();

	// QGLViewer restoration mechanism
	restoreStateFromFile();

	// Initialize light
	setLight( 1.08f, 1.99f );

	setMouseTracking( true );
	setAnimationPeriod( 0 );
	startAnimation();

	// Viewer settings :
	// - sets the backgroundColor() of the viewer and calls qglClearColor()
	setBackgroundColor( QColor( 150, 150, 150 ) );
	// Update GigaVoxels clear color
	_sampleCore->setClearColor( 150, 150, 150, 255 );
}

/******************************************************************************
 * Draw function called each frame
 ******************************************************************************/
void SampleViewer::draw()
{
	// Clear default frame buffer
	// glClearColor( 0.0f, 0.1f, 0.3f, 0.0f );					=> already done by setBackgroundColor()
	// glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );	=> already done in QGLViewr::preDraw() method

	glEnable( GL_DEPTH_TEST );
	glDisable( GL_LIGHTING );

	// Set light parameter
	setLight( _light[ 3 ], _light[ 4 ] );

	// Handle GigaVoxels draw
	_sampleCore->draw();

	// Draw the light
	if ( _controlLight )
	{
		drawLight();
	}
}

/******************************************************************************
 * Resize GL event handler
 *
 * @param width the new width
 * @param height the new height
 ******************************************************************************/
void SampleViewer::resizeGL( int width, int height )
{
	// Handle QGLViewer resize
	QGLViewer::resizeGL( width, height );

	// Handle GigaVoxels resize
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

	case Qt::Key_L:
		// Used to control light display
		_controlLight = !_controlLight;
		_moveLight = false;
		break;

	case Qt::Key_T:
		_sampleCore->toggleDisplayOctree();
		break;

	case Qt::Key_U:
		_sampleCore->togglePerfmonDisplay( 2 );
		break;

	case Qt::Key_E:
		// Used to display the Transfer Function Editor if previously closed
		if ( _transferFunctionEditor != NULL )
		{
			_transferFunctionEditor->show();
		}
		break;
	case Qt::Key_7:
			// Used to increase the first noise frequency
		_noiseParameters[0]++;
		_sampleCore->setNoiseParameters(_noiseParameters[0],_noiseParameters[1]);
		break;
	case Qt::Key_4:
			// Used to decrease the first noise frequency
		_noiseParameters[0]--;
		if (_noiseParameters[0]<NOISE_FIRST_FREQUENCY_LOWER_BOUND)
			_noiseParameters[0]=NOISE_FIRST_FREQUENCY_LOWER_BOUND;

		_sampleCore->setNoiseParameters(_noiseParameters[0],_noiseParameters[1]);

		break;
	case Qt::Key_8:
				// Used to increase the first noise amplitude
			_noiseParameters[1]+=0.001;
			if (_noiseParameters[1]>NOISE_SHELL_WIDTH_UPPER_BOUND)
				_noiseParameters[1]=NOISE_SHELL_WIDTH_UPPER_BOUND;

			_sampleCore->setNoiseParameters(_noiseParameters[0],_noiseParameters[1]);
			break;
	case Qt::Key_5:
				// Used to increase the first noise amplitude
			_noiseParameters[1]-=0.001;
			if (_noiseParameters[1]<0)
				_noiseParameters[1]=0;

			_sampleCore->setNoiseParameters(_noiseParameters[0],_noiseParameters[1]);

			break;


	default:
		QGLViewer::keyPressEvent( e );
		break;
	}
}

/******************************************************************************
 * Mouse press event handler
 *
 * @param e the event
 ******************************************************************************/
void SampleViewer::mousePressEvent( QMouseEvent* e )
{
	//if (_light1->grabsMouse())
	if ( _controlLight )
	{
		// Store delta x and y mouse movement
		_light[ 5 ] = e->x();
		_light[ 6 ] = e->y();

		// Update flag
		_moveLight = true;
	}
	else
	{
		QGLViewer::mousePressEvent( e );
	}
}

/******************************************************************************
 * Mouse move event handler
 *
 * @param e the event
 ******************************************************************************/
void SampleViewer::mouseMoveEvent( QMouseEvent* e )
{
	//if (_light1->grabsMouse())
	if ( _moveLight )
	{
		int mx = e->x();
		int my = e->y();

		_light[ 4 ] += ( _light[ 5 ] - e->x() ) / 100.0f;
		_light[ 5 ] = e->x();
		_light[ 3 ] += -( _light[ 6 ] - e->y() ) / 100.0f;
		_light[ 6 ] = e->y();

		if ( _light[ 3 ] < 0.0f )
		{
			_light[ 3 ] = 0.0f;
		}

		if ( _light[ 3 ] > (float)M_PI )
		{
			_light[3] = (float)M_PI;
		}

		setLight( _light[ 3 ], _light[ 4 ] );
	}
	else
	{
		QGLViewer::mouseMoveEvent( e );
	}
}

/******************************************************************************
 * Mouse release event handler
 *
 * @param e the event
 ******************************************************************************/
void SampleViewer::mouseReleaseEvent( QMouseEvent* e )
{
	QGLViewer::mouseReleaseEvent( e );
}

/******************************************************************************
 * Draw light
 ******************************************************************************/
void SampleViewer::drawLight() const
{
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();
	gluPerspective( 90.0f, 1.0f, 0.1f, 10.0f );

	glMatrixMode( GL_MODELVIEW );   // changes the current matrix to the modelview matrix
	glPushMatrix();
	glLoadIdentity();
	glColor3f( 1.0f, 1.0f, 0.5f );
	glTranslatef( 0.f, 0.f, -1.0f );
	glutSolidSphere( 0.05f, 20.f, 20.f );
	glTranslatef( 0.5f * _light[ 0 ], 0.5f * _light[ 1 ], 0.5f * _light[ 2 ] );
	glutSolidSphere( 0.05f, 20.f,20.f );
	glPopMatrix();

	glMatrixMode( GL_PROJECTION );
	glPopMatrix();

	glMatrixMode( GL_MODELVIEW );
}

/******************************************************************************
 * Set light
 *
 * @param theta ...
 * @param phi ...
 ******************************************************************************/
void SampleViewer::setLight( float theta, float phi )
{
	// Retrieve cartesian coordinates from spheric ones
	_light[ 0 ] = sinf( theta ) * cosf( phi );
	_light[ 1 ] = cosf( theta );
	_light[ 2 ] = sinf( theta ) * sinf( phi );

	// Store theta and phi parameters
	_light[ 3 ] = theta;
	_light[ 4 ] = phi;

	// Express the light direction in the view coordinate system
	float3 lightDirInView = make_float3( _light[ 0 ], _light[ 1 ], _light[ 2 ] );

	// Retrieve the model view matrix
	float4x4 modelViewMatrix;
	glGetFloatv( GL_MODELVIEW_MATRIX, modelViewMatrix._array );

	// Reset translation part
	modelViewMatrix._array[ 12 ] = 0.f;
	modelViewMatrix._array[ 13 ] = 0.f;
	modelViewMatrix._array[ 14 ] = 0.f;
	modelViewMatrix._array[ 15 ] = 1.f;

	// Compute inverse matrix
	float4x4 invModelViewMatrix;
	invModelViewMatrix = transpose( modelViewMatrix );

	// Retrieve light direction in world coordinate system then normalize
	float3 lightDirInWorld = mulRot( invModelViewMatrix, lightDirInView );
	float3 lightDir = normalize( lightDirInWorld );

	// Update the GigaVoxels pipeline
	_sampleCore->setLightPosition( lightDir.x, lightDir.y, lightDir.z );
}

/******************************************************************************
 * Slot called when at least one canal changed
 ******************************************************************************/
void SampleViewer::onFunctionChanged()
{
	if ( _transferFunctionEditor != NULL )
	{
		float* tab = new float[ 256 * 4 ];
		for (int i= 0; i < 256 ; ++i )
		{
			float x = i / 256.0f;
			float alpha = _transferFunctionEditor->evalf( 3, x );

			tab[ 4 * i + 0 ] = _transferFunctionEditor->evalf( 0, x ) * alpha;
			tab[ 4 * i + 1 ] = _transferFunctionEditor->evalf( 1, x ) * alpha;
			tab[ 4 * i + 2 ] = _transferFunctionEditor->evalf( 2, x ) * alpha;
			tab[ 4 * i + 3 ] = alpha;
		}

		_sampleCore->updateTransferFunction( tab, 256 );

		delete[] tab;
	}
}
