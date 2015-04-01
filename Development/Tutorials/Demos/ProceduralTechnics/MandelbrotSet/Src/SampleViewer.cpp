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
#include <GsGraphics/GsGraphicsCore.h>

// System
#include <cstdio>
#include <cstdlib>

// Qtfe
#include "Qtfe.h"

// STL
#include <iostream>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>

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
:	_sampleCore( NULL )
,	_light1( NULL )
,	_lightManipulation( false )
{
	// Window title
	setWindowTitle( tr( "Mandelbrot Set example" ) );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleViewer::~SampleViewer()
{
	//** Setups connection
	QObject::disconnect( _light1, SIGNAL( manipulated() ), this, SLOT( onLightFrameManipulated() ) );

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

	// LOG associated Graphics Core library properties/capabilities (i.e. OpenGL)
	GsGraphics::GsGraphicsCore::printInfo();

	// GigaVoxels pipeline initialization
	_sampleCore = new SampleCore();
	_sampleCore->init();

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

	// Read QGLViewer XML settings file if any
	restoreStateFromFile();

	// Viewer settings :
	// - sets the backgroundColor() of the viewer and calls qglClearColor()
	setBackgroundColor( QColor( 153, 153, 153 ) );
	// Update GigaVoxels clear color
	_sampleCore->setClearColor( 153, 153, 153, 255 );

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
	glDisable( GL_LIGHTING );
	// Update GigaVoxels light position
	_sampleCore->setLightPosition( 0.75f, 0.75f, 0.75f );

	// Viewer initialization
	setMouseTracking( false );
	setAnimationPeriod( 0 );
	startAnimation();

	//** Setups connection
	QObject::connect( _light1, SIGNAL( manipulated() ), this, SLOT( onLightFrameManipulated() ) );
}

/******************************************************************************
 * Draw function called each frame
 ******************************************************************************/
void SampleViewer::draw()
{
	// Clear default frame buffer
	// glClearColor( 0.0f, 0.1f, 0.3f, 0.0f );					// already done by SampleViewr::setBackgroundColor()
	// glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );	// already done in QGLViewr::preDraw() method

	// Ask rendition of the GigaVoxels pipeline
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
 * Resize GL event handler
 *
 * @param pWidth the new width
 * @param pHeight the new height
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

		case Qt::Key_E:
			// Used to display the Transfer Function Editor if previously closed
			if ( _transferFunctionEditor != NULL )
			{
				_transferFunctionEditor->show();
			}
			break;

		case Qt::Key_PageUp:
			camera()->setFlySpeed( camera()->flySpeed() * 1.5f );
			break;

		case Qt::Key_PageDown:
			camera()->setFlySpeed( camera()->flySpeed() / 1.5f );
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
			QGLViewer::keyPressEvent( e );
			break;
	}
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
