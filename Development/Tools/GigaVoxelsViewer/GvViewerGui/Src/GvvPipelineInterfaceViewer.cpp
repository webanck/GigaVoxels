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

#include "GvvPipelineInterfaceViewer.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// freeglut
#include <GL/freeglut.h>

#include "GvvPipelineInterface.h"
#include "GvvGLSceneInterface.h"

#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "GvvTransferFunctionEditor.h"
// - monitoring
#include "GvvPlotView.h"
#include "GvvCacheUsageView.h"
#include "GvvTimeBudgetMonitoringEditor.h"

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>

// System
#include <cassert>

// QGLViewer
#include <QGLViewer/manipulatedFrame.h>

// GigaSpace
#include <GsGraphics/GsGraphicsUtils.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerGui;

// GigaSpace
using namespace GsGraphics;

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
GvvPipelineInterfaceViewer::GvvPipelineInterfaceViewer( QWidget* parent, const QGLWidget* shareWidget, Qt::WindowFlags flags )
:	QGLViewer( parent, shareWidget, flags )
,	GvvPipelineManagerListener()
,	GvvGLSceneManagerListener()
,	mPipeline( NULL )
,	_scene( NULL )
{
	setBackgroundColor( Qt::green );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvPipelineInterfaceViewer::~GvvPipelineInterfaceViewer()
{
	//- test
	saveStateToFile();
}

/******************************************************************************
 * Set the pipeline.
 *
 * @param pPipeline The pipeline
 ******************************************************************************/
void GvvPipelineInterfaceViewer::setPipeline( GvViewerCore::GvvPipelineInterface* pPipeline )
{
	//assert( pPipeline != NULL );

	// Pipeline BEGIN
	//assert( mPipeline == NULL );
	mPipeline = pPipeline;
	// Pipeline END
}

/******************************************************************************
 * Get the pipeline.
 *
 * @return The pipeline
 ******************************************************************************/
const GvViewerCore::GvvPipelineInterface* GvvPipelineInterfaceViewer::getPipeline() const
{
	return mPipeline;
}

/******************************************************************************
 * Edit the pipeline.
 *
 * @return The pipeline
 ******************************************************************************/
GvViewerCore::GvvPipelineInterface* GvvPipelineInterfaceViewer::editPipeline()
{
	return mPipeline;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void GvvPipelineInterfaceViewer::init()
{
	if ( glewInit() != GLEW_OK )
	{
		exit( 1 );
	}

	// Pipeline BEGIN
	if ( mPipeline != NULL )
	{
	//	assert( mPipeline != NULL );
	//	mPipeline->init();

		//// TO DO
		//// - move this elsewhere
		//// - It is used to update Transfer function editor,
		//// by loading a default or associated transfer function
		//if ( mPipeline->hasTransferFunction() )
		//{
		//	GvvTransferFunctionEditor* editor = GvvApplication::get().getMainWindow()->getTransferFunctionEditor();
		//	if ( editor != NULL )
		//	{
		//		// Update referenced pipeline
		//		editor->setPipeline( mPipeline );

		//		// Load transfer function
		//		Qtfe* transferFunction = editor->getTransferFunction();
		//		if ( transferFunction != NULL )
		//		{
		//			if ( mPipeline->getTransferFunctionFilename() != NULL )
		//			{
		//				transferFunction->load( mPipeline->getTransferFunctionFilename() );
		//			}
		//		}
		//	}
		//}
	}
	//----------- A priori crash lorsque c'est appelé 2 fois ??------
	// Initialize CUDA with OpenGL Interoperability
	//cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );	// deprecated, use cudaSetDevice()
	//CUT_CHECK_ERROR( "cudaGLSetGLDevice" );
	//---------------------------------------------------------------
	// Pipeline END

	restoreStateFromFile();

	_manipulatedFrame = new qglviewer::ManipulatedFrame();
	_manipulatedFrame->setPosition( 1.0f, 1.0f, 1.0f );

	glEnable( GL_LIGHT1 );
	const GLfloat ambient[]  = {0.2f, 0.2f, 2.0f, 1.0f};
	const GLfloat diffuse[]  = {0.8f, 0.8f, 1.0f, 1.0f};
	const GLfloat specular[] = {0.0f, 0.0f, 1.0f, 1.0f};
	glLightfv( GL_LIGHT1, GL_AMBIENT,  ambient );
	glLightfv( GL_LIGHT1, GL_SPECULAR, specular );
	glLightfv( GL_LIGHT1, GL_DIFFUSE,  diffuse );
	glDisable( GL_LIGHT1 );
	glDisable( GL_LIGHTING );

	//** Setups connection
	QObject::connect( _manipulatedFrame, SIGNAL( manipulated() ), this, SLOT( onFrameManipulated() ) );

	// TO DO : add an accessor to modified that parameter
	setMouseTracking( true );
	//setMouseTracking( false );

	setAnimationPeriod( 0 );
	startAnimation();
}

/******************************************************************************
 * Main paint method, inherited from \c QGLWidget.
 *
 * Calls the following methods, in that order:
 * @param preDraw() (or preDrawStereo() if viewer displaysInStereo()) : places the camera in the world coordinate system.
 * @param draw() (or fastDraw() when the camera is manipulated) : main drawing method. Should be overloaded.
 * @param postDraw() : display of visual hints (world axis, FPS...)
 ******************************************************************************/
void GvvPipelineInterfaceViewer::paintGL()
{
	//if ( displaysInStereo() )
	//{
	//	for ( int view=1; view>=0; --view )
	//	{
	//		// Clears screen, set model view matrix with shifted matrix for ith buffer
	//		preDrawStereo( view );

	//		// Used defined method. Default is empty
	//		if ( camera()->frame()->isManipulated() )
	//		{
	//			fastDraw();
	//		}
	//		else
	//		{
	//			draw();
	//		}
	//		postDraw();
	//	}
	//}
	//else
	//{
		// Clears screen, set model view matrix...
		preDraw();
		
		// Used defined method. Default calls draw()
		/*if ( camera()->frame()->isManipulated() )
		{
			fastDraw();
		}
		else
		{*/
			draw();
		//}
		
		// Add visual hints: axis, camera, grid...
		//if ( _showFPS || _showGrid || _showAxis )
		if ( FPSIsDisplayed() || axisIsDrawn() || gridIsDrawn() )
		{
			postDraw();
		}
	//}

	Q_EMIT drawFinished( true );
}

/******************************************************************************
 * ...
 ******************************************************************************/
void GvvPipelineInterfaceViewer::draw()
{
	static unsigned int frame = 0;

	//--------------------------------------------
	//showEntireScene();
	//camera()->setZClippingCoefficient( 50.0f );
	//--------------------------------------------

	// Clear default frame buffer
	// glClearColor( 0.0f, 0.1f, 0.3f, 0.0f );					// already done by SampleViewer::setBackgroundColor()
	// glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );	// already done in QGLViewer::preDraw() method

	// Scene BEGIN
	if ( _scene != NULL )
	{
		_scene->draw();
	}
	// Scene END

	float lightPos[ 4 ] = { 1.0f, 1.0f, 1.0f, 1.0f };
	if ( mPipeline != NULL )
	{
		// TO DO : optimization
		//
		// Update internal GL matrices without call to GL context (glGetFloat())
		//camera()->getModelViewMatrix( _modelViewMatrix );	// extra copy inside...
		//camera()->getProjectionMatrix( _projectionMatrix );	// extra copy inside...
		//GsGraphicsUtils::copyGLMatrices( mPipeline->editModelViewMatrix(), _modelViewMatrix, mPipeline->editProjectionMatrix(), _projectionMatrix );

		if ( mPipeline->hasLight() )
		{
			_manipulatedFrame->getPosition( lightPos[ 0 ], lightPos[ 1 ], lightPos[ 2 ] );
	
			// Should not add a cudaMemcpy() call to update light position, if it is static
			//mPipeline->setLightPosition( lightPos[ 0 ], lightPos[ 1 ], lightPos[ 2 ] );
		}
	}

	// Pipeline BEGIN
	if ( mPipeline != NULL )
	{
		// Specify color to clear the color buffer
		const QColor& color = backgroundColor();
		mPipeline->setClearColor( color.red(), color.green(), color.blue(), color.alpha() );
		
		// Render the GigaVoxels scene
		mPipeline->draw();

		// Handle "time budget" monitoring if activated
		if ( mPipeline->hasTimeBudgetMonitoring() )
		{
			// Time budget monitoring view
			GvvTimeBudgetMonitoringEditor* timeBudgetMonitoringView = GvvApplication::get().getMainWindow()->getTimeBudgetMonitoringView();
			if ( timeBudgetMonitoringView != NULL )
			{
				const float frameDuration = mPipeline->getRendererElapsedTime();
				const float timeBudget = 1.f / static_cast< float >( mPipeline->getRenderingTimeBudget() ) * 1000.f; // (1/60Hz) x 1000 for time in milliseconds

				timeBudgetMonitoringView->onCurveChanged( frame, frameDuration );
			}
		}
		
		// Handle "data production" monitoring if activated
		if ( mPipeline->hasDataProductionMonitoring() )
		{
			// Cache plot view
			const unsigned int cacheNbNodeSubdivisionRequests = mPipeline->getCacheNbNodeSubdivisionRequests();
			const unsigned int cacheNbBrickLoadRequests = mPipeline->getCacheNbBrickLoadRequests();
			const unsigned int cacheNbUnusedNodes = mPipeline->getCacheNbUnusedNodes();
			const unsigned int cacheNbUnusedBricks = mPipeline->getCacheNbUnusedBricks();

			//	printf( "\nPRODUCER handled requests : [ node = %d ] [ brick = %d ]", cacheNbNodeSubdivisionRequests, cacheNbBrickLoadRequests );
			GvvPlotView* cachePlotView = GvvApplication::get().getMainWindow()->getCachePlotView();
			if ( cachePlotView != NULL )
			{
				cachePlotView->onCurveChanged( frame, cacheNbNodeSubdivisionRequests, cacheNbBrickLoadRequests, cacheNbUnusedNodes, cacheNbUnusedBricks );
			}
		}

		// Handle "cache" monitoring if activated
		if ( mPipeline->hasCacheMonitoring() )
		{
			// Cache usage view
			GvvCacheUsageView* cacheUsageView = GvvApplication::get().getMainWindow()->getCacheUsageView();
			if ( cacheUsageView != NULL )
			{
				cacheUsageView->update( mPipeline );
			}
		}
	}
	// Pipeline END

	// Draw light manipulator
	if ( mPipeline != NULL )
	{
		if ( mPipeline->hasLight() )
		{
			// Draw light manipulator
			glEnable( GL_LIGHT1 ); // must be enabled for drawLight()
			glLightfv( GL_LIGHT1, GL_POSITION, lightPos );
			glEnable( GL_DEPTH_TEST );
			if ( _manipulatedFrame->grabsMouse() )
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

	frame++;
}

/******************************************************************************
 * ...
 *
 * @param width ...
 * @param height ...
 ******************************************************************************/
void GvvPipelineInterfaceViewer::resizeGL( int width, int height )
{
	QGLViewer::resizeGL( width, height );

	// Pipeline BEGIN
	if ( mPipeline != NULL )
	{
		assert( mPipeline != NULL );
		mPipeline->resize( width, height );
	}
	// Pipeline END

	// Emit signal
	emit resized( width, height );
}

/******************************************************************************
 * ...
 *
 * @return
 ******************************************************************************/
QSize GvvPipelineInterfaceViewer::sizeHint() const
{
	return QSize( 512, 512 );
}

/******************************************************************************
 * ...
 *
 * @param e ...
 ******************************************************************************/
void GvvPipelineInterfaceViewer::keyPressEvent( QKeyEvent* e )
{
	switch ( e->key() )
	{
		case Qt::Key_Plus:
			// Pipeline BEGIN
			if ( mPipeline != NULL )
			{
				assert( mPipeline != NULL );
				mPipeline->incMaxVolTreeDepth();
			}
			// Pipeline END
			break;

		case Qt::Key_Minus:
			// Pipeline BEGIN
			if ( mPipeline != NULL )
			{
				assert( mPipeline != NULL );
				mPipeline->decMaxVolTreeDepth();
			}
			// Pipeline END
			break;

		case Qt::Key_C:
			// Pipeline BEGIN
			if ( mPipeline != NULL )
			{
				assert( mPipeline != NULL );
				mPipeline->clearCache();
			}
			// Pipeline END
			break;

		case Qt::Key_D:
			// Pipeline BEGIN
			if ( mPipeline != NULL )
			{
				assert( mPipeline != NULL );
				mPipeline->toggleDynamicUpdate();
			}
			// Pipeline END
			break;

		case Qt::Key_I:
			// Pipeline BEGIN
			if ( mPipeline != NULL )
			{
				assert( mPipeline != NULL );
				mPipeline->togglePerfmonDisplay( 1 );
			}
			// Pipeline END
			break;

		case Qt::Key_T:
			// Pipeline BEGIN
			if ( mPipeline != NULL )
			{
				assert( mPipeline != NULL );
				mPipeline->toggleDisplayOctree();
			}
			// Pipeline END
			break;

		case Qt::Key_U:
			// Pipeline BEGIN
			if ( mPipeline != NULL )
			{
				assert( mPipeline != NULL );
				mPipeline->togglePerfmonDisplay( 2 );
			}
			// Pipeline END
			break;

		default:
			QGLViewer::keyPressEvent( e );
			break;
	}
}

/******************************************************************************
 * Slot called when the transfer function of the associated editor has been modified
 ******************************************************************************/
void GvvPipelineInterfaceViewer::onTransferfunctionChanged()
{
	//assert( mPipeline != NULL );
	if ( mPipeline != NULL )
	{
		GvvTransferFunctionEditor* editor = GvvApplication::get().getMainWindow()->getTransferFunctionEditor();
		if ( editor != NULL )
		{
			Qtfe* transferFunction = editor->getTransferFunction();
			if ( transferFunction != NULL )
			{
				float* tab = new float[ 256 * 4 ];
				for ( int i = 0; i < 256 ; ++i )
				{
					float x = i / 256.0f;
					float alpha = transferFunction->evalf( 3, x );

					tab[ 4 * i + 0 ] = transferFunction->evalf( 0, x ) * alpha;
					tab[ 4 * i + 1 ] = transferFunction->evalf( 1, x ) * alpha;
					tab[ 4 * i + 2 ] = transferFunction->evalf( 2, x ) * alpha;
					tab[ 4 * i + 3 ] = alpha;
				}

				mPipeline->updateTransferFunction( tab, 256 );

				delete[] tab;
			}
		}
	}
}

/******************************************************************************
 * Add a pipeline.
 *
 * @param the pipeline to add
 ******************************************************************************/
void GvvPipelineInterfaceViewer::onPipelineAdded( GvvPipelineInterface* pPipeline )
{
	setPipeline( pPipeline );
}

/******************************************************************************
 * Add a pipeline.
 *
 * @param the pipeline to add
 ******************************************************************************/
void GvvPipelineInterfaceViewer::onPipelineRemoved( GvvPipelineInterface* pPipeline )
{
	setPipeline( NULL );
}

/******************************************************************************
 * Slot called when the ManipulatedFrame is manipulated
 ******************************************************************************/
void GvvPipelineInterfaceViewer::onFrameManipulated()
{
	//assert( mPipeline != NULL );
	if ( mPipeline != NULL )
	{
		float pos[ 4 ] = { 1.f, 1.f, 1.f, 1.f };
		_manipulatedFrame->getPosition( pos[ 0 ], pos[ 1 ], pos[ 2 ] );

		mPipeline->setLightPosition( pos[ 0 ], pos[ 1 ], pos[ 2 ] );
	}
}

/******************************************************************************
 * Add a scene.
 *
 * @param pScene the scene to add
 ******************************************************************************/
void GvvPipelineInterfaceViewer::onGLSceneAdded( GvvGLSceneInterface* pScene )
{
	_scene = pScene;
}

/******************************************************************************
 * Remove a scene.
 *
 * @param pScene the scene to remove
 ******************************************************************************/
void GvvPipelineInterfaceViewer::onGLSceneRemoved( GvvGLSceneInterface* pScene )
{
	_scene = NULL;
}

/******************************************************************************
 * Capture video by producing list of images
 *
 * @param flag to start and stop video generation
 *
 * @return flag to tell wheter or not it succeeds
 ******************************************************************************/
bool GvvPipelineInterfaceViewer::captureVideo( bool pFlag )
{
	QString dataRepository = QCoreApplication::applicationDirPath();
	dataRepository += QDir::separator();
	dataRepository += QString( "Data" );
	dataRepository += QDir::separator();
	dataRepository += QString( "Videos" );
	const QString filename = dataRepository + QDir::separator() + QString( "image" );
	setSnapshotFileName( filename );

	const QString format = "";
	//setSnapshotFormat( format );

	int counter = 0;
	//setSnapshotCounter( counter );

	int quality = 0;
	//setSnapshotQuality( quality );

	resize( 720, 576 );	// PAL DV format (use 720x480 for NTSC DV)

	if ( pFlag )
	{
		connect( this, SIGNAL( drawFinished( bool ) ), SLOT( saveSnapshot( bool ) ) );
	}
	else
	{
		disconnect( this, SIGNAL( drawFinished( bool ) ), this, SLOT( saveSnapshot( bool ) ) );
	}
	
	//ffmpeg.exe -i image-%04d.jpg -r 60 File-Out.mp4

	return true;
}
