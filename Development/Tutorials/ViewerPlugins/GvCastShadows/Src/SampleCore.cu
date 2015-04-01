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

#include "SampleCore.h"
 
/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/StaticRes3D.h>
#include <GvStructure/GvVolumeTree.h>
#include <GvStructure/GvDataProductionManager.h>
#include <GvUtils/GvSimplePipeline.h>
#include <GvUtils/GvSimpleHostProducer.h>
#include <GvUtils/GvDataLoader.h>
#include <GvUtils/GvSimpleHostShader.h>
#include <GvUtils/GvCommonGraphicsPass.h>
#include <GvCore/GvError.h>
#include <GvPerfMon/GvPerformanceMonitor.h>
#include <GvStructure/GvNode.h>

// Project
#include "Producer.h"
//#include "VolumeProducerBricks.h"
#include "VolumeTreeRendererGLSL.h"
//#include "ProducerTorusKernel.h"

// GvViewer
#include <GvvApplication.h>
#include <GvvApplication.h>
#include <GvvMainWindow.h>
#include <Gvv3DWindow.h>
#include <GvvPipelineInterfaceViewer.h>

// Cuda SDK
#include <helper_math.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QPoint>

// QGLViewer
#include <QGLViewer/qglviewer.h>
#include <QGLViewer/camera.h>

// STL
#include <string>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvRendering;

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerGui;


// STL
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * ...
 */
ObjectReceivingShadow* shadowReceiver = NULL;

/**
 * ...
 */
Mesh* shadowCaster = NULL;

/**
 * Defines the size allowed for each type of pool
 */
#define NODEPOOL_MEMSIZE	( 8U * 1024U * 1024U )		// 8 Mo
#define BRICKPOOL_MEMSIZE	( 128U * 1024U * 1024U )	// 128 Mo

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
SampleCore::SampleCore()
//:	GvvPipelineInterface()
:	GvViewerScene::GvvPipeline()
,	_pipeline( NULL )
,	_renderer( NULL )
,	mDisplayOctree( false )
,	mDisplayPerfmon( 0 )
,	mMaxVolTreeDepth( 16 )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
	delete _pipeline;
}

/******************************************************************************
 * Gets the name of this browsable
 *
 * @return the name of this browsable
 ******************************************************************************/
const char* SampleCore::getName() const
{
	return "ShadowCasting";
}

/******************************************************************************
 * Initialize the GigaVoxels pipeline
 ******************************************************************************/
void SampleCore::init()
{
	CUDAPM_INIT();

	// Initialize CUDA with OpenGL Interoperability
	if ( ! GvViewerGui::GvvApplication::get().isGPUComputingInitialized() )
	{
		//cudaGLSetGLDevice( gpuGetMaxGflopsDeviceId() );	// to do : deprecated, use cudaSetDevice()
		//GV_CHECK_CUDA_ERROR( "cudaGLSetGLDevice" );
		cudaSetDevice( gpuGetMaxGflopsDeviceId() );
		GV_CHECK_CUDA_ERROR( "cudaSetDevice" );
		
		GvViewerGui::GvvApplication::get().setGPUComputingInitialized( true );
	}

	// Compute the size of one element in the cache for nodes and bricks
	size_t nodeElemSize = NodeRes::numElements * sizeof( GvStructure::GvNode );
	//size_t brickElemSize = RealBrickRes::numElements * GvCore::DataTotalChannelSize< DataType >::value;

	// Compute how many we can fit into the given memory size
	size_t nodePoolNumElems = NODEPOOL_MEMSIZE / nodeElemSize;
	//size_t brickPoolNumElems = BRICKPOOL_MEMSIZE / brickElemSize;

	// Compute the resolution of the pools
	uint3 nodePoolRes = make_uint3( (uint)floorf( powf( (float)nodePoolNumElems, 1.0f / 3.0f ) ) ) * NodeRes::get();

	// Pipeline creation
	_pipeline = new PipelineType();

	// Producer creation
	ProducerType* producer = new ProducerType( 64 * 1024 * 1024, nodePoolRes.x * nodePoolRes.y * nodePoolRes.z );
	
	// Shader creation
	ShaderType* shader = new ShaderType();

	// Pipeline initialization
	const bool useGraphicsLibraryInteroperability = true;
	_pipeline->initialize( NODEPOOL_MEMSIZE, BRICKPOOL_MEMSIZE, producer, shader, useGraphicsLibraryInteroperability );

	// Renderer initialization
	_renderer = new RendererType( _pipeline->editDataStructure(), _pipeline->editCache() );
	assert( _renderer != NULL );
	_pipeline->addRenderer( _renderer );

	// Pipeline configuration
	_pipeline->editDataStructure()->setMaxDepth( mMaxVolTreeDepth );

	// Getting the sample viewer
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	sviewer = pipelineViewer;

	QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	
	QString shaderRepository = dataRepository + QDir::separator() + QString( "Shaders" );
	QString vertexShaderFilename = shaderRepository + QDir::separator() + QString( "GvCastShadows" ) + QDir::separator() + QString( "vert.glsl" );
	QString fragmentShaderFilename = shaderRepository + QDir::separator() + QString( "GvCastShadows" ) + QDir::separator() + QString( "frag.glsl" );
	
	QString meshRepository = dataRepository + QDir::separator() + QString( "3DModels" );
	QString meshFilename = meshRepository + QDir::separator() + QString( "dino.3ds" );
	
	// CREATING SHADOW RECEIVER
	shadowReceiver = new ObjectReceivingShadow();
	/*_pipeline->editRenderer()*/_renderer->getVolTreeChildArray()->unmapResource();
	/*_pipeline->editRenderer()*/_renderer->getVolTreeDataArray()->unmapResource();
	shadowReceiver->setVolTreeChildArray(/*_pipeline->editRenderer()*/_renderer->getVolTreeChildArray(), /*_pipeline->editRenderer()*/_renderer->getChildBufferName());
	shadowReceiver->setVolTreeDataArray(/*_pipeline->editRenderer()*/_renderer->getVolTreeDataArray(), /*_pipeline->editRenderer()*/_renderer->getDataBufferName());	
	shadowReceiver->init();
	/*_pipeline->editRenderer()*/_renderer->getVolTreeChildArray()->mapResource();
	/*_pipeline->editRenderer()*/_renderer->getVolTreeDataArray()->mapResource();

	// CREATING SHADOW CASTER
	GLuint vshader = useShader( GL_VERTEX_SHADER, vertexShaderFilename.toLatin1().constData() );
	GLuint fshader = useShader( GL_FRAGMENT_SHADER, fragmentShaderFilename.toLatin1().constData() );
	GLuint program = glCreateProgram();
	glAttachShader( program, vshader );
	glAttachShader( program, fshader );
	glLinkProgram( program );
	linkStatus( program );
	shadowCaster = new Mesh( program );
	shadowCasterFile = meshFilename.toStdString();
	shadowCaster->chargerMesh( shadowCasterFile );
	shadowCaster->creerVBO();
	scale = shadowCaster->getScaleFactor();
	shadowCaster->getTranslationFactors( translation );

	// VOXELIZATION
	//voxelization code on file shadowCasterFile
	//...
	//...
	/*************************************************************/
	// Retrieving the xml file 
	QString filename = dataRepository + QDir::separator() + QString( "Voxels" ) + QDir::separator() + QString( "Dino" ) + QDir::separator() + QString( "dino.xml" );
	GvUtils::GvDataLoader< DataType >* dataLoader = new GvUtils::GvDataLoader< DataType >( filename.toStdString(), BrickRes::get(), BrickBorderSize, true );
	producer->attachProducer( dataLoader );
}

/******************************************************************************
 * Draw function called of frame
 ******************************************************************************/
void SampleCore::draw()
{
	// Update light position
	// TO DO
	// - do it only if light has moved
	qglviewer::Vec light = qglviewer::Vec( _lightPosition.x, _lightPosition.y, _lightPosition.z );
	qglviewer::Vec lightV = sviewer->camera()->cameraCoordinatesOf( light );
	qglviewer::Vec wlight = sviewer->camera()->worldCoordinatesOf( lightV );
	setCameraLight( lightV.x, lightV.y, lightV.z );
	setWorldLight( wlight.x, wlight.y, wlight.z );

	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT( frame );
	CUDAPM_START_EVENT( app_init_frame );

	// Warning : already done
	//glClearColor( 0.0f, 0.1f, 0.3f, 0.0f );
	//glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );
	glClear( GL_STENCIL_BUFFER_BIT );

	glEnable( GL_DEPTH_TEST );

	glMatrixMode( GL_MODELVIEW );

	// Display the data structure (space partitioning)
	if ( mDisplayOctree )
	{
		glPushMatrix();
		glTranslatef(translation[0], translation[1], translation[2]);
		glScalef(scale, scale, scale);
		glTranslatef( -0.5f, -0.5f, -0.5f );//pour centrer la boite GV
		_pipeline->editDataStructure()->render();
		glPopMatrix();
	}

	// extract view transformations
	float4x4 viewMatrix;
	float4x4 projectionMatrix;
	// FIXME
	glPushMatrix();
	glTranslatef(translation[0], translation[1], translation[2]);
	glScalef(scale, scale, scale);
	glTranslatef( -0.5f, -0.5f, -0.5f );//pour centrer la boite GV
	// FIXME
	glGetFloatv( GL_MODELVIEW_MATRIX, viewMatrix._array );
	glGetFloatv( GL_PROJECTION_MATRIX, projectionMatrix._array );
	// FIXME
	glPopMatrix();

	// Build and extract tree transformations
	float4x4 modelMatrix;

	glPushMatrix();
	glLoadIdentity();
	glTranslatef( translation[ 0 ], translation[ 1 ], translation[ 2 ] );
	glScalef( scale, scale, scale );
	glTranslatef( -0.5f, -0.5f, -0.5f );//pour centrer la boite GV
	glGetFloatv( GL_MODELVIEW_MATRIX, modelMatrix._array );
	glPopMatrix();

	// Extract viewport
	GLint params[ 4 ];
	glGetIntegerv( GL_VIEWPORT, params );
	int4 viewport = make_int4( params[ 0 ], params[ 1 ], params[ 2 ], params[ 3 ] );

	CUDAPM_STOP_EVENT( app_init_frame );

	// Blending configuration
	glEnable( GL_BLEND );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	
	float newViewMatrix[ 16 ];
	//float newProjectionMatrix[ 16 ];
	glPushMatrix();
	glLoadIdentity();
	GV_CHECK_GL_ERROR();

	//------------------------------------------------------------------------------------------------
	bool found;
	qglviewer::Vec pixelCoords = sviewer->camera()->pointUnderPixel(
		QPoint( sviewer->camera()->screenWidth()/2, sviewer->camera()->screenHeight()/2), found ) 
		+ qglviewer::Vec( _worldLight.x, _worldLight.y, _worldLight.z );
	glPopMatrix();
	if ( found )
	{
		glMatrixMode( GL_MODELVIEW );
		glPushMatrix();
		glLoadMatrixf( modelMatrix._array );
		qglviewer::Vec lightToPixel = pixelCoords -qglviewer::Vec( _worldLight.x, _worldLight.y, _worldLight.z );
		cout << "lightToPixel " << lightToPixel.x << " " << lightToPixel.y << " " << lightToPixel.z << endl;

		qglviewer::Vec upVector = lightToPixel.orthogonalVec();
				
		gluLookAt( _worldLight.x, _worldLight.y, _worldLight.z, lightToPixel.x, lightToPixel.y, lightToPixel.z, upVector.x, upVector.y, upVector.z );
		//gluLookAt(_worldLight.x, _worldLight.y, _worldLight.z, pixelCoords.x, pixelCoords.y, pixelCoords.z, upVector.x, upVector.y, upVector.z);
		glTranslatef( _worldLight.x, _worldLight.y, _worldLight.z );
		glGetFloatv( GL_MODELVIEW_MATRIX, newViewMatrix);
		setNewMVMatrix( newViewMatrix );
		/*float objectCenter[ 3 ];
		mSampleCore->getCenter( objectCenter );
		GLdouble znear = 0.2 * ( qglviewer::Vec( objectCenter[ 0 ], objectCenter[ 1 ], objectCenter[ 2 ] ) - wlight).norm();
		GLdouble zfar = 2 * qglviewer::Vec( objectCenter[ 0 ], objectCenter[ 1 ], objectCenter[ 2 ] ).norm();
		gluPerspective( 150, 1, znear, zfar );
		glGetFloatv( GL_PROJECTION_MATRIX, newProjectionMatrix );*/
		glPopMatrix();
	}
	GV_CHECK_GL_ERROR();
	//------------------------------------------------------------------------------------------------

	// Render the GigaSpace scene
	//
	// TO DO
	// - ask for full brick production, or modify/optimize "light frustum"
	/*_pipeline->editRenderer()*/_renderer->setLightPosition( _cameraLight.x, _cameraLight.y, _cameraLight.z );
	_pipeline->execute( modelMatrix, viewMatrix, projectionMatrix, viewport );
	//_pipeline->execute( modelMatrix, newMVMatrix, /*newPMatrix*/projectionMatrix, viewport );
	GV_CHECK_GL_ERROR();

	// RENDER THE SHADOW RECEIVER AND RETRIEVE THE GIGAVOXELS INFO
	shadowReceiver->setModelMatrix( modelMatrix._array[ 0 ], modelMatrix._array[ 1 ], modelMatrix._array[ 2 ], modelMatrix._array[ 3 ],  
		modelMatrix._array[ 4 ], modelMatrix._array[ 5 ], modelMatrix._array[ 6 ], modelMatrix._array[ 7 ], 
		modelMatrix._array[ 8 ], modelMatrix._array[ 9 ], modelMatrix._array[ 10 ], modelMatrix._array[ 11 ],
		modelMatrix._array[ 12 ], modelMatrix._array[ 13 ], modelMatrix._array[ 14 ], modelMatrix._array[ 15 ] );
	shadowReceiver->setLightPosition( _cameraLight.x, _cameraLight.y, _cameraLight.z );
	shadowReceiver->setWorldLight( _worldLight.x, _worldLight.y, _worldLight.z );
	shadowReceiver->setTexBufferName( /*_pipeline->editRenderer()*/_renderer->getTexBufferName() );
	uint3 bsc = /*_pipeline->editRenderer()*/_renderer->getBrickCacheSize();
	shadowReceiver->setBrickCacheSize( bsc.x, bsc.y, bsc.z );
	float3 bpri = /*_pipeline->editRenderer()*/_renderer->getBrickPoolResInv();
	shadowReceiver->setBrickPoolResInv( bpri.x, bpri.y, bpri.z );
	shadowReceiver->setMaxDepth( /*_pipeline->editRenderer()*/_renderer->getMaxDepth() );
	GV_CHECK_GL_ERROR();
	shadowReceiver->render();
	GV_CHECK_GL_ERROR();

	// RENDER THE OBJECT CASTING THE SHADOW
	shadowCaster->setLightPosition( _cameraLight.x, _cameraLight.y, _cameraLight.z );
	shadowCaster->render();

	// TEST - optimization due to early unmap() graphics resource from GigaVoxels
	/*_pipeline->editRenderer()*/_renderer->nextFrame();

	CUDAPM_STOP_EVENT( frame );
	CUDAPM_STOP_FRAME;

	// Display the performance monitor
	if ( mDisplayPerfmon )
	{
		GvPerfMon::CUDAPerfMon::get().displayFrameGL( mDisplayPerfmon - 1 );
	}
}

/******************************************************************************
 * Resize the frame
 *
 * @param pWidth the new width
 * @param pHeight the new height
 ******************************************************************************/
void SampleCore::resize( int pWidth, int pHeight )
{
	mWidth = pWidth;
	mHeight = pHeight;

	// Re-init Perfmon subsystem
	CUDAPM_RESIZE( make_uint2( mWidth, mHeight ) );

	/*uchar* timersMask = GvPerfMon::CUDAPerfMon::get().getKernelTimerMask();
	cudaMemset( timersMask, 255, mWidth * mHeight );*/
}

/******************************************************************************
 * Clear the GigaVoxels cache
 ******************************************************************************/
void SampleCore::clearCache()
{
	_pipeline->clear();
}

/******************************************************************************
 * Toggle the display of the N-tree (octree) of the data structure
 ******************************************************************************/
void SampleCore::toggleDisplayOctree()
{
	mDisplayOctree = !mDisplayOctree;
}

/******************************************************************************
 * Get the appearance of the N-tree (octree) of the data structure
 ******************************************************************************/
void SampleCore::getDataStructureAppearance( bool& pShowNodeHasBrickTerminal, bool& pShowNodeHasBrickNotTerminal, bool& pShowNodeIsBrickNotInCache, bool& pShowNodeEmptyOrConstant
											, float& pNodeHasBrickTerminalColorR, float& pNodeHasBrickTerminalColorG, float& pNodeHasBrickTerminalColorB, float& pNodeHasBrickTerminalColorA
											, float& pNodeHasBrickNotTerminalColorR, float& pNodeHasBrickNotTerminalColorG, float& pNodeHasBrickNotTerminalColorB, float& pNodeHasBrickNotTerminalColorA
											, float& pNodeIsBrickNotInCacheColorR, float& pNodeIsBrickNotInCacheColorG, float& pNodeIsBrickNotInCacheColorB, float& pNodeIsBrickNotInCacheColorA
											, float& pNodeEmptyOrConstantColorR, float& pNodeEmptyOrConstantColorG, float& pNodeEmptyOrConstantColorB, float& pNodeEmptyOrConstantColorA ) const
{
	float4 nodeHasBrickTerminalColor;
	float4 nodeHasBrickNotTerminalColor;
	float4 nodeIsBrickNotInCacheColor;
	float4 nodeEmptyOrConstantColor;
										
	_pipeline->getDataStructure()->getDataStructureAppearance( pShowNodeHasBrickTerminal, pShowNodeHasBrickNotTerminal, pShowNodeIsBrickNotInCache, pShowNodeEmptyOrConstant
											, nodeHasBrickTerminalColor, nodeHasBrickNotTerminalColor, nodeIsBrickNotInCacheColor, nodeEmptyOrConstantColor );

	pNodeHasBrickTerminalColorR = nodeHasBrickTerminalColor.x;
	pNodeHasBrickTerminalColorG = nodeHasBrickTerminalColor.y;
	pNodeHasBrickTerminalColorB = nodeHasBrickTerminalColor.z;
	pNodeHasBrickTerminalColorA = nodeHasBrickTerminalColor.w;

	pNodeHasBrickNotTerminalColorR = nodeHasBrickNotTerminalColor.x;
	pNodeHasBrickNotTerminalColorG = nodeHasBrickNotTerminalColor.y;
	pNodeHasBrickNotTerminalColorB = nodeHasBrickNotTerminalColor.z;
	pNodeHasBrickNotTerminalColorA = nodeHasBrickNotTerminalColor.w;

	pNodeIsBrickNotInCacheColorR = nodeIsBrickNotInCacheColor.x;
	pNodeIsBrickNotInCacheColorG = nodeIsBrickNotInCacheColor.y;
	pNodeIsBrickNotInCacheColorB = nodeIsBrickNotInCacheColor.z;
	pNodeIsBrickNotInCacheColorA = nodeIsBrickNotInCacheColor.w;

	pNodeEmptyOrConstantColorR = nodeEmptyOrConstantColor.x;
	pNodeEmptyOrConstantColorG = nodeEmptyOrConstantColor.y;
	pNodeEmptyOrConstantColorB = nodeEmptyOrConstantColor.z;
	pNodeEmptyOrConstantColorA = nodeEmptyOrConstantColor.w;
}

/******************************************************************************
 * Set the appearance of the N-tree (octree) of the data structure
 ******************************************************************************/
void SampleCore::setDataStructureAppearance( bool pShowNodeHasBrickTerminal, bool pShowNodeHasBrickNotTerminal, bool pShowNodeIsBrickNotInCache, bool pShowNodeEmptyOrConstant
											, float pNodeHasBrickTerminalColorR, float pNodeHasBrickTerminalColorG, float pNodeHasBrickTerminalColorB, float pNodeHasBrickTerminalColorA
											, float pNodeHasBrickNotTerminalColorR, float pNodeHasBrickNotTerminalColorG, float pNodeHasBrickNotTerminalColorB, float pNodeHasBrickNotTerminalColorA
											, float pNodeIsBrickNotInCacheColorR, float pNodeIsBrickNotInCacheColorG, float pNodeIsBrickNotInCacheColorB, float pNodeIsBrickNotInCacheColorA
											, float pNodeEmptyOrConstantColorR, float pNodeEmptyOrConstantColorG, float pNodeEmptyOrConstantColorB, float pNodeEmptyOrConstantColorA )
{
	float4 nodeHasBrickTerminalColor = make_float4( pNodeHasBrickTerminalColorR, pNodeHasBrickTerminalColorG, pNodeHasBrickTerminalColorB, pNodeHasBrickTerminalColorA );
	float4 nodeHasBrickNotTerminalColor = make_float4( pNodeHasBrickNotTerminalColorR, pNodeHasBrickNotTerminalColorG, pNodeHasBrickNotTerminalColorB, pNodeHasBrickNotTerminalColorA );
	float4 nodeIsBrickNotInCacheColor = make_float4( pNodeIsBrickNotInCacheColorR, pNodeIsBrickNotInCacheColorG, pNodeIsBrickNotInCacheColorB, pNodeIsBrickNotInCacheColorA );
	float4 nodeEmptyOrConstantColor = make_float4( pNodeEmptyOrConstantColorR, pNodeEmptyOrConstantColorG, pNodeEmptyOrConstantColorB, pNodeEmptyOrConstantColorA );

	_pipeline->editDataStructure()->setDataStructureAppearance( pShowNodeHasBrickTerminal, pShowNodeHasBrickNotTerminal, pShowNodeIsBrickNotInCache, pShowNodeEmptyOrConstant
											, nodeHasBrickTerminalColor, nodeHasBrickNotTerminalColor, nodeIsBrickNotInCacheColor, nodeEmptyOrConstantColor );
}

/******************************************************************************
 * Toggle the GigaVoxels dynamic update mode
 ******************************************************************************/
void SampleCore::toggleDynamicUpdate()
{
	setDynamicUpdate( ! hasDynamicUpdate() );
}

/******************************************************************************
 * Get the dynamic update state
 *
 * @return the dynamic update state
 ******************************************************************************/
bool SampleCore::hasDynamicUpdate() const
{
	return _pipeline->hasDynamicUpdate();
}

/******************************************************************************
 * Set the dynamic update state
 *
 * @param pFlag the dynamic update state
 ******************************************************************************/
void SampleCore::setDynamicUpdate( bool pFlag )
{
	_pipeline->setDynamicUpdate( pFlag );
}

/******************************************************************************
 * Toggle the display of the performance monitor utility if
 * GigaVoxels has been compiled with the Performance Monitor option
 *
 * @param mode The performance monitor mode (1 for CPU, 2 for DEVICE)
 ******************************************************************************/
void SampleCore::togglePerfmonDisplay( uint mode )
{
	if ( mDisplayPerfmon )
	{
		mDisplayPerfmon = 0;

		GvPerfMon::CUDAPerfMon::_isActivated = false;
	}
	else
	{
		mDisplayPerfmon = mode;

		GvPerfMon::CUDAPerfMon::_isActivated = true;
	}
}

/******************************************************************************
 * Increment the max resolution of the data structure
 ******************************************************************************/
void SampleCore::incMaxVolTreeDepth()
{
	if ( mMaxVolTreeDepth < 32 )
	{
		mMaxVolTreeDepth++;
	}

	_pipeline->editDataStructure()->setMaxDepth( mMaxVolTreeDepth );
}

/******************************************************************************
 * Decrement the max resolution of the data structure
 ******************************************************************************/
void SampleCore::decMaxVolTreeDepth()
{
	if ( mMaxVolTreeDepth > 0 )
	{
		mMaxVolTreeDepth--;
	}

	_pipeline->editDataStructure()->setMaxDepth( mMaxVolTreeDepth );
}

/******************************************************************************
 * Get the node tile resolution of the data structure.
 *
 * @param pX the X node tile resolution
 * @param pY the Y node tile resolution
 * @param pZ the Z node tile resolution
 ******************************************************************************/
void SampleCore::getDataStructureNodeTileResolution( unsigned int& pX, unsigned int& pY, unsigned int& pZ ) const
{
	const uint3& nodeTileResolution = _pipeline->editDataStructure()->getNodeTileResolution().get();

	pX = nodeTileResolution.x;
	pY = nodeTileResolution.y;
	pZ = nodeTileResolution.z;
}

/******************************************************************************
 * Get the brick resolution of the data structure (voxels).
 *
 * @param pX the X brick resolution
 * @param pY the Y brick resolution
 * @param pZ the Z brick resolution
 ******************************************************************************/
void SampleCore::getDataStructureBrickResolution( unsigned int& pX, unsigned int& pY, unsigned int& pZ ) const
{
	const uint3& brickResolution = _pipeline->editDataStructure()->getBrickResolution().get();

	pX = brickResolution.x;
	pY = brickResolution.y;
	pZ = brickResolution.z;
}

/******************************************************************************
 * Get the max depth.
 *
 * @return the max depth
 ******************************************************************************/
unsigned int SampleCore::getRendererMaxDepth() const
{
	return _pipeline->editDataStructure()->getMaxDepth();
}

/******************************************************************************
 * Set the max depth.
 *
 * @param pValue the max depth
 ******************************************************************************/
void SampleCore::setRendererMaxDepth( unsigned int pValue )
{
	_pipeline->editDataStructure()->setMaxDepth( pValue );
}

/******************************************************************************
 * Set the request strategy indicating if, during data structure traversal,
 * priority of requests is set on brick loads or on node subdivisions first.
 *
 * @param pFlag the flag indicating the request strategy
 ******************************************************************************/
void SampleCore::setRendererPriorityOnBricks( bool pFlag )
{
	/*_pipeline->editRenderer()*/_renderer->setPriorityOnBricks( pFlag );
}

/******************************************************************************
 * Specify color to clear the color buffer
 *
 * @param pRed red component
 * @param pGreen green component
 * @param pBlue blue component
 * @param pAlpha alpha component
 ******************************************************************************/
void SampleCore::setClearColor( unsigned char pRed, unsigned char pGreen, unsigned char pBlue, unsigned char pAlpha )
{
	/*_pipeline->editRenderer()*/_renderer->setClearColor( make_uchar4( pRed, pGreen, pBlue, pAlpha ) );
}

/******************************************************************************
 * Tell wheter or not the pipeline has a light.
 *
 * @return the flag telling wheter or not the pipeline has a light
 ******************************************************************************/
bool SampleCore::hasLight() const
{
	return true;
}

/******************************************************************************
 * Get the light position
 *
 * @param pX the X light position
 * @param pY the Y light position
 * @param pZ the Z light position
 ******************************************************************************/
void SampleCore::getLightPosition( float& pX, float& pY, float& pZ ) const
{
	pX = _lightPosition.x;
	pY = _lightPosition.y;
	pZ = _lightPosition.z;
}

///******************************************************************************
// * Set the light position
// *
// * @param pX the X light position
// * @param pY the Y light position
// * @param pZ the Z light position
// ******************************************************************************/
//void SampleCore::setLightPosition( float pX, float pY, float pZ )
//{
//	// Update DEVICE memory with "light position"
//	//
//	// WARNING
//	// Apply inverse modelisation matrix applied on the GigaVoxels object to set light position correctly.
//	// Here a glTranslatef( -0.5f, -0.5f, -0.5f ) has been used.
//	_lightPosition.x = pX - _translation[ 0 ];
//	_lightPosition.y = pY - _translation[ 1 ];
//	_lightPosition.z = pZ - _translation[ 2 ];
//
//	// Update device memory
//	//GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cLightPosition, &_lightPosition, sizeof( _lightPosition ), 0, cudaMemcpyHostToDevice ) );
//}

/******************************************************************************
 * Get the translation
 *
 * @param pX the translation on x axis
 * @param pY the translation on y axis
 * @param pZ the translation on z axis
 ******************************************************************************/
void SampleCore::getTranslation( float& pX, float& pY, float& pZ ) const
{
	pX = _translation[ 0 ];
	pY = _translation[ 1 ];
	pZ = _translation[ 2 ];
}

/******************************************************************************
 * Set the translation
 *
 * @param pX the translation on x axis
 * @param pY the translation on y axis
 * @param pZ the translation on z axis
 ******************************************************************************/
void SampleCore::setTranslation( float pX, float pY, float pZ )
{
	_translation[ 0 ] = pX;
	_translation[ 1 ] = pY;
	_translation[ 2 ] = pZ;
}

/******************************************************************************
 * Get the rotation
 *
 * @param pAngle the rotation angle (in degree)
 * @param pX the x component of the rotation vector
 * @param pY the y component of the rotation vector
 * @param pZ the z component of the rotation vector
 ******************************************************************************/
void SampleCore::getRotation( float& pAngle, float& pX, float& pY, float& pZ ) const
{
	pAngle = _rotation[ 0 ];
	pX = _rotation[ 1 ];
	pY = _rotation[ 2 ];
	pZ = _rotation[ 3 ];
}

/******************************************************************************
 * Set the rotation
 *
 * @param pAngle the rotation angle (in degree)
 * @param pX the x component of the rotation vector
 * @param pY the y component of the rotation vector
 * @param pZ the z component of the rotation vector
 ******************************************************************************/
void SampleCore::setRotation( float pAngle, float pX, float pY, float pZ )
{
	_rotation[ 0 ] = pAngle;
	_rotation[ 1 ] = pX;;
	_rotation[ 2 ] = pY;;
	_rotation[ 3 ] = pZ;;
}

/******************************************************************************
 * Get the uniform scale
 *
 * @param pValue the uniform scale
 ******************************************************************************/
void SampleCore::getScale( float& pValue ) const
{
	pValue = _scale;
}

/******************************************************************************
 * Set the uniform scale
 *
 * @param pValue the uniform scale
 ******************************************************************************/
void SampleCore::setScale( float pValue )
{
	_scale = pValue;
}

/******************************************************************************
 * Get the node cache memory
 *
 * @return the node cache memory
 ******************************************************************************/
unsigned int SampleCore::getNodeCacheMemory() const
{
	return NODEPOOL_MEMSIZE / ( 1024U * 1024U );
}

/******************************************************************************
 * Set the node cache memory
 *
 * @param pValue the node cache memory
 ******************************************************************************/
void SampleCore::setNodeCacheMemory( unsigned int pValue )
{
}

/******************************************************************************
 * Get the brick cache memory
 *
 * @return the brick cache memory
 ******************************************************************************/
unsigned int SampleCore::getBrickCacheMemory() const
{
	return BRICKPOOL_MEMSIZE / ( 1024U * 1024U );
}

/******************************************************************************
 * Set the brick cache memory
 *
 * @param pValue the brick cache memory
 ******************************************************************************/
void SampleCore::setBrickCacheMemory( unsigned int pValue )
{
}

/******************************************************************************
 * Get the node cache capacity
 *
 * @return the node cache capacity
 ******************************************************************************/
unsigned int SampleCore::getNodeCacheCapacity() const
{
	return _pipeline->getCache()->getNodesCacheManager()->getNumElements();
}

/******************************************************************************
 * Set the node cache capacity
 *
 * @param pValue the node cache capacity
 ******************************************************************************/
void SampleCore::setNodeCacheCapacity( unsigned int pValue )
{
}

/******************************************************************************
 * Get the brick cache capacity
 *
 * @return the brick cache capacity
 ******************************************************************************/
unsigned int SampleCore::getBrickCacheCapacity() const
{
	return _pipeline->getCache()->getBricksCacheManager()->getNumElements();
}

/******************************************************************************
 * Set the brick cache capacity
 *
 * @param pValue the brick cache capacity
 ******************************************************************************/
void SampleCore::setBrickCacheCapacity( unsigned int pValue )
{
}

/******************************************************************************
 * Get the number of unused nodes in cache
 *
 * @return the number of unused nodes in cache
 ******************************************************************************/
unsigned int SampleCore::getCacheNbUnusedNodes() const
{
	return _pipeline->getCache()->getNodesCacheManager()->getNbUnusedElements();
}

/******************************************************************************
 * Get the number of unused bricks in cache
 *
 * @return the number of unused bricks in cache
 ******************************************************************************/
unsigned int SampleCore::getCacheNbUnusedBricks() const
{
	return _pipeline->getCache()->getBricksCacheManager()->getNbUnusedElements();
}

/******************************************************************************
 * Get the max number of requests of node subdivisions.
 *
 * @return the max number of requests
 ******************************************************************************/
unsigned int SampleCore::getCacheMaxNbNodeSubdivisions() const
{
	return _pipeline->getCache()->getMaxNbNodeSubdivisions();
}

/******************************************************************************
 * Set the max number of requests of node subdivisions.
 *
 * @param pValue the max number of requests
 ******************************************************************************/
void SampleCore::setCacheMaxNbNodeSubdivisions( unsigned int pValue )
{
	_pipeline->editCache()->setMaxNbNodeSubdivisions( pValue );
}

/******************************************************************************
 * Get the max number of requests of brick of voxel loads.
 *
 * @return the max number of requests
 ******************************************************************************/
unsigned int SampleCore::getCacheMaxNbBrickLoads() const
{
	return _pipeline->getCache()->getMaxNbBrickLoads();
}

/******************************************************************************
 * Set the max number of requests of brick of voxel loads.
 *
 * @param pValue the max number of requests
 ******************************************************************************/
void SampleCore::setCacheMaxNbBrickLoads( unsigned int pValue )
{
	_pipeline->editCache()->setMaxNbBrickLoads( pValue );
}

/******************************************************************************
 * Get the number of requests of node subdivisions the cache has handled.
 *
 * @return the number of requests
 ******************************************************************************/
unsigned int SampleCore::getCacheNbNodeSubdivisionRequests() const
{
	return _pipeline->getCache()->getNbNodeSubdivisionRequests();
}

/******************************************************************************
 * Get the number of requests of brick of voxel loads the cache has handled.
 *
 * @return the number of requests
 ******************************************************************************/
unsigned int SampleCore::getCacheNbBrickLoadRequests() const
{
	return _pipeline->getCache()->getNbBrickLoadRequests();
}

/******************************************************************************
 * Get the cache policy
 *
 * @return the cache policy
 ******************************************************************************/
unsigned int SampleCore::getCachePolicy() const
{
	return _pipeline->getCache()->getBricksCacheManager()->getPolicy();
}

/******************************************************************************
 * Set the cache policy
 *
 * @param pValue the cache policy
 ******************************************************************************/
void SampleCore::setCachePolicy( unsigned int pValue )
{
	_pipeline->editCache()->editNodesCacheManager()->setPolicy( static_cast< PipelineType::CacheType::NodesCacheManager::ECachePolicy>( pValue ) );
	_pipeline->editCache()->editBricksCacheManager()->setPolicy( static_cast< PipelineType::CacheType::BricksCacheManager::ECachePolicy>( pValue ) );
}

/******************************************************************************
 * Get the nodes cache usage
 *
 * @return the nodes cache usage
 ******************************************************************************/
unsigned int SampleCore::getNodeCacheUsage() const
{
	//const unsigned int nbProducedElements = _pipeline->getCache()->getNodesCacheManager()->_totalNumLoads;
	const unsigned int nbProducedElements = _pipeline->getCache()->getNodesCacheManager()->_numElemsNotUsed;
	const unsigned int nbElements = _pipeline->getCache()->getNodesCacheManager()->getNumElements();

	const unsigned int cacheUsage = static_cast< unsigned int >( 100.0f * static_cast< float >( nbElements - nbProducedElements ) / static_cast< float >( nbElements ) );

	//std::cout << "NODE cache usage [ " << nbProducedElements << " / "<< nbElements << " : " << cacheUsage << std::endl;

	return cacheUsage;
}

/******************************************************************************
 * Get the bricks cache usage
 *
 * @return the bricks cache usage
 ******************************************************************************/
unsigned int SampleCore::getBrickCacheUsage() const
{
	//const unsigned int nbProducedElements = _pipeline->getCache()->getBricksCacheManager()->_totalNumLoads;
	const unsigned int nbProducedElements = _pipeline->getCache()->getBricksCacheManager()->_numElemsNotUsed;
	const unsigned int nbElements = _pipeline->getCache()->getBricksCacheManager()->getNumElements();

	const unsigned int cacheUsage = static_cast< unsigned int >( 100.0f * static_cast< float >( nbElements - nbProducedElements ) / static_cast< float >( nbElements ) );

	//std::cout << "BRICK cache usage [ " << nbProducedElements << " / "<< nbElements << " : " << cacheUsage << std::endl;

	return cacheUsage;
}

/******************************************************************************
 * Get the number of tree leaf nodes
 *
 * @return the number of tree leaf nodes
 ******************************************************************************/
unsigned int SampleCore::getNbTreeLeafNodes() const
{
	return _pipeline->getCache()->_nbLeafNodes;
}

/******************************************************************************
 * Get the number of tree nodes
 *
 * @return the number of tree nodes
 ******************************************************************************/
unsigned int SampleCore::getNbTreeNodes() const
{
	return _pipeline->getCache()->_nbNodes;
}

/******************************************************************************
* Get the flag indicating wheter or not data production monitoring is activated
*
* @return the flag indicating wheter or not data production monitoring is activated
 ******************************************************************************/
bool SampleCore::hasDataProductionMonitoring() const
{
	return true;
}

/******************************************************************************
* Set the the flag indicating wheter or not data production monitoring is activated
*
* @param pFlag the flag indicating wheter or not data production monitoring is activated
 ******************************************************************************/
void SampleCore::setDataProductionMonitoring( bool pFlag )
{
}

/******************************************************************************
* Get the flag indicating wheter or not cache monitoring is activated
*
* @return the flag indicating wheter or not cache monitoring is activated
 ******************************************************************************/
bool SampleCore::hasCacheMonitoring() const
{
	return true;
}

/******************************************************************************
* Set the the flag indicating wheter or not cache monitoring is activated
*
* @param pFlag the flag indicating wheter or not cache monitoring is activated
 ******************************************************************************/
void SampleCore::setCacheMonitoring( bool pFlag )
{
}

/******************************************************************************
* Get the flag indicating wheter or not time budget monitoring is activated
*
* @return the flag indicating wheter or not time budget monitoring is activated
 ******************************************************************************/
bool SampleCore::hasTimeBudgetMonitoring() const
{
	return true;
}

/******************************************************************************
* Set the the flag indicating wheter or not time budget monitoring is activated
*
* @param pFlag the flag indicating wheter or not time budget monitoring is activated
 ******************************************************************************/
void SampleCore::setTimeBudgetMonitoring( bool pFlag )
{
}

/******************************************************************************
 *Tell wheter or not time budget is acivated
 *
 * @return a flag to tell wheter or not time budget is activated
 ******************************************************************************/
bool SampleCore::hasRenderingTimeBudget() const
{
	return true;
}

/******************************************************************************
 * Set the flag telling wheter or not time budget is acivated
 *
 * @param pFlag a flag to tell wheter or not time budget is activated
 ******************************************************************************/
void SampleCore::setRenderingTimeBudgetActivated( bool pFlag )
{
}

/******************************************************************************
 * Get the user requested time budget
 *
 * @return the user requested time budget
 ******************************************************************************/
unsigned int SampleCore::getRenderingTimeBudget() const
{
	return static_cast< unsigned int >( /*_pipeline->getRenderer()*/_renderer->getTimeBudget() );
}

/******************************************************************************
 * Set the user requested time budget
 *
 * @param pValue the user requested time budget
 ******************************************************************************/
void SampleCore::setRenderingTimeBudget( unsigned int pValue )
{
	/*_pipeline->editRenderer()*/_renderer->setTimeBudget( static_cast< float >( pValue ) );
}

/******************************************************************************
 * This method return the duration of the timer event between start and stop event
 *
 * @return the duration of the event in milliseconds
 ******************************************************************************/
float SampleCore::getRendererElapsedTime() const
{
	return /*_pipeline->editRenderer()*/_renderer->getElapsedTime();
}

///******************************************************************************
// * Tell wheter or not pipeline uses programmable shaders
// *
// * @return a flag telling wheter or not pipeline uses programmable shaders
// ******************************************************************************/
//bool SampleCore::hasProgrammableShaders() const
//{
//	return true;
//}
//
///******************************************************************************
// * Tell wheter or not pipeline has a given type of shader
// *
// * @param pShaderType the type of shader to test
// *
// * @return a flag telling wheter or not pipeline has a given type of shader
// ******************************************************************************/
//bool SampleCore::hasShaderType( unsigned int pShaderType ) const
//{
//	//return _shaderProgram->hasShaderType( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
//	RendererType* renderer = dynamic_cast< RendererType* >( _pipeline->editRenderer() );
//	return renderer->getShaderProgram()->hasShaderType( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
//}
//
///******************************************************************************
// * Get the source code associated to a given type of shader
// *
// * @param pShaderType the type of shader
// *
// * @return the associated shader source code
// ******************************************************************************/
//std::string SampleCore::getShaderSourceCode( unsigned int pShaderType ) const
//{
//	RendererType* renderer = dynamic_cast< RendererType* >( _pipeline->editRenderer() );
//	return renderer->getShaderProgram()->getShaderSourceCode( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
//}
//
///******************************************************************************
// * Get the filename associated to a given type of shader
// *
// * @param pShaderType the type of shader
// *
// * @return the associated shader filename
// ******************************************************************************/
//std::string SampleCore::getShaderFilename( unsigned int pShaderType ) const
//{
//	RendererType* renderer = dynamic_cast< RendererType* >( _pipeline->editRenderer() );
//	return renderer->getShaderProgram()->getShaderFilename( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
//}
//
///******************************************************************************
// * ...
// *
// * @param pShaderType the type of shader
// *
// * @return ...
// ******************************************************************************/
//bool SampleCore::reloadShader( unsigned int pShaderType )
//{
//	RendererType* renderer = dynamic_cast< RendererType* >( _pipeline->editRenderer() );
//	return renderer->getShaderProgram()->reloadShader( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
//}

/******************************************************************************
 * Set the light position in camera coordinates
 *
 * @param pX the X light position
 * @param pY the Y light position
 * @param pZ the Z light position
 ******************************************************************************/
void SampleCore::setLightPosition( float pX, float pY, float pZ )
{	
	_lightPosition = make_float3( pX, pY, pZ );
}

/******************************************************************************
 * Set the light position in world coordinates
 *
 * @param pX the X light position
 * @param pY the Y light position
 * @param pZ the Z light position
 ******************************************************************************/
void SampleCore::setWorldLight( float pX, float pY, float pZ )
{
	_worldLight = make_float3( pX, pY, pZ );
}

/******************************************************************************
 * Set the light position in world coordinates
 *
 * @param pX the X light position
 * @param pY the Y light position
 * @param pZ the Z light position
 ******************************************************************************/
void SampleCore::setCameraLight( float pX, float pY, float pZ )
{
	_cameraLight = make_float3( pX, pY, pZ );
}

/******************************************************************************
 * Returns the file name of the shadow casting object (OpenGL)
 ******************************************************************************/
string SampleCore::getShadowCasterFile()
{
	return shadowCasterFile;
}

/******************************************************************************
 * ...
 ******************************************************************************/
float SampleCore::getScale()
{
	return scale;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::getCenter( float center[ 3 ] )
{
	center[ 0 ] = translation[ 0 ];
	center[ 1 ] = translation[ 1 ];
	center[ 2 ] = translation[ 2 ];
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setNewMVMatrix( float m[ 16] )
{
	newMVMatrix._array[ 0 ] = m[ 0 ];
	newMVMatrix._array[ 1 ] = m[ 1 ];
	newMVMatrix._array[ 2 ] = m[ 2 ];
	newMVMatrix._array[ 3 ] = m[ 3 ];

	newMVMatrix._array[ 4 ] = m[ 4 ];
	newMVMatrix._array[ 5 ] = m[ 5 ];
	newMVMatrix._array[ 6 ] = m[ 6 ];
	newMVMatrix._array[ 7 ] = m[ 7 ];

	newMVMatrix._array[ 8 ] = m[ 8 ];
	newMVMatrix._array[ 9 ] = m[ 9 ];
	newMVMatrix._array[ 10 ] = m[ 10 ];
	newMVMatrix._array[ 11 ] = m[ 11 ];

	newMVMatrix._array[ 12 ] = m[ 12] ;
	newMVMatrix._array[ 13 ] = m[ 13 ];
	newMVMatrix._array[ 14 ] = m[ 14 ];
	newMVMatrix._array[ 15 ] = m[ 15 ];
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setNewPMatrix( float m[ 16 ] )
{
	newPMatrix._array[ 0 ] = m[ 0 ];
	newPMatrix._array[ 1 ] = m[ 1 ];
	newPMatrix._array[ 2 ] = m[ 2 ];
	newPMatrix._array[ 3 ] = m[ 3 ];

	newPMatrix._array[ 4 ] = m[ 4 ];
	newPMatrix._array[ 5 ] = m[ 5 ];
	newPMatrix._array[ 6 ] = m[ 6 ];
	newPMatrix._array[ 7 ] = m[ 7 ];

	newPMatrix._array[ 8 ] = m[ 8 ];
	newPMatrix._array[ 9 ] = m[ 9 ];
	newPMatrix._array[ 10 ] = m[ 10 ];
	newPMatrix._array[ 11 ] = m[ 11 ];

	newPMatrix._array[ 12 ] = m[ 12 ];
	newPMatrix._array[ 13 ] = m[ 13 ];
	newPMatrix._array[ 14 ] = m[ 14 ];
	newPMatrix._array[ 15 ] = m[ 15 ];
}

/******************************************************************************
 * Get the 3D model filename to load
 *
 * @return the 3D model filename to load
 ******************************************************************************/
string SampleCore::get3DModelFilename() const
{
	if ( shadowReceiver != NULL )
	{
		return shadowReceiver->get3DModelFilename();
	} else {
		return "";
	}
}

/******************************************************************************
 * Set the 3D model filename to load
 *
 * @param pFilename the 3D model filename to load
 ******************************************************************************/
void SampleCore::set3DModelFilename( const string& pFilename )
{
	// Store current shadow receiver state if any
	// ...
	
	// ---- Delete the 3D scene if needed ----
	
	if ( shadowReceiver != NULL )
	{
		delete shadowReceiver;
		shadowReceiver = NULL;

		// Clear the GigaVoxels cache
	//	_pipeline->editCache()->clearCache();
	}

	// Initialize proxy geometry (load the 3D scene)
	//
	// - find a way to modify internal buffer size
	// CREATING SHADOW RECEIVER
	shadowReceiver = new ObjectReceivingShadow();
	shadowReceiver->set3DModelFilename( pFilename.c_str() );
	/*_pipeline->editRenderer()*/_renderer->getVolTreeChildArray()->unmapResource();
	/*_pipeline->editRenderer()*/_renderer->getVolTreeDataArray()->unmapResource();
	shadowReceiver->setVolTreeChildArray(/*_pipeline->editRenderer()*/_renderer->getVolTreeChildArray(), /*_pipeline->editRenderer()*/_renderer->getChildBufferName());
	shadowReceiver->setVolTreeDataArray(/*_pipeline->editRenderer()*/_renderer->getVolTreeDataArray(), /*_pipeline->editRenderer()*/_renderer->getDataBufferName());	
	shadowReceiver->init();
	/*_pipeline->editRenderer()*/_renderer->getVolTreeChildArray()->mapResource();
	/*_pipeline->editRenderer()*/_renderer->getVolTreeDataArray()->mapResource();

	// Restore previous proxy geometry state
	// ...
	// Reset proxy geometry resources
	// ...
}

/******************************************************************************
 * Get the translation
 *
 * @param pX the translation on x axis
 * @param pY the translation on y axis
 * @param pZ the translation on z axis
 ******************************************************************************/
void SampleCore::getShadowReceiverTranslation( float& pX, float& pY, float& pZ ) const
{
	shadowReceiver->getTranslation( pX, pY, pZ );
}

/******************************************************************************
 * Set the translation
 *
 * @param pX the translation on x axis
 * @param pY the translation on y axis
 * @param pZ the translation on z axis
 ******************************************************************************/
void SampleCore::setShadowReceiverTranslation( float pX, float pY, float pZ )
{
	shadowReceiver->setTranslation( pX, pY, pZ );
}

/******************************************************************************
 * Get the rotation
 *
 * @param pAngle the rotation angle (in degree)
 * @param pX the x component of the rotation vector
 * @param pY the y component of the rotation vector
 * @param pZ the z component of the rotation vector
 ******************************************************************************/
void SampleCore::getShadowReceiverRotation( float& pAngle, float& pX, float& pY, float& pZ ) const
{
	shadowReceiver->getRotation( pAngle, pX, pY, pZ );
}

/******************************************************************************
 * Set the rotation
 *
 * @param pAngle the rotation angle (in degree)
 * @param pX the x component of the rotation vector
 * @param pY the y component of the rotation vector
 * @param pZ the z component of the rotation vector
 ******************************************************************************/
void SampleCore::setShadowReceiverRotation( float pAngle, float pX, float pY, float pZ )
{
	shadowReceiver->setRotation( pAngle, pX, pY, pZ );
}

/******************************************************************************
 * Get the uniform scale
 *
 * @param pValue the uniform scale
 ******************************************************************************/
void SampleCore::getShadowReceiverScale( float& pValue ) const
{
	shadowReceiver->getScale( pValue );
}

/******************************************************************************
 * Set the uniform scale
 *
 * @param pValue the uniform scale
 ******************************************************************************/
void SampleCore::setShadowReceiverScale( float pValue )
{
	shadowReceiver->setScale( pValue );
}

/******************************************************************************
 * Get the 3D model filename to load
 *
 * @return the 3D model filename to load
 ******************************************************************************/
string SampleCore::getShadowCaster3DModelFilename() const
{
	if ( shadowReceiver != NULL )
	{
		return shadowReceiver->get3DModelFilename();
	} else {
		return "";
	}
}

/******************************************************************************
 * Set the 3D model filename to load
 *
 * @param pFilename the 3D model filename to load
 ******************************************************************************/
void SampleCore::setShadowCaster3DModelFilename( const string& pFilename )
{
	// Store current shadow receiver state if any
	// ...
	
	// ---- Delete the 3D scene if needed ----
	
	if ( shadowReceiver != NULL )
	{
		delete shadowReceiver;
		shadowReceiver = NULL;

		// Clear the GigaVoxels cache
	//	_pipeline->editCache()->clearCache();
	}

	// Initialize proxy geometry (load the 3D scene)
	//
	// - find a way to modify internal buffer size
	// CREATING SHADOW RECEIVER
	shadowReceiver = new ObjectReceivingShadow();
	shadowReceiver->set3DModelFilename( pFilename.c_str() );
	/*_pipeline->editRenderer()*/_renderer->getVolTreeChildArray()->unmapResource();
	/*_pipeline->editRenderer()*/_renderer->getVolTreeDataArray()->unmapResource();
	shadowReceiver->setVolTreeChildArray(/*_pipeline->editRenderer()*/_renderer->getVolTreeChildArray(), /*_pipeline->editRenderer()*/_renderer->getChildBufferName());
	shadowReceiver->setVolTreeDataArray(/*_pipeline->editRenderer()*/_renderer->getVolTreeDataArray(), /*_pipeline->editRenderer()*/_renderer->getDataBufferName());	
	shadowReceiver->init();
	/*_pipeline->editRenderer()*/_renderer->getVolTreeChildArray()->mapResource();
	/*_pipeline->editRenderer()*/_renderer->getVolTreeDataArray()->mapResource();

	// Restore previous proxy geometry state
	// ...
	// Reset proxy geometry resources
	// ...
}

/******************************************************************************
 * Get the translation
 *
 * @param pX the translation on x axis
 * @param pY the translation on y axis
 * @param pZ the translation on z axis
 ******************************************************************************/
void SampleCore::getShadowCasterTranslation( float& pX, float& pY, float& pZ ) const
{
	shadowCaster->getTranslation( pX, pY, pZ );
}

/******************************************************************************
 * Set the translation
 *
 * @param pX the translation on x axis
 * @param pY the translation on y axis
 * @param pZ the translation on z axis
 ******************************************************************************/
void SampleCore::setShadowCasterTranslation( float pX, float pY, float pZ )
{
	shadowCaster->setTranslation( pX, pY, pZ );
}

/******************************************************************************
 * Get the rotation
 *
 * @param pAngle the rotation angle (in degree)
 * @param pX the x component of the rotation vector
 * @param pY the y component of the rotation vector
 * @param pZ the z component of the rotation vector
 ******************************************************************************/
void SampleCore::getShadowCasterRotation( float& pAngle, float& pX, float& pY, float& pZ ) const
{
	shadowCaster->getRotation( pAngle, pX, pY, pZ );
}

/******************************************************************************
 * Set the rotation
 *
 * @param pAngle the rotation angle (in degree)
 * @param pX the x component of the rotation vector
 * @param pY the y component of the rotation vector
 * @param pZ the z component of the rotation vector
 ******************************************************************************/
void SampleCore::setShadowCasterRotation( float pAngle, float pX, float pY, float pZ )
{
	shadowCaster->setRotation( pAngle, pX, pY, pZ );
}

/******************************************************************************
 * Get the uniform scale
 *
 * @param pValue the uniform scale
 ******************************************************************************/
void SampleCore::getShadowCasterScale( float& pValue ) const
{
	shadowCaster->getScale( pValue );
}

/******************************************************************************
 * Set the uniform scale
 *
 * @param pValue the uniform scale
 ******************************************************************************/
void SampleCore::setShadowCasterScale( float pValue )
{
	shadowCaster->setScale( pValue );
}
