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

#include "SampleCore.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/StaticRes3D.h>
#include <GvStructure/GvVolumeTree.h>
#include <GvStructure/GvDataProductionManager.h>
#include <GvUtils/GvSimplePipeline.h>
#include <GvCore/GvError.h>
#include <GvPerfMon/GvPerformanceMonitor.h>

// Project
#include "Producer.h"
#include "Shader.h"
#include "VolumeTreeRendererCUDA.h"

// GvViewer
#include <GvvApplication.h>
#include <GvvMainWindow.h>

// Cuda SDK
#include <helper_math.h>

// System
#include <cstdlib>
#include <ctime>
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvRendering;

// GigaVoxels viewer
using namespace GvViewerCore;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Defines the size allowed for each type of pool
 */
#define NODEPOOL_MEMSIZE	( 8U * 1024U * 1024U )		// 8 Mo
#define BRICKPOOL_MEMSIZE	( 256U * 1024U * 1024U )	// 256 Mo

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
:	GvvPipelineInterface()
,	_pipeline( NULL )
,	_colorTex( 0 )
,	_depthTex( 0 )
,	_frameBuffer( 0 )
,	_depthBuffer( 0 )
,	_displayOctree( false )
,	_displayPerfmon( 0 )
,	_maxVolTreeDepth( 0 )
{
	// Translation used to position the GigaVoxels data structure
	_translation[ 0 ] = -0.5f;
	_translation[ 1 ] = -0.5f;
	_translation[ 2 ] = -0.5f;

	// Light position
	_lightPosition = make_float3( 1.f, 1.f, 1.f );

	// Spheres ray-tracing parameters
	_nbSpheres = 0;
	_userDefinedMinLevelOfResolutionToHandleMode = false;
    _userDefinedMinLevelOfResolutionToHandle = 0;
	_automaticMinLevelOfResolutionToHandleMode = true;
	//_automaticMinLevelOfResolutionToHandle = 0;
	_sphereBrickIntersectionType = 0;
	_geometricCriteria = true;
	_minNbSpheresPerBrick = 1;
	_screenBasedCriteria = true;
	_absoluteSizeCriteria = true;
	_fixedSizeSphere = true;
	_meanSizeOfSpheres = false;
	_shaderUseUniformColor = false;
	_shaderUniformColor = make_float4( 1.f, 1.f, 1.f, 1.f );
    _shaderAnimation = false;
    _shaderBlurSphere = false;
    _shaderFog = false;
    _shaderFogDensity = 0.0f;
    _shading = false;
    _bugCorrection = true;
    _sphereDiameterCoeff = 1;
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
	// Delete the GigaVoxels pipeline
	delete _pipeline;

	// CUDA tip: clean up to ensure correct profiling
	//cudaError_t error = cudaDeviceReset();
}

/******************************************************************************
 * Gets the name of this browsable
 *
 * @return the name of this browsable
 ******************************************************************************/
const char* SampleCore::getName() const
{
	return "SphereRayTracing";
}

/******************************************************************************
 * Initialize the GigaVoxels pipeline
 ******************************************************************************/
void SampleCore::init()
{

    printf("*********** SAMPLE CORE INIT() ***********\n ");
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

	// Pipeline creation
	_pipeline = new PipelineType();
	ProducerType* producer = new ProducerType();
	ShaderType* shader = new ShaderType();

	// Pipeline initialization
	_pipeline->initialize( NODEPOOL_MEMSIZE, BRICKPOOL_MEMSIZE, producer, shader );

	// Pipeline configuration
	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );
	
	// Custom initialization
	// Note : this could be done via an XML settings file loaded at initialization
	setNbSpheres( 1 );
    setNbSpheresTotal ( 1 );

	// Sphere-brick intersection type
	//
	// 0 : sphere-sphere (brick are approximated by spheres)
	// 1 : sphere-box (brick are not approximated, it uses real sphere-box intersection test)
	setSphereBrickIntersectionType( 1 );
    setSphereRadiusFader( 1.0f );
	setGeometricCriteria( true );
	setMinNbSpheresPerBrick( 1 );
	setScreenBasedCriteria( true );
	setAbsoluteSizeCriteria( true );
    setSphereDiameterCoeff( 1 );
	setFixedSizeSphere( true );
	setFixedSizeSphereRadius( 0.01f );
	setMeanSizeOfSpheres( false );
	setShaderUniformColorMode( false );
	setShaderUniformColor( 1.f, 1.f, 1.f, 1.f );
    setShaderAnimation( false );
    setShaderFog( false );
    setShaderFogColor( 0.8f, 0.8f, 0.8f, 1.f);
    setFogDensity( 0.15f );
    setIlluminationCoeff( 1.0f );
	
    // Max depth
    //setRendererMaxDepth( 0 );
	_maxVolTreeDepth = 0;
	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );

	// Set light position
	setLightPosition( 1.f, 1.f, 1.f );

	// Configure cache
	_pipeline->editCache()->setMaxNbNodeSubdivisions( 500 );
	_pipeline->editCache()->setMaxNbBrickLoads( 300 );
	_pipeline->editCache()->editNodesCacheManager()->setPolicy( DataProductionManager::NodesCacheManager::eAllPolicies );
	_pipeline->editCache()->editBricksCacheManager()->setPolicy(DataProductionManager::BricksCacheManager::eAllPolicies );
}

/******************************************************************************
 * Draw function called of frame
 ******************************************************************************/
void SampleCore::draw()
{
	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT( frame );
	CUDAPM_START_EVENT( app_init_frame );

    const uchar4& color = _pipeline->getRenderer()->getClearColor();
    glClearColor( color.x/255.f, color.y/255.f, color.z/255.f, color.w/255.f );

    glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );

	glMatrixMode( GL_MODELVIEW );

	if ( _displayOctree )
	{
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );

		// Display the GigaVoxels N3-tree space partitioning structure
		glEnable( GL_DEPTH_TEST );
		glPushMatrix();
		// Translation used to position the GigaVoxels data structure
		//glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );
		_pipeline->editDataStructure()->render();
		glPopMatrix();
		glDisable( GL_DEPTH_TEST );

		// Clear the depth PBO (pixel buffer object) by reading from the previously cleared FBO (frame buffer object)
		glBindBuffer( GL_PIXEL_PACK_BUFFER, _depthBuffer );
		glReadPixels( 0, 0, _width, _height, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0 );
		glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
		GV_CHECK_GL_ERROR();
	}
	else
	{
		glClear( GL_COLOR_BUFFER_BIT );
	}

	glBindFramebuffer( GL_FRAMEBUFFER, 0 );

	// extract view transformations
	float4x4 viewMatrix;
	float4x4 projectionMatrix;
	glGetFloatv( GL_MODELVIEW_MATRIX, viewMatrix._array );
	glGetFloatv( GL_PROJECTION_MATRIX, projectionMatrix._array );

	// extract viewport
	GLint params[4];
	glGetIntegerv( GL_VIEWPORT, params );
	int4 viewport = make_int4(params[0], params[1], params[2], params[3]);

	// render the scene into textures
	CUDAPM_STOP_EVENT( app_init_frame );

	// Build the world transformation matrix
	float4x4 modelMatrix;
	glPushMatrix();
	glLoadIdentity();
	// Translation used to position the GigaVoxels data structure
	//glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );
	glGetFloatv( GL_MODELVIEW_MATRIX, modelMatrix._array );
	glPopMatrix();

	// Render
	_pipeline->execute( modelMatrix, viewMatrix, projectionMatrix, viewport );

	// Render the result to the screen
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();

	glDisable( GL_DEPTH_TEST );
	glEnable( GL_TEXTURE_RECTANGLE_EXT );
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _colorTex );

	// Draw a full screen quad
	GLint sMin = 0;
	GLint tMin = 0;
	GLint sMax = _width;
	GLint tMax = _height;
	glBegin( GL_QUADS );
	glColor3f( 1.0f, 1.0f, 1.0f );
	glTexCoord2i( sMin, tMin ); glVertex2i( -1, -1 );
	glTexCoord2i( sMax, tMin ); glVertex2i(  1, -1 );
	glTexCoord2i( sMax, tMax ); glVertex2i(  1,  1 );
	glTexCoord2i( sMin, tMax ); glVertex2i( -1,  1 );
	glEnd();

	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
	glDisable( GL_TEXTURE_RECTANGLE_EXT );

	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();

	// Update GigaVoxels info
	_pipeline->editRenderer()->nextFrame();

	CUDAPM_STOP_EVENT( frame );
	CUDAPM_STOP_FRAME;

	// Display the GigaVoxels performance monitor (if it has been activated during GigaVoxels compilation)
	if ( _displayPerfmon )
	{
		GvPerfMon::CUDAPerfMon::get().displayFrameGL( _displayPerfmon - 1 );
	}
}

/******************************************************************************
 * Resize the frame
 *
 * @param width the new width
 * @param height the new height
 ******************************************************************************/
void SampleCore::resize( int width, int height )
{
	_width = width;
	_height = height;

	// Reset default active frame region for rendering
	_pipeline->editRenderer()->setProjectedBBox( make_uint4( 0, 0, _width, _height ) );

	// Re-init Perfmon subsystem
	CUDAPM_RESIZE( make_uint2( _width, _height ) );

	// Create frame-dependent objects
	
	// Disconnect all registered graphics resources
	_pipeline->editRenderer()->resetGraphicsResources();
	
	// ...
	if (_depthBuffer)
	{
		glDeleteBuffers(1, &_depthBuffer);
	}

	if (_colorTex)
	{
		glDeleteTextures(1, &_colorTex);
	}
	if (_depthTex)
	{
		glDeleteTextures(1, &_depthTex);
	}

	if (_frameBuffer)
	{
		glDeleteFramebuffers(1, &_frameBuffer);
	}

	glGenTextures(1, &_colorTex);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, _colorTex);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);
	GV_CHECK_GL_ERROR();

	glGenBuffers(1, &_depthBuffer);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, _depthBuffer);
	glBufferData(GL_PIXEL_PACK_BUFFER, width * height * sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	GV_CHECK_GL_ERROR();

	glGenTextures(1, &_depthTex);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, _depthTex);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, GL_DEPTH24_STENCIL8_EXT, width, height, 0, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, NULL);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);
	GV_CHECK_GL_ERROR();

	glGenFramebuffers( 1, &_frameBuffer );
	glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_EXT, _colorTex, 0 );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, _depthTex, 0 );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, _depthTex, 0 );
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );
	GV_CHECK_GL_ERROR();

	// Create CUDA resources from OpenGL objects
	if ( _displayOctree )
	{
		_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	}
	else
	{
		_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
	}
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
	_displayOctree = !_displayOctree;

	// Disconnect all registered graphics resources
	_pipeline->editRenderer()->resetGraphicsResources();

	if ( _displayOctree )
	{
		_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	}
	else
	{
		_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
	}
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
	const bool status = _pipeline->hasDynamicUpdate();
	_pipeline->setDynamicUpdate( ! status );
}

/******************************************************************************
 * Toggle the display of the performance monitor utility if
 * GigaVoxels has been compiled with the Performance Monitor option
 *
 * @param mode The performance monitor mode (1 for CPU, 2 for DEVICE)
 ******************************************************************************/
void SampleCore::togglePerfmonDisplay( uint mode )
{
	if ( _displayPerfmon )
	{
		_displayPerfmon = 0;
	}
	else
	{
		_displayPerfmon = mode;
	}
}

/******************************************************************************
 * Increment the max resolution of the data structure
 ******************************************************************************/
void SampleCore::incMaxVolTreeDepth()
{
	if ( _maxVolTreeDepth < 32 )
	{
		_maxVolTreeDepth++;
	}

	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );
}

/******************************************************************************
 * Decrement the max resolution of the data structure
 ******************************************************************************/
void SampleCore::decMaxVolTreeDepth()
{
	if ( _maxVolTreeDepth > 0 )
	{
		_maxVolTreeDepth--;
	}

	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );
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
	const uint3& nodeTileResolution = _pipeline->getDataStructure()->getNodeTileResolution().get();

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
	const uint3& brickResolution = _pipeline->getDataStructure()->getBrickResolution().get();

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
	return _pipeline->getDataStructure()->getMaxDepth();
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
 * Set the request strategy indicating if, during data structure traversal,
 * priority of requests is set on brick loads or on node subdivisions first.
 *
 * @param pFlag the flag indicating the request strategy
 ******************************************************************************/
void SampleCore::setRendererPriorityOnBricks( bool pFlag )
{
	_pipeline->editRenderer()->setPriorityOnBricks( pFlag );
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
	_pipeline->editRenderer()->setClearColor( make_uchar4( pRed, pGreen, pBlue, pAlpha ) );
}

/******************************************************************************
 * Tell wheter or not the pipeline has a light.
 *
 * @return the flag telling wheter or not the pipeline has a light
 ******************************************************************************/
bool SampleCore::hasLight() const
{
	return false;
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

/******************************************************************************
 * Set the light position
 *
 * @param pX the X light position
 * @param pY the Y light position
 * @param pZ the Z light position
 ******************************************************************************/
void SampleCore::setLightPosition( float pX, float pY, float pZ )
{
    // Update DEVICE memory with "light position"
    //
    // WARNING
    // Apply inverse modelisation matrix applied on the GigaVoxels object to set light position correctly.
    // Here a glTranslatef( -0.5f, -0.5f, -0.5f ) has been used.
    _lightPosition.x = pX/* - _translation[ 0 ]*/;
    _lightPosition.y = pY/* - _translation[ 1 ]*/;
    _lightPosition.z = pZ/* - _translation[ 2 ]*/;

    // Update device memory
    GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cLightPosition, &_lightPosition, sizeof( _lightPosition ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Get the translation used to position the GigaVoxels data structure
 *
 * @param pX the x componenet of the translation
 * @param pX the y componenet of the translation
 * @param pX the z componenet of the translation
 ******************************************************************************/
void SampleCore::getTranslation( float& pX, float& pY, float& pZ ) const
{
	pX = _translation[ 0 ];
	pY = _translation[ 1 ];
	pZ = _translation[ 2 ];
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
	_pipeline->editCache()->editNodesCacheManager()->setPolicy( static_cast< DataProductionManager::NodesCacheManager::ECachePolicy>( pValue ) );
	_pipeline->editCache()->editBricksCacheManager()->setPolicy( static_cast< DataProductionManager::BricksCacheManager::ECachePolicy>( pValue ) );
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
 * ...
 ******************************************************************************/
unsigned int SampleCore::getNbSpheres() const
{
    return _nbSpheres;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setNbSpheres( unsigned int pValue )
{
    _nbSpheres = pValue;

    // Update producer
    _pipeline->editProducer()->setNbSpheres( pValue );

    // Update DEVICE memory
    GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cNbSpheres, &_nbSpheres, sizeof( _nbSpheres ), 0, cudaMemcpyHostToDevice ) );

    // Clear the cache
    clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
unsigned int SampleCore::getNbSpheresTotal() const
{
    return _nbSpheresTotal;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setNbSpheresTotal( unsigned int pValue )
{
    _nbSpheresTotal = pValue;

    unsigned int nbSphereByBrick = 0;

    // calcul du niveau de resolution necessaire a la production des etoiles
    unsigned int levelToHandle = 1 + logf(_nbSpheresTotal / 998) / log( 8.f );

    // calcul pour savoir combien il faut de spheres par brique
    nbSphereByBrick = _nbSpheresTotal / powf( 8, levelToHandle );

    // Update producer
    _pipeline->editProducer()->setNbSpheres( nbSphereByBrick );

    setUserDefinedMinLevelOfResolutionToHandle( levelToHandle );

    setRendererMaxDepth( levelToHandle );

    //printf("level To handle : %d\nnbSphere by brick : %d\n", levelToHandle, nbSphereByBrick);

    // Update DEVICE memory
    GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cNbSpheres, &nbSphereByBrick, sizeof( nbSphereByBrick ), 0, cudaMemcpyHostToDevice ) );

    // Clear the cache
    clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::regeneratePositions()
{
    _pipeline->editProducer()->generateNewParticleBuffer();
    
	// Clear the cache
    clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool SampleCore::getUserDefinedMinLevelOfResolutionToHandleMode() const
{
	return _userDefinedMinLevelOfResolutionToHandleMode;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setUserDefinedMinLevelOfResolutionToHandleMode( bool pFlag )
{
	_userDefinedMinLevelOfResolutionToHandleMode = pFlag;

	// Update DEVICE memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cMinLevelOfResolutionToHandle, &_userDefinedMinLevelOfResolutionToHandle, sizeof( _userDefinedMinLevelOfResolutionToHandle ), 0, cudaMemcpyHostToDevice ) );

	// Clear the cache
	clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
unsigned int SampleCore::getUserDefinedMinLevelOfResolutionToHandle() const
{
	return _userDefinedMinLevelOfResolutionToHandle;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setUserDefinedMinLevelOfResolutionToHandle( unsigned int pValue )
{
	_userDefinedMinLevelOfResolutionToHandle = pValue;

	// Update DEVICE memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cMinLevelOfResolutionToHandle, &_userDefinedMinLevelOfResolutionToHandle, sizeof( _userDefinedMinLevelOfResolutionToHandle ), 0, cudaMemcpyHostToDevice ) );

	// Clear the cache
	clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool SampleCore::getAutomaticMinLevelOfResolutionToHandleMode() const
{
	return _automaticMinLevelOfResolutionToHandleMode;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setAutomaticMinLevelOfResolutionToHandleMode( bool pFlag )
{
	_automaticMinLevelOfResolutionToHandleMode = pFlag;

	if ( pFlag )
	{
		unsigned int minLevelOfResolutionToHandle = 0;
		const unsigned int nbChildren = 8;	// for octree
		const unsigned int maxNbSphereByBrick = 998;		// 10 x 10 x 10 - 2		[ 2 first cache elements are used to write special data ]
		unsigned int minNbSphereByBrick = static_cast< unsigned int >( static_cast< float >( _nbSpheres ) / powf( nbChildren, minLevelOfResolutionToHandle ) );
		while ( minNbSphereByBrick > maxNbSphereByBrick )
		{
			minLevelOfResolutionToHandle++;

			minNbSphereByBrick /= powf( nbChildren, minLevelOfResolutionToHandle );
		}

		// Update DEVICE memory
		GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cMinLevelOfResolutionToHandle, &minLevelOfResolutionToHandle, sizeof( minLevelOfResolutionToHandle ), 0, cudaMemcpyHostToDevice ) );

		// Clear the cache
		clearCache();
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
unsigned int SampleCore::getAutomaticMinLevelOfResolutionToHandle() const
{
	//unsigned int minLevelOfResolutionToHandle = 0;
	//const unsigned int nbChildren = 8;	// for octree
	//const unsigned int maxNbSphereByBrick = 998;		// 10 x 10 x 10 - 2		[ 2 first cache elements are used to write special data ]
	//unsigned int minNbSphereByBrick = static_cast< unsigned int >( static_cast< float >( _nbSpheres ) / powf( nbChildren, minLevelOfResolutionToHandle ) );
	//while ( minNbSphereByBrick > maxNbSphereByBrick )
	//{
	//	minLevelOfResolutionToHandle++;

	//	minNbSphereByBrick /= powf( nbChildren, minLevelOfResolutionToHandle );
	//}

	//return minLevelOfResolutionToHandle;

	const unsigned int nbChildren = 8;	// for octree
	const unsigned int nbSpheres = static_cast< unsigned int >( static_cast< float >( _nbSpheres ) * powf( nbChildren, _userDefinedMinLevelOfResolutionToHandle ) );
	
	return nbSpheres;
}

///******************************************************************************
// * ...
// ******************************************************************************/
//void SampleCore::setAutomaticMinLevelOfResolutionToHandle( unsigned int pValue )
//{
//	_automaticMinLevelOfResolutionToHandle = pValue;
//
//	// Update DEVICE memory
//	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cAutomaticMinLevelOfResolutionToHandle, &_automaticMinLevelOfResolutionToHandle, sizeof( _automaticMinLevelOfResolutionToHandle ), 0, cudaMemcpyHostToDevice ) );
//
//	// Clear the cache
//	clearCache();
//}

/******************************************************************************
 * ...
 ******************************************************************************/
unsigned int SampleCore::getSphereBrickIntersectionType() const
{
	return _sphereBrickIntersectionType;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setSphereBrickIntersectionType( unsigned int pValue )
{
	_sphereBrickIntersectionType = pValue;

	// Update DEVICE memory with "voxel scale"
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cSphereBrickIntersectionType, &_sphereBrickIntersectionType, sizeof( _sphereBrickIntersectionType ), 0, cudaMemcpyHostToDevice ) );

	// Clear the cache
	clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
float SampleCore::getSphereRadiusFader() const
{
    return _sphereRadiusFader;//_pipeline->getProducer()->getSphereRadiusFader();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setSphereRadiusFader( float pValue )
{
	//_pipeline->editProducer()->setSphereRadiusFader( pValue );
    _sphereRadiusFader = pValue;

    // Update DEVICE memory with "voxel scale"
    GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cSphereRadiusFader, &_sphereRadiusFader, sizeof( _sphereRadiusFader ), 0, cudaMemcpyHostToDevice ) );

	// Clear the cache
	clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool SampleCore::hasGeometricCriteria() const
{
	return _geometricCriteria;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setGeometricCriteria( bool pFlag )
{
	_geometricCriteria = pFlag;

	// Update DEVICE memory with "voxel scale"
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cGeometricCriteria, &_geometricCriteria, sizeof( _geometricCriteria ), 0, cudaMemcpyHostToDevice ) );

	// Clear the cache
	clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
unsigned int SampleCore::getMinNbSpheresPerBrick() const
{
	return _minNbSpheresPerBrick;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setMinNbSpheresPerBrick( unsigned int pValue )
{
	_minNbSpheresPerBrick = pValue;

	// Update DEVICE memory with "voxel scale"
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cMinNbSpheresPerBrick, &_minNbSpheresPerBrick, sizeof( _minNbSpheresPerBrick ), 0, cudaMemcpyHostToDevice ) );

	// Clear the cache
	clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool SampleCore::hasScreenBasedCriteria() const
{
	return _screenBasedCriteria;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setScreenBasedCriteria( bool pFlag )
{
	_screenBasedCriteria = pFlag;

	// Update DEVICE memory with "voxel scale"
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cScreenBasedCriteria, &_screenBasedCriteria, sizeof( _screenBasedCriteria ), 0, cudaMemcpyHostToDevice ) );

	// Clear the cache
	clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool SampleCore::hasAbsoluteSizeCriteria() const
{
	return _absoluteSizeCriteria;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setAbsoluteSizeCriteria( bool pFlag )
{
	_absoluteSizeCriteria = pFlag;

	// Update DEVICE memory with "voxel scale"
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cAbsoluteSizeCriteria, &_absoluteSizeCriteria, sizeof( _absoluteSizeCriteria ), 0, cudaMemcpyHostToDevice ) );

	// Clear the cache
	clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool SampleCore::hasFixedSizeSphere() const
{
	return _fixedSizeSphere;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setFixedSizeSphere( bool pFlag )
{
    _fixedSizeSphere = pFlag;

    // Update DEVICE memory with "voxel scale"
    GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cFixedSizeSphere, &_fixedSizeSphere, sizeof( _fixedSizeSphere ), 0, cudaMemcpyHostToDevice ) );

    // Clear the cache
    clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
float SampleCore::getFixedSizeSphereRadius() const
{
    return _pipeline->getProducer()->getFixedSizeSphereRadius();
}

/******************************************************************************
 * ...
 ******************************************************************************/
unsigned int SampleCore::getSphereDiameterCoeff() const
{
    return _sphereDiameterCoeff;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setSphereDiameterCoeff( unsigned int pValue )
{
    _sphereDiameterCoeff = pValue;

    // Update DEVICE memory with "voxel scale"
    GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cCoeffAbsoluteSizeCriteria, &_sphereDiameterCoeff, sizeof( _sphereDiameterCoeff ), 0, cudaMemcpyHostToDevice ) );

    // Clear the cache
    clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
unsigned int SampleCore::getScreenSpaceCoeff() const
{
    return _screenSpaceCoeff;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setScreenSpaceCoeff( unsigned int pValue )
{
    _screenSpaceCoeff = pValue;

    // Update DEVICE memory with "voxel scale"
    GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cScreenSpaceCoeff, &_screenSpaceCoeff, sizeof( _screenSpaceCoeff ), 0, cudaMemcpyHostToDevice ) );

    // Clear the cache
    clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setFixedSizeSphereRadius( float pValue )
{
	_pipeline->editProducer()->setFixedSizeSphereRadius( pValue );

	// Clear the cache
	clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool SampleCore::hasMeanSizeOfSpheres() const
{
	return _meanSizeOfSpheres;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setMeanSizeOfSpheres( bool pFlag )
{
	_meanSizeOfSpheres = pFlag;

	// Update DEVICE memory with "voxel scale"
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cMeanSizeOfSpheres, &_meanSizeOfSpheres, sizeof( _meanSizeOfSpheres ), 0, cudaMemcpyHostToDevice ) );

	// Clear the cache
	clearCache();
}

/******************************************************************************
 * ...alpha
 ******************************************************************************/
bool SampleCore::hasShaderUniformColor() const
{
	return _shaderUseUniformColor;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setShaderUniformColorMode( bool pFlag )
{
	_shaderUseUniformColor = pFlag;

	// Update DEVICE memory with "voxel scale"
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cShaderUseUniformColor, &_shaderUseUniformColor, sizeof( _shaderUseUniformColor ), 0, cudaMemcpyHostToDevice ) );

	// Clear the cache
	clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
const float4& SampleCore::getShaderUniformColor() const
{
	return _shaderUniformColor;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setShaderUniformColor( float pR, float pG, float pB, float pA )
{
	_shaderUniformColor = make_float4( pR, pG, pB, pA );

	// Update DEVICE memory with "voxel scale"
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cShaderUniformColor, &_shaderUniformColor, sizeof( _shaderUniformColor ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool SampleCore::hasShaderAnimation() const
{
	return _shaderAnimation;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setShaderAnimation( bool pFlag )
{
	_shaderAnimation = pFlag;

	// Update DEVICE memory with "voxel scale"
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cShaderAnimation, &_shaderAnimation, sizeof( _shaderAnimation ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool SampleCore::hasShaderBlurSphere() const
{
    return _shaderBlurSphere;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setShaderBlurSphere( bool pFlag )
{
    _shaderBlurSphere = pFlag;

    // Update DEVICE memory with "voxel scale"
    GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cShaderBlurSphere, &_shaderBlurSphere, sizeof( _shaderBlurSphere ), 0, cudaMemcpyHostToDevice ) );
}


/******************************************************************************
 * ...
 ******************************************************************************/
bool SampleCore::hasShaderFog() const
{
    return _shaderFog;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setShaderFog( bool pFlag )
{
    _shaderFog = pFlag;

    // Update DEVICE memory with "voxel scale"
    GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cShaderFog, &_shaderFog, sizeof( _shaderFog ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * ...
 ******************************************************************************/
float SampleCore::getFogDensity() const
{

    return _shaderFogDensity;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setFogDensity( float pValue )
{
    _shaderFogDensity = pValue;

    // Update DEVICE memory with "voxel scale"
    GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cShaderFogDensity, &_shaderFogDensity, sizeof( _shaderFogDensity ), 0, cudaMemcpyHostToDevice ) );

}

/******************************************************************************
 * ...
 ******************************************************************************/
const float4& SampleCore::getShaderFogColor() const
{
    return _shaderFogColor;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setShaderFogColor( float pR, float pG, float pB, float pA )
{
    _shaderFogColor = make_float4( pR, pG, pB, pA );

    setClearColor(pR*255, pG*255, pB*255, pA*255);
    //_pipeline->editRenderer()setClearColor(make_uchar4(pR*255, pG*255, pB*255, pA*255));

    // Update DEVICE memory with "voxel scale"
    GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cShaderFogColor, &_shaderFogColor, sizeof( _shaderFogColor ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool SampleCore::IsLightSourceType() const
{
    return _shaderLightSourceType;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setLightSourceType( bool pFlag )
{
    _shaderLightSourceType = pFlag;

    // Update DEVICE memory with "voxel scale"
    GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cShaderLightSourceType, &_shaderLightSourceType, sizeof( _shaderLightSourceType ), 0, cudaMemcpyHostToDevice ) );

}

/******************************************************************************
 * ...
 ******************************************************************************/
bool SampleCore::hasShading() const
{
    return _shaderFog;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setShading( bool pFlag )
{
    _shading = pFlag;

    // Update DEVICE memory with "voxel scale"
    GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cShading, &_shading, sizeof( _shading ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool SampleCore::hasBugCorrection() const
{
    return _bugCorrection;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setBugCorrection( bool pFlag )
{
    _bugCorrection = pFlag;

    // Update DEVICE memory with "voxel scale"
    GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cShaderBugCorrection, &_bugCorrection, sizeof( _bugCorrection ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * ...
 ******************************************************************************/
float SampleCore::getIlluminationCoeff() const
{
    return _illuminationCoeff;//_pipeline->getProducer()->getSphereRadiusFader();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setIlluminationCoeff( float pValue )
{
    _illuminationCoeff = pValue;

    // Update DEVICE memory with "voxel scale"
    GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cSphereIlluminationCoeff, &_illuminationCoeff, sizeof( _illuminationCoeff ), 0, cudaMemcpyHostToDevice ) );
}
