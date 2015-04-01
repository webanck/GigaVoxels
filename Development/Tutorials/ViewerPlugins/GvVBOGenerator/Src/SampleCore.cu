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
#include <GvRendering/GvGraphicsInteroperabiltyHandler.h>
#include <GvPerfMon/GvPerformanceMonitor.h>
#include <GvCore/GvError.h>
//#include <GvUtils/GvProxyGeometryHandler.h>
#include <GvCore/GvVersion.h>
#include <GsCompute/GsDeviceManager.h>
#include <GvUtils/GvSimplePipeline.h>

// Project
#include "Producer.h"
#include "Shader.h"
#include "VolumeTreeRendererCUDA.h"
#include "VolumeTreeCache.h"
#include "ParticleSystem.h"

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
using namespace GvUtils;
using namespace GsGraphics;

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
 * GvKernel_UpdateVBO kernel
 *
 * This kernel update the VBO by dumping all used bricks content (i.e. points)
 *
 * @param pVBO VBO to update
 * @param pNbBricks number of bricks to process
 * @param pBrickAddressList list of brick addresses in cache (used to retrive positions where to fetch data)
 * @param pNbPointsList list of points inside each brick
 * @param pVboIndexOffsetList list of number of points for each used bricks
 * @param pDataStructure data structure in cache where to fecth data
 ******************************************************************************/
__global__
void KERNEL_UpdateVBO( float3* pVBO, const uint pNbPoints, const unsigned int nbFrame )
{
	// Retrieve global data index
	uint lineSize = __uimul( blockDim.x, gridDim.x );
	uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );

	// Check bounds
	if ( elem < pNbPoints )
	{
		const float x = 0.5f + 0.5f * cosf( nbFrame * 0.001f * 2.f * 3.141592f * ( (float)elem / (float)pNbPoints ) );
		const float y = 0.5f + 0.5f * sinf( nbFrame * 0.0001f * 2.f * 3.141592f * ( (float)elem / (float)pNbPoints ) );
		const float z = 0.5f + 0.5f * cosf( nbFrame * 0.005f * 2.f * 3.141592f * ( (float)elem / (float)pNbPoints ) );

		// Write to output global memory
		pVBO[ elem ] = make_float3( x, y, z );
	}
}

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
//,	_vbo( NULL )
{
	// Translation used to position the GigaVoxels data structure
	_translation[ 0 ] = -0.5f;
	_translation[ 1 ] = -0.5f;
	_translation[ 2 ] = -0.5f;

	// Light position
	_lightPosition = make_float3( 1.f, 1.f, 1.f );

	// Spheres ray-tracing parameters
	_nbPoints = 0;
	_userDefinedMinLevelOfResolutionToHandle = 0;
	_sphereBrickIntersectionType = 0;
	_geometricCriteria = true;
	_minNbPointsPerBrick = 1;
	_geometricCriteriaGlobalUsage = false;
	_apparentMinSizeCriteria = true;
	_apparentMinSize = 0.0f;
	_apparentMaxSizeCriteria = true;
	_apparentMaxSize = 0.0f;
	_shaderUseUniformColor = false;
	_shaderUniformColor = make_float4( 1.f, 1.f, 1.f, 1.f );
	_shaderAnimation = false;
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
	delete _pipeline;

	//delete _vbo;
	//delete _particleSystem;

	// CUDA tip: clean up to ensure correct profiling
	cudaError_t error = cudaDeviceReset();
}

/******************************************************************************
 * Gets the name of this browsable
 *
 * @return the name of this browsable
 ******************************************************************************/
const char* SampleCore::getName() const
{
	return "VBOGenerator";
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

	//---------------------------------------
	// TEST to avoid crash
	cudaDeviceSynchronize();
	//---------------------------------------

	// GigaVoxels API's version
	std::cout << "GigaVoxels API's version : " << GvCore::GvVersion::getVersion() << std::endl;
	// Test client architecture
	// If harware is compliant with le GigaVoxels Engine, launch the demo
	if ( GsCompute::GsDeviceManager::get().initialize() )
	{
	}
	else
	{
		std::cout << "\nThe program will now exit" << std::endl;
	}
	// Release memory
	//GsDeviceManager::get().finalize();

	// Compute the size of one element in the cache for nodes and bricks
	size_t nodeElemSize = NodeRes::numElements * sizeof( GvStructure::GvNode );
	size_t brickElemSize = RealBrickRes::numElements * GvCore::DataTotalChannelSize< DataType >::value;

	// Compute how many we can fit into the given memory size
	size_t nodePoolNumElems = NODEPOOL_MEMSIZE / nodeElemSize;
	size_t brickPoolNumElems = BRICKPOOL_MEMSIZE / brickElemSize;

	// Compute the resolution of the pools
	uint3 nodePoolRes = make_uint3( static_cast< uint >( floorf( powf( static_cast< float >( nodePoolNumElems ), 1.0f / 3.0f ) ) ) ) * NodeRes::get();
	uint3 brickPoolRes = make_uint3( static_cast< uint >( floorf( powf( static_cast< float >( brickPoolNumElems ), 1.0f / 3.0f ) ) ) ) * RealBrickRes::get();
	
	std::cout << "\nnodePoolRes: " << nodePoolRes << std::endl;
	std::cout << "brickPoolRes: " << brickPoolRes << std::endl;

	// Pipeline creation
	_pipeline = new PipelineType();
	ProducerType* producer = new ProducerType();
	ShaderType* shader = new ShaderType();

	// Pipeline initialization
	_pipeline->initialize( NODEPOOL_MEMSIZE, BRICKPOOL_MEMSIZE, producer, shader );

	// VBO
	//_vbo = new GvUtils::GvProxyGeometryHandler();
	//_vbo->initialize();

	// Cache initialization
	//_cache = new VolumeTreeCacheType( _volumeTree, nodePoolRes, brickPoolRes/*, _vbo*/ );

	// Producer initialization
	//_producer = new ProducerType();
	//_producer->initialize( _volumeTree, _cache );
	//_cache->addProducer( static_cast< VolumeTreeCacheType::ProducerType* >( _producer ) );

	// Renderer initialization
	//_renderer = new VolumeTreeRendererType( _volumeTree, _cache );

	// Configure cache
	_pipeline->editCache()->setMaxNbNodeSubdivisions( 500 );
	_pipeline->editCache()->setMaxNbBrickLoads( 300 );
	_pipeline->editCache()->editNodesCacheManager()->setPolicy( VolumeTreeCacheType::NodesCacheManager::eAllPolicies );
	_pipeline->editCache()->editBricksCacheManager()->setPolicy( VolumeTreeCacheType::BricksCacheManager::eAllPolicies );

	// TEST VBO
	//_vbo = _cache->_vbo;//_cache->getVBO();
	// Points definissant l'interval du cube
	//const float3 p1 = make_float3( 0.f, 0.f, 0.f );
	//const float3 p2 = make_float3( 1.f, 1.f, 1.f );
	//_particleSystem = new ParticleSystem( p1, p2 );
	_particleSystem  = _pipeline->editProducer()->_particleSystem;
	_pipeline->editCache()->editVBOCacheManager()->_particleSystem = _pipeline->editProducer()->_particleSystem;

	// Custom initialization
	// Note : this could be done via an XML settings file loaded at initialization
	//setNbPoints( 1 );
	setNbPoints( 998 );
	setUserDefinedMinLevelOfResolutionToHandle( 0 );
	// Sphere-brick intersection type
	//
	// 0 : sphere-sphere (brick are approximated by spheres)
	// 1 : sphere-box (brick are not approximated, it uses real sphere-box intersection test)
	setSphereBrickIntersectionType( 1 );
	setPointSizeFader( 1.0f );
	setGeometricCriteria( false );
	setMinNbPointsPerBrick( 1 );
	setApparentMinSizeCriteria( false );
	setApparentMinSize( 1.0f );
	setApparentMaxSizeCriteria( false );
	setApparentMaxSize( 1.0f );
	setShaderUniformColorMode( false );
	setShaderUniformColor( 1.f, 1.f, 1.f, 1.f );
	setShaderAnimation( false );

	// Max depth
	//setRendererMaxDepth( 0 );
	_maxVolTreeDepth = 0;
	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );

	// Set light position
	setLightPosition( 1.f, 1.f, 1.f );
}

/******************************************************************************
 * Draw function called of frame
 ******************************************************************************/
void SampleCore::draw()
{
	//static unsigned int nbFrame = 0;

	////-------------------------------------------------------------------------------------------------------------
	//// Update VBO content
	//// Map graphics resource
	//cudaStream_t stream = 0;
	//cudaError_t error = cudaGraphicsMapResources( 1, &_vbo->_d_vertices, stream );
	//assert( error == cudaSuccess );
	//// Get graphics resource's mapped address
	//float3* vboDevicePointer = NULL;
	//size_t size = 0;
	//error = cudaGraphicsResourceGetMappedPointer( (void**)&vboDevicePointer, &size, _vbo->_d_vertices );
	//assert( error == cudaSuccess );
	//// Update VBO
	//dim3 gridSize( 1, 1, 1 );
	//dim3 blockSize( 1024, 1, 1 );
	//KERNEL_UpdateVBO<<< gridSize, blockSize >>>( vboDevicePointer, 1000, nbFrame );
	//GV_CHECK_CUDA_ERROR( "KERNEL_UpdateVBO" );
	//// Unmap graphics resource
	//error = cudaGraphicsUnmapResources( 1, &_vbo->_d_vertices, stream );
	//assert( error == cudaSuccess );
	//// Render VBO
	//_vbo->_nbPoints = 1000;
	//_vbo->render();
	//// Update frame nb
	//nbFrame++;
	////-------------------------------------------------------------------------------------------------------------

	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT( frame );
	CUDAPM_START_EVENT( app_init_frame );

	////------------------------------------------------------------------
	//// Display the GigaVoxels N3-tree space partitioning structure
	//glEnable( GL_DEPTH_TEST );
	//glPushMatrix();
	//// Translation used to position the GigaVoxels data structure
	////glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );
	//_volumeTree->render();
	//glPopMatrix();
	//glDisable( GL_DEPTH_TEST );

	//// Renderer VBO
	//glEnable( GL_DEPTH_TEST );
	//glPushMatrix();
	//_vbo->render();
	//glPopMatrix();
	//glDisable( GL_DEPTH_TEST );

	//// TEST
	//glColor3f( 0.f, 1.f, 0.f );
	//glPointSize( 5.0f );
	//glEnable( GL_DEPTH_TEST );
	//glBegin( GL_POINTS );
	//glVertex3f( 0.5f, 0.5f, 0.5f );
	//glEnd();
	//glDisable( GL_DEPTH_TEST );
	//------------------------------------------------------------------

//	glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );

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

		//// Clear the depth PBO (pixel buffer object) by reading from the previously cleared FBO (frame buffer object)
		//glBindBuffer( GL_PIXEL_PACK_BUFFER, _depthBuffer );
		//glReadPixels( 0, 0, _width, _height, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0 );
		//glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
		//GV_CHECK_GL_ERROR();
	}
	else
	{
		glClear( GL_COLOR_BUFFER_BIT );
	}

//	glBindFramebuffer( GL_FRAMEBUFFER, 0 );

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

	// Renderer VBO
	//glEnable( GL_DEPTH_TEST );
	//glPushMatrix();
	//_vbo->render();
	_particleSystem->render( viewMatrix, projectionMatrix, viewport );
	//glPopMatrix();
	//glDisable( GL_DEPTH_TEST );

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
	GV_CHECK_CUDA_ERROR( "SampleCore::resize() + Disconnect all registered graphics resources" );
	
	// ...
	if (_depthBuffer)
	{
		glDeleteBuffers(1, &_depthBuffer);
		GV_CHECK_GL_ERROR();
	}

	if (_colorTex)
	{
		glDeleteTextures(1, &_colorTex);
		GV_CHECK_GL_ERROR();
	}
	if (_depthTex)
	{
		glDeleteTextures(1, &_depthTex);
		GV_CHECK_GL_ERROR();
	}

	if (_frameBuffer)
	{
		glDeleteFramebuffers(1, &_frameBuffer);
		GV_CHECK_GL_ERROR();
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
		//_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
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
	//	_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
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
//	_lightPosition.x = pX/* - _translation[ 0 ]*/;
//	_lightPosition.y = pY/* - _translation[ 1 ]*/;
//	_lightPosition.z = pZ/* - _translation[ 2 ]*/;
//
//	// Update device memory
//	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cLightPosition, &_lightPosition, sizeof( _lightPosition ), 0, cudaMemcpyHostToDevice ) );
//}

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
	float3 lightPosition = make_float3( pX, pY, pZ );
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cLightPosition, &lightPosition, sizeof( lightPosition ), 0, cudaMemcpyHostToDevice ) );
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
	_pipeline->editCache()->editNodesCacheManager()->setPolicy( static_cast< VolumeTreeCacheType::NodesCacheManager::ECachePolicy>( pValue ) );
	_pipeline->editCache()->editBricksCacheManager()->setPolicy( static_cast< VolumeTreeCacheType::BricksCacheManager::ECachePolicy>( pValue ) );
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
unsigned int SampleCore::getNbPoints() const
{
	return _nbPoints;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setNbPoints( unsigned int pValue )
{
	_nbPoints = pValue;

	// Update producer
	_pipeline->editProducer()->setNbPoints( pValue );
	
	// Update DEVICE memory with "voxel scale"
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cNbPoints, &_nbPoints, sizeof( _nbPoints ), 0, cudaMemcpyHostToDevice ) );

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
float SampleCore::getPointSizeFader() const
{
	return _pipeline->getProducer()->getPointSizeFader();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setPointSizeFader( float pValue )
{
	_pipeline->editProducer()->setPointSizeFader( pValue );

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
unsigned int SampleCore::getMinNbPointsPerBrick() const
{
	return _minNbPointsPerBrick;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setMinNbPointsPerBrick( unsigned int pValue )
{
	_minNbPointsPerBrick = pValue;

	// Update DEVICE memory with "voxel scale"
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cMinNbPointsPerBrick, &_minNbPointsPerBrick, sizeof( _minNbPointsPerBrick ), 0, cudaMemcpyHostToDevice ) );

	// Clear the cache
	clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool SampleCore::hasGeometricCriteriaGlobalUsage() const
{
	return _geometricCriteriaGlobalUsage;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setGeometricCriteriaGlobalUsage( bool pFlag )
{
	_geometricCriteriaGlobalUsage = pFlag;

	// Update DEVICE memory with "voxel scale"
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cGeometricCriteriaGlobalUsage, &_geometricCriteriaGlobalUsage, sizeof( _geometricCriteriaGlobalUsage ), 0, cudaMemcpyHostToDevice ) );

	// Clear the cache
	clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool SampleCore::hasApparentMinSizeCriteria() const
{
	return _apparentMinSizeCriteria;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setApparentMinSizeCriteria( bool pFlag )
{
	_apparentMinSizeCriteria = pFlag;

	// Update DEVICE memory with "voxel scale"
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cApparentMinSizeCriteria, &_apparentMinSizeCriteria, sizeof( _apparentMinSizeCriteria ), 0, cudaMemcpyHostToDevice ) );

	// Clear the cache
	//clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
float SampleCore::getApparentMinSize() const
{
	return _apparentMinSize;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setApparentMinSize( float pValue )
{
	_apparentMinSize = pValue;

	// Update DEVICE memory with "voxel scale"
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cApparentMinSize, &_apparentMinSize, sizeof( _apparentMinSize ), 0, cudaMemcpyHostToDevice ) );

	// Clear the cache
	//clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool SampleCore::hasApparentMaxSizeCriteria() const
{
	return _apparentMaxSizeCriteria;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setApparentMaxSizeCriteria( bool pFlag )
{
	_apparentMaxSizeCriteria = pFlag;

	// Update DEVICE memory with "voxel scale"
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cApparentMaxSizeCriteria, &_apparentMaxSizeCriteria, sizeof( _apparentMaxSizeCriteria ), 0, cudaMemcpyHostToDevice ) );

	// Clear the cache
	clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
float SampleCore::getApparentMaxSize() const
{
	return _apparentMaxSize;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setApparentMaxSize( float pValue )
{
	_apparentMaxSize = pValue;

	// Update DEVICE memory with "voxel scale"
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cApparentMaxSize, &_apparentMaxSize, sizeof( _apparentMaxSize ), 0, cudaMemcpyHostToDevice ) );

	// Clear the cache
	clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool SampleCore::hasShaderUniformColor() const
{
	//return _shaderUseUniformColor;

	return _particleSystem->hasShaderUniformColor();
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

	_particleSystem->setShaderUniformColorMode( pFlag );
}

/******************************************************************************
 * ...
 ******************************************************************************/
const float4& SampleCore::getShaderUniformColor() const
{
	//return _shaderUniformColor;
	return _particleSystem->getShaderUniformColor();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setShaderUniformColor( float pR, float pG, float pB, float pA )
{
	_shaderUniformColor = make_float4( pR, pG, pB, pA );

	// Update DEVICE memory with "voxel scale"
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cShaderUniformColor, &_shaderUniformColor, sizeof( _shaderUniformColor ), 0, cudaMemcpyHostToDevice ) );

	_particleSystem->setShaderUniformColor( pR, pG, pB, pA );
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool SampleCore::hasShaderAnimation() const
{
	//return _shaderAnimation;

	return _particleSystem->hasShaderAnimation();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setShaderAnimation( bool pFlag )
{
	_shaderAnimation = pFlag;

	// Update DEVICE memory with "voxel scale"
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cShaderAnimation, &_shaderAnimation, sizeof( _shaderAnimation ), 0, cudaMemcpyHostToDevice ) );

	_particleSystem->setShaderAnimation( pFlag );
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool SampleCore::hasTexture() const
{
	//return _hasTexture;

	return _particleSystem->hasTexture();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setTexture( bool pFlag )
{
	_hasTexture = pFlag;

	// GLSL uniform

	// Update DEVICE memory with "voxel scale"
	//GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cHasTexture, &_hasTexture, sizeof( _hasTexture ), 0, cudaMemcpyHostToDevice ) );

	_particleSystem->setTexture( pFlag );
}

/******************************************************************************
 * ...
 ******************************************************************************/
const std::string& SampleCore::getTextureFilename() const
{
	//return _textureFilename;

	return _particleSystem->getTextureFilename();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::setTextureFilename( const std::string& pFilename )
{
	_textureFilename = pFilename;

	_particleSystem->setTextureFilename( pFilename );
}

/******************************************************************************
 * Tell wheter or not pipeline uses programmable shaders
 *
 * @return a flag telling wheter or not pipeline uses programmable shaders
 ******************************************************************************/
bool SampleCore::hasProgrammableShaders() const
{
	return true;
}

/******************************************************************************
 * Tell wheter or not pipeline has a given type of shader
 *
 * @param pShaderType the type of shader to test
 *
 * @return a flag telling wheter or not pipeline has a given type of shader
 ******************************************************************************/
bool SampleCore::hasShaderType( unsigned int pShaderType ) const
{
	//return _shaderProgram->hasShaderType( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
	return false;
}

/******************************************************************************
 * Get the source code associated to a given type of shader
 *
 * @param pShaderType the type of shader
 *
 * @return the associated shader source code
 ******************************************************************************/
std::string SampleCore::getShaderSourceCode( unsigned int pShaderType ) const
{
	//return _shaderProgram->getShaderSourceCode( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
	return false;
}

/******************************************************************************
 * Get the filename associated to a given type of shader
 *
 * @param pShaderType the type of shader
 *
 * @return the associated shader filename
 ******************************************************************************/
std::string SampleCore::getShaderFilename( unsigned int pShaderType ) const
{
	//return _shaderProgram->getShaderFilename( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
	return false;
}

/******************************************************************************
 * ...
 *
 * @param pShaderType the type of shader
 *
 * @return ...
 ******************************************************************************/
bool SampleCore::reloadShader( unsigned int pShaderType )
{
	//return _shaderProgram->reloadShader( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
	return false;
}
