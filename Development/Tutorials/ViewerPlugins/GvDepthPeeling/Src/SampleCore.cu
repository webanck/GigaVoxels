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
#include <GvUtils/GvSimpleHostShader.h>
#include <GvUtils/GvCommonGraphicsPass.h>
#include <GvCore/GvError.h>
#include <GvPerfMon/GvPerformanceMonitor.h>
#include <GvUtils/GvShaderManager.h>

// Project
#include "ProducerKernel.h"
#include "ShaderKernel.h"
#include "Renderer.h"
#include "DepthPeeling.h"
#include "Mesh.h"

// GvViewer
#include <GvvApplication.h>
#include <GvvMainWindow.h>

// Cuda SDK
#include <helper_math.h>

// assimp
#include <assimp.h>
#include <aiScene.h>
#include <aiPostProcess.h>

// System
#include <cfloat>
#include <limits>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvRenderer;
using namespace GvUtils;

// GigaVoxels viewer
using namespace GvViewerCore;

// STL
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

// Defines the size allowed for each type of pool
#define NODEPOOL_MEMSIZE	( 8U * 1024U * 1024U )		// 8 Mo
#define BRICKPOOL_MEMSIZE	( 256U * 1024U * 1024U )	// 256 Mo

/**
 * Assimp library object to load 3D model (with a log mechanism)
 */
static struct aiLogStream stream;

#define BUFFER_OFFSET(i) ((void*)(i))

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

struct MyVertex
{
    float x, y, z;        //Vertex
    float nx, ny, nz;     //Normal
};

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
SampleCore::SampleCore()
:	GvvPipelineInterface()
,	_pipeline( NULL )
,	_graphicsEnvironment( NULL )
,	_displayOctree( false )
,	_displayPerfmon( 0 )
,	_maxVolTreeDepth( 0 )
,	_depthBuffer( 0 )
,	_colorTex( 0 )
,	_depthTex( 0 )
,	_frameBuffer( 0 )
,	_width( 0 )
,	_height( 0 )
,	_scene( NULL )
,	_filename()
,	_nbDepthPeelingLayers( 0 )
,	_shaderOpacity( 0.f )
,	_shaderMaterialOpacityProperty( 0.f )
,	_depthPeelingDepthMinResource( NULL )
,	_depthPeelingDepthMaxResource( NULL )
,	_depthPeelingInitProgram( 0 )
,	_depthPeelingCoreProgram( 0 )
,   _skybox()
,	_frontToBackPeeling( NULL )
,	_mesh( NULL )
{
	// Translation used to position the GigaVoxels data structure
	_translation[ 0 ] = -0.5f;
	_translation[ 1 ] = -0.5f;
	_translation[ 2 ] = -0.5f;

	// Light position
	_lightPosition = make_float3( 1.f, 1.f, 1.f );

	// Noise parameter(s)
	_hasHyperTexture = false;
	_noiseFirstFrequency = 0.f;
	_noiseStrength = 0.f;
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
	// Disconnect all registered graphics resources
	_pipeline->editRenderer()->resetGraphicsResources();

	// Delete the GigaVoxels pipeline
	delete _pipeline;

	// CUDA tip: clean up to ensure correct profiling
	//cudaError_t error = cudaDeviceReset();

	// Clean Assimp library ressources
	if ( _scene != NULL )
	{
		aiReleaseImport( _scene );
	}
}

/******************************************************************************
 * Gets the name of this browsable
 *
 * @return the name of this browsable
 ******************************************************************************/
const char* SampleCore::getName() const
{
	return "DepthPeeling";
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

	// Pipeline creation
	_pipeline = new PipelineType();
	ProducerType* producer = new ProducerType();
	ShaderType* shader = new ShaderType();

	// Pipeline initialization
    _pipeline->initialize( NODEPOOL_MEMSIZE, BRICKPOOL_MEMSIZE, producer, shader );

    // Pipeline configuration
	_pipeline->editDataStructure()->setMaxDepth( 5 );

	// Configure the Cache Management System
	_pipeline->editCache()->setMaxNbNodeSubdivisions( 500 );
	_pipeline->editCache()->setMaxNbBrickLoads( 300 );
	_pipeline->editCache()->editNodesCacheManager()->setPolicy( CacheType::NodesCacheManager::eAllPolicies );
	_pipeline->editCache()->editBricksCacheManager()->setPolicy( CacheType::BricksCacheManager::eAllPolicies );

	// Graphics environment creation
	_graphicsEnvironment = new GvCommonGraphicsPass();

	// Initialize the proxy geometry loader
	stream = aiGetPredefinedLogStream( aiDefaultLogStream_STDOUT, NULL );
	aiAttachLogStream( &stream );

	// Depth Peeling configuration
	setDepthPeelingNbLayers( 1 );

	// Data repository
	QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );

	// Load the proxy geometry
	//
	// @todo : check for file availability
	QString model3D = dataRepository + QDir::separator() + QString( "3DModels" ) + QDir::separator() + QString( "bunny.obj" );
	_filename = model3D.toLatin1().constData();
	set3DModelFilename( _filename );

	// Shaders management
	//
	// @todo : check for file availability
	QString vertexShaderFilename;
	QString fragmentShaderFilename;
	// Depth Peeling's initialization shader program
	vertexShaderFilename = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "GvDepthPeeling" ) + QDir::separator() + QString( "front_peeling_init_vertex.glsl" );
	fragmentShaderFilename = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "GvDepthPeeling" ) + QDir::separator() + QString( "front_peeling_init_fragment.glsl" );
	_depthPeelingInitProgram = GvUtils::GvShaderManager::createShaderProgram( vertexShaderFilename.toLatin1().constData(), NULL, fragmentShaderFilename.toLatin1().constData() );
	GvUtils::GvShaderManager::linkShaderProgram( _depthPeelingInitProgram );
	// Depth Peeling's core shader program
	vertexShaderFilename = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "GvDepthPeeling" ) + QDir::separator() + QString( "front_peeling_peel_vertex.glsl" );
	fragmentShaderFilename = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "GvDepthPeeling" ) + QDir::separator() + QString( "front_peeling_peel_fragment.glsl" );
	_depthPeelingCoreProgram = GvUtils::GvShaderManager::createShaderProgram( vertexShaderFilename.toLatin1().constData(), NULL, fragmentShaderFilename.toLatin1().constData() );
	GvUtils::GvShaderManager::linkShaderProgram( _depthPeelingCoreProgram );

	// Shader parameters
	setShaderOpacity( 0.5f );
	setShaderMaterialOpacityProperty( 1.f / 256.f );
	
	// Hypertexture initialization
	setHyperTexture( false );
	setNoiseFirstFrequency( 32.f );
	setNoiseStrength( 1.f );
	
    // Skybox initialization
    _skybox.init();

	// Front to back peeling
	_frontToBackPeeling = new DepthPeeling();

	// Mesh
	_mesh = new Mesh();
	_mesh->initialize();
}

/******************************************************************************
 * Draw function called of frame
 ******************************************************************************/
void SampleCore::draw()
{
	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT( frame );
	CUDAPM_START_EVENT( app_init_frame );

	// Reset glClearColor value because Depth Peeling management modifies it
	const float ucharToFloatColor = 1.f / 255.f;
	const uchar4& gvClearColor = _pipeline->editRenderer()->getClearColor();
	const float4 clearColor = make_float4( gvClearColor.x * ucharToFloatColor, gvClearColor.y * ucharToFloatColor, gvClearColor.z * ucharToFloatColor, gvClearColor.w * ucharToFloatColor );
	glClearColor( clearColor.x, clearColor.y, clearColor.z, 0.f );
	glClearDepth( 1.f );

	glMatrixMode( GL_MODELVIEW );

	glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );

	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );

	// Display the GigaVoxels N3-tree space partitioning structure
	glEnable( GL_DEPTH_TEST );
	glPushMatrix();

	// Translation used to position the GigaVoxels data structure
	glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );

	// Data Structure
	if ( _displayOctree )
	{
		_pipeline->editDataStructure()->displayDebugOctree();
	}

	// Proxy Geometry
	//drawProxyRecursive( _scene, _scene->mRootNode );

	// Sky Box
	//_skybox.draw();
	
	glPopMatrix();
	glDisable( GL_DEPTH_TEST );

	// Clear the depth PBO (pixel buffer object) by reading from the previously cleared FBO (frame buffer object)
	glBindBuffer( GL_PIXEL_PACK_BUFFER, _depthBuffer );
	glReadPixels( 0, 0, _width, _height, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0 );
	glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
	GV_CHECK_GL_ERROR();

	glBindFramebuffer( GL_FRAMEBUFFER, 0 );
	
    // Extract view transformations
	float4x4 viewMatrix;
	float4x4 projectionMatrix;
	glGetFloatv( GL_MODELVIEW_MATRIX, viewMatrix._array );
	glGetFloatv( GL_PROJECTION_MATRIX, projectionMatrix._array );

	// Extract viewport
	GLint params[ 4 ];
	glGetIntegerv( GL_VIEWPORT, params );
	int4 viewport = make_int4( params[ 0 ], params[ 1 ], params[ 2 ], params[ 3 ] );

	// render the scene into textures
	CUDAPM_STOP_EVENT( app_init_frame );

	// Build the world transformation matrix
	float4x4 modelMatrix;
	glPushMatrix();
	glLoadIdentity();
	// Translation used to position the GigaVoxels data structure
	glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );
	glGetFloatv( GL_MODELVIEW_MATRIX, modelMatrix._array );
	glPopMatrix();

	// -------- Depth Peeling - INIT stage [ BEGIN ] --------
	//
	//
	
	glPushMatrix();
	glTranslatef( -0.5f, -0.5f, -0.5f );

    glEnable( GL_DEPTH_TEST );

	// Init
    glBindFramebuffer( GL_FRAMEBUFFER, _depthPeelingFBO[ 0 ] );
	
	glDrawBuffer( GL_COLOR_ATTACHMENT0 );

    glClearColor( 0.f, 0.f, 0.f, 0.f );
    glClearDepth( 1.f );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	// Proxy Geometry
	glUseProgram( _depthPeelingInitProgram );
    drawScene();//ProxyRecursive( _scene, _scene->mRootNode );
	glUseProgram( 0 );

    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    glDisable( GL_DEPTH_TEST );

    glPopMatrix();

	// -------- Depth Peeling - INIT stage [ END ] --------

	// -------- Depth Peeling - CORE stage [ BEGIN ] --------
	//
	//

    glPushMatrix();
    glTranslatef( -0.5f, -0.5f, -0.5f );

	// Iterate through depth peeling's layers
	for ( unsigned int i = 0; i < _nbDepthPeelingLayers; i++ )
	{
		// ----  1st pass ----

        glEnable( GL_DEPTH_TEST );

		glBindFramebuffer( GL_FRAMEBUFFER, _depthPeelingFBO[ 1 ] );
		
		glDrawBuffer( GL_COLOR_ATTACHMENT0 );

        glClearColor( 0.f, 0.f, 0.f, 0.f );
        glClearDepth( 1.f );
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

		glUseProgram( _depthPeelingCoreProgram );

        glEnable( GL_TEXTURE_RECTANGLE_EXT );
        glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _depthPeelingTex[ 0 ] );
		GLint texId = glGetUniformLocation( _depthPeelingCoreProgram, "DepthTex" );
		glUniform1i( texId, 0 );
        GLint DepId = glGetUniformLocation( _depthPeelingCoreProgram, "DepthDep" );
        //glUniform1i( DepId, 0 );
		glUniform1i( DepId, 1 );

        //drawProxyRecursive( _scene, _scene->mRootNode );
        drawScene();
		
		glUseProgram( 0 );
		
        glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
        glDisable( GL_TEXTURE_RECTANGLE_EXT );

        glBindFramebuffer( GL_FRAMEBUFFER, 0 );

		glDisable( GL_DEPTH_TEST );

		// ----  2nd pass ----
		//
		// GigaVoxels rendering stage
		
		// Enable 8 colors max
		const float3 color = make_float3( ( ( i & 0x00000004) >> 2 ), ( ( i & 0x00000002 ) >> 1 ), ( i & 0x00000001 ) );
		GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cColor, &color, sizeof( color ), 0, cudaMemcpyHostToDevice ) );
		
		// GigaVoxels rendering
        _pipeline->execute( modelMatrix, viewMatrix, projectionMatrix, viewport );
		
		// ----  3nd pass ----

		if ( i < ( _nbDepthPeelingLayers - 1 ) )
		{
			glEnable( GL_DEPTH_TEST );

			glBindFramebuffer( GL_FRAMEBUFFER, _depthPeelingFBO[ 0 ] );

			glDrawBuffer( GL_COLOR_ATTACHMENT0 );

			glClearColor( 0.f, 0.f, 0.f, 0.f );
			glClearDepth( 1.f );
			glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

			glUseProgram( _depthPeelingCoreProgram );

			glEnable( GL_TEXTURE_RECTANGLE_EXT );
			glActiveTexture( GL_TEXTURE0 );
			glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _depthPeelingTex[ 1 ] );
			/*GLint*/ texId = glGetUniformLocation( _depthPeelingCoreProgram, "DepthTex" );
			glUniform1i( texId, 0 );
            DepId = glGetUniformLocation( _depthPeelingCoreProgram, "DepthDep" );
            //glUniform1i( DepId, 0 );
			glUniform1i( DepId, 1 );

			//drawProxyRecursive( _scene, _scene->mRootNode );
			drawScene();

			glUseProgram( 0 );

			glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
			glDisable( GL_TEXTURE_RECTANGLE_EXT );

			glBindFramebuffer( GL_FRAMEBUFFER, 0 );

			glDisable( GL_DEPTH_TEST );
		}
	}

	glPopMatrix();

	// -------- Depth Peeling - CORE stage [ END ] --------

	// Render the result to the screen
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
    glLoadIdentity();

	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();

	glEnable( GL_TEXTURE_RECTANGLE_EXT );
	glDisable( GL_DEPTH_TEST );

	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _colorTex );
	
	GLint sMin = 0;
	GLint tMin = 0;
	GLint sMax = _width;
	GLint tMax = _height;

	glBegin( GL_QUADS );
		glColor3f(1.0f, 1.0f, 1.0f);
		glTexCoord2i(sMin, tMin); glVertex2i(-1, -1);
		glTexCoord2i(sMax, tMin); glVertex2i( 1, -1);
		glTexCoord2i(sMax, tMax); glVertex2i( 1,  1);
		glTexCoord2i(sMin, tMax); glVertex2i(-1,  1);
	glEnd();

	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );

	glDisable( GL_TEXTURE_RECTANGLE_EXT );

	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();

	// TEST - optimization due to early unmap() graphics resource from GigaVoxels
	//_pipeline->editRenderer()->doPostRender();
	
	// Update GigaVoxels info
	_pipeline->editRenderer()->nextFrame();

	CUDAPM_STOP_EVENT(frame);
	CUDAPM_STOP_FRAME;

    // DEBUG
    glBindFramebuffer( GL_READ_FRAMEBUFFER,_depthPeelingFBO[ 0 ]);
    glBlitFramebuffer( 0,0,_width,_height,
                      0,0,_width/3,_height/3,
                      GL_COLOR_BUFFER_BIT,
                      GL_LINEAR );

    glBindFramebuffer( GL_READ_FRAMEBUFFER,_depthPeelingFBO[ 1 ]);
    glBlitFramebuffer( 0,0,_width,_height,
                      0,_height/3,_width/3,2*_height/3,
                      GL_COLOR_BUFFER_BIT,
                      GL_LINEAR );

	if ( _displayPerfmon )
	{
		GvPerfMon::CUDAPerfMon::getApplicationPerfMon().displayFrameGL( _displayPerfmon - 1 );
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
	// Reset default active frame region for rendering
	_pipeline->editRenderer()->setProjectedBBox( make_uint4( 0, 0, width, height ) );

	// Re-init Perfmon subsystem
	CUDAPM_RESIZE( make_uint2( width, height ) );

	// [ 1 ] - Reset graphics resources

	// Disconnect all registered graphics resources
	_pipeline->editRenderer()->resetGraphicsResources();
	
	// Reset graphics environment
	_graphicsEnvironment->reset( width, height );
	
	// Update internal variables
	_depthBuffer = _graphicsEnvironment->getDepthBuffer();
	_colorTex = _graphicsEnvironment->getColorTexture();
	_depthTex = _graphicsEnvironment->getDepthTexture();
	_frameBuffer = _graphicsEnvironment->getFrameBuffer();
	_width = _graphicsEnvironment->getWidth();
	_height = _graphicsEnvironment->getHeight();

	// [ 2 ] - Connect graphics resources

	// Create CUDA resources from OpenGL objects
	_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
	_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );

	// Depth Peeling resource management
	if ( _depthPeelingTex[ 0 ] != 0 || _depthPeelingTex[ 1 ] != 0 )
	{
		glDeleteTextures( 2, _depthPeelingTex );
	}
	if ( _depthPeelingFBO[ 0 ] != 0 || _depthPeelingFBO[ 1 ] != 0 )
	{
		glDeleteFramebuffers( 2, _depthPeelingFBO );
	}

	glGenFramebuffers( 2, _depthPeelingFBO );
    glGenTextures( 2, _depthPeelingTex );
    glGenTextures( 2, _depthPeelingDep );
	for ( unsigned int i = 0; i < 2; i++ )
	{
		// Depth texture used for depth peeling
		// - it will be attached to FBO as a "color attachment"
		glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _depthPeelingTex[ i ] );
		glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP );
		glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP );
		glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
		glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
		glTexImage2D( GL_TEXTURE_RECTANGLE_EXT, 0, GL_R32F, _width, _height, 0, GL_RED, GL_FLOAT, 0 );
		glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
		GV_CHECK_GL_ERROR();

		// Depth texture used for OpenGL depth-test
		// - it will be attached to FBO as a "depth attachment"
		glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _depthPeelingDep[ i ] );
        glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP );
        glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP );
        glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
		glTexImage2D( GL_TEXTURE_RECTANGLE_EXT, 0, GL_DEPTH_COMPONENT32F, _width, _height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL );
        GV_CHECK_GL_ERROR();

		glBindFramebuffer( GL_FRAMEBUFFER, _depthPeelingFBO[ i ] );
		glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_EXT, _depthPeelingTex[ i ], 0 );
		glFramebufferTexture2DEXT( GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_TEXTURE_RECTANGLE_EXT, _depthPeelingDep[ i ], 0 );
		GLenum fboStatus = glCheckFramebufferStatus( GL_FRAMEBUFFER );
		if ( fboStatus != GL_FRAMEBUFFER_COMPLETE )
		{
			std::cout << "FBO error, status : " << fboStatus << std::endl;
		}
		glBindFramebuffer( GL_FRAMEBUFFER, 0 );
		GV_CHECK_GL_ERROR();
	}
	GV_CUDA_SAFE_CALL( cudaGraphicsGLRegisterImage( &_depthPeelingDepthMinResource, _depthPeelingTex[ 0 ], GL_TEXTURE_RECTANGLE_EXT, cudaGraphicsRegisterFlagsReadOnly ) );
	GV_CUDA_SAFE_CALL( cudaGraphicsGLRegisterImage( &_depthPeelingDepthMaxResource, _depthPeelingTex[ 1 ], GL_TEXTURE_RECTANGLE_EXT, cudaGraphicsRegisterFlagsReadOnly ) );
	_pipeline->editRenderer()->setDepthPeelingDepthMinTexture( _depthPeelingDepthMinResource );
	_pipeline->editRenderer()->setDepthPeelingDepthMaxTexture( _depthPeelingDepthMaxResource );
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
										
	_pipeline->editDataStructure()->getDataStructureAppearance( pShowNodeHasBrickTerminal, pShowNodeHasBrickNotTerminal, pShowNodeIsBrickNotInCache, pShowNodeEmptyOrConstant
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
	return _pipeline->editRenderer()->dynamicUpdateState();
}

/******************************************************************************
 * Set the dynamic update state
 *
 * @param pFlag the dynamic update state
 ******************************************************************************/
void SampleCore::setDynamicUpdate( bool pFlag )
{
	_pipeline->editRenderer()->dynamicUpdateState() = pFlag;
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
 * Tell wheter or not the pipeline has a light.
 *
 * @return the flag telling wheter or not the pipeline has a light
 ******************************************************************************/
bool SampleCore::has3DModel() const
{
	return true;
}

/******************************************************************************
 * Get the 3D model filename to load
 *
 * @return the 3D model filename to load
 ******************************************************************************/
string SampleCore::get3DModelFilename() const
{
	return _filename;
}

/******************************************************************************
 * Get the depth peeling's number of layers
 *
 * @return the depth peeling's number of layers
 ******************************************************************************/
unsigned int SampleCore::getDepthPeelingNbLayers() const
{
	return _nbDepthPeelingLayers;
}

/******************************************************************************
 * Set the depth peeling's number of layers
 *
 * @param pValue the depth peeling's number of layers
 ******************************************************************************/
void SampleCore::setDepthPeelingNbLayers( unsigned int pValue )
{
	_nbDepthPeelingLayers = pValue;
}

/******************************************************************************
 * Get the shader opacity
 *
 * @return the shader opacity
 ******************************************************************************/
float SampleCore::getShaderOpacity() const
{
	return _shaderOpacity;
}

/******************************************************************************
 * Set the shader opacity
 *
 * @param pValue the shader opacity
 ******************************************************************************/
void SampleCore::setShaderOpacity( float pValue )
{
	_shaderOpacity = pValue;

	// Update DEVICE memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cShaderOpacity, &_shaderOpacity, sizeof( _shaderOpacity ), 0, cudaMemcpyHostToDevice ) );

	// Clear the GigaVoxels cache
	// - because, the shader color is set during the data production management
	_pipeline->editCache()->clearCache();
}

/******************************************************************************
 * Get the shader material opacity property
 *
 * @return the shader material opacity property
 ******************************************************************************/
float SampleCore::getShaderMaterialOpacityProperty() const
{
	return _shaderMaterialOpacityProperty;
}

/******************************************************************************
 * Set the shader material opacity property
 *
 * @param pValue the shader material opacity property
 ******************************************************************************/
void SampleCore::setShaderMaterialOpacityProperty( float pValue )
{
	_shaderMaterialOpacityProperty = pValue;

	const float invDistanceOfFullOpacity = 1.f / _shaderMaterialOpacityProperty;

	// Update DEVICE memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cInvDistanceOfFullOpacity, &invDistanceOfFullOpacity, sizeof( invDistanceOfFullOpacity ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Tell wheter or nor the hyper texture is activated
 *
 * @return a flag telling wheter or not the hyper texture is activated
 ******************************************************************************/
bool SampleCore::hasHyperTexture() const
{
	return _hasHyperTexture;
}

/******************************************************************************
 * Set the flag telling wheter or not the hyper texture is activated
 *
 * @param pFlag a flag telling wheter or not the hyper texture is activated
 ******************************************************************************/
void SampleCore::setHyperTexture( bool pFlag )
{
	_hasHyperTexture = pFlag;
	
	// Update device memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cHasHyperTexture, &_hasHyperTexture, sizeof( _hasHyperTexture ), 0, cudaMemcpyHostToDevice ) );

	// Clear cache
	clearCache();
}

/******************************************************************************
 * Get the noise first frequency
 *
 * @return the noise first frequency
 ******************************************************************************/
float SampleCore::getNoiseFirstFrequency() const
{
	return _noiseFirstFrequency;
}

/******************************************************************************
 * Set the noise first frequency
 *
 * @param pValue the noise first frequency
 ******************************************************************************/
void SampleCore::setNoiseFirstFrequency( float pValue )
{
	_noiseFirstFrequency = pValue;
	
	// Update device memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cNoiseFirstFrequency, &_noiseFirstFrequency, sizeof( _noiseFirstFrequency ), 0, cudaMemcpyHostToDevice ) );

	// Clear cache
	clearCache();
}

/******************************************************************************
 * Get the noise strength
 *
 * @return the noise strength
 ******************************************************************************/
float SampleCore::getNoiseStrength() const
{
	return _noiseStrength;
}

/******************************************************************************
 * Set the noise strength
 *
 * @param pValue the noise strength
 ******************************************************************************/
void SampleCore::setNoiseStrength( float pValue )
{
	_noiseStrength = pValue;
	
	// Update device memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cNoiseStrength, &_noiseStrength, sizeof( _noiseStrength ), 0, cudaMemcpyHostToDevice ) );

	// Clear cache
	clearCache();
}

/******************************************************************************
 * Set the 3D model filename to load
 *
 * @param pFilename the 3D model filename to load
 ******************************************************************************/
void SampleCore::set3DModelFilename( const string& pFilename )
{
	_filename = pFilename;

	// ---- Delete the 3D scene if needed ----

	if ( _scene != NULL )
	{
		aiReleaseImport( _scene );
		_scene = NULL;

		// Clear the GigaVoxels cache
		_pipeline->editCache()->clearCache();
	}

	// ---- Load the 3D scene ----

	_scene = aiImportFile( _filename.c_str(), 0 );//aiProcessPreset_TargetRealtime_Fast );

	// Scale the geometry
	float minx = +std::numeric_limits< float >::max();//FLT_MAX;
	float miny = +std::numeric_limits< float >::max();//FLT_MAX;
	float minz = +std::numeric_limits< float >::max();//FLT_MAX;
	float maxx = -std::numeric_limits< float >::max();//-FLT_MAX;
	float maxy = -std::numeric_limits< float >::max();//-FLT_MAX;
	float maxz = -std::numeric_limits< float >::max();//-FLT_MAX;

	for ( unsigned int meshIndex = 0; meshIndex < _scene->mNumMeshes; ++meshIndex )
	{
		const aiMesh* pMesh = _scene->mMeshes[ meshIndex ];

		for ( unsigned int vertexIndex = 0; vertexIndex < pMesh->mNumVertices; ++vertexIndex )
		{
			minx = std::min( minx, pMesh->mVertices[ vertexIndex ].x );
			miny = std::min( miny, pMesh->mVertices[ vertexIndex ].y );
			minz = std::min( minz, pMesh->mVertices[ vertexIndex ].z );
			maxx = std::max( maxx, pMesh->mVertices[ vertexIndex ].x );
			maxy = std::max( maxy, pMesh->mVertices[ vertexIndex ].y );
			maxz = std::max( maxz, pMesh->mVertices[ vertexIndex ].z );
		}
	}

	float scale = 0.95f / std::max( std::max( maxx - minx, maxy - miny ), maxz - minz );

	for ( unsigned int meshIndex = 0; meshIndex < _scene->mNumMeshes; ++meshIndex )
	{
		const aiMesh* pMesh = _scene->mMeshes[ meshIndex ];

		for ( unsigned int vertexIndex = 0; vertexIndex < pMesh->mNumVertices; ++vertexIndex )
		{
			pMesh->mVertices[ vertexIndex ].x = ( pMesh->mVertices[ vertexIndex ].x - ( maxx + minx ) * 0.5f ) * scale + 0.5f;
			pMesh->mVertices[ vertexIndex ].y = ( pMesh->mVertices[ vertexIndex ].y - ( maxy + miny ) * 0.5f ) * scale + 0.5f;
			pMesh->mVertices[ vertexIndex ].z = ( pMesh->mVertices[ vertexIndex ].z - ( maxz + minz ) * 0.5f ) * scale + 0.5f;
		}
	}

    // init VBO and IBO :
    glGenBuffers(1, &mVBO);
    glGenBuffers(1, &mIBO);

    // WARNING : we assume here that faces of the mesh are triangle. Plus we don't take of scene tree structure...

    // Computing number of vertices and triangles:
    unsigned int nbVertices = 0;
    mNbTriangle = 0;

    for (unsigned int meshIndex = 0; meshIndex < _scene->mNumMeshes; ++meshIndex) {
        nbVertices += _scene->mMeshes[meshIndex]->mNumVertices;
        mNbTriangle += _scene->mMeshes[meshIndex]->mNumFaces;
    }

    MyVertex *VBO = new MyVertex[nbVertices];
    unsigned int *IBO = new unsigned int[3*mNbTriangle];
    unsigned int offsetIBO = 0;
    unsigned int offsetVBO = 0;

    for (unsigned int meshIndex = 0; meshIndex < _scene->mNumMeshes; ++meshIndex)
        {
            const aiMesh *pMesh = _scene->mMeshes[meshIndex];

            // Storing vertices and normals into mVBO : X | Y | Z | Nx | Ny | Nz ... And storing index into IBO

            for (unsigned int faceIndex = 0; faceIndex < pMesh->mNumFaces; ++faceIndex)
                {
                    const struct aiFace *pFace = &pMesh->mFaces[faceIndex];

                    // Remark : we can compute different normal for same vertex, but new one overwrites the old one
                    for (unsigned int vertIndex = 0; vertIndex < pFace->mNumIndices; ++vertIndex)
                        {
                            int index = pFace->mIndices[vertIndex];

                            float normal[3];

                            // TO DO : Normaliser la normal
                            if (!pMesh->HasNormals()) {
                                // We compute normal with cross product :

                                // retrieve vertex index of the face
                                int a = pFace->mIndices[0];
                                int b = pFace->mIndices[1];
                                int c = pFace->mIndices[2];

                                float e1[3] = { pMesh->mVertices[b].x - pMesh->mVertices[a].x,
                                        pMesh->mVertices[b].y - pMesh->mVertices[a].y,
                                        pMesh->mVertices[b].z - pMesh->mVertices[a].z };

                                float e2[3] = { pMesh->mVertices[c].x - pMesh->mVertices[a].x,
                                        pMesh->mVertices[c].y - pMesh->mVertices[a].y,
                                        pMesh->mVertices[c].z - pMesh->mVertices[a].z };

                                VBO[offsetVBO + index].nx = e1[1]*e2[2] - e1[2]*e2[1];
                                VBO[offsetVBO + index].ny = e1[2]*e2[0] - e1[0]*e2[2];
                                VBO[offsetVBO + index].nz = e1[0]*e2[1] - e1[1]*e2[0];

                                // Normalizing the normal
                                float normal = sqrt ( VBO[offsetVBO + index].nx*VBO[offsetVBO + index].nx +
                                              VBO[offsetVBO + index].ny*VBO[offsetVBO + index].ny +
                                              VBO[offsetVBO + index].nz*VBO[offsetVBO + index].nz );
                                VBO[offsetVBO + index].nx /= normal;
                                VBO[offsetVBO + index].ny /= normal;
                                VBO[offsetVBO + index].nz /= normal;
                            } else {
                                VBO[offsetVBO + index].nx = pMesh->mNormals[index].x;
                                VBO[offsetVBO + index].ny = pMesh->mNormals[index].y;
                                VBO[offsetVBO + index].nz = pMesh->mNormals[index].z;
                            }


                            VBO[offsetVBO + index].x = pMesh->mVertices[index].x;
                            VBO[offsetVBO + index].y = pMesh->mVertices[index].y;
                            VBO[offsetVBO + index].z = pMesh->mVertices[index].z;

                            IBO[offsetIBO + vertIndex] = index;
                        }
                    offsetIBO += 3;
                }
            offsetVBO +=  _scene->mMeshes[meshIndex]->mNumVertices ;
        }

    glBindBuffer(GL_ARRAY_BUFFER, mVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(MyVertex)*nbVertices, &VBO[0].x, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);


    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*3*mNbTriangle, IBO, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // deleting tab used
    delete[] VBO;
    delete[] IBO;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::drawScene() const
{
    glBindBuffer(GL_ARRAY_BUFFER, mVBO);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, sizeof(MyVertex), BUFFER_OFFSET(0));   //The starting point of the VBO, for the vertices
    glEnableClientState(GL_NORMAL_ARRAY);
    glNormalPointer(GL_FLOAT, sizeof(MyVertex), BUFFER_OFFSET(3*sizeof(float)));   //The starting point of normals

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIBO);

    glDrawElements(GL_TRIANGLES, 3*mNbTriangle, GL_UNSIGNED_INT, BUFFER_OFFSET(0));   //The starting point of the IBO

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

}

/******************************************************************************
 * Draw the proxy geometry recursively
 *
 * @param pScene The 3D scene
 * @param pNode The node from which to draw geometry recursively (sub-nodes)
 ******************************************************************************/
void SampleCore::drawProxyRecursive( const struct aiScene* pScene, const struct aiNode* pNode )
{
	struct aiMatrix4x4 m = pNode->mTransformation;

	aiTransposeMatrix4( &m );
	
	glPushMatrix();
	glMultMatrixf( (float *)&m );

	// Iterate through meshes of the node
	for ( unsigned int meshIndex = 0; meshIndex < pScene->mNumMeshes; ++meshIndex )
	{
		// Retrieve current mesh
		const aiMesh* pMesh = pScene->mMeshes[ meshIndex ];

		// Iterate through faces of current mesh
		for ( unsigned int faceIndex = 0; faceIndex < pMesh->mNumFaces; ++faceIndex )
		{
			// Retrieve current face
			const struct aiFace *pFace = &pMesh->mFaces[ faceIndex ];

			GLenum face_mode;

			switch ( pFace->mNumIndices )
			{
				case 1: face_mode = GL_POINTS;
					break;

				case 2: face_mode = GL_LINES;
                    break;

				case 3: face_mode = GL_TRIANGLES;
					break;

				default: face_mode = GL_POLYGON;
					break;
			}

			glBegin( face_mode );

			// Iterate through vertices of current face
			for ( unsigned int vertIndex = 0; vertIndex < pFace->mNumIndices; ++vertIndex )
			{
				int index = pFace->mIndices[ vertIndex ];

				if ( pMesh->HasNormals() )
				{
                    glNormal3fv( &pMesh->mNormals[ index ].x );
				}

                glVertex3fv( &pMesh->mVertices[ index ].x );
			}

			glEnd();
		}
	}

	// Iterate through child nodes, and draw them recursively
	for ( unsigned int childIndex = 0; childIndex < pNode->mNumChildren; ++childIndex )
	{
		drawProxyRecursive( pScene, pNode->mChildren[ childIndex ] );
	}

	glPopMatrix();
}
