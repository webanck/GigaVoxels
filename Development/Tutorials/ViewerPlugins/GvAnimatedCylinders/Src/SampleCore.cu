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
#include <GvRendering/GvRendererCUDA.h>
#include <GvUtils/GvSimplePipeline.h>
#include <GvUtils/GvSimpleHostProducer.h>
#include <GvUtils/GvSimpleHostShader.h>
#include <GvUtils/GvCommonGraphicsPass.h>
#include <GvCore/GvError.h>
#include <GvPerfMon/GvPerformanceMonitor.h>
#include <GsGraphics/GsShaderProgram.h>

// Project
#include "ProducerKernel.h"
#include "ShaderKernel.h"

// GvViewer
#include <GvvApplication.h>
#include <GvvMainWindow.h>

// Cuda SDK
#include <helper_math.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>

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

// Defines the size allowed for each type of pool
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
//:	GvvPipelineInterface()
:	GvViewerScene::GvvPipeline()
,	_pipeline( NULL )
,	_renderer( NULL )
,	_graphicsEnvironment( NULL )
,	_displayOctree( false )
,	_displayPerfmon( 0 )
,	_maxVolTreeDepth( 0 )
,	_depthBuffer( 0 )
,	_colorTex( 0 )
,	_colorRenderBuffer( 0 )
,	_depthTex( 0 )
,	_frameBuffer( 0 )
,	_shapeColor( make_float3( 0.f, 0.f, 0.f ) )
,	_shapeOpacity( 0.f )
,	_shaderMaterialProperty( 0.f )
,	_textureSamplerUniformLocation( 0 )
{
	// Translation used to position the GigaVoxels data structure
	_translation[ 0 ] = -0.5f;
	_translation[ 1 ] = -0.5f;
	_translation[ 2 ] = -0.5f;

	// Rotation used to position the GigaVoxels data structure
	// _rotation[ 0 ] = 0.0f;
	// _rotation[ 1 ] = 0.0f;
	// _rotation[ 2 ] = 0.0f;
	// _rotation[ 3 ] = 0.0f;

	// Scale used to transform the GigaVoxels data structure
	_scale = 1.0f;

	// Light position
	// _lightPosition = make_float3(  1.f, 1.f, 1.f );
	// see the init method for it.
	setLightPosition(1.f, 1.f, 1.f);
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
	// Disconnect all registered graphics resources
	/*_pipeline->editRenderer()*/_renderer->resetGraphicsResources();

	// Delete the GigaVoxels pipeline
	delete _pipeline;

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
	return "AnimatedCylinders";
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

	// Renderer initialization
	_renderer = new RendererType( _pipeline->editDataStructure(), _pipeline->editCache() );
	assert( _renderer != NULL );
	_pipeline->addRenderer( _renderer );

	// Pipeline configuration
	_maxVolTreeDepth = 5;
	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );

	// Graphics environment creation
	_graphicsEnvironment = new GvCommonGraphicsPass();

	// Custom initialization
	// Note : this could be done via an XML settings file loaded at initialization
	// Note : the SET accessors call clearCache() which is useless...
	setShapeColor( make_float3( 1.f, 1.f, 1.f ) );
	setShapeOpacity( 1.f );
	// Full opacity is reached when distance ( 1.f / 512.f ) is traversed along a ray during ray-casting
	setShaderMaterialProperty( 1.f / 512.f );

	// Set light position
	setLightPosition( 1.f, 1.f, 0.5f );

	// Create and link a GLSL shader program
	QString dataRepository = QCoreApplication::applicationDirPath();
	dataRepository += QDir::separator();
	dataRepository += QString( "Data" );
	dataRepository += QDir::separator();
	dataRepository += QString( "Shaders" );
	dataRepository += QDir::separator();
	dataRepository += QString( "GvAnimatedCylinders" );
	dataRepository += QDir::separator();
	// Initialize points shader program
	QString vertexShaderFilename = dataRepository + QString( "fullscreenQuad_vert.glsl" );
	QString fragmentShaderFilename = dataRepository + QString( "fullscreenQuad_frag.glsl" );
	// Initialize shader program
	_shaderProgram = new GsShaderProgram();
	bool statusOK = _shaderProgram->addShader( GsShaderProgram::eVertexShader, vertexShaderFilename.toStdString() );
	assert( statusOK );
	if ( ! statusOK )
	{
		// TO DO
		// - handle error
	}
	statusOK = _shaderProgram->addShader( GsShaderProgram::eFragmentShader, fragmentShaderFilename.toStdString() );
	assert( statusOK );
	if ( ! statusOK )
	{
		// TO DO
		// - handle error
	}
	statusOK = _shaderProgram->link();
	assert( statusOK );
	if ( ! statusOK )
	{
		// TO DO
		// - handle error
	}

	// After linking has occurred, the command glGetUniformLocation can be used to obtain the location of a uniform variable
	// - beware, after "shader link", this location my changed
	_textureSamplerUniformLocation = glGetUniformLocation( _shaderProgram->_program, "uTextureSampler" );
	assert( _textureSamplerUniformLocation != -1 );
	if ( _textureSamplerUniformLocation == -1 )
	{
		// TO DO
		// - handle error
	}

	// Vertex position buffer initialization
	glGenBuffers( 1, &_fullscreenQuadVertexBuffer );
	glBindBuffer( GL_ARRAY_BUFFER, _fullscreenQuadVertexBuffer );
	GLsizeiptr fullscreenQuadVertexBufferSize = sizeof( GLfloat ) * 4/*nbVertices*/ * 2/*nb components per vertex*/;
	float2 fullscreenQuadVertices[ 4/*nbVertices*/ ] =
	{
		{ -1.0, -1.0 },
		{ 1.0, -1.0 },
		{ 1.0, 1.0 },
		{ -1.0, 1.0 }
	};
	glBufferData( GL_ARRAY_BUFFER, fullscreenQuadVertexBufferSize, &fullscreenQuadVertices[ 0 ], GL_STATIC_DRAW );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	// Vertex array object initialization
	glGenVertexArrays( 1, &_fullscreenQuadVAO );
	glBindVertexArray( _fullscreenQuadVAO );
	glEnableVertexAttribArray( 0 );	// vertex position
	glBindBuffer( GL_ARRAY_BUFFER, _fullscreenQuadVertexBuffer );
	glVertexAttribPointer( 0/*attribute index*/, 2/*nb components per vertex*/, GL_FLOAT/*type*/, GL_FALSE/*un-normalized*/, 0/*memory stride*/, static_cast< GLubyte* >( NULL )/*byte offset from buffer*/ );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
	glBindVertexArray( 0 );
}

/******************************************************************************
 * Draw function called of frame
 ******************************************************************************/
void SampleCore::draw()
{
	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT( frame );
	CUDAPM_START_EVENT( app_init_frame );

	updateElapsedTime();
	cacheFlushing();

	//glMatrixMode( GL_MODELVIEW );		// already done in the QLViewer::preDraw() method

	// TO DO : optimization
	//
	// - use glPushMatrix/lPushMatrix() and glEnable( GL_DEPTH_TEST ) / glDisable( GL_DEPTH_TEST ) only when required to reduce driver overhead

	// Handle image downscaling if activated
	int bufferWidth = _graphicsEnvironment->getBufferWidth();
	int bufferHeight = _graphicsEnvironment->getBufferHeight();
	if ( hasImageDownscaling() )
	{
		bufferWidth = _graphicsEnvironment->getImageDownscalingWidth();
		bufferHeight = _graphicsEnvironment->getImageDownscalingHeight();
		glViewport( 0, 0, bufferWidth, bufferHeight );
	}

	glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );
	if ( _displayOctree )
	{
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );

		// Display the GigaVoxels N3-tree space partitioning structure
		glEnable( GL_DEPTH_TEST );
		glPushMatrix();
		// Translation used to position the GigaVoxels data structure
		glScalef( _scale, _scale, _scale );
		glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );
		_pipeline->editDataStructure()->render();
		glPopMatrix();
		glDisable( GL_DEPTH_TEST );

		// Clear the depth PBO (pixel buffer object) by reading from the previously cleared FBO (frame buffer object)
		glBindBuffer( GL_PIXEL_PACK_BUFFER, _depthBuffer );
		glReadPixels( 0, 0, bufferWidth, bufferHeight, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0 );
		glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
		GV_CHECK_GL_ERROR();
	}
	else
	{
		glClear( GL_COLOR_BUFFER_BIT );
	}

	// Go back to default frame buffer
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );

	// Extract view transformations
	//
	// TO DO : optimization
	// - retrieve internal GL matrices without call to GL context (glGetFloatv())
	float4x4 viewMatrix;
	float4x4 projectionMatrix;
	glGetFloatv( GL_MODELVIEW_MATRIX, viewMatrix._array );	// Check what requires GigaSpace engine : is that pure View matrix, or could it be ModelView matrix ?
	glGetFloatv( GL_PROJECTION_MATRIX, projectionMatrix._array );

	// Extract viewport
	GLint params[ 4 ];
	glGetIntegerv( GL_VIEWPORT, params );
	int4 viewport = make_int4( params[ 0 ], params[ 1 ], params[ 2 ], params[ 3 ] );
	// Handle image downscaling if activated
	if ( hasImageDownscaling() )
	{
		// TO DO : clean this... it would better to send real viewport info and retrieve realBufferSize in the renderer ?
		viewport.z = bufferWidth;
		viewport.w = bufferHeight;
	}

	// render the scene into textures
	CUDAPM_STOP_EVENT( app_init_frame );

	// Build the world transformation matrix
	float4x4 modelMatrix;
	glPushMatrix();
	glLoadIdentity();
	// Translation used to position the GigaVoxels data structure
	glScalef( _scale, _scale, _scale );
	glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );
	glGetFloatv( GL_MODELVIEW_MATRIX, modelMatrix._array );
	glPopMatrix();

	// Render
	updateDisplacementMap();
	uint regularisationNb = 0U;
	cudaMemcpyToSymbol(cRegularisationNb, &regularisationNb, sizeof(regularisationNb), 0, cudaMemcpyHostToDevice);
	for(uint i=0U; i<15U; i++)
		_pipeline->execute( modelMatrix, viewMatrix, projectionMatrix, viewport );
	cudaMemcpyFromSymbol(&regularisationNb, cRegularisationNb, sizeof(regularisationNb), 0, cudaMemcpyDeviceToHost);
	std::cout << "Regularisations: " << regularisationNb << std::endl;

	if ( _graphicsEnvironment->getType() != 0 )
	{
		// Copy a block of pixels from the read framebuffer to the draw framebuffer
		//glBindFramebuffer( GL_DRAW_FRAMEBUFFER, 0 );	// already done before
		glBindFramebuffer( GL_READ_FRAMEBUFFER, _frameBuffer );
		//glReadBuffer( GL_COLOR_ATTACHMENT0 + TextureType ); => not use because we only have one color attachment
		/*GLint srcX0 = 0;
		GLint srcY0 = 0;
		GLint srcX1 = bufferWidth;
		GLint srcY1 = bufferHeight;
		GLint dstX0 = 0;
		GLint dstY0 = 0;
		GLint dstX1 = bufferWidth;
		GLint dstY1 = bufferHeight;
		GLbitfield mask = GL_COLOR_BUFFER_BIT;
		GLenum filter = GL_NEAREST;*/
		// Handle image downscaling if activated
		if ( hasImageDownscaling() )
		{
			//glBlitFramebuffer( srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter );

			int bufferWidth = _graphicsEnvironment->getBufferWidth();
			int bufferHeight = _graphicsEnvironment->getBufferHeight();
			int imageDownscalingWidth = _graphicsEnvironment->getImageDownscalingWidth();
			int imageDownscalingHeight = _graphicsEnvironment->getImageDownscalingHeight();
			glViewport( 0, 0, bufferWidth, bufferHeight );
			glBlitFramebuffer( 0, 0, imageDownscalingWidth, imageDownscalingHeight, 0, 0, bufferWidth, bufferHeight, GL_COLOR_BUFFER_BIT, GL_LINEAR );
		}
		else
		{
			//glBlitFramebuffer( srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter );

			glBlitFramebuffer( 0, 0, bufferWidth, bufferHeight, 0, 0, bufferWidth, bufferHeight, GL_COLOR_BUFFER_BIT, GL_NEAREST );
		}
	}
	else
	{
		// Render the result to the screen

		// Handle image downscaling if activated
		if ( hasImageDownscaling() )
		{
			glViewport( 0, 0, _graphicsEnvironment->getBufferWidth(), _graphicsEnvironment->getBufferHeight() );
		}

		glDisable( GL_DEPTH_TEST );

		// Draw fullscreen textured quad
		_shaderProgram->use();
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _colorTex );
		glUniform1i( _textureSamplerUniformLocation, 0 );
		glBindVertexArray( _fullscreenQuadVAO );
		glDrawArrays( GL_QUADS, 0, 4 );
		glBindVertexArray( 0 );
		glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
		glUseProgram( 0 );
	}

	// TEST - optimization due to early unmap() graphics resource from GigaVoxels
	///*_pipeline->editRenderer()*/_renderer->doPostRender();

	// Update GigaVoxels info
	/*_pipeline->editRenderer()*/_renderer->nextFrame();

	CUDAPM_STOP_EVENT( frame );
	CUDAPM_STOP_FRAME;

	if ( _displayPerfmon )
	{
		GvPerfMon::CUDAPerfMon::get().displayFrameGL( _displayPerfmon - 1 );
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
	// LOG
	//
	// @todo : check and avoid 0 values, replace by 1 and warn user
	if ( pWidth == 0 )
	{
		// TO DO
		// ...
	}
	if ( pHeight == 0 )
	{
		// TO DO
		// ...
	}

	// --------------------------
	// Reset default active frame region for rendering
	/*_pipeline->editRenderer()*/_renderer->setProjectedBBox( make_uint4( 0, 0, pWidth, pHeight ) );
	// Re-init Perfmon subsystem
	CUDAPM_RESIZE( make_uint2( pWidth, pHeight ) );
	// --------------------------

	// Update graphics environment
	_graphicsEnvironment->setBufferSize( pWidth, pHeight );

	// Reset graphics resources
	resetGraphicsresources();
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
	/*_pipeline->editRenderer()*/_renderer->resetGraphicsResources();

	if ( _displayOctree )
	{
		if ( _graphicsEnvironment->getType() == 0 )
		{
			/*_pipeline->editRenderer()*/_renderer->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		}
		else
		{
			/*_pipeline->editRenderer()*/_renderer->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorRenderBuffer, GL_RENDERBUFFER );
		}
		/*_pipeline->editRenderer()*/_renderer->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	}
	else
	{
		if ( _graphicsEnvironment->getType() == 0 )
		{
			/*_pipeline->editRenderer()*/_renderer->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		}
		else
		{
			/*_pipeline->editRenderer()*/_renderer->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorRenderBuffer, GL_RENDERBUFFER );
		}
	}
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
	if ( _displayPerfmon )
	{
		_displayPerfmon = 0;

		GvPerfMon::CUDAPerfMon::_isActivated = false;
	}
	else
	{
		_displayPerfmon = mode;

		GvPerfMon::CUDAPerfMon::_isActivated = true;
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
	// return false;
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
 * Set the pipeline and OpenGL light position.
 *
 * @param pX the X light position in the scene referential.
 * @param pY the Y light position in the scene referential.
 * @param pZ the Z light position in the scene referential.
 ******************************************************************************/
void SampleCore::setLightPosition(float pX, float pY, float pZ) {
	//Update the pipeline light position in the scene referential.
	_lightPosition.x = pX;
	_lightPosition.y = pY;
	_lightPosition.z = pZ;

	//Update DEVICE memory with "light position" in the volume tree referential.
	float3 lightPositionTree = make_float3(
		(pX - _translation[0])/_scale,
		(pY - _translation[1])/_scale,
		(pZ - _translation[2])/_scale
	);
	GV_CUDA_SAFE_CALL(cudaMemcpyToSymbol(cLightPositionTree, &lightPositionTree, sizeof(lightPositionTree), 0, cudaMemcpyHostToDevice));
	// Update DEVICE memory with "light position" in the scene referential.
	GV_CUDA_SAFE_CALL(cudaMemcpyToSymbol(cLightPositionScene, &_lightPosition, sizeof(_lightPosition), 0, cudaMemcpyHostToDevice));

	// Update the OpenGL light position in the scene referential.
	float position[4] = {pX, pY, pZ, 0.f};
	glLightfv(GL_LIGHT0, GL_POSITION, position);
	// cout << "position_change (scene):" << pX << "," << pY << "," << pZ << endl;
	// cout << "position_change (tree):" << lightPositionTree.x << "," << lightPositionTree.y << "," << lightPositionTree.z << endl;
	// cout << "old_position:" << old_position[0] << "," << old_position[1] << "," << old_position[2] << "," << old_position[3] << endl;
}

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
 * Get the shape color
 *
 * @return the shape color
 ******************************************************************************/
const float3& SampleCore::getShapeColor() const
{
	return _shapeColor;
}

/******************************************************************************
 * Set the shape color
 *
 * @param pColor the shape color
 ******************************************************************************/
void SampleCore::setShapeColor( const float3& pColor )
{
	_shapeColor = pColor;

	// Update device memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cShapeColor, &_shapeColor, sizeof( _shapeColor ), 0, cudaMemcpyHostToDevice ) );

	// Clear cache
	clearCache();
}

/******************************************************************************
 * Get the shape opacity
 *
 * @return the shape opacity
 ******************************************************************************/
float SampleCore::getShapeOpacity() const
{
	return _shapeOpacity;
}

/******************************************************************************
 * Set the shape opacity
 *
 * @param pValue the shape opacity
 ******************************************************************************/
void SampleCore::setShapeOpacity( float pValue )
{
	_shapeOpacity = pValue;

	// Update device memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cShapeOpacity, &_shapeOpacity, sizeof( _shapeOpacity ), 0, cudaMemcpyHostToDevice ) );

	// Clear cache
	clearCache();
}

/******************************************************************************
 * Get the shader material property (according to opacity)
 *
 * @return the shader material property (according to opacity)
 ******************************************************************************/
float SampleCore::getShaderMaterialProperty() const
{
	// Inverted value to speed code in shader
	return 1.f / _shaderMaterialProperty;
}

/******************************************************************************
 * Set the shader material property (according to opacity)
 *
 * @param pValue the shader material property (according to opacity)
 ******************************************************************************/
void SampleCore::setShaderMaterialProperty( float pValue )
{
	// Inverted value to speed code in shader
	_shaderMaterialProperty = 1.f / pValue;

	// Update device memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cShaderMaterialProperty, &_shaderMaterialProperty, sizeof( _shaderMaterialProperty ), 0, cudaMemcpyHostToDevice ) );
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
 * Tell wheter or not the pipeline uses image downscaling.
 *
 * @return the flag telling wheter or not the pipeline uses image downscaling
 ******************************************************************************/
bool SampleCore::hasImageDownscaling() const
{
	return _graphicsEnvironment->hasImageDownscaling();
}

/******************************************************************************
 * Set the flag telling wheter or not the pipeline uses image downscaling
 *
 * @param pFlag the flag telling wheter or not the pipeline uses image downscaling
 ******************************************************************************/
void SampleCore::setImageDownscaling( bool pFlag )
{
	// Update graphics environment
	_graphicsEnvironment->setImageDownscaling( pFlag );

	// Reset graphics resources
	resetGraphicsresources();
}

/******************************************************************************
 * Get the internal graphics buffer size
 *
 * @param pWidth the internal graphics buffer width
 * @param pHeight the internal graphics buffer height
 ******************************************************************************/
void SampleCore::getViewportSize( unsigned int& pWidth, unsigned int& pHeight ) const
{
	if ( _graphicsEnvironment != NULL )
	{
		pWidth = static_cast< unsigned int >( _graphicsEnvironment->getBufferWidth() );
		pHeight = static_cast< unsigned int >( _graphicsEnvironment->getBufferHeight() );
	}
}

/******************************************************************************
 * Set the internal graphics buffer size
 *
 * @param pWidth the internal graphics buffer width
 * @param pHeight the internal graphics buffer height
 ******************************************************************************/
void SampleCore::setViewportSize( unsigned int pWidth, unsigned int pHeight )
{
	// --------------------------
	// Reset default active frame region for rendering
	/*_pipeline->editRenderer()*/_renderer->setProjectedBBox( make_uint4( 0, 0, pWidth, pHeight ) );
	// Re-init Perfmon subsystem
	CUDAPM_RESIZE( make_uint2( pWidth, pHeight ) );
	// --------------------------

	// Update graphics environment
	_graphicsEnvironment->setBufferSize( pWidth, pHeight );

	// Reset graphics resources
	resetGraphicsresources();
}

/******************************************************************************
 * Get the internal graphics buffer size
 *
 * @param pWidth the internal graphics buffer width
 * @param pHeight the internal graphics buffer height
 ******************************************************************************/
void SampleCore::getGraphicsBufferSize( unsigned int& pWidth, unsigned int& pHeight ) const
{
	if ( _graphicsEnvironment != NULL )
	{
		pWidth = static_cast< unsigned int >( _graphicsEnvironment->getImageDownscalingWidth() );
		pHeight = static_cast< unsigned int >( _graphicsEnvironment->getImageDownscalingHeight() );
	}
}

/******************************************************************************
 * Set the internal graphics buffer size
 *
 * @param pWidth the internal graphics buffer width
 * @param pHeight the internal graphics buffer height
 ******************************************************************************/
void SampleCore::setGraphicsBufferSize( unsigned int pWidth, unsigned int pHeight )
{
	// --------------------------
	// Reset default active frame region for rendering
	/*_pipeline->editRenderer()*/_renderer->setProjectedBBox( make_uint4( 0, 0, pWidth, pHeight ) );
	// Re-init Perfmon subsystem
	CUDAPM_RESIZE( make_uint2( pWidth, pHeight ) );
	// --------------------------

	// Update graphics environment
	_graphicsEnvironment->setImageDownscalingSize( pWidth, pHeight );

	// Reset graphics resources
	resetGraphicsresources();
}

/******************************************************************************
 * Reset graphics resources
 ******************************************************************************/
void SampleCore::resetGraphicsresources()
{
	// [ 1 ] - Reset graphics resources

	// Disconnect all registered graphics resources
	/*_pipeline->editRenderer()*/_renderer->resetGraphicsResources();

	// Update graphics environment
	_graphicsEnvironment->reset();

	// Update internal variables
	_depthBuffer = _graphicsEnvironment->getDepthBuffer();
	_colorTex = _graphicsEnvironment->getColorTexture();
	_colorRenderBuffer = _graphicsEnvironment->getColorRenderBuffer();
	_depthTex = _graphicsEnvironment->getDepthTexture();
	_frameBuffer = _graphicsEnvironment->getFrameBuffer();

	// [ 2 ] - Connect graphics resources

	// Create CUDA resources from OpenGL objects
	if ( _displayOctree )
	{
		if ( _graphicsEnvironment->getType() == 0 )
		{
			/*_pipeline->editRenderer()*/_renderer->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		}
		else
		{
			/*_pipeline->editRenderer()*/_renderer->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorRenderBuffer, GL_RENDERBUFFER );
		}
		/*_pipeline->editRenderer()*/_renderer->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	}
	else
	{
		if ( _graphicsEnvironment->getType() == 0 )
		{
			/*_pipeline->editRenderer()*/_renderer->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		}
		else
		{
			/*_pipeline->editRenderer()*/_renderer->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorRenderBuffer, GL_RENDERBUFFER );
		}
	}
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
	return false;
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
	return _shaderProgram->hasShaderType( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
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
	return _shaderProgram->getShaderSourceCode( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
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
	return _shaderProgram->getShaderFilename( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
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
	return _shaderProgram->reloadShader( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
}


void SampleCore::updateElapsedTime() {
	static clock_t instant = clock();

	const clock_t newInstant = clock();
	_elapsedMiliseconds += (static_cast<float>(newInstant - instant)/static_cast<float>(CLOCKS_PER_SEC)) * 1000.f;

	_elapsedSeconds += _elapsedMiliseconds/1000U;
	_elapsedMiliseconds %= 1000U;

	// Update device memory
	GV_CUDA_SAFE_CALL(cudaMemcpyToSymbol(cElapsedSeconds, &_elapsedSeconds, sizeof(_elapsedSeconds), 0, cudaMemcpyHostToDevice));
	GV_CUDA_SAFE_CALL(cudaMemcpyToSymbol(cElapsedMiliseconds, &_elapsedMiliseconds, sizeof(_elapsedMiliseconds), 0, cudaMemcpyHostToDevice));

	instant = newInstant;

	// std::cout << _elapsedSeconds << " : " << _elapsedMiliseconds << std::endl;
}

bool SampleCore::cacheFlushing() {
	// static uint frame = 0U;
	// if(++frame > 15U) {
		// frame = 0;
		clearCache();
		return true;
	// } else return false;
}

void SampleCore::updateDisplacementMap() {
	//Parts in each dimension of the map (minimum 1).
	const size_t width = 1000U;
	const size_t height = 1000U;
	//Total number of cells in the map.
	const uint nbElelements = width*height;
	//Parameterization of the waves.
	const float loopSeconds = 10.f;
	const uint verticalWaves = 1U;
	const uint horizontalWaves = 0U;

	//Memory management variables.
	static bool initialized = false;
	static float data[nbElelements];
	static size_t offset;
	static size_t pitch;

	//First call to the function: initialize memory.
	if(!initialized) {
		initialized = true;
		GV_CUDA_SAFE_CALL(cudaMallocPitch((void**)&cDisplacementMapData, &pitch, width * sizeof(float), height));
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		GV_CUDA_SAFE_CALL(cudaBindTexture2D(&offset, &cDisplacementMap, cDisplacementMapData, &channelDesc, width, height, pitch));
		// Set mutable properties:
		cDisplacementMap.normalized=true;
		cDisplacementMap.addressMode[0]=cudaAddressModeWrap;
		cDisplacementMap.addressMode[1]=cudaAddressModeWrap;
		cDisplacementMap.filterMode= cudaFilterModePoint;
	}

	//Set the displacement map values.
	const float loopPart = 3.14f * (static_cast<float>(_elapsedSeconds) + static_cast<float>(_elapsedMiliseconds) / 1000.f) / loopSeconds;
	// std::cout << "seconds=" << _elapsedSeconds << std::endl;
	// std::cout << "mseconds=" << _elapsedMiliseconds << std::endl;
	// std::cout << "loopPart=" << loopPart << std::endl;
	//////////////////////////////////////////////////////////////////////waving patern as a 1 dimension line
	// for(uint i=0U; i<nbElelements; i++) data[i] = fabs(sin(static_cast<float>(i)+loopPart));
	//////////////////////////////////////////////////////////////////////waving patern as a 2 dimensions grid
	for(uint i=0U; i<height; i++) for(uint j=0U; j<width; j++) {
		const float verticalFrequency 	= 3.14f*static_cast<float>(verticalWaves)/static_cast<float>(height);
		const float verticalHeight 		= (
			verticalWaves > 0U ?
			(1.f + sin(static_cast<float>(i)*verticalFrequency+loopPart))/2.f :
			1.f
		);
		const float horizontalFrequency = 3.14f*static_cast<float>(horizontalWaves)/static_cast<float>(width);
		const float horizontalHeight 	= (
			horizontalWaves > 0U ?
			(1.f + sin(static_cast<float>(j)*horizontalFrequency+loopPart))/2.f :
			1.f
		);

		data[i*width + j] = 0.5f*verticalHeight + 0.5f*horizontalHeight;
		// std::cout << " " << data[i*width + j];
	}
	// std::cout << std::endl;
	//////////////////////////////////////////////////////////////////////

	// Copy data from host to device
	GV_CUDA_SAFE_CALL(cudaMemcpy2D((void*)cDisplacementMapData, pitch, (void*)data, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice));
}
