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
#include <GvCore/GvError.h>
#include <GvPerfMon/GvPerformanceMonitor.h>
#include <GvStructure/GvNode.h>

// Project
#include "ProducerKernel.h"
#include "VolumeTreeRendererGLSL.h"

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

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
:	_pipeline( NULL )
,	_renderer( NULL )
,	mColorTex( 0 )
,	mDepthTex( 0 )
,	mFrameBuffer( 0 )
,	mColorBuffer( 0 )
,	mDepthBuffer( 0 )
,	mColorResource( 0 )
,	mDepthResource( 0 )
,	mDisplayOctree( false )
,	mDisplayPerfmon( 0 )
,	mMaxVolTreeDepth( 6 )
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
 * Initialize the GigaVoxels pipeline
 ******************************************************************************/
void SampleCore::init()
{
	CUDAPM_INIT();

	// Initialize CUDA with OpenGL Interoperability
	//cudaGLSetGLDevice( gpuGetMaxGflopsDeviceId() );	// to do : deprecated, use cudaSetDevice()
	//GV_CHECK_CUDA_ERROR( "cudaGLSetGLDevice" );
	cudaSetDevice( gpuGetMaxGflopsDeviceId() );
	GV_CHECK_CUDA_ERROR( "cudaSetDevice" );

	// Compute the size of one element in the cache for nodes and bricks
	size_t nodeElemSize = NodeRes::numElements * sizeof( GvStructure::GvNode );
	//size_t brickElemSize = RealBrickRes::numElements * GvCore::DataTotalChannelSize< DataType >::value;

	// Compute how many we can fit into the given memory size
	size_t nodePoolNumElems = NODEPOOL_MEMSIZE / nodeElemSize;
	//size_t brickPoolNumElems = BRICKPOOL_MEMSIZE / brickElemSize;

	// Compute the resolution of the pools
	uint3 nodePoolRes = make_uint3( (uint)floorf( powf( (float)nodePoolNumElems, 1.0f / 3.0f ) ) ) * NodeRes::get();
	//uint3 brickPoolRes = make_uint3( (uint)floorf( powf( (float)brickPoolNumElems, 1.0f / 3.0f ) ) ) * RealBrickRes::get();

	//std::cout << "nodePoolRes: " << nodePoolRes << std::endl;
	//std::cout << "brickPoolRes: " << brickPoolRes << std::endl;

	// Pipeline creation
	_pipeline = new PipelineType();

	// Producer creation
	ProducerType* producer = new ProducerType();

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
}

/******************************************************************************
 * Draw function called of frame
 ******************************************************************************/
void SampleCore::draw()
{
	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT( frame );
	CUDAPM_START_EVENT( app_init_frame );

	glClearColor( 0.0f, 0.1f, 0.3f, 0.0f );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );

	glEnable( GL_DEPTH_TEST );

	glMatrixMode( GL_MODELVIEW);

	// Display the data structure (space partitioning)
	glPushMatrix();
	glTranslatef( -0.5f, -0.5f, -0.5f );
	if ( mDisplayOctree )
	{
		_pipeline->editDataStructure()->render();
	}
	glPopMatrix();

	// extract view transformations
	float4x4 viewMatrix;
	float4x4 projectionMatrix;
	// FIXME
	glPushMatrix();
	glTranslatef( -0.5f, -0.5f, -0.5f );
	// FIXME
	glGetFloatv( GL_MODELVIEW_MATRIX, viewMatrix._array );
	glGetFloatv( GL_PROJECTION_MATRIX, projectionMatrix._array );
	// FIXME
	glPopMatrix();

	// build and extract tree transformations
	float4x4 modelMatrix;

	glPushMatrix();
	glLoadIdentity();
	//glTranslatef(-0.5f, -0.5f, -0.5f);
	glGetFloatv( GL_MODELVIEW_MATRIX, modelMatrix._array );
	glPopMatrix();

	// extract viewport
	GLint params[4];
	glGetIntegerv( GL_VIEWPORT, params );
	int4 viewport = make_int4( params[0], params[1], params[2], params[3] );

	CUDAPM_STOP_EVENT( app_init_frame );

	// Render the scene into textures
	_pipeline->execute( modelMatrix, viewMatrix, projectionMatrix, viewport );

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
 * @param width the new width
 * @param height the new height
 ******************************************************************************/
void SampleCore::resize( int width, int height )
{
	mWidth = width;
	mHeight = height;

	// Re-init Perfmon subsystem
	CUDAPM_RESIZE( make_uint2( mWidth, mHeight ) );

	/*uchar *timersMask = GvPerfMon::CUDAPerfMon::get().getKernelTimerMask();
	cudaMemset(timersMask, 255, mWidth * mHeight);*/

	// Create frame-dependent objects
	if ( mColorResource )
	{
		GV_CUDA_SAFE_CALL( cudaGraphicsUnregisterResource(mColorResource) );
	}
	if ( mDepthResource )
	{
		GV_CUDA_SAFE_CALL( cudaGraphicsUnregisterResource(mDepthResource) );
	}

	if ( mColorBuffer )
	{
		glDeleteBuffers( 1, &mColorBuffer );
	}
	if ( mDepthBuffer )
	{
		glDeleteBuffers( 1, &mDepthBuffer );
	}

	if ( mColorTex )
	{
		glDeleteTextures( 1, &mColorTex );
	}
	if ( mDepthTex )
	{
		glDeleteTextures( 1, &mDepthTex );
	}

	if ( mFrameBuffer )
	{
		glDeleteFramebuffers( 1, &mFrameBuffer );
	}

	glGenBuffers( 1, &mColorBuffer );
	glBindBuffer( GL_PIXEL_PACK_BUFFER, mColorBuffer );
	glBufferData( GL_PIXEL_PACK_BUFFER, width * height * sizeof(GLubyte) * 4, NULL, GL_DYNAMIC_DRAW );
	glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
	GV_CHECK_GL_ERROR();

	glGenTextures( 1, &mColorTex );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, mColorTex );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexImage2D( GL_TEXTURE_RECTANGLE_EXT, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0  );
	GV_CHECK_GL_ERROR();

	glGenBuffers( 1, &mDepthBuffer );
	glBindBuffer( GL_PIXEL_PACK_BUFFER, mDepthBuffer );
	glBufferData( GL_PIXEL_PACK_BUFFER, width * height * sizeof(GLuint), NULL, GL_DYNAMIC_DRAW );
	glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
	GV_CHECK_GL_ERROR();

	glGenTextures( 1, &mDepthTex );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, mDepthTex );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexImage2D( GL_TEXTURE_RECTANGLE_EXT, 0, GL_DEPTH24_STENCIL8_EXT, width, height, 0, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, NULL );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
	GV_CHECK_GL_ERROR();

	glGenFramebuffers( 1, &mFrameBuffer );
	glBindFramebuffer( GL_FRAMEBUFFER, mFrameBuffer );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_EXT, mColorTex, 0 );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, mDepthTex, 0 );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, mDepthTex, 0 );
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );
	GV_CHECK_GL_ERROR();

	// Create CUDA resources from OpenGL objects
	GV_CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer( &mColorResource, mColorBuffer, cudaGraphicsRegisterFlagsNone ) );
	GV_CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer( &mDepthResource, mDepthBuffer, cudaGraphicsRegisterFlagsNone ) );

	// Pass resources to the renderer
	/*_pipeline->editRenderer()->setColorResource(mColorResource);
	_pipeline->editRenderer()->setDepthResource(mDepthResource);*/
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::clearCache()
{
	_pipeline->clear();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::toggleDisplayOctree()
{
	mDisplayOctree = !mDisplayOctree;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::toggleDynamicUpdate()
{
	const bool status = _pipeline->hasDynamicUpdate();
	_pipeline->setDynamicUpdate( ! status );
}

/******************************************************************************
 * ...
 *
 * @param mode ...
 ******************************************************************************/
void SampleCore::togglePerfmonDisplay( uint mode )
{
	if ( mDisplayPerfmon )
	{
		mDisplayPerfmon = 0;
	}
	else
	{
		mDisplayPerfmon = mode;
	}
}

/******************************************************************************
 * ...
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
 * ...
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
	float3 lightPos = make_float3( pX, pY, pZ );
	//GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cLightPosition, &lightPos, sizeof( lightPos ), 0, cudaMemcpyHostToDevice ) );
}
