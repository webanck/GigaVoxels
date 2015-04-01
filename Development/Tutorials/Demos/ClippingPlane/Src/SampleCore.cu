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
#include <GvRendering/GvRendererCUDA.h>
#include <GvUtils/GvSimplePipeline.h>
#include <GvUtils/GvSimpleHostProducer.h>
#include <GvUtils/GvSimpleHostShader.h>
#include <GvUtils/GvCommonGraphicsPass.h>
#include <GvCore/GvError.h>
#include <GvPerfMon/GvPerformanceMonitor.h>

// Project
#include "ProducerKernel.h"
#include "ShaderKernel.h"

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvRendering;
using namespace GvUtils;

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
:	_pipeline( NULL )
,	_renderer( NULL )
,	_graphicsEnvironment( NULL )
,	_colorTex( 0 )
,	_depthTex( 0 )
,	_frameBuffer( 0 )
,	_depthBuffer( 0 )
,	_displayOctree( false )
,	_displayPerfmon( 0 )
,	_maxVolTreeDepth( 5 )
{
	_clippingPlaneNbPoints = 0;
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
	delete _pipeline;
	delete _graphicsEnvironment;
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

	//// Compute the size of one element in the cache for nodes and bricks
	//size_t nodeElemSize = NodeRes::numElements * sizeof( GvStructure::OctreeNode );
	//size_t brickElemSize = RealBrickRes::numElements * GvCore::DataTotalChannelSize< DataType >::value;

	//// Compute how many we can fit into the given memory size
	//size_t nodePoolNumElems = NODEPOOL_MEMSIZE / nodeElemSize;
	//size_t brickPoolNumElems = BRICKPOOL_MEMSIZE / brickElemSize;

	//// Compute the resolution of the pools
	//uint3 nodePoolRes = make_uint3( static_cast< uint >( floorf( powf( static_cast< float >( nodePoolNumElems ), 1.0f / 3.0f ) ) ) ) * NodeRes::get();
	//uint3 brickPoolRes = make_uint3( static_cast< uint >( floorf( powf( static_cast< float >( brickPoolNumElems ), 1.0f / 3.0f ) ) ) ) * RealBrickRes::get();
	//
	//std::cout << "\nnodePoolRes: " << nodePoolRes << std::endl;
	//std::cout << "brickPoolRes: " << brickPoolRes << std::endl;

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
	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );

	// Graphics environment creation
	_graphicsEnvironment = new GvCommonGraphicsPass();
}

///******************************************************************************
// * Used to display the N-tree
// *
// * @param p1 Position
// * @param p2 Position
// ******************************************************************************/
//void drawCube( const float3& p1, const float3& p2 )
//{
//	glBegin(GL_QUADS);
//	// Front Face
//	glVertex3f(p1.x, p1.y, p2.z);	// Bottom Left Of The Texture and Quad
//	glVertex3f(p1.x, p2.y, p2.z);	// Top Left Of The Texture and Quad
//	glVertex3f(p2.x, p2.y, p2.z);	// Top Right Of The Texture and Quad
//	glVertex3f(p2.x, p1.y, p2.z);	// Bottom Right Of The Texture and Quad
//
//	// Back Face
//	glVertex3f(p1.x, p1.y, p1.z);	// Bottom Right Of The Texture and Quad
//	glVertex3f(p2.x, p1.y, p1.z);	// Bottom Left Of The Texture and Quad
//	glVertex3f(p2.x, p2.y, p1.z);	// Top Left Of The Texture and Quad
//	glVertex3f(p1.x, p2.y, p1.z);	// Top Right Of The Texture and Quad
//
//	// Top Face
//	glVertex3f(p1.x, p2.y, p1.z);	// Top Left Of The Texture and Quad
//	glVertex3f(p2.x, p2.y, p1.z);	// Top Right Of The Texture and Quad
//	glVertex3f(p2.x, p2.y, p2.z);	// Bottom Right Of The Texture and Quad
//	glVertex3f(p1.x, p2.y, p2.z);	// Bottom Left Of The Texture and Quad
//
//	// Bottom Face
//	glVertex3f(p1.x, p1.y, p1.z);	// Top Right Of The Texture and Quad
//	glVertex3f(p1.x, p1.y, p2.z);	// Bottom Right Of The Texture and Quad
//	glVertex3f(p2.x, p1.y, p2.z);	// Bottom Left Of The Texture and Quad
//	glVertex3f(p2.x, p1.y, p1.z);	// Top Left Of The Texture and Quad
//
//	// Right face
//	glVertex3f(p2.x, p1.y, p1.z);	// Bottom Right Of The Texture and Quad
//	glVertex3f(p2.x, p1.y, p2.z);	// Bottom Left Of The Texture and Quad
//	glVertex3f(p2.x, p2.y, p2.z);	// Top Left Of The Texture and Quad
//	glVertex3f(p2.x, p2.y, p1.z);	// Top Right Of The Texture and Quad
//
//	// Left Face
//	glVertex3f(p1.x, p1.y, p1.z);	// Bottom Left Of The Texture and Quad
//	glVertex3f(p1.x, p2.y, p1.z);	// Top Left Of The Texture and Quad
//	glVertex3f(p1.x, p2.y, p2.z);	// Top Right Of The Texture and Quad
//	glVertex3f(p1.x, p1.y, p2.z);	// Bottom Right Of The Texture and Quad
//
//	glEnd();
//}

/******************************************************************************
 * Draw function called of frame
 ******************************************************************************/
void SampleCore::setClippingPlaneGeometry( const float3* pPoints, unsigned int pNbPoints )
{
	_clippingPlaneNbPoints = pNbPoints;

	for ( unsigned int i = 0; i < pNbPoints; i++ )
	{
		_clippingPlaneGeometry[ i ] = pPoints[ i ];
	}
}

/******************************************************************************
 * Used to display the N-tree
 *
 * @param p1 Position
 * @param p2 Position
 ******************************************************************************/
void mydrawCube( const float3& p1, const float3& p2 )
{
	glBegin(GL_QUADS);
	// Front Face
	glVertex3f(p1.x, p1.y, p2.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p2.z);	// Top Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p2.z);	// Top Right Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p2.z);	// Bottom Right Of The Texture and Quad

	// Back Face
	glVertex3f(p1.x, p1.y, p1.z);	// Bottom Right Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p1.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p1.z);	// Top Left Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p1.z);	// Top Right Of The Texture and Quad

	// Top Face
	glVertex3f(p1.x, p2.y, p1.z);	// Top Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p1.z);	// Top Right Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p2.z);	// Bottom Right Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p2.z);	// Bottom Left Of The Texture and Quad

	// Bottom Face
	glVertex3f(p1.x, p1.y, p1.z);	// Top Right Of The Texture and Quad
	glVertex3f(p1.x, p1.y, p2.z);	// Bottom Right Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p2.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p1.z);	// Top Left Of The Texture and Quad

	// Right face
	glVertex3f(p2.x, p1.y, p1.z);	// Bottom Right Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p2.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p2.z);	// Top Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p1.z);	// Top Right Of The Texture and Quad

	// Left Face
	glVertex3f(p1.x, p1.y, p1.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p1.z);	// Top Left Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p2.z);	// Top Right Of The Texture and Quad
	glVertex3f(p1.x, p1.y, p2.z);	// Bottom Right Of The Texture and Quad

	glEnd();
}

/******************************************************************************
 * Draw function called of frame
 ******************************************************************************/
void SampleCore::renderClippinPlane()
{
	// Draw only if there are at least 3 points
	if ( _clippingPlaneNbPoints > 2 )
	{
		glColor3f( 1.f, 1.f, 1.f );
		glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
		mydrawCube( make_float3( 0.f, 0.f, 0.f ) + make_float3( -0.5f, -0.5f, -0.5f ), make_float3( 1.f, 1.f, 1.f ) + make_float3( -0.5f, -0.5f, -0.5f ) );
		glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

		// [ 1 ] - Draw clipping plane
		glColor3f( 0.f, 1.f, 1.f );
		//glBegin( GL_POLYGON );
		glLineWidth( 3.0f );
		glBegin( GL_LINE_LOOP );
		for ( unsigned int i = 0; i < _clippingPlaneNbPoints; i++ )
		{
			glVertex3f( _clippingPlaneGeometry[ i ].x, _clippingPlaneGeometry[ i ].y, _clippingPlaneGeometry[ i ].z );
		}
		glEnd();
	}

	//	// [ 2 ] - Draw clipping plane projected on bottom (z = 0)
	//	glBegin( GL_POLYGON );
	//	//glBegin( GL_LINE_LOOP );
	//	for ( unsigned int i = 0; i < _clippingPlaneNbPoints; i++ )
	//	{
	//		glVertex3f( _clippingPlaneGeometry[ i ].x, _clippingPlaneGeometry[ i ].y, 0.0f + (-0.5f ) );
	//	}
	//	glEnd();

	//	// [ 3 ] - Draw ribbon
	//	glBegin( GL_TRIANGLE_STRIP );
	//	//glBegin( GL_LINES );
	//	for ( unsigned int i = 0; i < _clippingPlaneNbPoints; i++ )
	//	{
	//		glVertex3f( _clippingPlaneGeometry[ i ].x, _clippingPlaneGeometry[ i ].y, 0.0f + (-0.5f ) );
	//		glVertex3f( _clippingPlaneGeometry[ i ].x, _clippingPlaneGeometry[ i ].y, _clippingPlaneGeometry[ i ].z );
	//	}
	//	glVertex3f( _clippingPlaneGeometry[ 0 ].x, _clippingPlaneGeometry[ 0 ].y, 0.0f + (-0.5f ) );
	//	glVertex3f( _clippingPlaneGeometry[ 0 ].x, _clippingPlaneGeometry[ 0 ].y, _clippingPlaneGeometry[ 0 ].z );
	//	glEnd();
	//}

	// ------------------------------------------
	glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE );
	glPushMatrix();
	glMultMatrixd( _clippingPlaneMatrix );
		//glPolygonMode( GL_FRONT_AND_BACK , GL_LINE );
		//glColor3f( 1.f, 1.f, 1.f );
		mydrawCube( make_float3( 0.f, 0.f, 0.f ) - make_float3( 3 * sqrtf( 3.f ) * 0.5f, 3 * sqrtf( 3.f ) * 0.5f, 3 * sqrtf( 3.f ) * 0.5f ) - make_float3( 0.0f, 0.0f, 3 * sqrtf( 3.f ) * 0.5f ), make_float3( 3 * sqrtf( 3.f ), 3 * sqrtf( 3.f ), 3 * sqrtf( 3.f ) ) - make_float3( 3 * sqrtf( 3.f ) * 0.5f, 3 * sqrtf( 3.f ) * 0.5f, 3 * sqrtf( 3.f ) * 0.5f ) - make_float3( 0.0f, 0.0f, 3 * sqrtf( 3.f ) * 0.5f ) );
		//glPolygonMode( GL_FRONT_AND_BACK , GL_FILL );
	glPopMatrix();
	glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );
	// ------------------------------------------

	// ------------------------------------------
	glPushMatrix();
	glMultMatrixd( _clippingPlaneMatrix );
		glPolygonMode( GL_FRONT_AND_BACK , GL_LINE );
		glColor3f( 1.f, 0.f, 0.f );
		mydrawCube( make_float3( 0.f, 0.f, 0.f ) - make_float3( 3 * sqrtf( 3.f ) * 0.5f, 3 * sqrtf( 3.f ) * 0.5f, 3 * sqrtf( 3.f ) * 0.5f ) - make_float3( 0.0f, 0.0f, 3 * sqrtf( 3.f ) * 0.5f ), make_float3( 3 * sqrtf( 3.f ), 3 * sqrtf( 3.f ), 3 * sqrtf( 3.f ) ) - make_float3( 3 * sqrtf( 3.f ) * 0.5f, 3 * sqrtf( 3.f ) * 0.5f, 3 * sqrtf( 3.f ) * 0.5f ) - make_float3( 0.0f, 0.0f, 3 * sqrtf( 3.f ) * 0.5f ) );
		glPolygonMode( GL_FRONT_AND_BACK , GL_FILL );
	glPopMatrix();
	// ------------------------------------------
}

/******************************************************************************
 * Draw function called of frame
 ******************************************************************************/
void SampleCore::setClippingPlaneMatrix( const GLdouble* pClippingPlaneMatrix )
{
	// to do : utilisr un memset...
	for ( unsigned int i = 0; i < 16; i++ )
	{
		_clippingPlaneMatrix[ i ] = pClippingPlaneMatrix[ i ];
	}
}

/******************************************************************************
 * Draw function called of frame
 ******************************************************************************/
void SampleCore::draw()
{
	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT( frame );
	CUDAPM_START_EVENT( app_init_frame );

	glMatrixMode( GL_MODELVIEW );

	glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );
//	if ( _displayOctree )
	//{
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );

		//// Display the GigaVoxels N3-tree space partitioning structure
		//glEnable( GL_DEPTH_TEST );
		//glPushMatrix();
		//// Translation used to position the GigaVoxels data structure
		//glTranslatef( -0.5f, -0.5f, -0.5f );
		//_pipeline->editDataStructure()->render();
		//glPopMatrix();
		//glDisable( GL_DEPTH_TEST );

		// Render clipping plane
		glEnable( GL_DEPTH_TEST );
	//	glEnable( GL_BLEND );
	//	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	//	glColor4f( 0.f, 1.f, 0.f, 1.0f );
		//glColor4f( 0.f, 1.f, 0.f, 0.2f );
		//glEnable( GL_CULL_FACE );
	//	glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE );
		renderClippinPlane();
		//glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );
	//	glDisable( GL_CULL_FACE );
	//	glDisable( GL_BLEND );
		glDisable( GL_DEPTH_TEST );

		// Clear the depth PBO (pixel buffer object) by reading from the previously cleared FBO (frame buffer object)
		glBindBuffer( GL_PIXEL_PACK_BUFFER, _depthBuffer );
		glReadPixels( 0, 0, _width, _height, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0 );
		glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
		GV_CHECK_GL_ERROR();
	//}
	//else
	//{
	//	glClear( GL_COLOR_BUFFER_BIT );
	//}
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );

	// Extract view transformations
	float4x4 viewMatrix;
	float4x4 projectionMatrix;
	glGetFloatv(GL_MODELVIEW_MATRIX, viewMatrix._array);
	glGetFloatv(GL_PROJECTION_MATRIX, projectionMatrix._array);

	// Extract viewport
	GLint params[4];
	glGetIntegerv(GL_VIEWPORT, params);
	int4 viewport = make_int4(params[0], params[1], params[2], params[3]);

	// render the scene into textures
	CUDAPM_STOP_EVENT(app_init_frame);

	// Build the world transformation matrix
	float4x4 modelMatrix;
	glPushMatrix();
	glLoadIdentity();
	// Translation used to position the GigaVoxels data structure
	glTranslatef( -0.5f, -0.5f, -0.5f );
	glGetFloatv( GL_MODELVIEW_MATRIX, modelMatrix._array );
	glPopMatrix();

	// Render
	_pipeline->execute( modelMatrix, viewMatrix, projectionMatrix, viewport );
	
	// Render the result to the screen
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	glEnable(GL_TEXTURE_RECTANGLE_EXT);
	glDisable(GL_DEPTH_TEST);

	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _colorTex );
	
	GLint sMin = 0;
	GLint tMin = 0;
	GLint sMax = _width;
	GLint tMax = _height;

	glBegin(GL_QUADS);
		glColor3f(1.0f, 1.0f, 1.0f);
		glTexCoord2i(sMin, tMin); glVertex2i(-1, -1);
		glTexCoord2i(sMax, tMin); glVertex2i( 1, -1);
		glTexCoord2i(sMax, tMax); glVertex2i( 1,  1);
		glTexCoord2i(sMin, tMax); glVertex2i(-1,  1);
	glEnd();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);

	glDisable(GL_TEXTURE_RECTANGLE_EXT);

	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	// TEST - optimization due to early unmap() graphics resource from GigaVoxels
	///*_pipeline->editRenderer()*/_renderer->doPostRender();
	
	// Update GigaVoxels info
	/*_pipeline->editRenderer()*/_renderer->nextFrame();

	CUDAPM_STOP_EVENT(frame);
	CUDAPM_STOP_FRAME;

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
	/*_pipeline->editRenderer()*/_renderer->setProjectedBBox( make_uint4( 0, 0, _width, _height ) );

	// Re-init Perfmon subsystem
	CUDAPM_RESIZE(make_uint2(_width, _height));

	// Create frame-dependent objects

	// Disconnect all registered graphics resources
	/*_pipeline->editRenderer()*/_renderer->resetGraphicsResources();
	
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

	glGenFramebuffers(1, &_frameBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, _frameBuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_EXT, _colorTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, _depthTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, _depthTex, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	GV_CHECK_GL_ERROR();

	// Create CUDA resources from OpenGL objects
//	if ( _displayOctree )
	//{
		/*_pipeline->editRenderer()*/_renderer->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		/*_pipeline->editRenderer()*/_renderer->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	//}
	//else
	//{
	//	/*_pipeline->editRenderer()*/_renderer->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
	//}
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

	//// Disconnect all registered graphics resources
	///*_pipeline->editRenderer()*/_renderer->resetGraphicsResources();

	//if ( _displayOctree )
	//{
	//	/*_pipeline->editRenderer()*/_renderer->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
	//	/*_pipeline->editRenderer()*/_renderer->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	//}
	//else
	//{
	//	/*_pipeline->editRenderer()*/_renderer->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
	//}
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
	// Due to glTranslatef( -0.5f, -0.5f, -0.5f ) used to place the GigaVoxels object centered at [0;0;0],
	// light position must be displaced accordingly.
	float3 lightPos = make_float3( pX, pY, pZ );
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cLightPosition, &lightPos, sizeof( lightPos ), 0, cudaMemcpyHostToDevice ) );
}
