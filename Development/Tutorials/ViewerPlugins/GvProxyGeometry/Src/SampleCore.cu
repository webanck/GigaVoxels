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
#include <GvRendering/GvVolumeTreeRendererCUDA.h>
#include <GvUtils/GvProxyGeometry.h>
#include <GvPerfMon/GvPerformanceMonitor.h>
#include <GvCore/GvError.h>

// Project
#include "SphereProducer.h"
#include "SphereShader.h"

// GvViewer
#include <GvvApplication.h>
#include <GvvMainWindow.h>

// Cuda SDK
#include <helper_math.h>

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
:	GvvPipelineInterface()
,	_hasProxyGeometryOptimisation( false )
,	_colorTex( 0 )
,	_depthTex( 0 )
,	_frameBuffer( 0 )
,	_depthBuffer( 0 )
,	_displayOctree( false )
,	_displayPerfmon( 0 )
,	_maxVolTreeDepth( 5 )
,	_proxyGeometry( NULL )
{
	// Translation used to position the GigaVoxels data structure
	_translation[ 0 ] = -0.5f;
	_translation[ 1 ] = -0.5f;
	_translation[ 2 ] = -0.5f;

	// Light position
	_lightPosition = make_float3( 1.f, 1.f, 1.f );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
	delete _volumeTreeRenderer;
	delete _volumeTreeCache;
	delete _volumeTree;
	delete _producer;
	delete _proxyGeometry;
}

/******************************************************************************
 * Gets the name of this browsable
 *
 * @return the name of this browsable
 ******************************************************************************/
const char* SampleCore::getName() const
{
	return "ProxyGeometry";
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
	size_t nodeElemSize = NodeRes::numElements * sizeof( GvStructure::OctreeNode );
	size_t brickElemSize = RealBrickRes::numElements * GvCore::DataTotalChannelSize< DataType >::value;

	// Compute how many we can fit into the given memory size
	size_t nodePoolNumElems = NODEPOOL_MEMSIZE / nodeElemSize;
	size_t brickPoolNumElems = BRICKPOOL_MEMSIZE / brickElemSize;

	// Compute the resolution of the pools
	uint3 nodePoolRes = make_uint3( static_cast< uint >( floorf( powf( static_cast< float >( nodePoolNumElems ), 1.0f / 3.0f ) ) ) ) * NodeRes::get();
	uint3 brickPoolRes = make_uint3( static_cast< uint >( floorf( powf( static_cast< float >( brickPoolNumElems ), 1.0f / 3.0f ) ) ) ) * RealBrickRes::get();
	
	std::cout << "\nnodePoolRes: " << nodePoolRes << std::endl;
	std::cout << "brickPoolRes: " << brickPoolRes << std::endl;

	// Producer initialization
	_producer = new ProducerType();

	// Data structure initialization
	_volumeTree = new VolumeTreeType( nodePoolRes, brickPoolRes, 0 );
	_volumeTree->setMaxDepth( _maxVolTreeDepth );

	// Cache initialization
	_volumeTreeCache = new VolumeTreeCacheType( _volumeTree, _producer, nodePoolRes, brickPoolRes );

	// Renderer initialization
	_volumeTreeRenderer = new VolumeTreeRendererType( _volumeTree, _volumeTreeCache, _producer );

	// Proxy geometry  initialization
	_proxyGeometry = new GvProxyGeometry();
	setProxyGeometryOptimisation( true );
}

/******************************************************************************
 * Draw function called of frame
 ******************************************************************************/
void SampleCore::draw()
{
	// Peformance Monitor
	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT( frame );
	CUDAPM_START_EVENT( app_init_frame );

	// Compute the 2D AABB of a 3D model projected on screen
	GLdouble minX = 0.0;
	GLdouble minY = 0.0;
	GLdouble maxX = 0.0;
	GLdouble maxY = 0.0;
	if ( _hasProxyGeometryOptimisation )
	{
		// TO DO :
		// - replace gluProject() code to loop over proxy geometry (and avoid the ModelView/Projection multiplication)
		// - store buffered version of the OpenGL matrices (following the camera ?)
		//
		// TO DO :
		// WARNING, if the object pass behind the camera, there is also a 2D projected BBox... with Z > 1
		// Sam thing with object at far plane...
		// Frustum culling should be done first in 3D...
		// FIX this...
		GLdouble myModelViewMatrix[ 16 ];
		GLdouble myProjectionMatrix[ 16 ];
		glPushMatrix();
		glTranslatef( -0.5f, -0.5f, -0.5f );
		glGetDoublev( GL_MODELVIEW_MATRIX, myModelViewMatrix );
		glPopMatrix();
		glGetDoublev( GL_PROJECTION_MATRIX, myProjectionMatrix );
		GLint myViewport[ 4 ];
		glGetIntegerv( GL_VIEWPORT, myViewport );
		_proxyGeometry->getProjected2DBBox( myModelViewMatrix, myProjectionMatrix, myViewport, minX, minY, maxX, maxY );
		// View frustum culling optimization :
		// - don't render when outside of viewport
		if ( ( static_cast< int >( maxX ) == 0 )
			|| ( static_cast< int >( minX ) == _width )
			|| ( static_cast< int >( maxY ) == 0 )
			|| ( static_cast< int >( minY ) == _height )
			)
		{
			CUDAPM_STOP_EVENT( app_init_frame );
			CUDAPM_STOP_EVENT( frame );
			CUDAPM_STOP_FRAME;

			return;
		}

		/*_volumeTreeRenderer->_2DBBoxLB = make_uint2( minX, minY );
		_volumeTreeRenderer->_2DBBoxTR = make_uint2( maxX, maxY );*/

		// Set default active frame region for rendering
		//_volumeTreeRenderer->setProjectedBBox( make_uint4( minX, minY, maxX, maxY ) );
		_volumeTreeRenderer->setProjectedBBox( make_uint4( minX, minY, maxX - minX, maxY - minY ) );
	}
	else
	{
		/*_volumeTreeRenderer->_2DBBoxLB = make_uint2( 0, 0 );
		_volumeTreeRenderer->_2DBBoxTR = make_uint2( _width, _height );*/

		// Set default active frame region for rendering
		_volumeTreeRenderer->setProjectedBBox( make_uint4( 0, 0, _width, _height ) );
	}

	glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );

	//glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );

	//// Draw the octree where the sphere will be
	//glEnable( GL_DEPTH_TEST );
	//glPushMatrix();
	//// Translation used to position the GigaVoxels data structure
	//glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );
	//if ( _displayOctree )
	//{
	//	_volumeTree->displayDebugOctree();
	//}
	//glPopMatrix();
	//glDisable( GL_DEPTH_TEST );

	//// Copy the current scene into PBO
	//glBindBuffer( GL_PIXEL_PACK_BUFFER, _colorBuffer );
	//glReadPixels( 0, 0, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
	////glReadPixels( minX, minY, maxX - minX, maxY - minY, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
	//glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	//GV_CHECK_GL_ERROR();

	//glBindBuffer( GL_PIXEL_PACK_BUFFER, _depthBuffer );
	//glReadPixels( 0, 0, _width, _height, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0 );
	////glReadPixels( minX, minY, maxX - minX, maxY - minY, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0 );
	//glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	//GV_CHECK_GL_ERROR();

	if ( _displayOctree )
	{
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );

		// Display the GigaVoxels N3-tree space partitioning structure
		glEnable( GL_DEPTH_TEST );
		glPushMatrix();
		// Transformations used to position the GigaVoxels data structure
		/*glRotatef( _rotation[ 0 ], _rotation[ 1 ], _rotation[ 2 ], _rotation[ 3 ] );
		glScalef( _scale, _scale, _scale );*/
		glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );
		_volumeTree->displayDebugOctree();
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

	// Peformance Monitor
	// Render the scene into textures
	CUDAPM_STOP_EVENT( app_init_frame );

	// Extract view transformations
	float4x4 viewMatrix;
	float4x4 projectionMatrix;
	glGetFloatv( GL_MODELVIEW_MATRIX, viewMatrix._array );
	glGetFloatv( GL_PROJECTION_MATRIX, projectionMatrix._array );

	// Extract viewport
	GLint params[ 4 ];
	glGetIntegerv( GL_VIEWPORT, params );
	int4 viewport = make_int4( params[ 0 ], params[ 1 ], params[ 2 ], params[ 3 ] );

	// Build the world transformation matrix
	float4x4 modelMatrix;
	float x = -0.5f;
	float y = -0.5f;
	float z = -0.5f;
	glPushMatrix();
	glLoadIdentity();
	glTranslatef( x, y, z );
	glGetFloatv( GL_MODELVIEW_MATRIX, modelMatrix._array );
	glPopMatrix();

	// Render the GigaVoxels scene
	_volumeTreeRenderer->render( modelMatrix, viewMatrix, projectionMatrix, viewport );
	
	//// Upload changes into the textures
	//glBindBuffer( GL_PIXEL_UNPACK_BUFFER, _colorBuffer );
	//glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _colorTex );
	//if ( ! _hasProxyGeometryOptimisation )
	//{
	//	// Copy all buffer
	//	glTexSubImage2D( GL_TEXTURE_RECTANGLE_EXT, 0, 0, 0, _width, _height, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
	//}
	//else
	//{
	//	// Copy only a sub-part of the buffer
	//	glPushClientAttrib( GL_CLIENT_PIXEL_STORE_BIT );
	//	glPixelStorei( GL_UNPACK_ROW_LENGTH, _width );
	//	glPixelStorei( GL_UNPACK_SKIP_PIXELS, static_cast< int >( minX ) );
	//	glPixelStorei( GL_UNPACK_SKIP_ROWS, static_cast< int >( minY ) );
	//	glTexSubImage2D( GL_TEXTURE_RECTANGLE_EXT, 0,
	//					static_cast< int >( minX ), static_cast< int >( minY ),
	//					static_cast< int >( maxX ) - static_cast< int >( minX ), static_cast< int >( maxY ) - static_cast< int >( minY ),
	//					GL_RGBA, GL_UNSIGNED_BYTE, 0 );
	//	glPopClientAttrib();
	//}
	//glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
	//glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
		
	// Render the result to the screen
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();

	glEnable( GL_TEXTURE_RECTANGLE_EXT );
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _colorTex );

	GLint sMin = 0;
	GLint tMin = 0;
	GLint sMax = 0;
	GLint tMax = 0;

	// Draw a full screen quad
	if ( ! _hasProxyGeometryOptimisation )
	{
		sMin = 0;
		tMin = 0;
		sMax = _width;
		tMax = _height;

		glBegin( GL_QUADS );

			glColor3f( 1.0f, 1.0f, 1.0f );

			glTexCoord2i( sMin, tMin ); glVertex2i( -1, -1 );
			glTexCoord2i( sMax, tMin ); glVertex2i(  1, -1 );
			glTexCoord2i( sMax, tMax ); glVertex2i(  1,  1 );
			glTexCoord2i( sMin, tMax ); glVertex2i( -1,  1 );

		glEnd();
	}
	else
	{
		sMin = minX;
		tMin = minY;
		sMax = maxX;
		tMax = maxY;

		glBegin( GL_QUADS );

			glColor3f( 1.0f, 1.0f, 1.0f );

			/*glTexCoord2i( sMin, tMin ); glVertex2f( 2.f * ( minX / _width ) - 1.f, 2.f * ( minY / _height ) - 1.f );
			glTexCoord2i( sMax, tMin ); glVertex2f( 2.f * ( maxX / _width ) - 1.f, 2.f * ( minY / _height ) - 1.f );
			glTexCoord2i( sMax, tMax ); glVertex2f( 2.f * ( maxX / _width ) - 1.f, 2.f * ( maxY / _height ) - 1.f );
			glTexCoord2i( sMin, tMax ); glVertex2f( 2.f * ( minX / _width ) - 1.f, 2.f * ( maxY / _height ) - 1.f );*/

			glTexCoord2i( sMin, tMin ); glVertex3f( 2.f * ( minX / _width ) - 1.f, 2.f * ( minY / _height ) - 1.f, 0.f );
			glTexCoord2i( sMax, tMin ); glVertex3f( 2.f * ( maxX / _width ) - 1.f, 2.f * ( minY / _height ) - 1.f, 0.f );
			glTexCoord2i( sMax, tMax ); glVertex3f( 2.f * ( maxX / _width ) - 1.f, 2.f * ( maxY / _height ) - 1.f, 0.f );
			glTexCoord2i( sMin, tMax ); glVertex3f( 2.f * ( minX / _width ) - 1.f, 2.f * ( maxY / _height ) - 1.f, 0.f );

		glEnd();
	}

	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
	glDisable( GL_TEXTURE_RECTANGLE_EXT );

	// Texture : final color write on 2D projected BBox
	if ( _hasProxyGeometryOptimisation )
	{
		glBegin( GL_LINE_LOOP );
			glColor3f( 1.0f, 1.0f, 1.0f );
			glVertex2f( 2.f * ( minX / _width ) - 1.f, 2.f * ( minY / _height ) - 1.f );
			glVertex2f( 2.f * ( maxX / _width ) - 1.f, 2.f * ( minY / _height ) - 1.f );
			glVertex2f( 2.f * ( maxX / _width ) - 1.f, 2.f * ( maxY / _height ) - 1.f );
			glVertex2f( 2.f * ( minX / _width ) - 1.f, 2.f * ( maxY / _height ) - 1.f );
		glEnd();
	}

	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();

	// TEST - optimization due to early unmap() graphics resource from GigaVoxels
	//_volumeTreeRenderer->doPostRender();
	
	// Update GigaVoxels info
	_volumeTreeRenderer->nextFrame();

	CUDAPM_STOP_EVENT( frame );
	CUDAPM_STOP_FRAME;

	// Display the GigaVoxels performance monitor (if it has been activated during GigaVoxels compilation)
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
	_width = width;
	_height = height;

	// Reset default active frame region for rendering
	_volumeTreeRenderer->setProjectedBBox( make_uint4( 0, 0, _width, _height ) );

	// Re-init Perfmon subsystem
	CUDAPM_RESIZE(make_uint2(_width, _height));

	// Disconnect all registered graphics resources
	_volumeTreeRenderer->resetGraphicsResources();

	// Create frame-dependent objects

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
	if ( _displayOctree )
	{
		_volumeTreeRenderer->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		_volumeTreeRenderer->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	}
	else
	{
		_volumeTreeRenderer->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
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
	_volumeTreeRenderer->resetGraphicsResources();

	if ( _displayOctree )
	{
		_volumeTreeRenderer->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		_volumeTreeRenderer->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	}
	else
	{
		_volumeTreeRenderer->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
	}
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

	_volumeTree->setMaxDepth( _maxVolTreeDepth );
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

	_volumeTree->setMaxDepth( _maxVolTreeDepth );
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
	const uint3& nodeTileResolution = _volumeTree->getNodeTileResolution().get();

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
	const uint3& brickResolution = _volumeTree->getBrickResolution().get();

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
	return _volumeTree->getMaxDepth();
}

/******************************************************************************
 * Set the max depth.
 *
 * @param pValue the max depth
 ******************************************************************************/
void SampleCore::setRendererMaxDepth( unsigned int pValue )
{
	_volumeTree->setMaxDepth( pValue );
}

/******************************************************************************
 * Get the max number of requests of node subdivisions.
 *
 * @return the max number of requests
 ******************************************************************************/
unsigned int SampleCore::getCacheMaxNbNodeSubdivisions() const
{
	return _volumeTreeCache->getMaxNbNodeSubdivisions();
}

/******************************************************************************
 * Set the max number of requests of node subdivisions.
 *
 * @param pValue the max number of requests
 ******************************************************************************/
void SampleCore::setCacheMaxNbNodeSubdivisions( unsigned int pValue )
{
	_volumeTreeCache->setMaxNbNodeSubdivisions( pValue );
}

/******************************************************************************
 * Get the max number of requests of brick of voxel loads.
 *
 * @return the max number of requests
 ******************************************************************************/
unsigned int SampleCore::getCacheMaxNbBrickLoads() const
{
	return _volumeTreeCache->getMaxNbBrickLoads();
}

/******************************************************************************
 * Set the max number of requests of brick of voxel loads.
 *
 * @param pValue the max number of requests
 ******************************************************************************/
void SampleCore::setCacheMaxNbBrickLoads( unsigned int pValue )
{
	_volumeTreeCache->setMaxNbBrickLoads( pValue );
}

/******************************************************************************
 * Set the request strategy indicating if, during data structure traversal,
 * priority of requests is set on brick loads or on node subdivisions first.
 *
 * @param pFlag the flag indicating the request strategy
 ******************************************************************************/
void SampleCore::setRendererPriorityOnBricks( bool pFlag )
{
	_volumeTreeRenderer->setPriorityOnBricks( pFlag );
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
	_volumeTreeRenderer->setClearColor( make_uchar4( pRed, pGreen, pBlue, pAlpha ) );
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
	_lightPosition.x = pX - _translation[ 0 ];
	_lightPosition.y = pY - _translation[ 1 ];
	_lightPosition.z = pZ - _translation[ 2 ];

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
 * Tell wheter or not the proxy geometry optimization is activated.
 *
 * @return the flag telling wheter or not the proxy geometry optimization is activated
 ******************************************************************************/
bool SampleCore::hasProxyGeometryOptimisation() const
{
	return _hasProxyGeometryOptimisation;
}

/******************************************************************************
 * Set the flag telling wheter or not the proxy geometry optimization is activated.
 *
 * @param pFlag a flag telling wheter or not the proxy geometry optimization is activated
 ******************************************************************************/
void SampleCore::setProxyGeometryOptimisation( bool pFlag )
{
	_hasProxyGeometryOptimisation = pFlag;
}
