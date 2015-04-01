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
#include <GvStructure/GvVolumeTreeCache.h>
#include <GvRenderer/GvVolumeTreeRendererCUDA.h>
#include <GvRenderer/GvGraphicsInteroperabiltyHandler.h>
#include <GvPerfMon/CUDAPerfMon.h>
#include <GvCore/GvError.h>

// Project
#include "ProducerLoad.h"
#include "ShaderLoad.h"
#include "VolumeProducerBricks.h"

// GvViewer
#include <GvvApplication.h>
#include <GvvMainWindow.h>

// Cuda SDK
#include <helper_math.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvRenderer;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

// Defines the size allowed for each type of pool
#define NODEPOOL_MEMSIZE	(8*1024*1024)		// 8 Mo
#define BRICKPOOL_MEMSIZE	(256*1024*1024)		// 256 Mo

/******************************************************************************
 * Constructor
 ******************************************************************************/
SampleCore::SampleCore()
:	GvvPipelineInterface()
,	mColorTex( 0 )
,	mDepthTex( 0 )
,	mFrameBuffer( 0 )
,	mDepthBuffer( 0 )
,	mDisplayOctree( false )
,	mDisplayPerfmon( 0 )
,	mMaxVolTreeDepth( 16 )
{
	_dofParameters = make_float3( 0.f, 0.f, 0.f );
	_lightPosition = make_float3(  1.f, 1.f, 1.f );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
	delete mRenderer;
	delete mVolumeTree;
	delete mProducer;

	// TO DO : destroy cache
	// ...
}

/******************************************************************************
 * Gets the name of this browsable
 *
 * @return the name of this browsable
 ******************************************************************************/
const char* SampleCore::getName() const
{
	return "DepthOfField";
}

/******************************************************************************
 * Init
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

	// Here we compute the size of node and brick pools.
	size_t voxelFullSize = GvCore::DataTotalChannelSize<DataType>::value;

	size_t nodePoolNumElems = NODEPOOL_MEMSIZE / sizeof(GvStructure::OctreeNode);
	size_t brickPoolNumElems = BRICKPOOL_MEMSIZE / voxelFullSize;

	float nodePoolResF = powf((float)nodePoolNumElems, 1.0f / 3.0f);
	uint nodePoolResAxis = (uint)ceilf(nodePoolResF);
	uint3 nodePoolRes = make_uint3(nodePoolResAxis);

	float brickPoolResF = powf((float)brickPoolNumElems, 1.0f / 3.0f);
	uint brickPoolResAxis = (uint)/*ceilf*/(brickPoolResF);
	uint3 brickPoolRes = make_uint3(brickPoolResAxis);

	std::cout << "voxelSize " << voxelFullSize << std::endl;
	std::cout << "nodePoolRes: " << nodePoolRes << std::endl;
	std::cout << "brickPoolRes: " << brickPoolRes << std::endl;

	// Sata structure
	mVolumeTree = new VolumeTreeType( nodePoolRes, brickPoolRes, 0 );

	QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	QString filename = dataRepository + QDir::separator() + QString( "Voxels" ) + QDir::separator() + QString( "xyzrgb_dragon512_BR8_B1" ) + QDir::separator() + QString( "xyzrgb_dragon512" );
	VolumeProducerBricks< DataType >* volProducerBrick =
		//new VolumeProducerBricks< DataType >( "../../media/voxels/invalidData/meshCompressed",
		new VolumeProducerBricks< DataType >( filename.toStdString(),
			make_uint3( 512 ), BrickRes::get(), BrickBorderSize, true );

	// Producer
	mProducer = new ProducerType(64 * 1024 * 1024, nodePoolRes.x * nodePoolRes.y * nodePoolRes.z);
	mProducer->attachProducer(volProducerBrick);

	// Cache
	mVolumeTreeCache = new VolumeTreeCacheType( mVolumeTree, mProducer, nodePoolRes, brickPoolRes );

	// Renderer
	mRenderer = new RendererType( mVolumeTree, mVolumeTreeCache, mProducer );

	// Configure the renderer
	mVolumeTree->setMaxDepth( mMaxVolTreeDepth );

	// Custom initialization
	// Note : this could be done via an XML settings file loaded at initialization
	setAperture( 1.f );
	setFocalLength( 4.f );
	setPlaneInFocus( 3.f );
}

/******************************************************************************
 * Draw
 ******************************************************************************/
void SampleCore::draw()
{
	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT( frame );
	CUDAPM_START_EVENT( app_init_frame );

	glMatrixMode( GL_MODELVIEW );
	
	glBindFramebuffer( GL_FRAMEBUFFER, mFrameBuffer );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );

	if ( mDisplayOctree )
	{
		// Display the GigaVoxels N3-tree space partitioning structure
		glEnable( GL_DEPTH_TEST );
		glPushMatrix();
		glTranslatef( -0.5f, -0.5f, -0.5f );
		mVolumeTree->displayDebugOctree();
		glPopMatrix();
		glDisable( GL_DEPTH_TEST );
	}

	// Clear the depth PBO (pixel buffer object) by reading from the previously cleared FBO (frame buffer object)
	glBindBuffer( GL_PIXEL_PACK_BUFFER, mDepthBuffer );
	glReadPixels( 0, 0, mWidth, mHeight, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0 );
	glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
	GV_CHECK_GL_ERROR();

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

	//for ( float z = -3.5f; z <= 2.5f; z += 2.0f )
	for ( float z = -3.5f; z <= 2.5f; z += 1.5f )
	//for ( float z = -2.5f; z <= 1.5f; z += 1.0f )
	{
		// Build and extract tree transformations
		float4x4 modelMatrix;

		glPushMatrix();
		glLoadIdentity();
		glTranslatef( -0.5f, -0.5f, z );
		glGetFloatv( GL_MODELVIEW_MATRIX, modelMatrix._array );
		glPopMatrix();

		// Render the scene into textures
		mRenderer->render( modelMatrix, viewMatrix, projectionMatrix, viewport );
	}

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
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, mColorTex );

	// Draw a full screen quad
	GLint sMin = 0;
	GLint tMin = 0;
	GLint sMax = mWidth;
	GLint tMax = mHeight;
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

	// TEST - optimization due to early unmap() graphics resource from GigaVoxels
	//_renderer->doPostRender();
	
	// Update GigaVoxels info
	mRenderer->nextFrame();

	CUDAPM_STOP_EVENT( frame );
	CUDAPM_STOP_FRAME;

	// Display the GigaVoxels performance monitor (if it has been activated during GigaVoxels compilation)
	if ( mDisplayPerfmon )
	{
		GvPerfMon::CUDAPerfMon::getApplicationPerfMon().displayFrameGL( mDisplayPerfmon - 1 );
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

	// Reset default active frame region for rendering
	mRenderer->setProjectedBBox( make_uint4( 0, 0, mWidth, mHeight ) );

	// Re-init Perfmon subsystem
	CUDAPM_RESIZE(make_uint2(mWidth, mHeight));

	/*uchar *timersMask = GvPerfMon::CUDAPerfMon::getApplicationPerfMon().getKernelTimerMask();
	cudaMemset(timersMask, 255, mWidth * mHeight);*/

	// Disconnect all registered graphics resources
	mRenderer->resetGraphicsResources();

	// Create frame-dependent objects
	
	if (mDepthBuffer)
		glDeleteBuffers(1, &mDepthBuffer);

	if (mColorTex)
		glDeleteTextures(1, &mColorTex);
	if (mDepthTex)
		glDeleteTextures(1, &mDepthTex);

	if (mFrameBuffer)
		glDeleteFramebuffers(1, &mFrameBuffer);

	glGenTextures(1, &mColorTex);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, mColorTex);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);
	GV_CHECK_GL_ERROR();

	glGenBuffers(1, &mDepthBuffer);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, mDepthBuffer);
	glBufferData(GL_PIXEL_PACK_BUFFER, width * height * sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	GV_CHECK_GL_ERROR();

	glGenTextures(1, &mDepthTex);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, mDepthTex);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, GL_DEPTH24_STENCIL8_EXT, width, height, 0, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, NULL);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);
	GV_CHECK_GL_ERROR();

	glGenFramebuffers(1, &mFrameBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, mFrameBuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_EXT, mColorTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, mDepthTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, mDepthTex, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	GV_CHECK_GL_ERROR();

	// Create CUDA resources from OpenGL objects
	mRenderer->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, mColorTex, GL_TEXTURE_RECTANGLE_EXT );
	mRenderer->connect( GvGraphicsInteroperabiltyHandler::eDepthReadWriteSlot, mDepthBuffer );
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
 ******************************************************************************/
void SampleCore::togglePerfmonDisplay(uint mode)
{
	if (mDisplayPerfmon)
		mDisplayPerfmon = 0;
	else
		mDisplayPerfmon = mode;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::incMaxVolTreeDepth()
{
	if (mMaxVolTreeDepth < 32)
		mMaxVolTreeDepth++;

	mVolumeTree->setMaxDepth(mMaxVolTreeDepth);
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::decMaxVolTreeDepth()
{
	if (mMaxVolTreeDepth > 0)
		mMaxVolTreeDepth--;

	mVolumeTree->setMaxDepth(mMaxVolTreeDepth);
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
	const uint3& nodeTileResolution = mVolumeTree->getNodeTileResolution().get();

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
	const uint3& brickResolution = mVolumeTree->getBrickResolution().get();

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
	return mVolumeTree->getMaxDepth();
}

/******************************************************************************
 * Set the max depth.
 *
 * @param pValue the max depth
 ******************************************************************************/
void SampleCore::setRendererMaxDepth( unsigned int pValue )
{
	mVolumeTree->setMaxDepth( pValue );
}

/******************************************************************************
 * Get the max number of requests of node subdivisions.
 *
 * @return the max number of requests
 ******************************************************************************/
unsigned int SampleCore::getCacheMaxNbNodeSubdivisions() const
{
	return mVolumeTreeCache->getMaxNbNodeSubdivisions();
}

/******************************************************************************
 * Set the max number of requests of node subdivisions.
 *
 * @param pValue the max number of requests
 ******************************************************************************/
void SampleCore::setCacheMaxNbNodeSubdivisions( unsigned int pValue )
{
	mVolumeTreeCache->setMaxNbNodeSubdivisions( pValue );
}

/******************************************************************************
 * Get the max number of requests of brick of voxel loads.
 *
 * @return the max number of requests
 ******************************************************************************/
unsigned int SampleCore::getCacheMaxNbBrickLoads() const
{
	return mVolumeTreeCache->getMaxNbBrickLoads();
}

/******************************************************************************
 * Set the max number of requests of brick of voxel loads.
 *
 * @param pValue the max number of requests
 ******************************************************************************/
void SampleCore::setCacheMaxNbBrickLoads( unsigned int pValue )
{
	mVolumeTreeCache->setMaxNbBrickLoads( pValue );
}

/******************************************************************************
 * Set the request strategy indicating if, during data structure traversal,
 * priority of requests is set on brick loads or on node subdivisions first.
 *
 * @param pFlag the flag indicating the request strategy
 ******************************************************************************/
void SampleCore::setRendererPriorityOnBricks( bool pFlag )
{
	mRenderer->setPriorityOnBricks( pFlag );
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
	mRenderer->setClearColor( make_uchar4( pRed, pGreen, pBlue, pAlpha ) );
}

/******************************************************************************
 * Get the aperture
 *
 * @return the aperture
 ******************************************************************************/
float SampleCore::getAperture() const
{
	return _dofParameters.x;
}

/******************************************************************************
 * Set the aperture
 *
 * @param pValue the new value
 ******************************************************************************/
void SampleCore::setAperture( float pValue )
{
	_dofParameters.x = pValue;

	// Update device memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cDofParameters, (&_dofParameters), sizeof( _dofParameters ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Get the focal length
 *
 * @return the focal length
 ******************************************************************************/
float SampleCore::getFocalLength() const
{
	return _dofParameters.y;
}

/******************************************************************************
 * Set the focal length
 *
 * @param pValue the new value
 ******************************************************************************/
void SampleCore::setFocalLength( float pValue )
{
	_dofParameters.y = pValue;

	// Update device memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cDofParameters, (&_dofParameters), sizeof( _dofParameters ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Get the plane in focus
 *
 * @return the plane in focus
 ******************************************************************************/
float SampleCore::getPlaneInFocus() const
{
	return _dofParameters.z;
}

/******************************************************************************
 * Set the plane in focus
 *
 * @param pValue the new value
 ******************************************************************************/
void SampleCore::setPlaneInFocus( float pValue )
{
	_dofParameters.z = pValue;

	// Update device memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cDofParameters, &_dofParameters, sizeof( _dofParameters ), 0, cudaMemcpyHostToDevice ) );
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
	// Here no glTranslate() has been used, so nothing to do.
	_lightPosition.x = pX;
	_lightPosition.y = pY;
	_lightPosition.z = pZ;

	// Update device memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cLightPosition, &_lightPosition, sizeof( _lightPosition ), 0, cudaMemcpyHostToDevice ) );
}
