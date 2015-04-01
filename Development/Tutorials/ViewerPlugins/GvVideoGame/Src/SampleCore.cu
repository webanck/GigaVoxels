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
#include <GvUtils/GvDataLoader.h>
#include <GvRenderer/GvGraphicsInteroperabiltyHandler.h>
#include <GvPerfMon/CUDAPerfMon.h>
#include <GvCore/GvError.h>

// Project
#include "ProducerLoad.h"
#include "ShaderLoad.h"

// GigaVoxels
#include <GvUtils/GvPipeline.h>

// GvViewer
#include <GvvApplication.h>

// Cuda SDK
#include <helper_math.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>

//-------------------------------------------
#include "GvgMD2Model.h"
#include <GvgTerrain.h>
// GvViewer
#include <GvvApplication.h>
#include <GvvMainWindow.h>
#include <Gvv3DWindow.h>
#include <GvvPipelineInterfaceViewer.h>
//-------------------------------------------

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvRenderer;

// STL
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

// Defines the size allowed for each type of pool
#define NODEPOOL_MEMSIZE	( 8U * 1024U * 1024U )		// 8 Mo
#define BRICKPOOL_MEMSIZE	( 256U * 1024U * 1024U )	// 256 Mo

///**
// * Tag name identifying a space profile element
// */
//const char* SampleCore::cTypeName = "GigaVoxelsPipeline";

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

//template< typename TDataTypeList, int TChannel >
//void listDataTypes()
//{
//	// Typedef to access the channel in the data type list
//	typedef typename Loki::TL::TypeAt< TDataTypeList, TChannel >::Result ChannelType;
//
//	std::cout << GvCore::typeToString< ChannelType >() << std::endl;
////	// Build filename according to GigaVoxels internal syntax
////	std::stringstream filename;
////	filename << mFileName << "_BR" << mBrickSize << "_B" << mBorderSize << "_L" << mLevel
////		<< "_C" << TChannel << "_" << GvCore::typeToString< ChannelType >() << mFileExt;
////
////	// Store generated filename
////	mResult->push_back( filename.str() );
//}


/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
SampleCore::SampleCore()
:	GvViewerCore::GvvPipelineInterface()
,	mColorTex(0)
,	mDepthTex(0)
,	mFrameBuffer(0)
,	mDepthBuffer(0)
,	mDisplayOctree(false)
,	mDisplayPerfmon( 0 )
//,	mMaxVolTreeDepth(16)
,	mMaxVolTreeDepth( 6 )
,	_filename()
,	_resolution( 0 )
{
	_lightPosition = make_float3(  1.f, 1.f, 1.f );

	_MD2model = NULL;
	_terrain = NULL;
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
//	delete mPipeline;
//	mPipeline = NULL;

//	delete mVolumeTreeRenderer;
	//delete mVolumeTreeCache;
	//delete mVolumeTree;
	/*delete mProducer;
	mProducer = NULL;*/
	
//	delete mPipeline->editRenderer();
//	mPipeline->editRenderer() = NULL;
//	delete mPipeline->editCache();
	//mPipeline->editCache() = NULL;
//	delete mPipeline->editDataStructure();
	//mPipeline->editDataStructure() = NULL;
	delete mPipeline;
	mPipeline = NULL;

	delete mProducer;
	mProducer = NULL;
}

///******************************************************************************
// * Returns the type of this browsable. The type is used for retrieving
// * the context menu or when requested or assigning an icon to the
// * corresponding item
// *
// * @return the type name of this browsable
// ******************************************************************************/
//const char* SampleCore::getTypeName() const
//{
//	return cTypeName;
//}

/******************************************************************************
 * Gets the name of this browsable
 *
 * @return the name of this browsable
 ******************************************************************************/
const char* SampleCore::getName() const
{
	return "VideoGame";
}

/******************************************************************************
 * ...
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
	size_t brickElemSize = RealBrickRes::numElements * GvCore::DataTotalChannelSize<DataType>::value;

	// Compute how many we can fit into the given memory size
	size_t nodePoolNumElems = NODEPOOL_MEMSIZE / nodeElemSize;
	size_t brickPoolNumElems = BRICKPOOL_MEMSIZE / brickElemSize;

	// Compute the resolution of the pools
	uint3 nodePoolRes = make_uint3((uint)floorf(powf((float)nodePoolNumElems, 1.0f / 3.0f))) * NodeRes::get();
	uint3 brickPoolRes = make_uint3((uint)floorf(powf((float)brickPoolNumElems, 1.0f / 3.0f))) * RealBrickRes::get();

	std::cout << "" << std::endl;
	std::cout << "nodePoolRes: " << nodePoolRes << std::endl;
	std::cout << "brickPoolRes: " << brickPoolRes << std::endl;

	// Instanciate our objects
	QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	QString filename = dataRepository + QDir::separator() + QString( "Voxels" ) + QDir::separator() + QString( "xyzrgb_dragon512_BR8_B1" ) + QDir::separator() + QString( "xyzrgb_dragon512" );
	set3DModelFilename( filename.toLatin1().constData() );
	//QString filename( get3DModelFilename().c_str() );
	//unsigned int dataResolution = get3DModelResolution();
	unsigned int dataResolution = 512;
	// TO DO :
	// Test empty and existence of filename
	GvUtils::GvDataLoader< DataType >* dataLoader = new GvUtils::GvDataLoader< DataType >(
														filename.toStdString(),
														make_uint3( dataResolution ), BrickRes::get(), BrickBorderSize, true );

	// Producer
	mProducer = new ProducerType( 64 * 1024 * 1024, nodePoolRes.x * nodePoolRes.y * nodePoolRes.z );
	mProducer->attachProducer( dataLoader );

	//mVolumeTree = new VolumeTreeType(nodePoolRes, brickPoolRes, 0);
	//mVolumeTree->setMaxDepth( mMaxVolTreeDepth );

	//mVolumeTreeCache = new VolumeTreeCacheType(mVolumeTree, mProducer, nodePoolRes, brickPoolRes);
	//mVolumeTreeRenderer = new VolumeTreeRendererType(mVolumeTree, mVolumeTreeCache, mProducer);
	//gluProject
	//-----------------------------------------------
	// TEST
	//PipelineType* pipeline = new PipelineType();
	mPipeline = new PipelineType();
	mPipeline->initialize( NODEPOOL_MEMSIZE, BRICKPOOL_MEMSIZE, mProducer );
	//mVolumeTree = pipeline->editDataStructure();
	//mVolumeTreeCache = pipeline->editCache();
	//mVolumeTreeRenderer = pipeline->editRenderer();
	//mVolumeTree->setMaxDepth( mMaxVolTreeDepth );
	mPipeline->editDataStructure()->setMaxDepth( mMaxVolTreeDepth );
	//-----------------------------------------------

	////------------------------------------------------------------
	//// Typedef to access the channel in the data type list
	//for ( int i = 0; i < GvCore::DataNumChannels< DataType >::value; i++ )
	//{
	//	listDataTypes< DataType, Loki::Int2Type< i > ) >();
	//}
	////------------------------------------------------------------

	//-----------------------------------------------
	// TEST
	mPipeline->editCache()->setMaxNbNodeSubdivisions( 100 );
	mPipeline->editCache()->setMaxNbBrickLoads( 100 );
	//-----------------------------------------------

	// Custom initialization
	// Note : this could be done via an XML settings file loaded at initialization
	// Need to initialize CUDA memory with light position
	float x,y,z;
	getLightPosition( x,y,z );
	setLightPosition( x,y,z );

	//-------------------------------------------
	_MD2model = new GvgMD2Model();
	_MD2model->load( "yoshi.md2" );

	_terrain = GvgTerrain::create();
	_terrain->initialize();

	GvViewerGui::GvvApplication& application = GvViewerGui::GvvApplication::get();
	GvViewerGui::GvvMainWindow* mainWindow = application.getMainWindow();
	GvViewerGui::Gvv3DWindow* window3D = mainWindow->get3DWindow();
	window3D->getPipelineViewer()->setSceneRadius( 1000.f );

	//--------------------------------------------
	//showEntireScene();
	window3D->getPipelineViewer()->camera()->setZClippingCoefficient( 50.0f );
	//--------------------------------------------

	//-------------------------------------------
}

/******************************************************************************
 * ...
 ******************************************************************************/
void draw()
{
	//glEnable( GL_LIGHTING );
	//glEnable( GL_LIGHT1 );

	//-------------------------------------------
	//glPushMatrix();
	//glScalef( 1.f / 40.f, 1.f / 40.f, 1.f / 40.f );
	//_MD2model->draw( 0 );
	//glPopMatrix();
	//-------------------------------------------

	//glDisable( GL_LIGHTING );
	//glDisable( GL_LIGHT1 );
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::draw()
{
	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT( frame );
	CUDAPM_START_EVENT( app_init_frame );

	glBindFramebuffer( GL_FRAMEBUFFER, mFrameBuffer );

	glMatrixMode( GL_MODELVIEW );
	
	/*if ( mDisplayOctree )
	{*/
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );

		//-------------------------------------------
	glEnable( GL_DEPTH_TEST );

	//glEnable( GL_LIGHTING );
	//glEnable( GL_LIGHT0 );
//	glEnable( GL_LIGHT1 );
//	glEnable( GL_TEXTURE_2D );
	glPushMatrix();
	glTranslatef( 0.f, 0.f, 0.5f );
	glRotatef( 90.0f, -1.0f, 0.0f, 0.0f );
	glScalef( 1.f / 80.f, 1.f / 80.f, 1.f / 80.f );
	static int keyframe = 0;
	_MD2model->draw( static_cast< int >( static_cast< float >( keyframe ) * 0.5f ) % _MD2model->_nbKeyframes );
	keyframe++;
	glPopMatrix();
//	glDisable( GL_LIGHT1 );
//	glDisable( GL_LIGHT0 );
//	glDisable( GL_LIGHTING );
	//
	/*static bool terrainInitialized = false;
	if ( ! terrainInitialized )
	{
		_terrain->initialize();
		terrainInitialized = true;
	}*/
	glEnable( GL_LIGHTING );
	glEnable( GL_LIGHT0 );
	glEnable( GL_LIGHT1 );
	glColorMaterial( GL_FRONT, GL_DIFFUSE );
	glEnable( GL_COLOR_MATERIAL );
	glPushMatrix();
	glTranslatef( 0.f, -0.9f, 0.0f );
	_terrain->render();
	glPopMatrix();
	glDisable( GL_COLOR_MATERIAL );
	glDisable( GL_LIGHT1 );
	glDisable( GL_LIGHT0 );
	glDisable( GL_LIGHTING );
	//
//	glDisable( GL_TEXTURE_2D );

	glDisable( GL_DEPTH_TEST );
	//-------------------------------------------

	if ( mDisplayOctree )
	{
		// Display the GigaVoxels N3-tree space partitioning structure
		glEnable( GL_DEPTH_TEST );
		glPushMatrix();
		glTranslatef( -0.5f, -0.5f, -0.5f );
		mPipeline->editDataStructure()->displayDebugOctree();
		glPopMatrix();
		glDisable( GL_DEPTH_TEST );
	}

	// Clear the depth PBO (pixel buffer object) by reading from the previously cleared FBO (frame buffer object)
		glBindBuffer( GL_PIXEL_PACK_BUFFER, mDepthBuffer );
		glReadPixels( 0, 0, mWidth, mHeight, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0 );
		glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
		GV_CHECK_GL_ERROR();

	/*}
	else
	{
		glClear( GL_COLOR_BUFFER_BIT );
	}*/

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
	glTranslatef( -0.5f, -0.5f, -0.5f );
	glGetFloatv( GL_MODELVIEW_MATRIX, modelMatrix._array );
	glPopMatrix();

	// Render
	mPipeline->editRenderer()->render( modelMatrix, viewMatrix, projectionMatrix, viewport );

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
	//_volumeTreeRenderer->doPostRender();
	
	// Update GigaVoxels info
	mPipeline->editRenderer()->nextFrame();

	CUDAPM_STOP_EVENT( frame );
	CUDAPM_STOP_FRAME;

	// Display the GigaVoxels performance monitor (if it has been activated during GigaVoxels compilation)
	if ( mDisplayPerfmon )
	{
		GvPerfMon::CUDAPerfMon::getApplicationPerfMon().displayFrameGL( mDisplayPerfmon - 1 );
	}
}

/******************************************************************************
 * ...
 *
 * @param width ...
 * @param height ...
 ******************************************************************************/
void SampleCore::resize(int width, int height)
{
	mWidth = width;
	mHeight = height;

	// Reset default active frame region for rendering
	mPipeline->editRenderer()->setProjectedBBox( make_uint4( 0, 0, mWidth, mHeight ) );

	// Re-init Perfmon subsystem
	CUDAPM_RESIZE(make_uint2(mWidth, mHeight));

	/*uchar *timersMask = GvPerfMon::CUDAPerfMon::getApplicationPerfMon().getKernelTimerMask();
	cudaMemset(timersMask, 255, mWidth * mHeight);*/

	// Create frame-dependent objects

	// ...
	/*if ( mDisplayOctree )
	{*/
		// Disconnect all registered graphics resources
		mPipeline->editRenderer()->resetGraphicsResources();
	/*}
	else
	{
		mPipeline->editRenderer()->disconnect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot );
	}*/

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
	//if ( mDisplayOctree )
	//{
	//	mPipeline->editRenderer()->disconnect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot );

		mPipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, mColorTex, GL_TEXTURE_RECTANGLE_EXT );
		mPipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, mDepthBuffer );
	/*}
	else
	{
		mPipeline->editRenderer()->disconnect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot );
		mPipeline->editRenderer()->disconnect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot );
		
		mPipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, mColorTex, GL_TEXTURE_RECTANGLE_EXT );
	}*/
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::clearCache()
{
	//mVolumeTreeRenderer->clearCache();
	mPipeline->editRenderer()->clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::toggleDisplayOctree()
{
	//if ( mDisplayOctree )
	//{
	//	// old
	//	mPipeline->editRenderer()->disconnect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot );
	//	mPipeline->editRenderer()->disconnect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot );
	//	
	//	// new
	//	mPipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, mColorTex, GL_TEXTURE_RECTANGLE_EXT );
	//}
	//else
	//{
	//	// old
	//	mPipeline->editRenderer()->disconnect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot );

	//	// new
	//	mPipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, mColorTex, GL_TEXTURE_RECTANGLE_EXT );
	//	mPipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, mDepthBuffer );
	//}

	mDisplayOctree = !mDisplayOctree;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::toggleDynamicUpdate()
{
	//mVolumeTreeRenderer->dynamicUpdateState() = !mVolumeTreeRenderer->dynamicUpdateState();
	mPipeline->editRenderer()->dynamicUpdateState() = !mPipeline->editRenderer()->dynamicUpdateState();
}

/******************************************************************************
 * ...
 *
 * @param mode ...
 ******************************************************************************/
void SampleCore::togglePerfmonDisplay( uint mode )
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

	//mVolumeTree->setMaxDepth( mMaxVolTreeDepth );
	mPipeline->editDataStructure()->setMaxDepth( mMaxVolTreeDepth );
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::decMaxVolTreeDepth()
{
	if (mMaxVolTreeDepth > 0)
		mMaxVolTreeDepth--;

	//mVolumeTree->setMaxDepth( mMaxVolTreeDepth );
	mPipeline->editDataStructure()->setMaxDepth( mMaxVolTreeDepth );
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
	const uint3& nodeTileResolution = mPipeline->editDataStructure()->getNodeTileResolution().get();

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
	const uint3& brickResolution = mPipeline->editDataStructure()->getBrickResolution().get();

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
	return mPipeline->editDataStructure()->getMaxDepth();
}

/******************************************************************************
 * Set the max depth.
 *
 * @param pValue the max depth
 ******************************************************************************/
void SampleCore::setRendererMaxDepth( unsigned int pValue )
{
	mPipeline->editDataStructure()->setMaxDepth( pValue );
}

/******************************************************************************
 * Get the max number of requests of node subdivisions.
 *
 * @return the max number of requests
 ******************************************************************************/
unsigned int SampleCore::getCacheMaxNbNodeSubdivisions() const
{
	return mPipeline->editCache()->getMaxNbNodeSubdivisions();
}

/******************************************************************************
 * Set the max number of requests of node subdivisions.
 *
 * @param pValue the max number of requests
 ******************************************************************************/
void SampleCore::setCacheMaxNbNodeSubdivisions( unsigned int pValue )
{
	mPipeline->editCache()->setMaxNbNodeSubdivisions( pValue );
}

/******************************************************************************
 * Get the max number of requests of brick of voxel loads.
 *
 * @return the max number of requests
 ******************************************************************************/
unsigned int SampleCore::getCacheMaxNbBrickLoads() const
{
	return mPipeline->editCache()->getMaxNbBrickLoads();
}

/******************************************************************************
 * Set the max number of requests of brick of voxel loads.
 *
 * @param pValue the max number of requests
 ******************************************************************************/
void SampleCore::setCacheMaxNbBrickLoads( unsigned int pValue )
{
	mPipeline->editCache()->setMaxNbBrickLoads( pValue );
}

/******************************************************************************
 * Set the request strategy indicating if, during data structure traversal,
 * priority of requests is set on brick loads or on node subdivisions first.
 *
 * @param pFlag the flag indicating the request strategy
 ******************************************************************************/
void SampleCore::setRendererPriorityOnBricks( bool pFlag )
{
	mPipeline->editRenderer()->setPriorityOnBricks( pFlag );
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
	mPipeline->editRenderer()->setClearColor( make_uchar4( pRed, pGreen, pBlue, pAlpha ) );
}

/******************************************************************************
 * Tell wheter or not the pipeline has a transfer function.
 *
 * @return the flag telling wheter or not the pipeline has a transfer function
 ******************************************************************************/
bool SampleCore::hasTransferFunction() const
{
	return false;
}

/******************************************************************************
 * Update the associated transfer function
 *
 * @param the new transfer function data
 * @param the size of the transfer function
 ******************************************************************************/
void SampleCore::updateTransferFunction( float* pData, unsigned int pSize )
{
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
 * Set the 3D model filename to load
 *
 * @param pFilename the 3D model filename to load
 ******************************************************************************/
void SampleCore::set3DModelFilename( const string& pFilename )
{
	_filename = pFilename;
}

/******************************************************************************
 * Get the 3D model resolution
 *
 * @return the 3D model resolution
 ******************************************************************************/
unsigned int SampleCore::get3DModelResolution() const
{
	return _resolution;
}

/******************************************************************************
 * Set the 3D model resolution
 *
 * @param pValue the 3D model resolution
 ******************************************************************************/
void SampleCore::set3DModelResolution( unsigned int pValue )
{
	_resolution = pValue;
}
