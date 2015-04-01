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

// GigVoxels
#include <GvCore/StaticRes3D.h>
#include <GvStructure/GvVolumeTree.h>
#include <GvStructure/GvVolumeTreeCache.h>
#include <GvRenderer/GvVolumeTreeRendererCUDA.h>
#include <GvPerfMon/CUDAPerfMon.h>
#include <GvUtils/GvDataLoader.h>
#include <GvRenderer/GvGraphicsInteroperabiltyHandler.h>
#include <GvConfig.h>
#include <GvCore/GvError.h>

// Dynamic Load
#include "ProducerLoad.h"
#include "ShaderLoad.h"

// GigaVoxels
#include <GvUtils/GvPipeline.h>

// GvViewer
#include <GvvApplication.h>

// Cuda GPU Computing SDK
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

// STL
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

// Defines the size allowed for each type of pool
#define NODEPOOL_MEMSIZE	(8*1024*1024)		// 8 Mo
#define BRICKPOOL_MEMSIZE	(256*1024*1024)		// 256 Mo

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
,	_threshold( 0.f )
,	_fullOpacityDistance( 0.f )
,	_gradientStep( 0.f )
,	_pipeline( NULL )
{
	_lightPosition = make_float3(  1.f, 1.f, 1.f );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
//	delete _pipeline;
//	_pipeline = NULL;

//	delete mVolumeTreeRenderer;
	//delete mVolumeTreeCache;
	//delete mVolumeTree;
	/*delete mProducer;
	mProducer = NULL;*/
	
//	delete _pipeline->editRenderer();
//	_pipeline->editRenderer() = NULL;
//	delete _pipeline->editCache();
	//_pipeline->editCache() = NULL;
//	delete _pipeline->editDataStructure();
	//_pipeline->editDataStructure() = NULL;
	delete _pipeline;
	_pipeline = NULL;

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
	return "RawDataLoader";
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
	//QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	//QString filename = dataRepository + QDir::separator() + QString( "Voxels" ) + QDir::separator() + QString( "xyzrgb_dragon512_BR8_B1" ) + QDir::separator() + QString( "xyzrgb_dragon512" );
	//set3DModelFilename( filename.toLatin1().constData() );
	QString filename( get3DModelFilename().c_str() );
	unsigned int dataResolution = get3DModelResolution();
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
	
	//-----------------------------------------------
	// TEST
	//PipelineType* pipeline = new PipelineType();
	_pipeline = new PipelineType();
	_pipeline->initialize( NODEPOOL_MEMSIZE, BRICKPOOL_MEMSIZE, mProducer );
	//mVolumeTree = pipeline->editDataStructure();
	//mVolumeTreeCache = pipeline->editCache();
	//mVolumeTreeRenderer = pipeline->editRenderer();
	//mVolumeTree->setMaxDepth( mMaxVolTreeDepth );
	_pipeline->editDataStructure()->setMaxDepth( mMaxVolTreeDepth );
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
	_pipeline->editCache()->setMaxNbNodeSubdivisions( 100 );
	_pipeline->editCache()->setMaxNbBrickLoads( 100 );
	//-----------------------------------------------

	// Create the transfer function
	//ShaderLoad::createTransferFunction( 256 );

	// Custom initialization
	// Note : this could be done via an XML settings file loaded at initialization
	// Need to initialize CUDA memory with light position
	float x,y,z;
	getLightPosition( x, y, z );
	setLightPosition( x, y, z );
	setThreshold( 0.f );	// no threshold by default
	setFullOpacityDistance( dataResolution ); // the distance ( 1 / FullOpacityDistance ) is the distance after which opacity is full.
	setGradientStep( 0.25f );

	_myTransferFunction = new GvUtils::GvTransferFunction();
	_myTransferFunction->create( 256 );
	_myTransferFunction->bindToTextureReference( &transerFunctionTexture, "transerFunctionTexture", true, cudaFilterModeLinear, cudaAddressModeClamp );
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::draw()
{
	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT( frame );
	CUDAPM_START_EVENT( app_init_frame );

	glMatrixMode( GL_MODELVIEW );

	glBindFramebuffer( GL_FRAMEBUFFER, mFrameBuffer );
	if ( mDisplayOctree )
	{
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );

		// Display the GigaVoxels N3-tree space partitioning structure
		glEnable( GL_DEPTH_TEST );
		glPushMatrix();
		// Translation used to position the GigaVoxels data structure
		//glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );
		glTranslatef( -0.5f, -0.5f, -0.5f );
		_pipeline->editDataStructure()->displayDebugOctree();
		glPopMatrix();
		glDisable( GL_DEPTH_TEST );

		// Clear the depth PBO (pixel buffer object) by reading from the previously cleared FBO (frame buffer object)
		glBindBuffer( GL_PIXEL_PACK_BUFFER, mDepthBuffer );
		glReadPixels( 0, 0, mWidth, mHeight, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0 );
		glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
		GV_CHECK_GL_ERROR();
	}
	else
	{
		glClear( GL_COLOR_BUFFER_BIT );
	}
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
	//glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );
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
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, mColorTex );
	
	GLint sMin = 0;
	GLint tMin = 0;
	GLint sMax = mWidth;
	GLint tMax = mHeight;

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
	//_pipeline->editRenderer()->doPostRender();
	
	// Update GigaVoxels info
	_pipeline->editRenderer()->nextFrame();

	CUDAPM_STOP_EVENT(frame);
	CUDAPM_STOP_FRAME;

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
	_pipeline->editRenderer()->setProjectedBBox( make_uint4( 0, 0, mWidth, mHeight ) );

	// Re-init Perfmon subsystem
	CUDAPM_RESIZE(make_uint2(mWidth, mHeight));

	/*uchar *timersMask = GvPerfMon::CUDAPerfMon::getApplicationPerfMon().getKernelTimerMask();
	cudaMemset(timersMask, 255, mWidth * mHeight);*/

	// Disconnect all registered graphics resources
	_pipeline->editRenderer()->resetGraphicsResources();

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
	if ( mDisplayOctree )
	{
		_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, mColorTex, GL_TEXTURE_RECTANGLE_EXT );
		_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, mDepthBuffer );
	}
	else
	{
		_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, mColorTex, GL_TEXTURE_RECTANGLE_EXT );
	}
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

	// Disconnect all registered graphics resources
	_pipeline->editRenderer()->resetGraphicsResources();

	if ( mDisplayOctree )
	{
		_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, mColorTex, GL_TEXTURE_RECTANGLE_EXT );
		_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, mDepthBuffer );
	}
	else
	{
		_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, mColorTex, GL_TEXTURE_RECTANGLE_EXT );
	}
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
	_pipeline->editDataStructure()->setMaxDepth( mMaxVolTreeDepth );
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::decMaxVolTreeDepth()
{
	if (mMaxVolTreeDepth > 0)
		mMaxVolTreeDepth--;

	//mVolumeTree->setMaxDepth( mMaxVolTreeDepth );
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
 * Get the max number of requests of node subdivisions.
 *
 * @return the max number of requests
 ******************************************************************************/
unsigned int SampleCore::getCacheMaxNbNodeSubdivisions() const
{
	return _pipeline->editCache()->getMaxNbNodeSubdivisions();
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
	return _pipeline->editCache()->getMaxNbBrickLoads();
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
 * Tell wheter or not the pipeline has a transfer function.
 *
 * @return the flag telling wheter or not the pipeline has a transfer function
 ******************************************************************************/
bool SampleCore::hasTransferFunction() const
{
	return true;
}

/******************************************************************************
 * Update the associated transfer function
 *
 * @param the new transfer function data
 * @param the size of the transfer function
 ******************************************************************************/
void SampleCore::updateTransferFunction( float* pData, unsigned int pSize )
{
	//float4* tf = ShaderLoad::getTransferFunction();
	float4* tf = _myTransferFunction->editData();
	//uint size = ShaderLoad::getTransferFunctionRes();
	unsigned int size = _myTransferFunction->getResolution();
	assert( size == pSize );
	for ( unsigned int i = 0; i < size; ++i )
	{
		tf[ i ] = make_float4( pData[ 4 * i ], pData[ 4 * i + 1 ], pData[ 4 * i + 2 ], pData[ 4 * i + 3 ] );
	}
	//ShaderLoad::transferFunctionUpdated();
	_myTransferFunction->updateDeviceMemory();

	// Update cache
	// NOTE : no need toclear the cache because the transfer funtion is applied in the Shader, not the Producer
	//_pipeline->clear();
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

/******************************************************************************
 * Get the threshold
 *
 * @return the threshold
 ******************************************************************************/
float SampleCore::getThreshold() const
{
	return _threshold;
}

/******************************************************************************
 * Set the threshold
 *
 * @param pValue the threshold
 ******************************************************************************/
void SampleCore::setThreshold( float pValue )
{
	_threshold = pValue;

	// Update device memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cThreshold, &_threshold, sizeof( _threshold ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Get the full opacity distance
 *
 * @return the full opacity distance
 ******************************************************************************/
float SampleCore::getFullOpacityDistance() const
{
	return _fullOpacityDistance;
}

/******************************************************************************
 * Set the full opacity distance
 *
 * @param pValue the full opacity distance
 ******************************************************************************/
void SampleCore::setFullOpacityDistance( float pValue )
{
	_fullOpacityDistance = pValue;

	// Update device memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cFullOpacityDistance, &_fullOpacityDistance, sizeof( _fullOpacityDistance ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Get the gradient step
 *
 * @return the gradient step
 ******************************************************************************/
float SampleCore::getGradientStep() const
{
	return _gradientStep;
}

/******************************************************************************
 * Set the gradient step
 *
 * @param pValue the gradient step
 ******************************************************************************/
void SampleCore::setGradientStep( float pValue )
{
	_gradientStep = pValue;
	
	// Update device memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cGradientStep, &_gradientStep, sizeof( _gradientStep ), 0, cudaMemcpyHostToDevice ) );
}
