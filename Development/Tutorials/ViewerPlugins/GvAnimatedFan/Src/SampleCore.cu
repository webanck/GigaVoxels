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
#include <GvUtils/GvSimpleHostShader.h>
#include <GvUtils/GvDataLoader.h>
#include <GvPerfMon/GvPerformanceMonitor.h>
#include <GvCore/GvError.h>

// Project
#include "Producer.h"
#include "ShaderKernel.h"

// GvViewer
#include <GvvApplication.h>

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
using namespace GvRendering;

// STL
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

// Defines the size allowed for each type of pool
#define NODEPOOL_MEMSIZE	( 8U * 1024U * 1024U )		// 8 Mo
#define BRICKPOOL_MEMSIZE	( 1024U * 1024U * 1024U )	// 256 Mo

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
,	_colorTex( 0 )
,	_depthTex( 0 )
,	_frameBuffer( 0 )
,	_depthBuffer( 0 )
,	_displayOctree( false )
,	_displayPerfmon( 0 )
,	_maxVolTreeDepth( 6 )
,	_filename()
,	_resolution( 0 )
{
	// Translation used to position the GigaVoxels data structure
	_translation[ 0 ] = -0.5f;
	_translation[ 1 ] = -0.5f;
	_translation[ 2 ] = -0.5f;

	float truc = 0;
	cudaMemcpyToSymbol( cThreshold, &truc,  sizeof( truc ), 0, cudaMemcpyHostToDevice ) ;

	// Rotation used to position the GigaVoxels data structure
	_rotation[ 0 ] = 0.0f;
	_rotation[ 1 ] = 0.0f;
	_rotation[ 2 ] = 0.0f;
	_rotation[ 3 ] = 0.0f;

	// Scale used to transform the GigaVoxels data structure
	_scale = 1.0f;

	// Light position
	_lightPosition = make_float3(  1.f, 1.f, 1.f );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
	// ...
	if ( _displayOctree )
	{
		_pipeline->editRenderer()->disconnect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot );
		_pipeline->editRenderer()->disconnect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot );
	}
	else
	{
		_pipeline->editRenderer()->disconnect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot );
	}

	delete _pipeline;
	_pipeline = NULL;

	// CUDA tip: clean up to ensure correct profiling
	//cudaError_t error = cudaDeviceReset();
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
	return "AnimatedFan";
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::init()
{
	CUDAPM_INIT();

	if ( ! GvViewerGui::GvvApplication::get().isGPUComputingInitialized() )
	{
		//cudaGLSetGLDevice( gpuGetMaxGflopsDeviceId() );	// to do : deprecated, use cudaSetDevice()
		//GV_CHECK_CUDA_ERROR( "cudaGLSetGLDevice" );
		cudaSetDevice( gpuGetMaxGflopsDeviceId() );
		GV_CHECK_CUDA_ERROR( "cudaSetDevice" );
		
		GvViewerGui::GvvApplication::get().setGPUComputingInitialized( true );
	}
	
	// Compute the size of one element in the cache for nodes and bricks
	size_t nodeElemSize = PipelineType::NodeTileResolution::numElements * sizeof( GvStructure::GvNode );
	//size_t brickElemSize = RealBrickRes::numElements * GvCore::DataTotalChannelSize< PipelineType::DataTypeList >::value;

	// Compute how many we can fit into the given memory size
	size_t nodePoolNumElems = NODEPOOL_MEMSIZE / nodeElemSize;
	//size_t brickPoolNumElems = BRICKPOOL_MEMSIZE / brickElemSize;

	// Compute the resolution of the pools
	uint3 nodePoolRes = make_uint3((uint)floorf(powf((float)nodePoolNumElems, 1.0f / 3.0f))) * NodeRes::get();
	//uint3 brickPoolRes = make_uint3((uint)floorf(powf((float)brickPoolNumElems, 1.0f / 3.0f))) * RealBrickRes::get();

	//std::cout << "" << std::endl;
	//std::cout << "nodePoolRes: " << nodePoolRes << std::endl;
	//std::cout << "brickPoolRes: " << brickPoolRes << std::endl;

	// Instanciate our objects
	//QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	//QString filename = dataRepository + QDir::separator() + QString( "Voxels" ) + QDir::separator() + QString( "xyzrgb_dragon512_BR8_B1" ) + QDir::separator() + QString( "xyzrgb_dragon512" );
	//set3DModelFilename( filename.toLatin1().constData() );
	QString filename( get3DModelFilename().c_str() );

	
	// unsigned int dataResolution = get3DModelResolution();
	// TO DO :
	// Test empty and existence of filename
	
	GvUtils::GvDataLoader< DataType >* dataLoader = new GvUtils::GvDataLoader< DataType >(
														filename.toStdString(), PipelineType::BrickTileResolution::get(), PipelineType::BrickTileBorderSize, true );

	// Producer initialization
	_producer = new ProducerType( 64 * 1024 * 1024, nodePoolRes.x * nodePoolRes.y * nodePoolRes.z );
	_producer->attachProducer( dataLoader );

	// Shader creation
	ShaderType* shader = new ShaderType();

	// Pipeline initialization
	_pipeline = new PipelineType();
	_pipeline->initialize( NODEPOOL_MEMSIZE, BRICKPOOL_MEMSIZE, _producer, shader );
	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );
	
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

	// Custom initialization
	// Note : this could be done via an XML settings file loaded at initialization
	// Need to initialize CUDA memory with light position
	float x,y,z;
	getLightPosition( x,y,z );
	setLightPosition( x,y,z );
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::draw()
{
	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT( frame );
	CUDAPM_START_EVENT( app_init_frame );

	glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );

	glMatrixMode( GL_MODELVIEW );

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
	glScalef( _scale, _scale, _scale );
	glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );
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

	// TEST - optimization due to early unmap() graphics resource from GigaVoxels
	//_volumeTreeRenderer->doPostRender();
	
	// Update GigaVoxels info
	_pipeline->editRenderer()->nextFrame();

	CUDAPM_STOP_EVENT( frame );
	CUDAPM_STOP_FRAME;

	// Display the GigaVoxels performance monitor (if it has been activated during GigaVoxels compilation)
	if ( _displayPerfmon )
	{
		GvPerfMon::CUDAPerfMon::get().displayFrameGL( _displayPerfmon - 1 );
		
		// SORTIE CONSOLE
		// GvPerfMon::CUDAPerfMon::getApplicationPerfMon().displayFrame();

		// HOST
		//GvPerfMon::CUDAPerfMon::getApplicationPerfMon().displayFrameGL( 1 );
		
		// DEVICE
		//GvPerfMon::CUDAPerfMon::getApplicationPerfMon().displayFrameGL( 0 );
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
	_width = width;
	_height = height;

	// Reset default active frame region for rendering
	_pipeline->editRenderer()->setProjectedBBox( make_uint4( 0, 0, _width, _height ) );

	// Re-init Perfmon subsystem
	CUDAPM_RESIZE(make_uint2(_width, _height));

	/*uchar *timersMask = GvPerfMon::CUDAPerfMon::getApplicationPerfMon().getKernelTimerMask();
	cudaMemset(timersMask, 255, _width * _height);*/

	// Disconnect all registered graphics resources
	_pipeline->editRenderer()->resetGraphicsResources();

	// Create frame-dependent objects

	if (_depthBuffer)
		glDeleteBuffers(1, &_depthBuffer);

	if (_colorTex)
		glDeleteTextures(1, &_colorTex);
	if (_depthTex)
		glDeleteTextures(1, &_depthTex);

	if (_frameBuffer)
		glDeleteFramebuffers(1, &_frameBuffer);

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
		_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	}
	else
	{
		_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
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
	if (_displayPerfmon)
		_displayPerfmon = 0;
	else
		_displayPerfmon = mode;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::incMaxVolTreeDepth()
{
	if (_maxVolTreeDepth < 32)
		_maxVolTreeDepth++;

	//mVolumeTree->setMaxDepth( _maxVolTreeDepth );
	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::decMaxVolTreeDepth()
{
	if (_maxVolTreeDepth > 0)
		_maxVolTreeDepth--;

	//mVolumeTree->setMaxDepth( _maxVolTreeDepth );
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

///******************************************************************************
// * Get the 3D model resolution
// *
// * @return the 3D model resolution
// ******************************************************************************/
//unsigned int SampleCore::get3DModelResolution() const
//{
//	return _resolution;
//}
//
///******************************************************************************
// * Set the 3D model resolution
// *
// * @param pValue the 3D model resolution
// ******************************************************************************/
//void SampleCore::set3DModelResolution( unsigned int pValue )
//{
//	_resolution = pValue;
//}

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


void SampleCore::setThreshold (double value) {

	threshold = value;
	cudaMemcpyToSymbol( cThreshold, &threshold,  sizeof( threshold ), 0, cudaMemcpyHostToDevice ) ;



}