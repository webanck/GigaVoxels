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
#include <GvUtils/GvDataLoader.h>
#include <GvUtils/GvSimpleHostShader.h>
#include <GvUtils/GvCommonGraphicsPass.h>
#include <GvCore/GvError.h>
#include <GvPerfMon/GvPerformanceMonitor.h>
#include <GvStructure/GvNode.h>

// Project
#include "Producer.h"
//#include "VolumeProducerBricks.h"
#include "VolumeTreeRendererGLSL.h"
//#include "ProducerTorusKernel.h"

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
GlossyObject* glossyObject = NULL;
Mesh* environmentObject = NULL;
CubeMap* cubeMap = NULL;
float scale = 1.0;
float translation[3] = {0.0};

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
,	mDisplayOctree( false )
,	mDisplayPerfmon( 0 )
,	mMaxVolTreeDepth( 16 )
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
void SampleCore::init(SampleViewer* sv)
{
	CUDAPM_INIT();
	cudaSetDevice( gpuGetMaxGflopsDeviceId() );
	GV_CHECK_CUDA_ERROR( "cudaSetDevice" );

	// Compute the size of one element in the cache for nodes and bricks
	size_t nodeElemSize = NodeRes::numElements * sizeof( GvStructure::GvNode );
	size_t brickElemSize = RealBrickRes::numElements * GvCore::DataTotalChannelSize< DataType >::value;

	// Compute how many we can fit into the given memory size
	size_t nodePoolNumElems = NODEPOOL_MEMSIZE / nodeElemSize;
	size_t brickPoolNumElems = BRICKPOOL_MEMSIZE / brickElemSize;

	// Compute the resolution of the pools
	uint3 nodePoolRes = make_uint3( (uint)floorf( powf( (float)nodePoolNumElems, 1.0f / 3.0f ) ) ) * NodeRes::get();

	// Pipeline creation
	_pipeline = new PipelineType();

	// Producer creation
	ProducerType* producer = new ProducerType( 64 * 1024 * 1024, nodePoolRes.x * nodePoolRes.y * nodePoolRes.z );
	
	// Shader creation
	ShaderType* shader = new ShaderType();

	// Pipeline initialization
	const bool useGraphicsLibraryInteroperability = true;
	_pipeline->initialize( NODEPOOL_MEMSIZE, BRICKPOOL_MEMSIZE, producer, shader, useGraphicsLibraryInteroperability );

	// Pipeline configuration
	_pipeline->editDataStructure()->setMaxDepth( mMaxVolTreeDepth );

	//getting the sample viewer
	sviewer = sv;

	/****************CREATING GLOSSY OBJECT****************/
	glossyObject = new GlossyObject();
	_pipeline->editRenderer()->getVolTreeChildArray()->unmapResource();
	_pipeline->editRenderer()->getVolTreeDataArray()->unmapResource();
	glossyObject->setVolTreeChildArray(_pipeline->editRenderer()->getVolTreeChildArray(), _pipeline->editRenderer()->getChildBufferName());
	glossyObject->setVolTreeDataArray(_pipeline->editRenderer()->getVolTreeDataArray(), _pipeline->editRenderer()->getDataBufferName());	
	glossyObject->init();
	_pipeline->editRenderer()->getVolTreeChildArray()->mapResource();
	_pipeline->editRenderer()->getVolTreeDataArray()->mapResource();
	
	/****************CREATING CUBE MAP****************/
	/*cubeMap = new CubeMap( "../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/CubeMapTextures/NissiBeach2/posx.jpg",
						   "../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/CubeMapTextures/NissiBeach2/negx.jpg",
						   "../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/CubeMapTextures/NissiBeach2/posy.jpg",
						   "../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/CubeMapTextures/NissiBeach2/negy.jpg",
						   "../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/CubeMapTextures/NissiBeach2/posz.jpg",
						   "../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/CubeMapTextures/NissiBeach2/negz.jpg");*/
	cubeMap = new CubeMap( "../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/CubeMapTextures/NissiBeach2/posx.png",
						   "../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/CubeMapTextures/NissiBeach2/negx.png",
						   "../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/CubeMapTextures/NissiBeach2/posy.png",
						   "../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/CubeMapTextures/NissiBeach2/negy.png",
						   "../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/CubeMapTextures/NissiBeach2/posz.png",
						   "../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/CubeMapTextures/NissiBeach2/negz.png");
	cubeMap->Load();
	cubeMap->init();

	/****************CREATING ENVIRONMENT******************/
	GLuint vshader = useShader(GL_VERTEX_SHADER, "../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/Res/vert.glsl");
	GLuint fshader = useShader(GL_FRAGMENT_SHADER, "../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/Res/frag.glsl");
	GLuint program = glCreateProgram();
	glAttachShader(program, vshader);
	glAttachShader(program, fshader);
	glLinkProgram(program);
	linkStatus(program);
	environmentObject = new Mesh(program);
	//string environmentObjectFile = "Data/3DModels/MickeyMouse.obj";	
	string environmentObjectFile = "Data/3DModels/dino.3ds";	
	environmentObject->chargerMesh(environmentObjectFile);
	environmentObject->creerVBO();
	scale = environmentObject->getScaleFactor();
	environmentObject->getTranslationFactors(translation);

	/************************VOXELIZATION***********************/
	//voxelization code on file environmentObjectFile
	//...
	//...
	/*************************************************************/

	//retrieving the xml file 
	QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	QString filename = dataRepository + QDir::separator() + QString( "Voxels" ) + QDir::separator() + QString( /*"xyzrgb_dragon512_BR8_B1"*/"Dino") + QDir::separator() + QString( /*"xyzrgb_dragon.xml"*/"dino.xml");
	GvUtils::GvDataLoader< DataType >* dataLoader = new GvUtils::GvDataLoader< DataType >( filename.toStdString(), BrickRes::get(), BrickBorderSize, true );
	producer->attachProducer( dataLoader );

	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);
	//glCullFace(GL_FRONT);
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


	glMatrixMode( GL_MODELVIEW);

	// Display the data structure (space partitioning)
	if ( mDisplayOctree )
	{
		glPushMatrix();
		glTranslatef(translation[0], translation[1], translation[2]);
		glScalef(scale, scale, scale);
		glTranslatef( -0.5f, -0.5f, -0.5f );//pour centrer la boite GV
		_pipeline->editDataStructure()->render();
		glPopMatrix();
	}

	// extract view transformations
	float4x4 viewMatrix;
	float4x4 projectionMatrix;
	// FIXME
	glPushMatrix();
	glTranslatef(translation[0], translation[1], translation[2]);
	glScalef(scale, scale, scale);
	glTranslatef( -0.5f, -0.5f, -0.5f );//pour centrer la boite GV
	// FIXME
	glGetFloatv( GL_MODELVIEW_MATRIX, viewMatrix._array );
	glGetFloatv( GL_PROJECTION_MATRIX, projectionMatrix._array );
	// FIXME
	glPopMatrix();

	// build and extract tree transformations
	float4x4 modelMatrix;

	glPushMatrix();
	glLoadIdentity();
	glTranslatef(translation[0], translation[1], translation[2]);
	glScalef(scale, scale, scale);
	glTranslatef( -0.5f, -0.5f, -0.5f);//pour centrer la boite GV
	glGetFloatv( GL_MODELVIEW_MATRIX, modelMatrix._array );
	glPopMatrix();

	// extract viewport
	GLint params[4];
	glGetIntegerv( GL_VIEWPORT, params );
	int4 viewport = make_int4( params[0], params[1], params[2], params[3] );

	CUDAPM_STOP_EVENT( app_init_frame );
	
	glEnable (GL_BLEND);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	/******************RENDER GIGAVOXELS SCENE******************/
	_pipeline->editRenderer()->setLightPosition(lightPos.x, lightPos.y, lightPos.z);
	_pipeline->execute( modelMatrix, viewMatrix, projectionMatrix, viewport );

	/******************RENDER THE CUBE MAP*********************/
	cubeMap->render();
	float cubeModelMatrix[16];
	cubeMap->getCubeModelMatrix(cubeModelMatrix);

	glEnable(GL_DEPTH_TEST);
	
	/******************RENDER THE GLOSSY OBJECT AND RETRIEVE THE GIGAVOXELS INFO*********************/
	glossyObject->setModelMatrix(modelMatrix._array[0],modelMatrix._array[1], modelMatrix._array[2], modelMatrix._array[3],  
							modelMatrix._array[4],modelMatrix._array[5], modelMatrix._array[6], modelMatrix._array[7], 
							modelMatrix._array[8],modelMatrix._array[9], modelMatrix._array[10], modelMatrix._array[11],
							modelMatrix._array[12],modelMatrix._array[13], modelMatrix._array[14], modelMatrix._array[15]);

	glossyObject->setCubeModelMatrix(cubeModelMatrix);
	glossyObject->setLightPosition(lightPos.x, lightPos.y, lightPos.z);
	glossyObject->setWorldLight(worldLight.x, worldLight.y, worldLight.z);
	glossyObject->setWorldCameraPosition(worldCamPos.x, worldCamPos.y, worldCamPos.z);
	glossyObject->setTexBufferName(_pipeline->editRenderer()->getTexBufferName());
	glossyObject->setCubeMapTextureID(cubeMap->getTextureID());
	uint3 bsc = _pipeline->editRenderer()->getBrickCacheSize();
	glossyObject->setBrickCacheSize(bsc.x, bsc.y, bsc.z);
	float3 bpri = _pipeline->editRenderer()->getBrickPoolResInv();
	glossyObject->setBrickPoolResInv(bpri.x, bpri.y, bpri.z);
	glossyObject->setMaxDepth(_pipeline->editRenderer()->getMaxDepth());
	GV_CHECK_GL_ERROR();
	glossyObject->render();
	GV_CHECK_GL_ERROR();
	
	/******************RENDER THE OBJECT CASTING ITS REFLECTION*******************/
	environmentObject->setLightPosition(lightPos.x, lightPos.y, lightPos.z);
	environmentObject->render();



	// TEST - optimization due to early unmap() graphics resource from GigaVoxels
	_pipeline->editRenderer()->nextFrame();

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
 * Set the light position in camera coordinates
 *
 * @param pX the X light position
 * @param pY the Y light position
 * @param pZ the Z light position
 ******************************************************************************/
void SampleCore::setLightPosition( float pX, float pY, float pZ )
{	
	lightPos = make_float3( pX, pY, pZ );
}

/******************************************************************************
 * Set the light position in world coordinates
 *
 * @param pX the X light position
 * @param pY the Y light position
 * @param pZ the Z light position
 ******************************************************************************/
void SampleCore::setWorldLight( float pX, float pY, float pZ ) {
	worldLight = make_float3(pX, pY, pZ);
}

void SampleCore::setWorldCamera(float x, float y, float z) {
	worldCamPos.x = x;
	worldCamPos.y = y;
	worldCamPos.z = z;
} 

/******************************************************************************
 * Returns the file name of the shadow casting object (OpenGL)
 ******************************************************************************/
string SampleCore::getShadowCasterFile() {
	return shadowCasterFile;
}
