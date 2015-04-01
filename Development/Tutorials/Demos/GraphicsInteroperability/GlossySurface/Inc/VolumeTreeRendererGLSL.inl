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
//class GvGraphicsInteroperabiltyHandler;
/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GvError.h>
#include <GvUtils/GvShaderManager.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 *
 * It initializes all OpenGL-related stuff
 *
 * @param pVolumeTree data structure to render
 * @param pVolumeTreeCache cache
 * @param pProducer producer of data
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::VolumeTreeRendererGLSL( TVolumeTreeType* pVolumeTree, TVolumeTreeCacheType* pVolumeTreeCache )
:	GvRendering::GvRenderer< TVolumeTreeType, TVolumeTreeCacheType >( pVolumeTree, pVolumeTreeCache )
{

	// Initialize graphics interoperability
	_graphicsInteroperabiltyHandler = new GvRendering::GvGraphicsInteroperabiltyHandler();

	// Create a buffer object
	glGenBuffers( 1, &_textBuffer );
	glBindBuffer( GL_TEXTURE_BUFFER, _textBuffer );
	// Creates and initializes buffer object's data store 
	glBufferData( GL_TEXTURE_BUFFER, 8192 * sizeof( GLfloat ), NULL, GL_STATIC_DRAW );
	glBindBuffer( GL_TEXTURE_BUFFER, 0 );
	GV_CHECK_GL_ERROR();

	// Create a buffer texture
	glGenTextures( 1, &_textBufferTBO );
	glBindTexture( GL_TEXTURE_BUFFER, _textBufferTBO );
	// Attach the storage for a buffer object to the active buffer texture
	glTexBuffer( GL_TEXTURE_BUFFER, GL_R32F, _textBuffer );
	glBindTexture( GL_TEXTURE_BUFFER, 0 );
	GV_CHECK_GL_ERROR();

	// Retrieve useful GigaVoxels arrays
	/*GvCore::Array3DGPULinear< uint >* */ volTreeChildArray = this->_volumeTree->_childArray;
	/*GvCore::Array3DGPULinear< uint >* */ volTreeDataArray = this->_volumeTree->_dataArray;
	GvCore::Array3DGPULinear< uint >* updateBufferArray = this->_volumeTreeCache->getUpdateBuffer();
	GvCore::Array3DGPULinear< uint >* nodeTimeStampArray = this->_volumeTreeCache->getNodesCacheManager()->getTimeStampList();
	GvCore::Array3DGPULinear< uint >* brickTimeStampArray = this->_volumeTreeCache->getBricksCacheManager()->getTimeStampList();

	// Unmap GigaVoxels graphics resources from CUDA environment in order to be used by OpenGL
	volTreeChildArray->unmapResource();
	volTreeDataArray->unmapResource();
	updateBufferArray->unmapResource();
	nodeTimeStampArray->unmapResource();
	brickTimeStampArray->unmapResource();
GV_CHECK_GL_ERROR();
	// Create a buffer texture associated to the GigaVoxels update buffer array
	glGenTextures( 1, &_updateBufferTBO );
	glBindTexture( GL_TEXTURE_BUFFER, _updateBufferTBO );
	GV_CHECK_GL_ERROR();
	// Attach the storage of buffer object to buffer texture
	glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, updateBufferArray->getBufferName() );
	glBindTexture( GL_TEXTURE_BUFFER, 0 );

GV_CHECK_GL_ERROR();
	// Create a buffer texture associated to the GigaVoxels node time stamp array
	glGenTextures( 1, &_nodeTimeStampTBO );
	glBindTexture( GL_TEXTURE_BUFFER, _nodeTimeStampTBO );
	// Attach the storage of buffer object to buffer texture
	glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, nodeTimeStampArray->getBufferName() );
	glBindTexture( GL_TEXTURE_BUFFER, 0 );
GV_CHECK_GL_ERROR();
	// Create a buffer texture associated to the GigaVoxels brick time stamp array
	glGenTextures( 1, &_brickTimeStampTBO );
	glBindTexture( GL_TEXTURE_BUFFER, _brickTimeStampTBO );
	// Attach the storage of buffer object to buffer texture
	glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, brickTimeStampArray->getBufferName() );
	glBindTexture( GL_TEXTURE_BUFFER, 0 );
GV_CHECK_GL_ERROR();
	// Create a buffer texture associated to the GigaVoxels data structure's child array
	glGenTextures( 1, &_childArrayTBO );
	glBindTexture( GL_TEXTURE_BUFFER, _childArrayTBO );
	// Attach the storage of buffer object to buffer texture
	childBufferName = volTreeChildArray->getBufferName();
	glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, volTreeChildArray->getBufferName() );
	glBindTexture( GL_TEXTURE_BUFFER, 0 );
GV_CHECK_GL_ERROR();
	// Create a buffer texture associated to the GigaVoxels data structure's data array
	glGenTextures( 1, &_dataArrayTBO );
	glBindTexture( GL_TEXTURE_BUFFER, _dataArrayTBO );
	// Attach the storage of buffer object to buffer texture
	dataBufferName = volTreeDataArray->getBufferName();
	glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, volTreeDataArray->getBufferName() );
	glBindTexture( GL_TEXTURE_BUFFER, 0 );

	// Check for OpenGL error(s)
	GV_CHECK_GL_ERROR();

	// Map GigaVoxels graphics resources in order to be used by CUDA environment
	volTreeChildArray->mapResource();
	volTreeDataArray->mapResource();
	updateBufferArray->mapResource();
	nodeTimeStampArray->mapResource();
	brickTimeStampArray->mapResource();

	// Create and link a GLSL shader program
	//QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	//QString vertexShaderFilename = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "RendererGLSLSphere" ) + QDir::separator() + QString( "rayCastVert.glsl" );
	QString vertexShaderFilename = QString("../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/Res/gvCastingReflectionVert.glsl");
	//QString vertexShaderFilename = QString( "rayCastVert.glsl" );
	//QFileInfo vertexShaderFileInfo( vertexShaderFilename );
	//if ( ( ! vertexShaderFileInfo.isFile() ) || ( ! vertexShaderFileInfo.isReadable() ) )
	//{
	//	// Idea
	//	// Maybe use Qt function : bool QFileInfo::permission ( QFile::Permissions permissions ) const

	//	// TO DO
	//	// Handle error : free memory and exit
	//	// ...
	//	std::cout << "ERROR. Check filename : " << vertexShaderFilename.toLatin1().constData() << std::endl;
	//}
	//QString fragmentShaderFilename = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "RendererGLSLSphere" ) + QDir::separator() + QString( "rayCastFrag.glsl" );
	QString fragmentShaderFilename = QString("../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/Res/gvCastingReflectionFrag.glsl");
	//QString fragmentShaderFilename = QString( "rayCastFrag.glsl" );
	//QFileInfo fragmentShaderFileInfo( fragmentShaderFilename );
	//if ( ( ! fragmentShaderFileInfo.isFile() ) || ( ! fragmentShaderFileInfo.isReadable() ) )
	//{
	//	// Idea
	//	// Maybe use Qt function : bool QFileInfo::permission ( QFile::Permissions permissions ) const

	//	// TO DO
	//	// Handle error : free memory and exit
	//	// ...
	//	std::cout << "ERROR. Check filename : " << fragmentShaderFilename.toLatin1().constData() << std::endl;
	//}
	_rayCastProg = GvUtils::GvShaderManager::createShaderProgram( vertexShaderFilename.toLatin1().constData(), NULL, fragmentShaderFilename.toLatin1().constData() );
	GvUtils::GvShaderManager::linkShaderProgram( _rayCastProg );
GV_CHECK_GL_ERROR();
	_volTreeChildArrayLoc = glGetUniformLocation( _rayCastProg, "uNodePoolChildArray" );
	_volTreeDataArrayLoc = glGetUniformLocation( _rayCastProg, "uNodePoolDataArray" );
	_updateBufferArrayLoc = glGetUniformLocation( _rayCastProg, "uUpdateBufferArray" );
	_nodeTimeStampArrayLoc = glGetUniformLocation( _rayCastProg, "uNodeTimeStampArray" );
	_brickTimeStampArrayLoc = glGetUniformLocation( _rayCastProg, "uBrickTimeStampArray" );
	_currentTimeLoc = glGetUniformLocation( _rayCastProg, "uCurrentTime" );
	GV_CHECK_GL_ERROR();
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::~VolumeTreeRendererGLSL()
{
	// Destroy TBO
	if ( _updateBufferTBO )
	{
		glDeleteTextures( 1, &_updateBufferTBO );
	}

	// Destroy TBO
	if (_nodeTimeStampTBO)
	{
		glDeleteTextures( 1, &_nodeTimeStampTBO );
	}

	// Destroy TBO
	if ( _brickTimeStampTBO )
	{
		glDeleteTextures( 1, &_brickTimeStampTBO );
	}

	// Destroy TBO
	if ( _childArrayTBO )
	{
		glDeleteTextures( 1, &_childArrayTBO );
	}

	// Destroy TBO
	if ( _dataArrayTBO )
	{
		glDeleteTextures( 1, &_dataArrayTBO );
	}

	// TO DO
	//
	// Destroy :
	// _textBuffer
	// _textBufferTBO
	// ...
}

/******************************************************************************
 * This function is the specific implementation method called
 * by the parent GvIRenderer::render() method during rendering.
 *
 * @param pModelMatrix the current model matrix
 * @param pViewMatrix the current view matrix
 * @param pProjectionMatrix the current projection matrix
 * @param pViewport the viewport configuration
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::render( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	//_graphicsInteroperabiltyHandler->mapResources();
	//bindGraphicsResources();
	// Call internal render method
	doRender( pModelMatrix, pViewMatrix, pProjectionMatrix, pViewport );
	//unbindGraphicsResources();
	//_graphicsInteroperabiltyHandler->unmapResources();
}

/******************************************************************************
 * Start the rendering process.
 *
 * @param pModelMatrix the current model matrix
 * @param pViewMatrix the current view matrix
 * @param pProjectionMatrix the current projection matrix
 * @param pViewport the viewport configuration
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::doRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	CUDAPM_START_EVENT( vsrender_pre_frame );
	CUDAPM_START_EVENT( vsrender_pre_frame_mapbuffers );
	CUDAPM_STOP_EVENT( vsrender_pre_frame_mapbuffers );
	CUDAPM_STOP_EVENT( vsrender_pre_frame );

	// Create a render view context to access to useful variables during (view matrix, model matrix, etc...)
	GvRendering::GvRendererContext viewContext;

	// Extract zNear, zFar as well as the distance in view space
	// from the center of the screen to each side of the screen.
	float fleft   = pProjectionMatrix._array[ 14 ] * ( pProjectionMatrix._array[ 8 ] - 1.0f ) / ( pProjectionMatrix._array[ 0 ] * ( pProjectionMatrix._array[ 10 ] - 1.0f ) );
	float fright  = pProjectionMatrix._array[ 14 ] * ( pProjectionMatrix._array[ 8 ] + 1.0f ) / ( pProjectionMatrix._array[ 0 ] * ( pProjectionMatrix._array[ 10 ] - 1.0f ) );
	float ftop    = pProjectionMatrix._array[ 14 ] * ( pProjectionMatrix._array[ 9 ] + 1.0f ) / ( pProjectionMatrix._array[ 5 ] * ( pProjectionMatrix._array[ 10 ] - 1.0f ) );
	float fbottom = pProjectionMatrix._array[ 14 ] * ( pProjectionMatrix._array[ 9 ] - 1.0f ) / ( pProjectionMatrix._array[ 5 ] * ( pProjectionMatrix._array[ 10 ] - 1.0f ) );
	float fnear   = pProjectionMatrix._array[ 14 ] / ( pProjectionMatrix._array[ 10 ] - 1.0f );
	float ffar    = pProjectionMatrix._array[ 14 ] / ( pProjectionMatrix._array[ 10 ] + 1.0f );

	//float2 viewSurfaceVS[2];
	//viewSurfaceVS[0]=make_float2(fleft, fbottom);
	//viewSurfaceVS[1]=make_float2(fright, ftop);

	//float2 viewSurfaceVS_Size=viewSurfaceVS[1]-viewSurfaceVS[0];
	/////////////////////////////////////////////
GV_CHECK_GL_ERROR();
	// transfor matrices
	float4x4 invprojectionMatrixT=transpose(inverse(pProjectionMatrix));
	float4x4 invViewMatrixT=transpose(inverse(pViewMatrix));

	float4x4 projectionMatrixT=transpose(pProjectionMatrix);
	float4x4 viewMatrixT=transpose(pViewMatrix);
	float4x4 modelMatrixT=transpose(pModelMatrix);
	//float4x4 modelMatrixT=pModelMatrix;
	CUDAPM_START_EVENT(vsrender_copyconsts_frame);
GV_CHECK_GL_ERROR();
	viewContext.invViewMatrix=invViewMatrixT;
	viewContext.viewMatrix=viewMatrixT;
	viewContext.modelMatrix = modelMatrixT;
	//viewContext.invProjMatrix=invprojectionMatrixT;
	//viewContext.projMatrix=projectionMatrixT;
GV_CHECK_GL_ERROR();
	// Store frustum parameters
	viewContext.frustumNear = fnear;
	viewContext.frustumNearINV = 1.0f / fnear;
	viewContext.frustumFar = ffar;
	viewContext.frustumRight = fright;
	viewContext.frustumTop = ftop;
	viewContext.frustumC = pProjectionMatrix._array[ 10 ]; // - ( ffar + fnear ) / ( ffar - fnear );
	viewContext.frustumD = pProjectionMatrix._array[ 14 ]; // ( -2.0f * ffar * fnear ) / ( ffar - fnear );
GV_CHECK_GL_ERROR();
	float3 viewPlanePosWP = mul( viewContext.invViewMatrix, make_float3( fleft, fbottom, -fnear ) );
	viewContext.viewCenterWP = mul( viewContext.invViewMatrix, make_float3( 0.0f, 0.0f, 0.0f ) );
	viewContext.viewPlaneDirWP = viewPlanePosWP - viewContext.viewCenterWP;

	// Resolution dependant stuff
	viewContext.frameSize = make_uint2( pViewport.z, pViewport.w );
	//float2 pixelSize=viewSurfaceVS_Size/make_float2((float)viewContext.frameSize.x, (float)viewContext.frameSize.y);
	//viewContext.pixelSize=pixelSize;
	viewContext.viewPlaneXAxisWP = ( mul( viewContext.invViewMatrix, make_float3( fright, fbottom, -fnear ) ) - viewPlanePosWP ) / static_cast< float >( viewContext.frameSize.x );
	viewContext.viewPlaneYAxisWP = ( mul( viewContext.invViewMatrix, make_float3( fleft, ftop, -fnear ) ) - viewPlanePosWP ) / static_cast< float >( viewContext.frameSize.y );

	//_graphicsInteroperabiltyHandler->setRendererContextInfo( viewContext );
GV_CHECK_GL_ERROR();GV_CHECK_GL_ERROR();
	// Copy data to CUDA memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_renderViewContext, &viewContext, sizeof( viewContext ) ) );
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_currentTime, &(this->_currentTime), sizeof( this->_currentTime ), 0, cudaMemcpyHostToDevice ) );

	CUDAPM_STOP_EVENT( vsrender_copyconsts_frame );
	GV_CHECK_GL_ERROR();
	// Specify values of uniform variables for shader program object
	glProgramUniform3fEXT( _rayCastProg, glGetUniformLocation( _rayCastProg, "uViewPos" ), viewContext.viewCenterWP.x, viewContext.viewCenterWP.y, viewContext.viewCenterWP.z );
	glProgramUniform3fEXT( _rayCastProg, glGetUniformLocation( _rayCastProg, "uViewPlane" ), viewContext.viewPlaneDirWP.x, viewContext.viewPlaneDirWP.y, viewContext.viewPlaneDirWP.z );
	glProgramUniform3fEXT( _rayCastProg, glGetUniformLocation( _rayCastProg, "uViewAxisX" ), viewContext.viewPlaneXAxisWP.x, viewContext.viewPlaneXAxisWP.y, viewContext.viewPlaneXAxisWP.z );
	glProgramUniform3fEXT( _rayCastProg, glGetUniformLocation( _rayCastProg, "uViewAxisY" ), viewContext.viewPlaneYAxisWP.x, viewContext.viewPlaneYAxisWP.y, viewContext.viewPlaneYAxisWP.z );
	//glProgramUniform2fEXT( _rayCastProg, glGetUniformLocation( _rayCastProg, "frameSize" ), static_cast< float >( viewContext.frameSize.x ), static_cast< float >( viewContext.frameSize.y ) );
	//glProgramUniformMatrix4fvEXT( _rayCastProg, glGetUniformLocation( _rayCastProg, "modelViewMat" ), 1, GL_FALSE, pViewMatrix._array );
GV_CHECK_GL_ERROR();

	//glProgramUniform4iEXT( _rayCastProg, glGetUniformLocation( _rayCastProg, "viewPort" ), pViewport.x, pViewport.y, pViewport.z, pViewport.w);
	
	
	glProgramUniform3fEXT( _rayCastProg, glGetUniformLocation( _rayCastProg, "lightPos" ), _lightPos.x, _lightPos.y, _lightPos.z);
	GV_CHECK_GL_ERROR();
	//float voxelSizeMultiplier = 1.0f;
	//GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( GvRendering::k_voxelSizeMultiplier, (&voxelSizeMultiplier), sizeof( voxelSizeMultiplier ), 0, cudaMemcpyHostToDevice ) );

	//uint numLoop = 0;

#ifdef _DEBUG
	// DEBUG : text buffer
	{
		/*glBindBuffer( GL_TEXTURE_BUFFER, _textBuffer );
		GLfloat* textBuffer = (GLfloat *)glMapBuffer( GL_TEXTURE_BUFFER, GL_WRITE_ONLY );

		for ( int i = 0; i < 8192; i++ )
		{
			textBuffer[ i ] = 0.0f;
		}

		glUnmapBuffer( GL_TEXTURE_BUFFER );
		glBindBuffer( GL_TEXTURE_BUFFER, 0 );*/
	}
#endif

	CUDAPM_START_EVENT_GPU( gv_rendering );

	/*if ( this->_dynamicUpdate )
	{
	}
	else
	{
	}*/

	// Disable writing into the depth buffer
	//glEnable(GL_DEPTH_TEST);
	glDepthMask( GL_FALSE );

	// Activate blending
	glEnable( GL_BLEND );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

	// Installs program object as part of current rendering state
	glUseProgram( _rayCastProg );

	// Retrieve useful GigaVoxels arrays
	/*GvCore::Array3DGPULinear< uint >**/ volTreeChildArray = this->_volumeTree->_childArray;
	/*GvCore::Array3DGPULinear< uint >**/ volTreeDataArray = this->_volumeTree->_dataArray;
	GvCore::Array3DGPULinear< uint >* updateBufferArray = this->_volumeTreeCache->getUpdateBuffer();
	GvCore::Array3DGPULinear< uint >* nodeTimeStampArray = this->_volumeTreeCache->getNodesCacheManager()->getTimeStampList();
	GvCore::Array3DGPULinear< uint >* brickTimeStampArray = this->_volumeTreeCache->getBricksCacheManager()->getTimeStampList();
	GV_CHECK_GL_ERROR();
	// Unmap GigaVoxels graphics resources from CUDA environment in order to be used by OpenGL
	updateBufferArray->unmapResource();
	nodeTimeStampArray->unmapResource();
	brickTimeStampArray->unmapResource();
	volTreeChildArray->unmapResource();
	volTreeDataArray->unmapResource();
	this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 0 >() )->unmapResource();
	GV_CHECK_GL_ERROR();
	// Returns location of uniform variables
	/*GLint volTreeChildArrayLoc = glGetUniformLocation( _rayCastProg, "uNodePoolChildArray" );
	GLint volTreeDataArrayLoc = glGetUniformLocation( _rayCastProg, "uNodePoolDataArray" );
	GLint updateBufferArrayLoc = glGetUniformLocation( _rayCastProg, "uUpdateBufferArray" );
	GLint nodeTimeStampArrayLoc = glGetUniformLocation( _rayCastProg, "uNodeTimeStampArray" );
	GLint brickTimeStampArrayLoc = glGetUniformLocation( _rayCastProg, "uBrickTimeStampArray" );
	GLint currentTimeLoc = glGetUniformLocation( _rayCastProg, "uCurrentTime" );
*/
	// Note :
	// glBindImageTextureEXT() command binds a single level of a texture to an image unit
	// for the purpose of reading and writing it from shaders.
	//
	// Specification :
	// void glBindImageTexture( GLuint  unit,  GLuint  texture,  GLint  level,  GLboolean  layered,  GLint  layer,  GLenum  access,  GLenum  format );
	GV_CHECK_GL_ERROR();
	glBindImageTextureEXT( 0, _updateBufferTBO, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI );
	glUniform1i( _updateBufferArrayLoc, 0 );
	GV_CHECK_GL_ERROR();
	glBindImageTextureEXT( 1, _nodeTimeStampTBO, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI );
	glUniform1i( _nodeTimeStampArrayLoc, 1 );

	glBindImageTextureEXT(2, _brickTimeStampTBO, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI );
	glUniform1i( _brickTimeStampArrayLoc, 2 );

	glBindImageTextureEXT(3, _childArrayTBO, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI );
	glUniform1i( _volTreeChildArrayLoc, 3 );

	glBindImageTextureEXT( 4, _dataArrayTBO, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI );
	glUniform1i( _volTreeDataArrayLoc, 4 );
	glUniform1ui( _currentTimeLoc, this->_currentTime );

	glBindImageTextureEXT( 7, _textBufferTBO, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F );
	glUniform1i( glGetUniformLocation( _rayCastProg, "d_textBuffer" ), 7 );

	// Retrieve node pool and brick pool resolution
	uint3 nodePoolRes = this->_volumeTree->_nodePool->getChannel( Loki::Int2Type< 0 >() )->getResolution();
	uint3 brickPoolRes = this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 0 >() )->getResolution();

	// Retrieve node cache and brick cache size
	uint3 nodeCacheSize = nodeTimeStampArray->getResolution();
	brickCacheSize = brickTimeStampArray->getResolution();

	glUniform3ui( glGetUniformLocation( _rayCastProg, "nodeCacheSize" ),
		nodeCacheSize.x, nodeCacheSize.y, nodeCacheSize.z );
	
	glUniform3ui( glGetUniformLocation( _rayCastProg, "uBrickCacheSize" ),
		brickCacheSize.x, brickCacheSize.y, brickCacheSize.z );
	GV_CHECK_GL_ERROR();
	GLint dataPoolLoc = glGetUniformLocation( _rayCastProg, "uDataPool" );
	glUniform1i( dataPoolLoc, 0 );

	glUniform3f( glGetUniformLocation( _rayCastProg, "nodePoolResInv" ),
		1.0f / (GLfloat)nodePoolRes.x, 1.0f / (GLfloat)nodePoolRes.y, 1.0f / (GLfloat)nodePoolRes.z );
	GV_CHECK_GL_ERROR();
	brickPoolResInv = make_float3(1.0f / (GLfloat)brickPoolRes.x, 1.0f / (GLfloat)brickPoolRes.y, 1.0f / (GLfloat)brickPoolRes.z);
	glUniform3f( glGetUniformLocation( _rayCastProg, "uBrickPoolResInv" ),
		1.0f / (GLfloat)brickPoolRes.x, 1.0f / (GLfloat)brickPoolRes.y, 1.0f / (GLfloat)brickPoolRes.z );

	GV_CHECK_GL_ERROR();
	maxDepth = this->_volumeTree->getMaxDepth();
	GLint res = glGetUniformLocation( _rayCastProg, "uMaxDepth" );
	glUniform1ui( glGetUniformLocation( _rayCastProg, "uMaxDepth" ), this->_volumeTree->getMaxDepth() );

	glUniform1f( glGetUniformLocation( _rayCastProg, "frustumC" ), viewContext.frustumC );
	glUniform1f( glGetUniformLocation( _rayCastProg, "frustumD" ), viewContext.frustumD );

	// Check for OpenGL error(s)
	GV_CHECK_GL_ERROR();

	// Bind user data as 3D texture for rendering
	//
	// Content of one voxel has been defined as :
	// typedef Loki::TL::MakeTypelist< uchar4 >::Result DataType;
	glActiveTexture( GL_TEXTURE0 );
	texBufferName = this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 0 >() )->getBufferName(); 
	glBindTexture( GL_TEXTURE_3D, this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 0 >() )->getBufferName() );

	GLuint vploc = glGetAttribLocation( _rayCastProg, "iPosition" );

	// Set projection matrix to identity
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();

	// Set model/view matrix to identity
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();

	// Draw a quad on full screen
	glBegin( GL_QUADS );
	glVertexAttrib4f( vploc, -1.0f, -1.0f, 0.0f, 1.0f );
	glVertexAttrib4f( vploc,  1.0f, -1.0f, 0.0f, 1.0f );
	glVertexAttrib4f( vploc,  1.0f,  1.0f, 0.0f, 1.0f );
	glVertexAttrib4f( vploc, -1.0f,  1.0f, 0.0f, 1.0f );
	glEnd();

	// Restore previous projection matrix
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();

	// Restore previous model/view matrix
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();

	// Unbind user data as 3D texture for rendering
	glBindTexture( GL_TEXTURE_3D, 0 );

	// Stop using shader program
	//
	// Programmable processors will be disabled and fixed functionality will be used
	// for both vertex and fragment processing.
	glUseProgram( 0 );

	//glMemoryBarrierEXT( GL_BUFFER_UPDATE_BARRIER_BIT_EXT );
	//glMemoryBarrierEXT( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT_EXT );
	//glMemoryBarrierEXT( GL_ALL_BARRIER_BITS_EXT );

	// Check for OpenGL error(s)
	GV_CHECK_GL_ERROR();

#ifdef _DEBUG
	// DEBUG : text buffer
	/*glBindBuffer( GL_TEXTURE_BUFFER, _textBuffer );
	GLfloat* textBuffer = (GLfloat *)glMapBuffer( GL_TEXTURE_BUFFER, GL_READ_ONLY );

	for ( int i = 0; i < 8192; i++ )
	{
		if ( textBuffer[ i ] != 0.0f )
		{
			printf( "\ntextBuffer[ %d ] = %f", i, textBuffer[ i ] );
		}
	}

	glUnmapBuffer( GL_TEXTURE_BUFFER );
	glBindBuffer( GL_TEXTURE_BUFFER, 0 );*/
#endif

	// Map GigaVoxels graphics resources in order to be used by CUDA environment
	updateBufferArray->mapResource();
	nodeTimeStampArray->mapResource();
	brickTimeStampArray->mapResource();
	volTreeChildArray->mapResource();
	volTreeDataArray->mapResource();
	this->_volumeTree->_dataPool->getChannel(Loki::Int2Type<0>())->mapResource();

	// Disable blending and enable writing into depth buffer
	glDisable( GL_BLEND );
	//glEnable( GL_DEPTH_TEST );
	glDepthMask( GL_TRUE );

	CUDAPM_STOP_EVENT_GPU( gv_rendering );

	CUDAPM_START_EVENT( vsrender_post_frame );
	CUDAPM_START_EVENT( vsrender_post_frame_unmapbuffers );
	CUDAPM_STOP_EVENT( vsrender_post_frame_unmapbuffers );
	CUDAPM_STOP_EVENT( vsrender_post_frame );

	GV_CHECK_CUDA_ERROR( "RendererVolTreeGLSL::render" );
}



template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::setLightPosition(float x, float y, float z) {
	_lightPos = make_float3(x, y, z);
}

template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
uint3 VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::getBrickCacheSize() {
	return this->brickCacheSize;
}

template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
float3 VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::getBrickPoolResInv() {
	return brickPoolResInv;
}

template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
uint VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::getMaxDepth() {
	return maxDepth;
}


template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
GvCore::Array3DGPULinear< uint >* VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::getVolTreeChildArray() {
	//volTreeChildArray->unmapResource();
	return volTreeChildArray;
}

template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
GvCore::Array3DGPULinear< uint >* VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::getVolTreeDataArray() {
	//volTreeDataArray->unmapResource();
	return volTreeDataArray;
}

template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
GLint VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::getChildBufferName() {
	return childBufferName;
}

template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
GLint VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::getDataBufferName() {
	return dataBufferName;
}

template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
GLint VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::getTexBufferName() {
	return texBufferName;
}



/******************************************************************************
 * Attach an OpenGL texture or renderbuffer object to an internal graphics resource 
 * that will be mapped to a color or depth slot used during rendering.
 *
 * @param pGraphicsResourceSlot the associated internal graphics resource (color or depth) and its type of access (read, write or read/write)
 * @param pImage the OpenGL texture or renderbuffer object
 * @param pTarget the target of the OpenGL texture or renderbuffer object
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
bool VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::connect( GvRendering::GvGraphicsInteroperabiltyHandler::GraphicsResourceSlot pGraphicsResourceSlot, GLuint pImage, GLenum pTarget )
{
	return _graphicsInteroperabiltyHandler->connect( pGraphicsResourceSlot, pImage, pTarget );
}

/******************************************************************************
 * Attach an OpenGL buffer object (i.e. a PBO, a VBO, etc...) to an internal graphics resource 
 * that will be mapped to a color or depth slot used during rendering.
 *
 * @param pGraphicsResourceSlot the associated internal graphics resource (color or depth) and its type of access (read, write or read/write)
 * @param pBuffer the OpenGL buffer
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
bool VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::connect( GvRendering::GvGraphicsInteroperabiltyHandler::GraphicsResourceSlot pGraphicsResourceSlot, GLuint pBuffer )
{
	return _graphicsInteroperabiltyHandler->connect( pGraphicsResourceSlot, pBuffer );
}

/******************************************************************************
 * Disconnect all registered graphics resources
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
bool VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::resetGraphicsResources()
{
	return _graphicsInteroperabiltyHandler->reset();
}

template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
bool VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::bindGraphicsResources()
{
	// Iterate through graphics resources info
	std::vector< std::pair< GvRendering::GvGraphicsInteroperabiltyHandler::GraphicsResourceSlot, GvRendering::GvGraphicsResource* > >& graphicsResources = _graphicsInteroperabiltyHandler->editGraphicsResources();
	for ( int i = 0; i < graphicsResources.size(); i++ )
	{
		// Get current graphics resources info
		std::pair< GvRendering::GvGraphicsInteroperabiltyHandler::GraphicsResourceSlot, GvRendering::GvGraphicsResource* >& graphicsResourceInfo = graphicsResources[ i ];
		GvRendering::GvGraphicsInteroperabiltyHandler::GraphicsResourceSlot graphicsResourceSlot = graphicsResourceInfo.first;
		GvRendering::GvGraphicsResource* graphicsResource  = graphicsResourceInfo.second;
		assert( graphicsResource != NULL );

		// [ 2 ] - Bind array to texture or surface if needed
		if ( graphicsResource->getMemoryType() == GvRendering::GvGraphicsResource::eCudaArray )
		{
			struct cudaArray* imageArray = static_cast< struct cudaArray* >( graphicsResource->getMappedAddress() );

			cudaError_t error;
			switch ( graphicsResourceSlot )
			{
			case GvRendering::GvGraphicsInteroperabiltyHandler::eColorReadSlot:
					error = cudaBindTextureToArray( GvRendering::_inputColorTexture, imageArray );
					break;

				case GvRendering::GvGraphicsInteroperabiltyHandler::eColorWriteSlot:
				case GvRendering::GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot:
					error = cudaBindSurfaceToArray( GvRendering::_colorSurface, imageArray );
					break;

				case GvRendering::GvGraphicsInteroperabiltyHandler::eDepthReadSlot:
					error = cudaBindTextureToArray( GvRendering::_inputDepthTexture, imageArray );
					break;

				case GvRendering::GvGraphicsInteroperabiltyHandler::eDepthWriteSlot:
				case GvRendering::GvGraphicsInteroperabiltyHandler::eDepthReadWriteSlot:
					error = cudaBindSurfaceToArray( GvRendering::_depthSurface, imageArray );
					break;

				default:
					assert( false );
					break;
			}
		}
	}

	return false;
}

/******************************************************************************
 * Unbind all graphics resources used by the GL interop handler during rendering.
 *
 * Internally, it unbinds textures and surfaces to arrays associated to mapped graphics reources.
 *
 * NOTE : this method should be in the GvGraphicsInteroperabiltyHandler but it seems that
 * there are conflicts with textures ans surfaces symbols. The binding succeeds but not the
 * read/write operations.
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
bool VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::unbindGraphicsResources()
{
	// Iterate through graphics resources info
	std::vector< std::pair< GvRendering::GvGraphicsInteroperabiltyHandler::GraphicsResourceSlot, GvRendering::GvGraphicsResource* > >& graphicsResources = _graphicsInteroperabiltyHandler->editGraphicsResources();
	for ( int i = 0; i < graphicsResources.size(); i++ )
	{
		// Get current graphics resources info
		std::pair< GvRendering::GvGraphicsInteroperabiltyHandler::GraphicsResourceSlot, GvRendering::GvGraphicsResource* >& graphicsResourceInfo = graphicsResources[ i ];
		GvRendering::GvGraphicsInteroperabiltyHandler::GraphicsResourceSlot graphicsResourceSlot = graphicsResourceInfo.first;
		GvRendering::GvGraphicsResource* graphicsResource  = graphicsResourceInfo.second;
		assert( graphicsResource != NULL );

		// [ 2 ] - Bind array to texture or surface if needed
		if ( graphicsResource->getMemoryType() == GvRendering::GvGraphicsResource::eCudaArray )
		{
			struct cudaArray* imageArray = static_cast< struct cudaArray* >( graphicsResource->getMappedAddress() );

			cudaError_t error;
			switch ( graphicsResourceSlot )
			{
			case GvRendering::GvGraphicsInteroperabiltyHandler::eColorReadSlot:
					error = cudaUnbindTexture( GvRendering::_inputColorTexture );
					break;

				case GvRendering::GvGraphicsInteroperabiltyHandler::eColorWriteSlot:
				case GvRendering::GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot:
					// There is no "unbind surface" function in CUDA
					break;

				case GvRendering::GvGraphicsInteroperabiltyHandler::eDepthReadSlot:
					error = cudaUnbindTexture( GvRendering::_inputDepthTexture );
					break;

				case GvRendering::GvGraphicsInteroperabiltyHandler::eDepthWriteSlot:
				case GvRendering::GvGraphicsInteroperabiltyHandler::eDepthReadWriteSlot:
					// There is no "unbind surface" function in CUDA
					break;

				default:
					assert( false );
					break;
			}
		}
	}

	return false;
}


