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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GvError.h>
#include <GsGraphics/GsShaderProgram.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>

// STL
#include <iostream>

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
template< typename TDataStructureType, typename TVolumeTreeCacheType >
RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::RendererGLSL( TDataStructureType* pVolumeTree, TVolumeTreeCacheType* pVolumeTreeCache )
:	GvRendering::GvRenderer< TDataStructureType, TVolumeTreeCacheType >( pVolumeTree, pVolumeTreeCache )
,	_coneApertureScale( 0.f )
,	_maxNbLoops( 0 )
,	_positionLoc( -1 )
{
	bool statusOK = false;

	// Retrieve useful GigaVoxels arrays
	_nodeBuffer = this->_volumeTree->_childArray;
	_dataBuffer = this->_volumeTree->_dataArray;
	_requestBuffer = this->_volumeTreeCache->getUpdateBuffer();
	_nodeTimestampBuffer = this->_volumeTreeCache->getNodesCacheManager()->getTimeStampList();
	_brickTimestampBuffer = this->_volumeTreeCache->getBricksCacheManager()->getTimeStampList();

	// Create buffer textures associated to the GigaVoxels/GiagSpace buffers
	// - update buffer array
	glGenTextures( 1, &_updateBufferTBO );
	// - node time stamp array
	glGenTextures( 1, &_nodeTimeStampTBO );
	// - brick time stamp array
	glGenTextures( 1, &_brickTimeStampTBO );
	// - data structure's child array
	glGenTextures( 1, &_childArrayTBO );
	// - data structure's data array
	glGenTextures( 1, &_dataArrayTBO );
	
	// Attach the storage of buffer objects to buffer textures
	// - update buffer array
	glBindTexture( GL_TEXTURE_BUFFER, _updateBufferTBO );
	glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, _requestBuffer->getBufferName() );
	// - node time stamp array
	glBindTexture( GL_TEXTURE_BUFFER, _nodeTimeStampTBO );
	glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, _nodeTimestampBuffer->getBufferName() );
	// - brick time stamp array
	glBindTexture( GL_TEXTURE_BUFFER, _brickTimeStampTBO );
	glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, _brickTimestampBuffer->getBufferName() );
	// - data structure's child array
	glBindTexture( GL_TEXTURE_BUFFER, _childArrayTBO );
	glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, _nodeBuffer->getBufferName() );
	// - data structure's data array
	glBindTexture( GL_TEXTURE_BUFFER, _dataArrayTBO );
	glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, _dataBuffer->getBufferName() );
	
	// Reset GL state
	glBindTexture( GL_TEXTURE_BUFFER, 0 );

	// Check for OpenGL error(s)
	GV_CHECK_GL_ERROR();

	// Initialize shader program
	statusOK = initializeShaderProgram();
	assert( statusOK );

	// Check for OpenGL error(s)
	GV_CHECK_GL_ERROR();

	// Unmap GigaVoxels graphics resources from CUDA environment in order to be used by OpenGL
	//_graphicsResources = new cudaGraphicsResource[ 7 ];
	_graphicsResources[ 0 ] = _requestBuffer->getGraphicsResource();
	_graphicsResources[ 1 ] = _nodeTimestampBuffer->getGraphicsResource();
	_graphicsResources[ 2 ] = _brickTimestampBuffer->getGraphicsResource();
	_graphicsResources[ 3 ] = _nodeBuffer->getGraphicsResource();
	_graphicsResources[ 4 ] = _dataBuffer->getGraphicsResource();
	_graphicsResources[ 5 ] = this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 0 >() )->getGraphicsResource();
	_graphicsResources[ 6 ] = this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 1 >() )->getGraphicsResource();
	
	// Settings
	_coneApertureScale = 1.333f;
	_maxNbLoops = 200;

	// OpenGL initialization
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA ); // to avoid redundant call
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TDataStructureType, typename TVolumeTreeCacheType >
RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::~RendererGLSL()
{
	// Finalize shader program
	finalizeShaderProgram();

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
}

#ifdef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_GVRENDERERCUDA
/******************************************************************************
 * pre-render stage
 ******************************************************************************/
template< typename TDataStructureType, typename TVolumeTreeCacheType >
inline void RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::preRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
}
/******************************************************************************
* post-render stage
 ******************************************************************************/
template< typename TDataStructureType, typename TVolumeTreeCacheType >
inline void RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::postRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
}
#endif // GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_GVRENDERERCUDA

/******************************************************************************
 * This function is the specific implementation method called
 * by the parent GvIRenderer::render() method during rendering.
 *
 * @param pModelMatrix the current model matrix
 * @param pViewMatrix the current view matrix
 * @param pProjectionMatrix the current projection matrix
 * @param pViewport the viewport configuration
 ******************************************************************************/
template< typename TDataStructureType, typename TVolumeTreeCacheType >
void RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::render( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	// Call internal render method
	doRender( pModelMatrix, pViewMatrix, pProjectionMatrix, pViewport );
}

/******************************************************************************
 * Start the rendering process.
 *
 * @param pModelMatrix the current model matrix
 * @param pViewMatrix the current view matrix
 * @param pProjectionMatrix the current projection matrix
 * @param pViewport the viewport configuration
 ******************************************************************************/
template< typename TDataStructureType, typename TVolumeTreeCacheType >
void RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::doRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	CUDAPM_START_EVENT( vsrender_pre_frame );
	CUDAPM_START_EVENT( vsrender_pre_frame_mapbuffers );
	CUDAPM_STOP_EVENT( vsrender_pre_frame_mapbuffers );
	CUDAPM_STOP_EVENT( vsrender_pre_frame );

	// Unmap GigaVoxels graphics resources from CUDA environment in order to be used by OpenGL
	GV_CUDA_SAFE_CALL( cudaGraphicsUnmapResources( 7, &_graphicsResources[ 0 ], 0 ) );
	
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

	float2 viewSurfaceVS[ 2 ];
	viewSurfaceVS[ 0 ] = make_float2( fleft, fbottom );
	viewSurfaceVS[ 1 ] = make_float2( fright, ftop );
	float2 viewSurfaceVS_Size = viewSurfaceVS[ 1 ] - viewSurfaceVS[ 0 ];
	
	// transfor matrices
	float4x4 invprojectionMatrixT = transpose( inverse( pProjectionMatrix ) );
	float4x4 invViewMatrixT = transpose( inverse( pViewMatrix ) );

	float4x4 projectionMatrixT = transpose( pProjectionMatrix );
	float4x4 viewMatrixT = transpose( pViewMatrix );

	CUDAPM_START_EVENT( vsrender_copyconsts_frame );

	viewContext.invViewMatrix = invViewMatrixT;
	viewContext.viewMatrix = viewMatrixT;
	//viewContext.invProjMatrix = invprojectionMatrixT;
	//viewContext.projMatrix = projectionMatrixT;

	// Store frustum parameters
	viewContext.frustumNear = fnear;
	viewContext.frustumNearINV = 1.0f / fnear;
	viewContext.frustumFar = ffar;
	viewContext.frustumRight = fright;
	viewContext.frustumTop = ftop;
	viewContext.frustumC = pProjectionMatrix._array[ 10 ]; // - ( ffar + fnear ) / ( ffar - fnear );
	viewContext.frustumD = pProjectionMatrix._array[ 14 ]; // ( -2.0f * ffar * fnear ) / ( ffar - fnear );

	float3 viewPlanePosWP = mul( viewContext.invViewMatrix, make_float3( fleft, fbottom, -fnear ) );
	viewContext.viewCenterWP = mul( viewContext.invViewMatrix, make_float3( 0.0f, 0.0f, 0.0f ) );
	viewContext.viewPlaneDirWP = viewPlanePosWP - viewContext.viewCenterWP;

	// Resolution dependant stuff
	viewContext.frameSize = make_uint2( pViewport.z, pViewport.w );
	float2 pixelSize = viewSurfaceVS_Size / make_float2( (float)viewContext.frameSize.x, (float)viewContext.frameSize.y );
	//viewContext.pixelSize=pixelSize;
	viewContext.viewPlaneXAxisWP = ( mul( viewContext.invViewMatrix, make_float3( fright, fbottom, -fnear ) ) - viewPlanePosWP ) / static_cast< float >( viewContext.frameSize.x );
	viewContext.viewPlaneYAxisWP = ( mul( viewContext.invViewMatrix, make_float3( fleft, ftop, -fnear ) ) - viewPlanePosWP ) / static_cast< float >( viewContext.frameSize.y );
	
	CUDAPM_STOP_EVENT( vsrender_copyconsts_frame );

	// Disable writing into the depth buffer
	//glEnable( GL_DEPTH_TEST );
	glDepthMask( GL_FALSE );

	// Activate blending
	glEnable( GL_BLEND );
	//glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA ); // to avoid redundant call

	// Installs program object as part of current rendering state
	_shaderProgram->use();

	// Viewing System uniform parameters
	// - glProgramUniform3fEXT => requires OpenGL 4.1
	glProgramUniform3fEXT( _shaderProgram->_program, _viewPosLoc, viewContext.viewCenterWP.x, viewContext.viewCenterWP.y, viewContext.viewCenterWP.z );
	glProgramUniform3fEXT( _shaderProgram->_program, _viewPlaneLoc, viewContext.viewPlaneDirWP.x, viewContext.viewPlaneDirWP.y, viewContext.viewPlaneDirWP.z );
	glProgramUniform3fEXT( _shaderProgram->_program, _viewAxisXLoc, viewContext.viewPlaneXAxisWP.x, viewContext.viewPlaneXAxisWP.y, viewContext.viewPlaneXAxisWP.z );
	glProgramUniform3fEXT( _shaderProgram->_program, _viewAxisYLoc, viewContext.viewPlaneYAxisWP.x, viewContext.viewPlaneYAxisWP.y, viewContext.viewPlaneYAxisWP.z );
	glProgramUniform2fEXT( _shaderProgram->_program, _pixelSizeLoc, pixelSize.x, pixelSize.y );
	glProgramUniform1fEXT( _shaderProgram->_program, _frustumNearInvLoc, viewContext.frustumNearINV );
		
	//glProgramUniform2fEXT( _shaderProgram->_program, glGetUniformLocation( _shaderProgram->_program, "frameSize" ), static_cast< float >( viewContext.frameSize.x ), static_cast< float >( viewContext.frameSize.y ) );
	//glProgramUniformMatrix4fvEXT( _shaderProgram->_program, glGetUniformLocation( _shaderProgram->_program, "modelViewMat" ), 1, GL_FALSE, pViewMatrix._array );

	// Update time
	glUniform1ui( _currentTimeLoc, this->_currentTime );

	CUDAPM_START_EVENT_GPU( gv_rendering );

	// TODO
	// - add dynamic update management
	/*if ( this->_dynamicUpdate )
	{
	}
	else
	{
	}*/

	// Note :
	// glBindImageTextureEXT() command binds a single level of a texture to an image unit
	// for the purpose of reading and writing it from shaders.
	//
	// Specification :
	// void glBindImageTexture( GLuint  unit,  GLuint  texture,  GLint  level,  GLboolean  layered,  GLint  layer,  GLenum  access,  GLenum  format );

	// Data Production Management
	// - buffer of requests
	glBindImageTextureEXT( 0, _updateBufferTBO, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI );
	// - nodes time stamps buffer
	glBindImageTextureEXT( 1, _nodeTimeStampTBO, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI );
	// - bricks time stamps buffer
	glBindImageTextureEXT( 2, _brickTimeStampTBO, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI );
	// Node Pool
	// - child array
	glBindImageTextureEXT( 3, _childArrayTBO, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI );
	// - data array
	glBindImageTextureEXT( 4, _dataArrayTBO, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI );
	
	// Bind user data as 3D texture for rendering
	//
	// Content of one voxel has been defined as :
	// typedef Loki::TL::MakeTypelist< uchar4 >::Result DataType;
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_3D, this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 0 >() )->getBufferName() );
	glActiveTexture( GL_TEXTURE1 );
	glBindTexture( GL_TEXTURE_3D, this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 1 >() )->getBufferName() );

	// Draw a fullscreen quad
	// - TODO : use programmable shader program
	glBegin( GL_QUADS );
	glVertexAttrib2f( _positionLoc, -1.0f, -1.0f );
	glVertexAttrib2f( _positionLoc,  1.0f, -1.0f );
	glVertexAttrib2f( _positionLoc,  1.0f,  1.0f );
	glVertexAttrib2f( _positionLoc, -1.0f,  1.0f );
	glEnd();

	// Unbind user data as 3D texture for rendering
	glBindTexture( GL_TEXTURE_3D, 0 );	// deprecated

	// Stop using shader program
	//
	// Programmable processors will be disabled and fixed functionality will be used
	// for both vertex and fragment processing.
	glUseProgram( 0 );

	//glMemoryBarrierEXT( GL_BUFFER_UPDATE_BARRIER_BIT_EXT );
	//glMemoryBarrierEXT( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT_EXT );
	//glMemoryBarrierEXT( GL_ALL_BARRIER_BITS_EXT );

	// Disable blending and enable writing into depth buffer
	glDisable( GL_BLEND );
	//glEnable( GL_DEPTH_TEST );
	glDepthMask( GL_TRUE );

	CUDAPM_STOP_EVENT_GPU( gv_rendering );

	// Map GigaVoxels graphics resources in order to be used by CUDA environment
	GV_CUDA_SAFE_CALL( cudaGraphicsMapResources( 7, &_graphicsResources[ 0 ], 0 ) );

	// Copy data to CUDA memory
	// - use async copy first
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbolAsync( k_renderViewContext, &viewContext, sizeof( viewContext ) ) );
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbolAsync( k_currentTime, &(this->_currentTime), sizeof( this->_currentTime ), 0, cudaMemcpyHostToDevice ) )

	CUDAPM_START_EVENT( vsrender_post_frame );
	CUDAPM_START_EVENT( vsrender_post_frame_unmapbuffers );
	CUDAPM_STOP_EVENT( vsrender_post_frame_unmapbuffers );
	CUDAPM_STOP_EVENT( vsrender_post_frame );

	GV_CHECK_CUDA_ERROR( "RendererGLSL::render" );
}

/******************************************************************************
 * Get the cone aperture scale
 *
 * @return the cone aperture scale
 ******************************************************************************/
template< typename TDataStructureType, typename TVolumeTreeCacheType >
float RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::getConeApertureScale() const
{
	return _coneApertureScale;
}

/******************************************************************************
 * Set the cone aperture scale
 *
 * @param pValue the cone aperture scale
 ******************************************************************************/
template< typename TDataStructureType, typename TVolumeTreeCacheType >
void RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::setConeApertureScale( float pValue )
{
	_coneApertureScale = pValue;

	// Cone aperture management
	glProgramUniform1fEXT( _shaderProgram->_program, _coneApertureScaleLoc, _coneApertureScale );
}

/******************************************************************************
 * Get the max number of loops during the main GigaSpace pipeline pass (GLSL shader)
 *
 * @return the max number of loops
 ******************************************************************************/
template< typename TDataStructureType, typename TVolumeTreeCacheType >
unsigned int RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::getMaxNbLoops() const
{
	return _maxNbLoops;
}

/******************************************************************************
 * Set the max number of loops during the main GigaSpace pipeline pass (GLSL shader)
 *
 * @param pValue the max number of loops
 ******************************************************************************/
template< typename TDataStructureType, typename TVolumeTreeCacheType >
void RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::setMaxNbLoops( unsigned int pValue )
{
	_maxNbLoops = pValue;

	// GigaSpace pipeline uniform parameters
	glProgramUniform1uiEXT( _shaderProgram->_program, _maxNbLoopsLoc, _maxNbLoops );
}

/******************************************************************************
 * Set the max depth
 *
 * @param pValue the max depth
 ******************************************************************************/
template< typename TDataStructureType, typename TVolumeTreeCacheType >
void RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::setMaxDepth( unsigned int pValue )
{	
	// GigaSpace pipeline uniform parameters
	//glProgramUniform1uiEXT( _shaderProgram->_program, _maxDepthLoc, pValue );
	glProgramUniform1uiEXT( _shaderProgram->_program, _maxDepthLoc, this->_volumeTree->getMaxDepth() );
}

/******************************************************************************
 * Initialize shader program
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
template< typename TDataStructureType, typename TVolumeTreeCacheType >
bool RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::initializeShaderProgram()
{
	bool statusOK = true;

	// Retrieve data repository
	QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	
	// Create and link a GLSL shader program
	QString vertexShaderFilename = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "GvSimpleShapeGLSL" ) + QDir::separator() + QString( "gigaspace_vert.glsl" );
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

	QString fragmentShaderFilename = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "GvSimpleShapeGLSL" ) + QDir::separator() + QString( "gigaspace_frag.glsl" );
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

	// Initialize shader program
	_shaderProgram = new GsGraphics::GsShaderProgram();
	statusOK = _shaderProgram->addShader( GsGraphics::GsShaderProgram::eVertexShader, vertexShaderFilename.toStdString() );
	assert( statusOK );
	statusOK = _shaderProgram->addShader( GsGraphics::GsShaderProgram::eFragmentShader, fragmentShaderFilename.toStdString() );
	assert( statusOK );
	statusOK = _shaderProgram->link();
	assert( statusOK );

	// Shader program initialization
	// - store/cache uniforms and attributes locations
	// - set uniforms values
	statusOK = initializeShaderProgramUniforms();
	assert( statusOK );
		
	return statusOK;
}

/******************************************************************************
 * Finalize shader program
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
template< typename TDataStructureType, typename TVolumeTreeCacheType >
bool RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::finalizeShaderProgram()
{
	bool statusOK = true;

	delete _shaderProgram;
	_shaderProgram = NULL;

	return statusOK;
}

/******************************************************************************
 * Initialize shader program
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
template< typename TDataStructureType, typename TVolumeTreeCacheType >
bool RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::initializeShaderProgramUniforms()
{
	bool statusOK = true;

	// Store locations of uniform variables
	// - TODO : check errors
	_nodeBufferLoc = glGetUniformLocation( _shaderProgram->_program, "uNodePoolChildArray" );
	if ( _nodeBufferLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_dataBufferLoc = glGetUniformLocation( _shaderProgram->_program, "uNodePoolDataArray" );
	if ( _dataBufferLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_requestBufferLoc = glGetUniformLocation( _shaderProgram->_program, "uUpdateBufferArray" );
	if ( _requestBufferLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_nodeTimestampBufferLoc = glGetUniformLocation( _shaderProgram->_program, "uNodeTimeStampArray" );
	if ( _nodeTimestampBufferLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_brickTimestampBufferLoc = glGetUniformLocation( _shaderProgram->_program, "uBrickTimeStampArray" );
	if ( _brickTimestampBufferLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_currentTimeLoc = glGetUniformLocation( _shaderProgram->_program, "uCurrentTime" );
	if ( _currentTimeLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();

	//-------------------------------------------------------------------------
	_viewPosLoc = glGetUniformLocation( _shaderProgram->_program, "uViewPos" );
	if ( _viewPosLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_viewPlaneLoc = glGetUniformLocation( _shaderProgram->_program, "uViewPlane" );
	if ( _viewPlaneLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_viewAxisXLoc = glGetUniformLocation( _shaderProgram->_program, "uViewAxisX" );
	if ( _viewAxisXLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_viewAxisYLoc = glGetUniformLocation( _shaderProgram->_program, "uViewAxisY" );
	if ( _viewAxisYLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_pixelSizeLoc = glGetUniformLocation( _shaderProgram->_program, "uPixelSize" );
	if ( _pixelSizeLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_frustumNearInvLoc = glGetUniformLocation( _shaderProgram->_program, "uFrustumNearInv" );
	if ( _frustumNearInvLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_coneApertureScaleLoc = glGetUniformLocation( _shaderProgram->_program, "uConeApertureScale" );
	if ( _coneApertureScaleLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_maxNbLoopsLoc = glGetUniformLocation( _shaderProgram->_program, "uMaxNbLoops" );
	if ( _maxNbLoopsLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	
	// Retrieve node pool and brick pool resolution
	//_nodePoolResInvLoc = glGetUniformLocation( _shaderProgram->_program, "nodePoolResInv" );
	_brickPoolResInvLoc = glGetUniformLocation( _shaderProgram->_program, "uBrickPoolResInv" );
	if ( _brickPoolResInvLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	// Retrieve node cache and brick cache size
	//_nodeCacheSizeLoc = glGetUniformLocation( _shaderProgram->_program, "nodeCacheSize" );
	_brickCacheSizeLoc = glGetUniformLocation( _shaderProgram->_program, "uBrickCacheSize" );
	if ( _brickCacheSizeLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	// Data Pool
	// - channel 0
	_dataPool_Channel_0_Loc = glGetUniformLocation( _shaderProgram->_program, "uDataPoolChannel0" );
	if ( _dataPool_Channel_0_Loc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	// - channel 1
	_dataPool_Channel_1_Loc = glGetUniformLocation( _shaderProgram->_program, "uDataPoolChannel1" );
	if ( _dataPool_Channel_1_Loc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	// Max depth
	_maxDepthLoc = glGetUniformLocation( _shaderProgram->_program, "uMaxDepth" );
	if ( _maxDepthLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}

	// Fullscreen quad position
	_positionLoc = glGetAttribLocation( _shaderProgram->_program, "iPosition" );
	if ( _positionLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}

	// Initialize constant uniform parameters
	// - install program object as part of current rendering state
	_shaderProgram->use();
	// Retrieve node pool and brick pool resolution
	//uint3 nodePoolRes = this->_volumeTree->_nodePool->getChannel( Loki::Int2Type< 0 >() )->getResolution();
	uint3 brickPoolRes = this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 0 >() )->getResolution();
	//glUniform3f( _nodePoolResInvLoc, 1.0f / (GLfloat)nodePoolRes.x, 1.0f / (GLfloat)nodePoolRes.y, 1.0f / (GLfloat)nodePoolRes.z );
	glUniform3f( _brickPoolResInvLoc, 1.0f / (GLfloat)brickPoolRes.x, 1.0f / (GLfloat)brickPoolRes.y, 1.0f / (GLfloat)brickPoolRes.z );
	// Retrieve node cache and brick cache size
	//uint3 nodeCacheSize = _nodeTimestampBuffer->getResolution();
	uint3 brickCacheSize = _brickTimestampBuffer->getResolution();
	//glUniform3ui( _nodeCacheSizeLoc, nodeCacheSize.x, nodeCacheSize.y, nodeCacheSize.z );
	glUniform3ui( _brickCacheSizeLoc, brickCacheSize.x, brickCacheSize.y, brickCacheSize.z );

	glProgramUniform1uiEXT( _shaderProgram->_program, _maxNbLoopsLoc, _maxNbLoops );
	glProgramUniform1fEXT( _shaderProgram->_program, _coneApertureScaleLoc, _coneApertureScale );
	
	// Data Production Management
	// - buffer of requests
	glUniform1i( _requestBufferLoc, 0 );
	// - nodes time stamps buffer
	glUniform1i( _nodeTimestampBufferLoc, 1 );
	// - bricks time stamps buffer
	glUniform1i( _brickTimestampBufferLoc, 2 );

	// Node Pool
	// - child array
	glUniform1i( _nodeBufferLoc, 3 );
	// - data array
	glUniform1i( _dataBufferLoc, 4 );

	// Update time
	glUniform1ui( _currentTimeLoc, this->_currentTime );

	// Data Pool
	// - channel 0
	glUniform1i( _dataPool_Channel_0_Loc, 0 );
	// - channel 1
	glUniform1i( _dataPool_Channel_1_Loc, 1 );

	// Max depth
	glUniform1ui( _maxDepthLoc, this->_volumeTree->getMaxDepth() );

	// Uninstall program object as part of current rendering state
	GsGraphics::GsShaderProgram::unuse();

	return statusOK;
}
