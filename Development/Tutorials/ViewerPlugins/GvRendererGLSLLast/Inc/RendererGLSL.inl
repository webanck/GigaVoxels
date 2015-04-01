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
//#include <GsGraphics/GsShaderProgram.h>

#include "ProxyGeometry.h"
#include "Mesh.h"

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>

// STL
#include <iostream>

//--------------------------------------------------------------
// CUDA - NSight
//#define GV_NSIGHT_PROLIFING

#ifdef GV_NSIGHT_PROLIFING
	#include "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\nvToolsExt\\include\\nvToolsExtCuda.h"
#endif
//--------------------------------------------------------------

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
,	_proxyGeometry( NULL )
,	_hasProxyGeometry( false )
,	_isProxyGeometryVisible( false )
{
	std::cout << "TEST" << std::endl;

	bool statusOK = false;

	// Retrieve useful GigaVoxels arrays
	_nodeBuffer = this->_volumeTree->_childArray;
	_dataBuffer = this->_volumeTree->_dataArray;
	_requestBuffer = this->_volumeTreeCache->getUpdateBuffer();
	_nodeTimestampBuffer = this->_volumeTreeCache->getNodesCacheManager()->getTimeStampList();
	_brickTimestampBuffer = this->_volumeTreeCache->getBricksCacheManager()->getTimeStampList();

	// Unmap GigaVoxels graphics resources from CUDA environment in order to be used by OpenGL
	_nodeBuffer->unmapResource();
	_dataBuffer->unmapResource();
	_requestBuffer->unmapResource();
	_nodeTimestampBuffer->unmapResource();
	_brickTimestampBuffer->unmapResource();

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

	// Map GigaVoxels graphics resources in order to be used by CUDA environment
	_nodeBuffer->mapResource();
	_dataBuffer->mapResource();
	_requestBuffer->mapResource();
	_nodeTimestampBuffer->mapResource();
	_brickTimestampBuffer->mapResource();

	// Initialize shader program
	statusOK = initializeShaderProgram();
	assert( statusOK );

	// Check for OpenGL error(s)
	GV_CHECK_GL_ERROR();

		std::cout << "TEST 222" << std::endl;

	/////////////////////////////////////////////////////////////////////
	//_graphicsResources = new cudaGraphicsResource[ 6 ];
	/*_graphicsResources[ 0 ] = _requestBuffer->_bufferResource;
	_graphicsResources[ 1 ] = _nodeTimestampBuffer->_bufferResource;
	_graphicsResources[ 2 ] = _brickTimestampBuffer->_bufferResource;
	_graphicsResources[ 3 ] = _nodeBuffer->_bufferResource;
	_graphicsResources[ 4 ] = _dataBuffer->_bufferResource;
	_graphicsResources[ 5 ] = this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 0 >() )->_bufferResource;*/
	/////////////////////////////////////////////////////////////////////

	// Settings
	_coneApertureScale = 1.333f;
	_maxNbLoops = 200;

	// Initialize proxy geometry
	_proxyGeometry = new ProxyGeometry();
	const QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	const QString meshRepository = dataRepository + QDir::separator() + QString( "3DModels" ) + QDir::separator() + QString( "stanford_bunny" );
	const QString meshFilename = meshRepository + QDir::separator() + QString( "bunny.obj" );
	_proxyGeometry->set3DModelFilename( meshFilename.toStdString() );
	/*bool*/ statusOK = _proxyGeometry->initialize();
	assert( statusOK );
	// Register proxy geometry
	//_pipeline->editRenderer()->setProxyGeometry( _proxyGeometry );

	// Reset proxy geometry resources
	//_pipeline->editRenderer()->unregisterProxyGeometryGraphicsResources();
	_proxyGeometry->setBufferSize( /*pWidth*/512, /*pHeight*/512 );
	//_pipeline->editRenderer()->registerProxyGeometryGraphicsResources();

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

	// Finalize proxy geometry
	finalizeProxyGeometry();

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
	if ( _hasProxyGeometry )
	{
		// Generate depth maps from mesh
		// - min depth from closest faces
		// - max depth from farthest faces
		// => we get a "shell" from the mesh
		//------------------------------------------------------
		// extract view transformations
		float4x4 viewMatrix;
		float4x4 projectionMatrix;
		glGetFloatv( GL_MODELVIEW_MATRIX, viewMatrix._array );
		glGetFloatv( GL_PROJECTION_MATRIX, projectionMatrix._array );
		// extract viewport
		GLint params[ 4 ];
		glGetIntegerv( GL_VIEWPORT, params );
		int4 viewport = make_int4( params[ 0 ], params[ 1 ], params[ 2 ], params[ 3 ] );
		//------------------------------------------------------
		glEnable( GL_DEPTH_TEST );
		glDisable( GL_CULL_FACE );
		float4x4 proxyGeometryModelViewMatrix;
		glPushMatrix();
		// Add Model transformation to lie between -0.5 and 0.5
		const IMesh* mesh = _proxyGeometry->getMesh();
		const float minX = mesh->_minX;
		const float minY = mesh->_minY;
		const float minZ = mesh->_minZ;
		const float maxX = mesh->_maxX;
		const float maxY = mesh->_maxY;
		const float maxZ = mesh->_maxZ;
		const float uniformScale = 0.99f / std::max( std::max( maxX - minX, maxY - minY ), maxZ - minZ );
		glScalef( uniformScale, uniformScale, uniformScale );
		const float3 translate = make_float3( - ( minX + maxX ) * 0.5f, - ( minY + maxY ) * 0.5f, - ( minZ + maxZ ) * 0.5f );
		glTranslatef( translate.x, translate.y, translate.z );
		glGetFloatv( GL_MODELVIEW_MATRIX, proxyGeometryModelViewMatrix._array );
		// TO DO : add a screen based criteria to stop division => ...
		_proxyGeometry->render( proxyGeometryModelViewMatrix, projectionMatrix, viewport );
		// Display proxy geometry
		if ( _isProxyGeometryVisible )
		{
			const_cast< IMesh* >( mesh )->render( proxyGeometryModelViewMatrix, projectionMatrix, viewport );
		}
		glPopMatrix();
	}
	
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

	float2 viewSurfaceVS[ 2 ];
	viewSurfaceVS[ 0 ] = make_float2( fleft, fbottom );
	viewSurfaceVS[ 1 ] = make_float2( fright, ftop );
	float2 viewSurfaceVS_Size = viewSurfaceVS[ 1 ] - viewSurfaceVS[ 0 ];
	
	// transfor matrices
	float4x4 invprojectionMatrixT = transpose( inverse( pProjectionMatrix ) );
	float4x4 invViewMatrixT = transpose( inverse( pViewMatrix ) );

	float4x4 projectionMatrixT=transpose(pProjectionMatrix);
	float4x4 viewMatrixT=transpose(pViewMatrix);

	CUDAPM_START_EVENT(vsrender_copyconsts_frame);

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

	// Copy data to CUDA memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_renderViewContext, &viewContext, sizeof( viewContext ) ) );
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_currentTime, &(this->_currentTime), sizeof( this->_currentTime ), 0, cudaMemcpyHostToDevice ) );
	
	// TEST -------------------------------------
	//GV_CUDA_SAFE_CALL( cudaDeviceSynchronize() );
	// TEST -------------------------------------

	CUDAPM_STOP_EVENT( vsrender_copyconsts_frame );

	///////////// TEST
	_shaderProgram->use();
	///////////// TEST

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
	//glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA ); // to avoid redundant call

	// Installs program object as part of current rendering state
	//_shaderProgram->use();

	// Unmap GigaVoxels graphics resources from CUDA environment in order to be used by OpenGL
	// non-overlapping range
	//--------------------------------------------------------------
#ifdef GV_NSIGHT_PROLIFING
	nvtxRangeId_t idUnmapResources = nvtxRangeStartA( "UNMAP resources" );
#endif
	//--------------------------------------------------------------
	_requestBuffer->unmapResource();
	_nodeTimestampBuffer->unmapResource();
	_brickTimestampBuffer->unmapResource();
	_nodeBuffer->unmapResource();
	_dataBuffer->unmapResource();
	this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 0 >() )->unmapResource();
#ifdef GV_NSIGHT_PROLIFING
	// TEST -------------------------------------
	GV_CUDA_SAFE_CALL( cudaDeviceSynchronize() );
	// TEST -------------------------------------
	//--------------------------------------------------------------
	nvtxRangeEnd( idUnmapResources );
#endif
	//--------------------------------------------------------------

	/////////////////////////////////////////////////////////////////////
	//
	//C:\Program Files\NVIDIA GPU Computing Toolkit\nvToolsExt\lib\x64nvToolsExt64_1.lib
	//
#ifdef GV_NSIGHT_PROLIFING
	nvtxRangeId_t idTestMapResources = nvtxRangeStartA( "MAP test resources" );
	_requestBuffer->mapResource();
	_nodeTimestampBuffer->mapResource();
	_brickTimestampBuffer->mapResource();
	_nodeBuffer->mapResource();
	_dataBuffer->mapResource();
	this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 0 >() )->mapResource();
	GV_CUDA_SAFE_CALL( cudaDeviceSynchronize() );
	nvtxRangeEnd( idTestMapResources );
	//--------------------------------------------------------------
	nvtxRangeId_t idEnhancedUnmapResources = nvtxRangeStartA( "enhanced UNMAP resources" );
#endif
	//GV_CUDA_SAFE_CALL( cudaGraphicsUnmapResources( 6, &_graphicsResources[ 0 ], 0 ) );
#ifdef GV_NSIGHT_PROLIFING
	GV_CUDA_SAFE_CALL( cudaDeviceSynchronize() );
	nvtxRangeEnd( idEnhancedUnmapResources );
#endif
	/////////////////////////////////////////////////////////////////////

#ifdef GV_NSIGHT_PROLIFING
	nvtxRangeId_t idSetGLUniforms = nvtxRangeStartA( "Set GL uniforms" );
#endif

	// Note :
	// glBindImageTextureEXT() command binds a single level of a texture to an image unit
	// for the purpose of reading and writing it from shaders.
	//
	// NOTE : requires OpenGL 4.2
	//
	// Specification :
	// void glBindImageTexture( GLuint  unit,  GLuint  texture,  GLint  level,  GLboolean  layered,  GLint  layer,  GLenum  access,  GLenum  format );

	// Data Production Management
	//
	// - buffer of requests
	glBindImageTextureEXT( 0, _updateBufferTBO, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI );
	//glUniform1i( _requestBufferLoc, 0 );
	// - nodes time stamps buffer
	glBindImageTextureEXT( 1, _nodeTimeStampTBO, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI );
	//glUniform1i( _nodeTimestampBufferLoc, 1 );
	// - bricks time stamps buffer
	glBindImageTextureEXT( 2, _brickTimeStampTBO, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI );
	//glUniform1i( _brickTimestampBufferLoc, 2 );

	// Node Pool
	//
	// - child array
	glBindImageTextureEXT( 3, _childArrayTBO, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI );
	//glUniform1i( _nodeBufferLoc, 3 );
	// - data array
	glBindImageTextureEXT( 4, _dataArrayTBO, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI );
	//glUniform1i( _dataBufferLoc, 4 );

	// Update time
	glUniform1ui( _currentTimeLoc, this->_currentTime );

	// Data Pool
	glUniform1i( _dataPoolLoc, 0 );

	// Max depth
	glUniform1ui( _maxDepthLoc, this->_volumeTree->getMaxDepth() );

	// Check for OpenGL error(s)
	GV_CHECK_GL_ERROR();

	// Bind user data as 3D texture for rendering
	//
	// Content of one voxel has been defined as :
	// typedef Loki::TL::MakeTypelist< uchar4 >::Result DataType;
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_3D, this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 0 >() )->getBufferName() );

	GLuint vploc = glGetAttribLocation( _shaderProgram->_program, "iPosition" );

	// Proxy geometry
	if ( _hasProxyGeometry )
	{
		GLint uProxyGeometryFrontFacesLocation = glGetUniformLocation( _shaderProgram->_program, "uProxyGeometryFrontFaces" );
		if ( uProxyGeometryFrontFacesLocation < 0 )
		{
			std::cout << "uProxyGeometryFrontFacesLocation" << std::endl;
		}
		glUniform1i( uProxyGeometryFrontFacesLocation, 1 );
		GLint uProxyGeometryBackFacesLocation = glGetUniformLocation( _shaderProgram->_program, "uProxyGeometryBackFaces" );
		if ( uProxyGeometryBackFacesLocation < 0 )
		{
			std::cout << "uProxyGeometryBackFacesLocation" << std::endl;
		}
		glUniform1i( uProxyGeometryBackFacesLocation, 2 );
		glActiveTexture( GL_TEXTURE1 );
		glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _proxyGeometry->_depthMinTex );
		glActiveTexture( GL_TEXTURE2 );
		glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _proxyGeometry->_depthMaxTex );
	}
	
	// TEST -------------------------------------
#ifdef GV_NSIGHT_PROLIFING
	glFlush();
	glFinish();
#endif
	// TEST -------------------------------------

#ifdef GV_NSIGHT_PROLIFING
	nvtxRangeEnd( idSetGLUniforms );
#endif

#ifdef GV_NSIGHT_PROLIFING
	nvtxRangeId_t idGLRender = nvtxRangeStartA( "GL render" );
#endif

	// Draw a fullscreen quad
	// - TODO : use programmable shader program
	glBegin( GL_QUADS );
	glVertexAttrib2f( vploc, -1.0f, -1.0f );
	glVertexAttrib2f( vploc,  1.0f, -1.0f );
	glVertexAttrib2f( vploc,  1.0f,  1.0f );
	glVertexAttrib2f( vploc, -1.0f,  1.0f );
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

	// Check for OpenGL error(s)
	GV_CHECK_GL_ERROR();

	// TEST -------------------------------------
#ifdef GV_NSIGHT_PROLIFING
	glFlush();
	glFinish();
#endif
	// TEST -------------------------------------

#ifdef GV_NSIGHT_PROLIFING
	nvtxRangeEnd( idGLRender );
#endif

	// Map GigaVoxels graphics resources in order to be used by CUDA environment
	//--------------------------------------------------------------
#ifdef GV_NSIGHT_PROLIFING
	nvtxRangeId_t idMapResources = nvtxRangeStartA( "MAP resources" );
#endif
	//--------------------------------------------------------------
	_requestBuffer->mapResource();
	_nodeTimestampBuffer->mapResource();
	_brickTimestampBuffer->mapResource();
	_nodeBuffer->mapResource();
	_dataBuffer->mapResource();
	this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 0 >() )->mapResource();
	// TEST -------------------------------------
#ifdef GV_NSIGHT_PROLIFING
	GV_CUDA_SAFE_CALL( cudaDeviceSynchronize() );
	// TEST -------------------------------------
	//--------------------------------------------------------------
	nvtxRangeEnd( idMapResources );
#endif
	//--------------------------------------------------------------

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
 * Get the flag indicating wheter or not using proxy geometry is activated
 *
 * @return the flag indicating wheter or not using proxy geometry is activated
 ******************************************************************************/
template< typename TDataStructureType, typename TVolumeTreeCacheType >
bool RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::hasProxyGeometry() const
{
	return _hasProxyGeometry;
}

/******************************************************************************
 * Set the flag indicating wheter or not using proxy geometry is activated
 *
 * @param pFlag the flag indicating wheter or not using proxy geometry is activated
 ******************************************************************************/
template< typename TDataStructureType, typename TVolumeTreeCacheType >
void RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::setProxyGeometry( bool pFlag )
{
	_hasProxyGeometry = pFlag;
}

/******************************************************************************
 * Get the flag indicating wheter or not using proxy geometry is activated
 *
 * @return the flag indicating wheter or not using proxy geometry is activated
 ******************************************************************************/
template< typename TDataStructureType, typename TVolumeTreeCacheType >
bool RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::isProxyGeometryVisible() const
{
	return _isProxyGeometryVisible;
}

/******************************************************************************
 * Set the flag indicating wheter or not using proxy geometry is activated
 *
 * @param pFlag the flag indicating wheter or not using proxy geometry is activated
 ******************************************************************************/
template< typename TDataStructureType, typename TVolumeTreeCacheType >
void RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::setProxyGeometryVisible( bool pFlag )
{
	_isProxyGeometryVisible = pFlag;
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
	bool statusOK = false;

	// Retrieve data repository
	QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	
	// Create and link a GLSL shader program
	QString vertexShaderFilename = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "GvRendererGLSL" ) + QDir::separator() + QString( "gigaspace_vert.glsl" );
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

	QString fragmentShaderFilename = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "GvRendererGLSL" ) + QDir::separator() + QString( "gigaspace_frag.glsl" );
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

	std::cout << "TEST 2" << std::endl;

	// Store locations of uniform variables
	// - TODO : check errors
	_nodeBufferLoc = glGetUniformLocation( _shaderProgram->_program, "uNodePoolChildArray" );
	assert( _nodeBufferLoc != -1 );
	if ( _nodeBufferLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_dataBufferLoc = glGetUniformLocation( _shaderProgram->_program, "uNodePoolDataArray" );
	assert( _dataBufferLoc != -1 );
	if ( _dataBufferLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_requestBufferLoc = glGetUniformLocation( _shaderProgram->_program, "uUpdateBufferArray" );
	assert( _requestBufferLoc != -1 );
	if ( _requestBufferLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_nodeTimestampBufferLoc = glGetUniformLocation( _shaderProgram->_program, "uNodeTimeStampArray" );
	assert( _nodeTimestampBufferLoc != -1 );
	if ( _nodeTimestampBufferLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_brickTimestampBufferLoc = glGetUniformLocation( _shaderProgram->_program, "uBrickTimeStampArray" );
	assert( _brickTimestampBufferLoc != -1 );
	if ( _brickTimestampBufferLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_currentTimeLoc = glGetUniformLocation( _shaderProgram->_program, "uCurrentTime" );
	assert( _currentTimeLoc != -1 );
	if ( _currentTimeLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();

	//-------------------------------------------------------------------------
	_viewPosLoc = glGetUniformLocation( _shaderProgram->_program, "uViewPos" );
	assert( _viewPosLoc != -1 );
	if ( _viewPosLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_viewPlaneLoc = glGetUniformLocation( _shaderProgram->_program, "uViewPlane" );
	assert( _viewPlaneLoc != -1 );
	if ( _viewPlaneLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_viewAxisXLoc = glGetUniformLocation( _shaderProgram->_program, "uViewAxisX" );
	assert( _viewAxisXLoc != -1 );
	if ( _viewAxisXLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_viewAxisYLoc = glGetUniformLocation( _shaderProgram->_program, "uViewAxisY" );
	assert( _viewAxisYLoc != -1 );
	if ( _viewAxisYLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_pixelSizeLoc = glGetUniformLocation( _shaderProgram->_program, "uPixelSize" );
	assert( _pixelSizeLoc != -1 );
	if ( _pixelSizeLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_frustumNearInvLoc = glGetUniformLocation( _shaderProgram->_program, "uFrustumNearInv" );
	assert( _frustumNearInvLoc != -1 );
	if ( _frustumNearInvLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_coneApertureScaleLoc = glGetUniformLocation( _shaderProgram->_program, "uConeApertureScale" );
	assert( _coneApertureScaleLoc != -1 );
	if ( _coneApertureScaleLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	_maxNbLoopsLoc = glGetUniformLocation( _shaderProgram->_program, "uMaxNbLoops" );
	assert( _maxNbLoopsLoc != -1 );
	if ( _maxNbLoopsLoc == -1 )
	{
		std::cout << "error" << std::endl;
	}
	GV_CHECK_GL_ERROR();
	//-------------------------------------------------------------------------

	// Retrieve node pool and brick pool resolution
	_nodePoolResInvLoc = glGetUniformLocation( _shaderProgram->_program, "nodePoolResInv" );
	_brickPoolResInvLoc = glGetUniformLocation( _shaderProgram->_program, "uBrickPoolResInv" );
	// Retrieve node cache and brick cache size
	_nodeCacheSizeLoc = glGetUniformLocation( _shaderProgram->_program, "nodeCacheSize" );
	_brickCacheSizeLoc = glGetUniformLocation( _shaderProgram->_program, "uBrickCacheSize" );
	// Data Pool
	_dataPoolLoc = glGetUniformLocation( _shaderProgram->_program, "uDataPool" );
	// Max depth
	_maxDepthLoc = glGetUniformLocation( _shaderProgram->_program, "uMaxDepth" );
	
	// Initialize constant uniform parameters
	// - install program object as part of current rendering state
	_shaderProgram->use();
	//-------------------------------------------------------
	// Retrieve node pool and brick pool resolution
	uint3 nodePoolRes = this->_volumeTree->_nodePool->getChannel( Loki::Int2Type< 0 >() )->getResolution();
	uint3 brickPoolRes = this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 0 >() )->getResolution();
	//glUniform3f( _nodePoolResInvLoc, 1.0f / (GLfloat)nodePoolRes.x, 1.0f / (GLfloat)nodePoolRes.y, 1.0f / (GLfloat)nodePoolRes.z );
	glUniform3f( _brickPoolResInvLoc, 1.0f / (GLfloat)brickPoolRes.x, 1.0f / (GLfloat)brickPoolRes.y, 1.0f / (GLfloat)brickPoolRes.z );
	// Retrieve node cache and brick cache size
	uint3 nodeCacheSize = _nodeTimestampBuffer->getResolution();
	uint3 brickCacheSize = _brickTimestampBuffer->getResolution();
	//glUniform3ui( _nodeCacheSizeLoc, nodeCacheSize.x, nodeCacheSize.y, nodeCacheSize.z );
	glUniform3ui( _brickCacheSizeLoc, brickCacheSize.x, brickCacheSize.y, brickCacheSize.z );
	//-------------------------------------------------------
	// glBindImageTextureEXT() command binds a single level of a texture to an image unit
	// for the purpose of reading and writing it from shaders.
	//
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

	// - uninstall program object as part of current rendering state
	GsGraphics::GsShaderProgram::unuse();

	std::cout << "TEST 3" << std::endl;

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
	bool statusOK = false;

	delete _shaderProgram;
	_shaderProgram = NULL;

	return true;
}

/******************************************************************************
 * Initialize shader program
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
template< typename TDataStructureType, typename TVolumeTreeCacheType >
bool RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::initializeProxyGeometry()
{
	bool statusOK = false;

	// Retrieve data repository
	QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );

	// Initialize proxy geometry
	_proxyGeometry = new ProxyGeometry();
	
	const QString meshRepository = dataRepository + QDir::separator() + QString( "3DModels" ) + QDir::separator() + QString( "stanford_bunny" );
	const QString meshFilename = meshRepository + QDir::separator() + QString( "bunny.obj" );
	_proxyGeometry->set3DModelFilename( meshFilename.toStdString() );
	statusOK = _proxyGeometry->initialize();
	assert( statusOK );
	// Register proxy geometry
	//_pipeline->editRenderer()->setProxyGeometry( _proxyGeometry );

	// Reset proxy geometry resources
	//_pipeline->editRenderer()->unregisterProxyGeometryGraphicsResources();
	_proxyGeometry->setBufferSize( /*pWidth*/512, /*pHeight*/512 );
	//_pipeline->editRenderer()->registerProxyGeometryGraphicsResources();

	return statusOK;
}

/******************************************************************************
 * Finalize proxy geometry
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
template< typename TDataStructureType, typename TVolumeTreeCacheType >
bool RendererGLSL< TDataStructureType, TVolumeTreeCacheType >
::finalizeProxyGeometry()
{
	bool statusOK = false;

	delete _proxyGeometry;
	_proxyGeometry = NULL;

	return statusOK;
}
