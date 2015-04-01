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

// Cuda SDK
#include <helper_cuda.h>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 *
 * @param pDataStructure data structure to render
 * @param pDataProductionManager data production manager that will handle requests emitted during rendering
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManagerType, typename TShaderType >
RendererCUDA< TDataStructureType, TDataProductionManagerType, TShaderType >::
RendererCUDA( TDataStructureType* pDataStructure, TDataProductionManagerType* pDataProductionManager )
:	GvRendering::GvRendererCUDA< TDataStructureType, TDataProductionManagerType, TShaderType >( pDataStructure, pDataProductionManager )
,	_rayMinResource( NULL )
,	_rayMaxResource( NULL )
,	_proxyGeometry( NULL )
{
	GV_CHECK_CUDA_ERROR( "RendererCUDA::RenderesrCUDA prestart" );
	
	// Configure texture samplers
	rayMinTex.normalized = 0;
	rayMinTex.addressMode[ 0 ] = cudaAddressModeClamp;
	rayMinTex.addressMode[ 1 ] = cudaAddressModeClamp;
	rayMinTex.addressMode[ 2 ] = cudaAddressModeClamp;
	rayMinTex.filterMode = cudaFilterModePoint;

	rayMaxTex.normalized = 0;
	rayMaxTex.addressMode[ 0 ] = cudaAddressModeClamp;
	rayMaxTex.addressMode[ 1 ] = cudaAddressModeClamp;
	rayMaxTex.addressMode[ 2 ] = cudaAddressModeClamp;
	rayMaxTex.filterMode = cudaFilterModePoint;
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManagerType, typename TShaderType >
RendererCUDA< TDataStructureType, TDataProductionManagerType, TShaderType >::~RendererCUDA()
{
}

/******************************************************************************
 * This function is the specific implementation method called
 * by the parent GvIRenderer::render() method during rendering.
 *
 * @param pModelMatrix the current model matrix
 * @param pViewMatrix the current view matrix
 * @param pProjectionMatrix the current projection matrix
 * @param pViewport the viewport configuration* ...
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManagerType, typename TShaderType >
void RendererCUDA< TDataStructureType, TDataProductionManagerType, TShaderType >
::render( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	assert( _rayMinResource != NULL && _rayMaxResource != NULL );

	// Initialize frame objects
	uint2 frameSize = make_uint2( pViewport.z - pViewport.x, pViewport.w - pViewport.y );
	this->initFrameObjects( frameSize );

	// Map graphics resources
	CUDAPM_START_EVENT( vsrender_pre_frame_mapbuffers );
	this->_graphicsInteroperabiltyHandler->mapResources();
	cudaArray* rayMinArray;
	GV_CUDA_SAFE_CALL( cudaGraphicsMapResources( 1, &_rayMinResource, 0 ) );
	GV_CUDA_SAFE_CALL( cudaGraphicsSubResourceGetMappedArray( &rayMinArray, _rayMinResource, 0, 0 ) );
	cudaArray* rayMaxArray;
	GV_CUDA_SAFE_CALL( cudaGraphicsMapResources( 1, &_rayMaxResource, 0 ) );
	GV_CUDA_SAFE_CALL( cudaGraphicsSubResourceGetMappedArray( &rayMaxArray, _rayMaxResource, 0, 0 ) );
	this->bindGraphicsResources();
	GV_CUDA_SAFE_CALL( cudaBindTextureToArray( rayMinTex, rayMinArray ) );
	GV_CUDA_SAFE_CALL( cudaBindTextureToArray( rayMaxTex, rayMaxArray ) );
	CUDAPM_STOP_EVENT( vsrender_pre_frame_mapbuffers );

	// Start rendering
	doRender( pModelMatrix, pViewMatrix, pProjectionMatrix, pViewport );

	// Unmap graphics resources
	CUDAPM_START_EVENT( vsrender_post_frame_unmapbuffers );
	GV_CUDA_SAFE_CALL( cudaUnbindTexture( rayMinTex ) );
	GV_CUDA_SAFE_CALL( cudaUnbindTexture( rayMaxTex ) );
	this->unbindGraphicsResources();
	GV_CUDA_SAFE_CALL( cudaGraphicsUnmapResources( 1, &_rayMinResource, 0 ) );
	GV_CUDA_SAFE_CALL( cudaGraphicsUnmapResources( 1, &_rayMaxResource, 0 ) );
	this->_graphicsInteroperabiltyHandler->unmapResources();
	CUDAPM_STOP_EVENT( vsrender_post_frame_unmapbuffers );
}

/******************************************************************************
 * Start the rendering process.
 *
 * @param pModelMatrix the current model matrix
 * @param pViewMatrix the current view matrix
 * @param pProjectionMatrix the current projection matrix
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManagerType, typename TShaderType >
void RendererCUDA< TDataStructureType, TDataProductionManagerType, TShaderType >
::doRender( const float4x4& modelMatrix, const float4x4& viewMatrix, const float4x4& projMatrix, const int4& pViewport )
{
	// Create a render view context to access to useful variables during (view matrix, model matrix, etc...)
	GvRendering::GvRendererContext viewContext;

	// Extract zNear, zFar as well as the distance in view space
	// from the center of the screen to each side of the screen.
	float fleft   = projMatrix._array[ 14 ] * ( projMatrix._array[ 8 ] - 1.0f ) / ( projMatrix._array[ 0 ] * ( projMatrix._array[ 10 ] - 1.0f ) );
	float fright  = projMatrix._array[ 14 ] * ( projMatrix._array[ 8 ] + 1.0f ) / ( projMatrix._array[ 0 ] * ( projMatrix._array[ 10 ] - 1.0f ) );
	float ftop    = projMatrix._array[ 14 ] * ( projMatrix._array[ 9 ] + 1.0f ) / ( projMatrix._array[ 5 ] * ( projMatrix._array[ 10 ] - 1.0f ) );
	float fbottom = projMatrix._array[ 14 ] * ( projMatrix._array[ 9 ] - 1.0f ) / ( projMatrix._array[ 5 ] * ( projMatrix._array[ 10 ] - 1.0f ) );
	float fnear   = projMatrix._array[ 14 ] / ( projMatrix._array[ 10 ] - 1.0f );
	float ffar    = projMatrix._array[ 14 ] / ( projMatrix._array[ 10 ] + 1.0f );
	
	float2 viewSurfaceVS[ 2 ];
	viewSurfaceVS[ 0 ] = make_float2( fleft, fbottom );
	viewSurfaceVS[ 1 ] = make_float2( fright, ftop );

	float3 viewPlane[ 2 ];
	viewPlane[ 0 ] = make_float3( fleft, fbottom, fnear );
	viewPlane[ 1 ] = make_float3( fright, ftop, fnear );
	// float3 viewSize = ( viewPlane[ 1 ] - viewPlane[ 0 ] );

	// Projected 2D Bounding Box of the GigaVoxels 3D BBox.
	// It holds its bottom left corner and its size.
	// ( bottomLeftCorner.x, bottomLeftCorner.y, frameSize.x, frameSize.y )
	viewContext._projectedBBox = this->_projectedBBox;
	
	float2 viewSurfaceVS_Size = viewSurfaceVS[ 1 ] - viewSurfaceVS[ 0 ];
	
	// Transform matrices
	float4x4 invModelMatrixT = transpose( inverse( modelMatrix ) );
	float4x4 invViewMatrixT = transpose( inverse( viewMatrix ) );

	float4x4 modelMatrixT = transpose( modelMatrix );
	float4x4 viewMatrixT = transpose( viewMatrix );

	//float4x4 viewMatrix=(inverse(invViewMatrix));

	viewContext.invViewMatrix = invViewMatrixT;
	viewContext.viewMatrix = viewMatrixT;
	viewContext.invModelMatrix = invModelMatrixT;
	viewContext.modelMatrix = modelMatrixT;

	// Store frustum parameters
	viewContext.frustumNear = fnear;
	viewContext.frustumNearINV = 1.0f / fnear;
	viewContext.frustumFar = ffar;
	viewContext.frustumRight = fright;
	viewContext.frustumTop = ftop;
	viewContext.frustumC = projMatrix._array[ 10 ]; // - ( ffar + fnear ) / ( ffar - fnear );
	viewContext.frustumD = projMatrix._array[ 14 ]; // ( -2.0f * ffar * fnear ) / ( ffar - fnear );

	// Graphics resource settings
	viewContext._clearColor = this->_clearColor;
	viewContext._clearDepth = this->_clearDepth;
	/*bindGraphicsResources();*/
	this->_graphicsInteroperabiltyHandler->setRendererContextInfo( viewContext );
	// TO DO : add texture offsets !!!!!!!!!!!!
	// ...
	
	// WORLD
	float3 viewPlanePosWP = mul( viewContext.invViewMatrix, make_float3( fleft, fbottom, -fnear ) );
	viewContext.viewCenterWP = mul( viewContext.invViewMatrix, make_float3( 0.0f, 0.0f, 0.0f ) );
	viewContext.viewPlaneDirWP = viewPlanePosWP - viewContext.viewCenterWP;
	// TREE
	float3 viewPlanePosTP = mul( viewContext.invModelMatrix, viewPlanePosWP );
	viewContext.viewCenterTP = mul( viewContext.invModelMatrix, viewContext.viewCenterWP );
	viewContext.viewPlaneDirTP = viewPlanePosTP - viewContext.viewCenterTP;

	// Resolution dependant stuff
	viewContext.frameSize = this->_frameSize;
	float2 pixelSize = viewSurfaceVS_Size / make_float2( static_cast< float >( viewContext.frameSize.x ), static_cast< float >( viewContext.frameSize.y ) );
	viewContext.pixelSize = pixelSize;
	/*viewContext.viewPlaneXAxisWP = ( mul( viewContext.invViewMatrix, make_float3( fright, fbottom, -fnear ) ) - viewPlanePosWP ) / static_cast< float >( viewContext.frameSize.x );
	viewContext.viewPlaneYAxisWP = ( mul( viewContext.invViewMatrix, make_float3( fleft, ftop, -fnear ) ) - viewPlanePosWP ) / static_cast< float >( viewContext.frameSize.y );*/
	// WORLD
	viewContext.viewPlaneXAxisWP = mul( viewContext.invViewMatrix, make_float3( fright, fbottom, -fnear ) );
	viewContext.viewPlaneYAxisWP = mul( viewContext.invViewMatrix, make_float3( fleft, ftop, -fnear ) );
	// TREE
	viewContext.viewPlaneXAxisTP = mul( viewContext.invModelMatrix, viewContext.viewPlaneXAxisWP );
	viewContext.viewPlaneYAxisTP = mul( viewContext.invModelMatrix, viewContext.viewPlaneYAxisWP );
	// WORLD
	viewContext.viewPlaneXAxisWP = ( viewContext.viewPlaneXAxisWP - viewPlanePosWP ) / static_cast< float >( viewContext.frameSize.x );
	viewContext.viewPlaneYAxisWP = ( viewContext.viewPlaneYAxisWP - viewPlanePosWP ) / static_cast< float >( viewContext.frameSize.y );
	// TREE
	viewContext.viewPlaneXAxisTP = ( viewContext.viewPlaneXAxisTP - viewPlanePosTP ) / static_cast< float >( viewContext.frameSize.x );
	viewContext.viewPlaneYAxisTP = ( viewContext.viewPlaneYAxisTP - viewPlanePosTP ) / static_cast< float >( viewContext.frameSize.y );

	CUDAPM_START_EVENT( vsrender_copyconsts_frame );

		int maxVolTreeDepth = this->_volumeTree->getMaxDepth();

		// TEST, OPTIM : early preRender pass
		//this->_volumeTreeCache->preRenderPass();

		GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_renderViewContext, &viewContext, sizeof( viewContext ) ) );
		GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_currentTime, (&this->_currentTime), sizeof( this->_currentTime ), 0, cudaMemcpyHostToDevice) );
		GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_maxVolTreeDepth, &maxVolTreeDepth, sizeof( maxVolTreeDepth ), 0, cudaMemcpyHostToDevice) );
	
		// TO DO : move this "Performance monitor" piece of code in another place
	#ifdef USE_CUDAPERFMON
		// Performance monitor
		if ( GvPerfMon::CUDAPerfMon::get()._requestResize ) 
		{
			// Update device memory
			GvCore::Array3DGPULinear< GvCore::uint64 >* my_d_timersArray = GvPerfMon::CUDAPerfMon::get().getKernelTimerArray();
			GvCore::Array3DKernelLinear< GvCore::uint64 > h_timersArray = my_d_timersArray->getDeviceArray();
			GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( GvPerfMon::k_timersArray, &h_timersArray, sizeof( h_timersArray ), 0, cudaMemcpyHostToDevice ) );
		
			// Update device memory
			uchar* my_d_timersMask = GvPerfMon::CUDAPerfMon::get().getKernelTimerMask();
			GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( GvPerfMon::k_timersMask, &my_d_timersMask, sizeof( my_d_timersMask ), 0, cudaMemcpyHostToDevice ) );

			// Update the performnace monitor's state
			GvPerfMon::CUDAPerfMon::get()._requestResize = false;
		}
	#endif
	
	CUDAPM_STOP_EVENT( vsrender_copyconsts_frame );

	//float voxelSizeMultiplier = 1.0f;
	//GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( GvRendering::k_voxelSizeMultiplier, (&voxelSizeMultiplier), sizeof( voxelSizeMultiplier ), 0, cudaMemcpyHostToDevice ) );
	
	CUDAPM_START_EVENT_GPU( gv_rendering );

	dim3 blockSize( RenderBlockResolution::x, RenderBlockResolution::y, 1 );
	dim3 gridSize( iDivUp( this->_frameSize.x, RenderBlockResolution::x ), iDivUp( this->_frameSize.y, RenderBlockResolution::y ), 1 );
	// FUTUR optimization
	//
	//dim3 gridSize( iDivUp( /*projectedBBoxSize*/_projectedBBox.z, RenderBlockResolution::x ), iDivUp( /*projectedBBoxSize*/_projectedBBox.w, RenderBlockResolution::y ), 1 );
	
	if ( this->_dynamicUpdate )
	{
		if ( this->_hasPriorityOnBricks )
		{
			// Priority on brick is set to TRUE to force loading data at low resolution first
			RenderKernel< RenderBlockResolution, false, true, TShaderType >
							<<< gridSize, blockSize >>>(
								this->_volumeTree->volumeTreeKernel,
								this->_volumeTreeCache->getKernelObject() );
		}
		else
		{
			RenderKernel< RenderBlockResolution, false, false, TShaderType >
							<<< gridSize, blockSize >>>(
								this->_volumeTree->volumeTreeKernel,
								this->_volumeTreeCache->getKernelObject() );
		}
	}
	else
	{
		RenderKernel< RenderBlockResolution, false, false, TShaderType >
						<<< gridSize, blockSize >>>(
								this->_volumeTree->volumeTreeKernel,
								this->_volumeTreeCache->getKernelObject() );
	}

	GV_CHECK_CUDA_ERROR( "RenderKernelSimple" );

	CUDAPM_STOP_EVENT_GPU( gv_rendering );
}

/******************************************************************************
 * Register the graphics resources associated to proxy geometry
 *
 * @return a flag to tell wheter or not it succeeds
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManagerType, typename TShaderType >
bool RendererCUDA< TDataStructureType, TDataProductionManagerType, TShaderType >
::registerProxyGeometryGraphicsResources()
{
	assert( _proxyGeometry != NULL );

	GV_CUDA_SAFE_CALL( cudaGraphicsGLRegisterImage( &_rayMinResource, _proxyGeometry->_depthMinTex, GL_TEXTURE_RECTANGLE, cudaGraphicsRegisterFlagsReadOnly ) );
	GV_CUDA_SAFE_CALL( cudaGraphicsGLRegisterImage( &_rayMaxResource, _proxyGeometry->_depthMaxTex, GL_TEXTURE_RECTANGLE, cudaGraphicsRegisterFlagsReadOnly ) );
}
	
/******************************************************************************
 * Unregister the graphics resources associated to proxy geometry
 *
 * @return a flag to tell wheter or not it succeeds
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManagerType, typename TShaderType >
bool RendererCUDA< TDataStructureType, TDataProductionManagerType, TShaderType >
::unregisterProxyGeometryGraphicsResources()
{
	if ( _rayMinResource )
	{
		GV_CUDA_SAFE_CALL( cudaGraphicsUnregisterResource( _rayMinResource ) );
	}

	if ( _rayMaxResource )
	{
		GV_CUDA_SAFE_CALL( cudaGraphicsUnregisterResource( _rayMaxResource ) );
	}
}

/******************************************************************************
 * Get the associated proxy geometry
 *
 * @return the proxy geometry
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManagerType, typename TShaderType >
const ProxyGeometry* RendererCUDA< TDataStructureType, TDataProductionManagerType, TShaderType >
::getProxyGeometry() const
{
	return _proxyGeometry;
}

/******************************************************************************
 * Get the associated proxy geometry
 *
 * @return the proxy geometry
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManagerType, typename TShaderType >
ProxyGeometry* RendererCUDA< TDataStructureType, TDataProductionManagerType, TShaderType >
::editProxyGeometry()
{
	return _proxyGeometry;
}

/******************************************************************************
 * Set the associated proxy geometry
 *
 * @param pProxyGeometry the proxy geometry
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManagerType, typename TShaderType >
void RendererCUDA< TDataStructureType, TDataProductionManagerType, TShaderType >
::setProxyGeometry( ProxyGeometry* pProxyGeometry )
{
	_proxyGeometry = pProxyGeometry;
}
