/*
 * Copyright (C) 2011 Fabrice Neyret <Fabrice.Neyret@imag.fr>
 * Copyright (C) 2011 Cyril Crassin <Cyril.Crassin@icare3d.org>
 * Copyright (C) 2011 Morgan Armand <morgan.armand@gmail.com>
 *
 * This file is part of Gigavoxels.
 *
 * Gigavoxels is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Gigavoxels is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Gigavoxels.  If not, see <http://www.gnu.org/licenses/>.
 */

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 *
 * @param volumeTree ...
 * @param gpuProd ...
 * @param nodePoolRes ...
 * @param brickPoolRes ...
 ******************************************************************************/
template< typename BvhTreeType, typename BvhTreeCacheType, class ProducerType >
BvhTreeRenderer< BvhTreeType, BvhTreeCacheType, ProducerType >::BvhTreeRenderer(BvhTreeType *bvhTree, BvhTreeCacheType *bvhTreeCache, ProducerType *gpuProd)
{
	_bvhTree			= bvhTree;
	_bvhTreeCache		= bvhTreeCache;
	gpuProducer			= gpuProd;

	_userParam=make_float4(1.0f, 0.0f, 0.0f, 0.0f);
	_frameSize=make_uint2(0,0);

	//Init frame dependant buffers//
	//Deferred lighting infos
	d_rayOutputColor		= 0;
	d_rayOutputNormal		= 0;
	d_inFrameColor			= 0;
	d_inFrameDepth			= 0;
	d_outFrameColor			= 0;
	d_outFrameDepth			= 0;
	d_rayBufferT			= 0;
	d_rayBufferTmin			= 0;
	d_rayBufferMaskedAt		= 0;
	d_rayBufferStackIndex	= 0;
	d_rayBufferStackVals	= 0;

	currentDebugRay			= make_int2(-1, -1);
	debugDisplayTimes		= false;

	_currentTime			= 10;
	_dynamicUpdate			= true;

	//Init everything
	//initFrameObjects(fs);
	///////////////////////////////

	_colorResource = NULL;
	_depthResource = NULL;

	this->cuda_Init();
	this->_bvhTree->cuda_Init();
	this->_bvhTree->initCache( gpuProd->getBVHTrianglesManager() );

	// FIXME
	this->_maxVolTreeDepth = 6;
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename BvhTreeType, typename BvhTreeCacheType, class ProducerType >
BvhTreeRenderer< BvhTreeType, BvhTreeCacheType, ProducerType >::~BvhTreeRenderer()
{
	//Frame dependant buffers
	deleteFrameObjects();
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename BvhTreeType, typename BvhTreeCacheType, class ProducerType >
void BvhTreeRenderer< BvhTreeType, BvhTreeCacheType, ProducerType >::cuda_Init()
{
	cudaDeviceProp deviceProps;
	cudaGetDeviceProperties( &deviceProps, gpuGetMaxGflopsDeviceId() );	// TO DO : handle the case where user could want an other device

	std::cout<<"\n******Device properties******\n";
	std::cout<<"Name: "<<deviceProps.name<<"\n";
	std::cout<<"Compute capability: "<<deviceProps.major<<"."<<deviceProps.minor<<"\n";
	std::cout<<"Compute mode: "<<deviceProps.computeMode<<"\n";
	std::cout<<"Can map host memory: "<<deviceProps.canMapHostMemory<<"\n";
	std::cout<<"Can overlap transfers and kernels: "<<deviceProps.deviceOverlap<<"\n";
	std::cout<<"Kernels timeout: "<<deviceProps.kernelExecTimeoutEnabled<<"\n";
	std::cout<<"Integrated chip: "<<deviceProps.integrated<<"\n";
	std::cout<<"Global memory: "<<deviceProps.totalGlobalMem/1024/1024<<"MB\n";
	std::cout<<"Shared memory: "<<deviceProps.sharedMemPerBlock/1024<<"KB\n";
	std::cout<<"Clock rate: "<<deviceProps.clockRate/1000<<"MHz\n";
	std::cout<<"*****************************\n\n";

	//GV_CHECK_CUDA_ERROR("VoxelSceneRenderer::cuda_Init start");
	//GV_CHECK_CUDA_ERROR("VoxelSceneRenderer::cuda_Init end");
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename BvhTreeType, typename BvhTreeCacheType, class ProducerType >
void BvhTreeRenderer< BvhTreeType, BvhTreeCacheType, ProducerType >::cuda_Destroy()
{
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename BvhTreeType, typename BvhTreeCacheType, class ProducerType >
void BvhTreeRenderer< BvhTreeType, BvhTreeCacheType, ProducerType >::deleteFrameObjects()
{
	// Deferred lighting infos
	if (d_rayOutputColor)
		delete d_rayOutputColor;
	if (d_rayOutputNormal)
		delete d_rayOutputNormal;
	
	if (d_inFrameColor)
		delete d_inFrameColor;
	if (d_inFrameDepth)
		delete d_inFrameDepth;
	if (d_outFrameColor)
		delete d_outFrameColor;
	if (d_outFrameDepth)
		delete d_outFrameDepth;

	if (d_rayBufferT)
		delete d_rayBufferT;
	if (d_rayBufferTmin)
		delete d_rayBufferTmin;
	if (d_rayBufferMaskedAt)
		delete d_rayBufferMaskedAt;

	if (d_rayBufferStackIndex)
		delete d_rayBufferStackIndex;
	if (d_rayBufferStackVals)
		delete d_rayBufferStackVals;
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename BvhTreeType, typename BvhTreeCacheType, class ProducerType >
void BvhTreeRenderer< BvhTreeType, BvhTreeCacheType, ProducerType >::initFrameObjects(const uint2 &fs)
{
	if (_frameSize.x != fs.x || _frameSize.y != fs.y)
	{
		_frameSize = fs;

		// Destruct frame based objects
		deleteFrameObjects();

		d_rayOutputColor		= new GvCore::Array3DGPULinear< uchar4 >(make_uint3(_frameSize.x, _frameSize.y, 1));
		d_rayOutputNormal		= new GvCore::Array3DGPULinear<float4>(make_uint3(_frameSize.x, _frameSize.y, 1));

		d_inFrameColor			= new GvCore::Array3DGPULinear<uchar4>(NULL, make_uint3(_frameSize.x, _frameSize.y, 1));
		d_inFrameDepth			= new GvCore::Array3DGPULinear<float>(NULL, make_uint3(_frameSize.x, _frameSize.y, 1));
		d_outFrameColor			= new GvCore::Array3DGPULinear<uchar4>(NULL, make_uint3(_frameSize.x, _frameSize.y, 1));
		d_outFrameDepth			= new GvCore::Array3DGPULinear<float>(NULL, make_uint3(_frameSize.x, _frameSize.y, 1));

		/////////////////////
		d_rayBufferTmin			= new GvCore::Array3DGPULinear<float>(make_uint3(_frameSize.x * _frameSize.y, 1, 1));			// 1 per ray
		d_rayBufferT			= new GvCore::Array3DGPULinear<float>(make_uint3(_frameSize.x * _frameSize.y, 1, 1));			// 1 per ray
		d_rayBufferMaskedAt		= new GvCore::Array3DGPULinear<int>(make_uint3(_frameSize.x * _frameSize.y, 1, 1));				// 1 per ray

		uint numBlocks= (_frameSize.x / RenderBlockResolution::x) * (_frameSize.y / RenderBlockResolution::y);
		d_rayBufferStackIndex	= new GvCore::Array3DGPULinear<int>(make_uint3(numBlocks, 1, 1));								//1 per tile
		d_rayBufferStackVals	= new GvCore::Array3DGPULinear<uint>(make_uint3(numBlocks * BVH_TRAVERSAL_STACK_SIZE, 1, 1));
	}

	GV_CHECK_CUDA_ERROR("VoxelSceneRenderer::initFrameObjects");
}

/******************************************************************************
 * ...
 *
 * @param modelMatrix ...
 * @param viewMatrix ...
 * @param projectionMatrix ...
 * @param viewport ...
 ******************************************************************************/
template< typename BvhTreeType, typename BvhTreeCacheType, class ProducerType >
void BvhTreeRenderer< BvhTreeType, BvhTreeCacheType, ProducerType >
::renderImpl(const float4x4 &modelMatrix, const float4x4 &viewMatrix, const float4x4 &projectionMatrix, const int4 &viewport)
{
	assert(_colorResource != NULL && _depthResource != NULL);// && "You must set the input buffers first");

	initFrameObjects(make_uint2(viewport.z/* - viewport.x*/, viewport.w/* - viewport.y*/));

	size_t bufferSize;

	CUDAPM_START_EVENT(vsrender_pre_frame_mapbuffers);

	GV_CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &_colorResource, 0));
	GV_CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)d_inFrameColor->getDataStoragePtrAddress(), &bufferSize, _colorResource));

	GV_CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &_depthResource, 0));
	GV_CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)d_inFrameDepth->getDataStoragePtrAddress(), &bufferSize, _depthResource));

	CUDAPM_STOP_EVENT(vsrender_pre_frame_mapbuffers);

	// Use the same buffer as input and output of the ray-tracing
	d_outFrameColor->manualSetDataStorage(d_inFrameColor->getPointer());
	d_outFrameDepth->manualSetDataStorage(d_inFrameDepth->getPointer());

	doRender(modelMatrix, viewMatrix, projectionMatrix, viewport );

	CUDAPM_START_EVENT(vsrender_post_frame_unmapbuffers);
	GV_CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &_colorResource, 0));
	GV_CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &_depthResource, 0));
	CUDAPM_STOP_EVENT(vsrender_post_frame_unmapbuffers);
}

/******************************************************************************
 * ...
 *
 * @param modelMatrix ...
 * @param viewMatrix ...
 * @param projMatrix ...
 ******************************************************************************/
template< typename BvhTreeType, typename BvhTreeCacheType, class ProducerType >
void BvhTreeRenderer< BvhTreeType, BvhTreeCacheType, ProducerType >
::doRender(const float4x4 &modelMatrix, const float4x4 &viewMatrix, const float4x4 &projMatrix, const int4& pViewport )
{
	// Extract zNear, zFar as well as the distance in view space from the center of the screen
	// to each side of the screen.
	float fleft, fright, fbottom, ftop, fnear, ffar;
	fleft=projMatrix._array[14]*(projMatrix._array[8]-1.0f)/(projMatrix._array[0]*(projMatrix._array[10]-1.0f));
	fright=projMatrix._array[14]*(projMatrix._array[8]+1.0f)/(projMatrix._array[0]*(projMatrix._array[10]-1.0f));
	ftop=projMatrix._array[14]*(projMatrix._array[9]+1.0f)/((projMatrix._array[10]-1.0f)*projMatrix._array[5]);
	fbottom=projMatrix._array[14]*(projMatrix._array[9]-1.0f)/((projMatrix._array[10]-1.0f)*projMatrix._array[5]);
	fnear=projMatrix._array[14]/(projMatrix._array[10]-1.0f);
	ffar=projMatrix._array[14]/(projMatrix._array[10]+1.0f);

	float2 viewSurfaceVS[2];
	viewSurfaceVS[0]=make_float2(fleft, fbottom);
	viewSurfaceVS[1]=make_float2(fright, ftop);

	float3 viewPlane[2];
	viewPlane[0]=make_float3(fleft, fbottom, fnear);
	viewPlane[1]=make_float3(fright, ftop, fnear);
	float3 viewSize=(viewPlane[1]-viewPlane[0]);

	float2 viewSurfaceVS_Size=viewSurfaceVS[1]-viewSurfaceVS[0];
	///////////////////////////////////////////

	//transfor matrices
	float4x4 invModelMatrixT=transpose(inverse(modelMatrix));
	float4x4 invViewMatrixT=transpose(inverse(viewMatrix));

	float4x4 modelMatrixT=transpose(modelMatrix);
	float4x4 viewMatrixT=transpose(viewMatrix);

	GvRendering::GvRendererContext viewContext;

	viewContext.invViewMatrix=invViewMatrixT;
	viewContext.viewMatrix=viewMatrixT;
	viewContext.invModelMatrix=invModelMatrixT;
	viewContext.modelMatrix=modelMatrixT;

	viewContext.frustumNear=fnear;
	viewContext.frustumNearINV=1.0f/fnear;
	viewContext.frustumFar=ffar;
	viewContext.frustumRight=fright;
	viewContext.frustumTop=ftop;
	viewContext.frustumC= (-ffar+fnear)/(ffar-fnear);
	viewContext.frustumD= (-2.0f*ffar*fnear)/(ffar-fnear);
	/*viewContext.inFrameColor = d_inFrameColor->getDeviceArray();
	viewContext.inFrameDepth = d_inFrameDepth->getDeviceArray();
	viewContext.outFrameColor = d_outFrameColor->getDeviceArray();
	viewContext.outFrameDepth = d_outFrameDepth->getDeviceArray();*/
	viewContext._graphicsResources[ GvRendering::GvGraphicsInteroperabiltyHandler::eColorInput ] = d_inFrameColor->getPointer();//*(void **)d_inFrameColor->getDataStoragePtrAddress();
	viewContext._graphicsResourceAccess[ GvRendering::GvGraphicsInteroperabiltyHandler::eColorInput ] = GvRendering::GvGraphicsResource::ePointer;
	viewContext._graphicsResources[ GvRendering::GvGraphicsInteroperabiltyHandler::eDepthInput ] = d_inFrameDepth->getPointer();//*(void **)d_inFrameDepth->getDataStoragePtrAddress();
	viewContext._graphicsResourceAccess[ GvRendering::GvGraphicsInteroperabiltyHandler::eDepthInput ] = GvRendering::GvGraphicsResource::ePointer;
	viewContext._graphicsResources[ GvRendering::GvGraphicsInteroperabiltyHandler::eColorOutput ] = d_outFrameColor->getPointer();//*(void **)d_outFrameColor->getDataStoragePtrAddress();
	viewContext._graphicsResourceAccess[ GvRendering::GvGraphicsInteroperabiltyHandler::eColorOutput ] = GvRendering::GvGraphicsResource::ePointer;
	viewContext._graphicsResources[ GvRendering::GvGraphicsInteroperabiltyHandler::eDepthOutput ] = d_outFrameDepth->getPointer();//*(void **)d_outFrameDepth->getDataStoragePtrAddress();
	viewContext._graphicsResourceAccess[ GvRendering::GvGraphicsInteroperabiltyHandler::eDepthOutput ] = GvRendering::GvGraphicsResource::ePointer;

	float3 viewPlanePos=mul(viewContext.invViewMatrix, make_float3(fleft, fbottom, -fnear));
	viewContext.viewCenterWP = mul(viewContext.invViewMatrix, make_float3(0.0f, 0.0f, 0.0f));
	viewContext.viewPlaneDirWP = viewPlanePos-viewContext.viewCenterWP;

	///Resolution dependant stuff///
	viewContext.frameSize=_frameSize;
	float2 pixelSize=viewSurfaceVS_Size/make_float2((float)viewContext.frameSize.x, (float)viewContext.frameSize.y);
	viewContext.pixelSize=pixelSize;
	viewContext.viewPlaneXAxisWP=(mul(viewContext.invViewMatrix, make_float3(fright, fbottom, -fnear))-viewPlanePos)/(float)viewContext.frameSize.x;
	viewContext.viewPlaneYAxisWP=(mul(viewContext.invViewMatrix, make_float3(fleft, ftop, -fnear))-viewPlanePos)/(float)viewContext.frameSize.y;

	CUDAPM_START_EVENT(vsrender_copyconsts_frame);

	GV_CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_renderViewContext, &viewContext, sizeof(viewContext)));
	GV_CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_currentTime, (&this->_currentTime), sizeof(this->_currentTime), 0, cudaMemcpyHostToDevice));
	GV_CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_maxVolTreeDepth, (&this->_maxVolTreeDepth), sizeof(this->_maxVolTreeDepth), 0, cudaMemcpyHostToDevice));
//	GV_CUDA_SAFE_CALL(cudaMemcpyToSymbol(k_userParam, (&this->_userParam), sizeof(this->_userParam), 0, cudaMemcpyHostToDevice));

	CUDAPM_STOP_EVENT(vsrender_copyconsts_frame);

#if 0

	dim3 blockSize(RenderBlockResolution::x, RenderBlockResolution::y, 1);
	dim3 gridSize(iDivUp(_frameSize.x,RenderBlockResolution::x), iDivUp(_frameSize.y, RenderBlockResolution::y), 1);

	RenderKernelSimple<RenderBlockResolution>
		<<<gridSize, blockSize>>>(this->_bvhTree->getKernelObject());

#else
	dim3 blockSize( RenderBlockResolution::x, RenderBlockResolution::y, 1 );
	dim3 gridSize( iDivUp( _frameSize.x, RenderBlockResolution::x ), iDivUp( _frameSize.y, RenderBlockResolution::y ), 1 );

	RenderKernelInitRayState< RenderBlockResolution >
							<<< gridSize, blockSize, 0, 0 >>>(
									this->d_rayBufferT->getPointer(),
									this->d_rayBufferMaskedAt->getPointer(),
									this->d_rayBufferStackIndex->getPointer(),
									this->d_rayBufferStackVals->getPointer() );
	cudaDeviceSynchronize();
	GV_CHECK_CUDA_ERROR( "RenderKernelInitRayState" );

	uint maxNumLoop = 4;

	uint totalNumberOfUpdates = 0;

	uint numLoop = 0;
	uint numUpdates = 0;

	do
	{
		_bvhTreeCache->preRenderPass();

		CUDAPM_START_EVENT(gv_rendering);

		if (_dynamicUpdate)
		{
			RenderKernelContinueRays<RenderBlockResolution, 1, true>
				<<<gridSize, blockSize, 0, 0>>>(
						this->_bvhTree->getKernelObject(),
						//this->d_rayBufferTmin->getPointer(),
						this->d_rayBufferT->getPointer(),
						this->d_rayBufferMaskedAt->getPointer(),
						this->d_rayBufferStackIndex->getPointer(),
						this->d_rayBufferStackVals->getPointer() );
		}
		else
		{
			RenderKernelContinueRays<RenderBlockResolution, 1, false>
				<<<gridSize, blockSize, 0, 0>>>(
						this->_bvhTree->getKernelObject(),
						//this->d_rayBufferTmin->getPointer(),
						this->d_rayBufferT->getPointer(),
						this->d_rayBufferMaskedAt->getPointer(),
						this->d_rayBufferStackIndex->getPointer(),
						this->d_rayBufferStackVals->getPointer() );

		}

		cudaDeviceSynchronize();
		GV_CHECK_CUDA_ERROR("RenderKernelContinueRays");

		CUDAPM_STOP_EVENT(gv_rendering);

		CUDAPM_START_EVENT(dataProduction_handleRequests);

		//Bricks loading
		if (_dynamicUpdate)
		{
			//dynamicUpdate=false;

			/*uint maxNumSubdiv;
			uint maxNumBrickLoad;
			if(false){
				maxNumSubdiv=0xFFFFFFFF;
				maxNumBrickLoad=0xFFFFFFFF;
			}else{
				maxNumSubdiv=500;
				maxNumBrickLoad=30;
			}*/

			uint updateRes = _bvhTreeCache->handleRequests(/*maxNumSubdiv, maxNumBrickLoad*/);

			numUpdates = updateRes;
			totalNumberOfUpdates += updateRes;
		}
		else
		{
			numUpdates=0;
		}

		CUDAPM_STOP_EVENT(dataProduction_handleRequests);

		numLoop++;
	}
	while ( numUpdates && numLoop < maxNumLoop );
#endif
}

/******************************************************************************
 * ...
 *
 * @param colorResource ...
 ******************************************************************************/
template< typename BvhTreeType, typename BvhTreeCacheType, class ProducerType >
void BvhTreeRenderer< BvhTreeType, BvhTreeCacheType, ProducerType >::setColorResource(struct cudaGraphicsResource *colorResource)
{
	_colorResource = colorResource;
}

/******************************************************************************
 * ...
 *
 * @param depthResource ...
 ******************************************************************************/
template< typename BvhTreeType, typename BvhTreeCacheType, class ProducerType >
void BvhTreeRenderer< BvhTreeType, BvhTreeCacheType, ProducerType >::setDepthResource(struct cudaGraphicsResource *depthResource)
{
	_depthResource = depthResource;
}
