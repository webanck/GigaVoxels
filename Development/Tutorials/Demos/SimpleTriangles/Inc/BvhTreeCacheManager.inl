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
#include <GvCache/GvCacheManagerResources.h>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 *
 * @param cacheSize ...
 * @param elemSize ...
 ******************************************************************************/
template< unsigned int TId, typename ElementRes >
GPUCacheManager< TId, ElementRes >::GPUCacheManager( uint3 cacheSize, uint3 elemSize )
:	_cacheSize( cacheSize )
{
	_elemsCacheSize = _cacheSize / elemSize;

	d_TimeStampArray = new GvCore::Array3DGPULinear< uint >( _elemsCacheSize );
	d_TimeStampArray->fill( 0 );

	uint numElements = _elemsCacheSize.x * _elemsCacheSize.y * _elemsCacheSize.z - cNumLockedElements;

	d_elemAddressList		= new thrust::device_vector< uint >( numElements );
	d_elemAddressListTmp	= new thrust::device_vector< uint >( numElements );

	d_TempMaskList			= new thrust::device_vector< uint >( numElements );
	d_TempMaskList2			= new thrust::device_vector< uint >( numElements );

	// Initialize
	thrust::fill( d_TempMaskList->begin(), d_TempMaskList->end(), 0 );
	thrust::fill( d_TempMaskList2->begin(), d_TempMaskList2->end(), 0 );

	// TODO
	/*uint3 pageTableRes=d_pageTableArray->getResolution();
	uint pageTableResLinear=pageTableRes.x*pageTableRes.y*pageTableRes.z;*/
	//uint pageTableResLinear = 4000000;//BVH_VERTEX_POOL_SIZE;
	uint pageTableResLinear = 4147426;//BVH_NODE_POOL_SIZE

	//d_TempUpdateMaskList = GPUCacheManagerResources::getTempUsageMask1(pageTableRes.x*pageTableRes.y*pageTableRes.z); 
	d_TempUpdateMaskList	= new thrust::device_vector< uint >( pageTableResLinear );
	d_UpdateCompactList		= new thrust::device_vector< uint >( pageTableResLinear );

	// Init
	thrust::host_vector< uint > tmpelemaddress;

	for ( uint pos = 0; pos < _elemsCacheSize.x; pos++ )
	{
		tmpelemaddress.push_back( pos );
	}

	// Dont use element zero !
	thrust::copy( tmpelemaddress.begin() + cNumLockedElements, tmpelemaddress.end(), d_elemAddressList->begin() );

#if USE_CUDPP_LIBRARY
	uint cudppNumElem = std::max( pageTableResLinear, numElements );
	scanplan = GvCache::GvCacheManagerResources::getScanplan( cudppNumElem );
	GV_CUDA_SAFE_CALL( cudaMalloc( (void**) &d_numElementsPtr, sizeof( size_t ) ) );
#endif
}

/******************************************************************************
 * Handle requests
 *
 * @param updateList input list of node addresses that have been updated with "subdivision" or "load/produce" requests
 * @param numUpdateElems maximum number of elements to process
 * @param updateMask a unique given type of requests to take into account
 * @param maxNumElems ...
 * @param numValidNodes ...
 * @param gpuPool associated pool (nodes or bricks)
 *
 * @return ...
 ******************************************************************************/
template< unsigned int TId, typename ElementRes >
template< typename GPUPoolType, typename TProducerType >
uint GPUCacheManager< TId, ElementRes >
::genericWrite( uint* updateList, uint numUpdateElems, uint updateMask,
			   uint maxNumElems, uint numValidNodes, const GPUPoolType& gpuPool, TProducerType* pProducer )
{
	assert( pProducer != NULL );
	// TO DO : check pProducer at run-time

	uint numElems = 0;
	//uint providerId = GPUProviderType::ProviderId::value;

	if ( numUpdateElems > 0 )
	{
		const uint cacheId = Id::value;

		// [ 1 ] - Create the list of nodes that will be concerned by the data production management
		//
		// Only nodes whose request correponds to the given "updateMask" will be selected.
		//
		// The resulting list will be placed in [ d_UpdateCompactList ]

		//CUDAPM_START_EVENT_CHANNEL( 0, providerId, gpucache_nodes_manageUpdates );
		numElems = createUpdateList( updateList, numUpdateElems, updateMask );
		//CUDAPM_STOP_EVENT_CHANNEL( 0, providerId, gpucache_nodes_manageUpdates );

		// Prevent loading more than the cache size
		numElems = std::min( numElems, getNumElements() );
		
		//-----------------------------------
		// QUESTION : à quoi sert le test ?
		// - ça arrive sur la "simple sphere" quand on augmente trop le depth
		//-----------------------------------
		//if (numElems > numElemsNotUsed)
		//{
		//	std::cout << "CacheManager<" << providerId << ">: Warning: "
		//		<< numElemsNotUsed << " slots available!" << std::endl;
		//}

		///numElems = std::min(numElems, numElemsNotUsed);	// Prevent replacing elements in use
		///numElems = std::min(numElems, maxNumElems);		// Smooth loading

		if ( numElems > 0 )
		{
			//std::cout << "CacheManager<" << providerId << ">: " << numElems << " requests" << std::endl;

			// Invalidation phase
			//totalNumLoads += numElems;
			//lastNumLoads = numElems;

			/*CUDAPM_START_EVENT_CHANNEL( 1, providerId, gpucache_bricks_bricksInvalidation );*/
			//invalidateElements( numElems, numValidNodes );
			/*CUDAPM_STOP_EVENT_CHANNEL( 1, providerId, gpucache_bricks_bricksInvalidation );*/

			//CUDAPM_START_EVENT_CHANNEL( 0, providerId, gpucache_nodes_subdivKernel );
			//CUDAPM_START_EVENT_CHANNEL( 1, providerId, gpucache_bricks_gpuFetchBricks );
			//CUDAPM_EVENT_NUMELEMS_CHANNEL( 1, providerId, gpucache_bricks_gpuFetchBricks, numElems );

			// Write new elements into the cache
			thrust::device_vector< uint >* nodesAddressCompactList = d_UpdateCompactList;	// list of nodes to produce
			thrust::device_vector< uint >* elemsAddressCompactList = d_elemAddressList;		// ...

#if CUDAPERFMON_CACHE_INFO==1
			{
				dim3 blockSize(64, 1, 1);
				uint numBlocks=iDivUp(numElems, blockSize.x);
				dim3 gridSize=dim3( std::min( numBlocks, 32768U) , iDivUp(numBlocks,32768U), 1);

				SyntheticInfo_Update_DataWrite< ElementRes, AddressType ><<<gridSize, blockSize, 0>>>(
					d_CacheStateBufferArray->getPointer(), numElems,
					thrust::raw_pointer_cast(&(*elemsAddressCompactList)[0]),
					elemsCacheSize);

				GV_CHECK_CUDA_ERROR("SyntheticInfo_Update_DataWrite");
			}

			numPagesWrited = numElems;
#endif
			// Ask the HOST producer to generate its data
			pProducer->produceData( numElems, nodesAddressCompactList, elemsAddressCompactList, Id() );

			//CUDAPM_STOP_EVENT_CHANNEL( 0, providerId, gpucache_nodes_subdivKernel );
			//CUDAPM_STOP_EVENT_CHANNEL( 1, providerId, gpucache_bricks_gpuFetchBricks );
		}
	}

	return numElems;
}

/******************************************************************************
 * Create the list of nodes that will be concerned by the data production management
 *
 * @param inputList input list of node addresses that have been updated with "subdivision" or "load/produce" requests
 * @param inputNumElem maximum number of elements to process
 * @param testFlag a unique given type of requests to take into account
 *
 * @return the number of requests that the manager will have to handle
 ******************************************************************************/
template< unsigned int TId, typename ElementRes >
uint GPUCacheManager< TId, ElementRes >
::createUpdateList( uint* inputList, uint inputNumElem, uint pTestFlag )
{
	// ---- [ 1 ] ---- 1st step
	//
	// Fill the buffer of masks of valid elements whose attribute is equal to "pTestFlag"
	// Result is placed in "_d_TempUpdateMaskList"

	CUDAPM_START_EVENT( gpucachemgr_createUpdateList_createMask );

	// Set kernel execution configuration
	dim3 blockSize( 64, 1, 1 );
	uint numBlocks = iDivUp( inputNumElem, blockSize.x );
	dim3 gridSize = dim3( std::min( numBlocks, 65535U ) , iDivUp( numBlocks,65535U ), 1 );

	// Given a flag indicating a type of request (subdivision or load), and the updated global buffer of requests,
	// it fills a resulting mask buffer.
	// In fact, this mask indicates, for each element of the usage list, if it was used in the current rendering pass
	CacheManagerCreateUpdateMask<<< gridSize, blockSize, 0 >>>( inputNumElem, inputList, /*output*/(uint*)thrust::raw_pointer_cast( &(*d_TempUpdateMaskList)[ 0 ] ), pTestFlag );
	GV_CHECK_CUDA_ERROR( "UpdateCreateSubdivMask" );

	CUDAPM_STOP_EVENT( gpucachemgr_createUpdateList_createMask );

	// ---- [ 2 ] ---- 2nd step
	//
	// Concatenate only previous valid elements from input data in "inputList" into the buffer of requests
	// Result is placed in "_d_UpdateCompactList"

	CUDAPM_START_EVENT( gpucachemgr_createUpdateList_elemsReduction );

#if USE_CUDPP_LIBRARY
	
	// Stream compaction
	cudppCompact( scanplan,
		/*output*/thrust::raw_pointer_cast( &(*d_UpdateCompactList)[ 0 ] ), /*nbValidElements*/d_numElementsPtr,
		/*input*/inputList, /*isValid*/(uint*)thrust::raw_pointer_cast( &(*d_TempUpdateMaskList)[ 0 ] ),
		/*nbElements*/inputNumElem );
	GV_CHECK_CUDA_ERROR( "cudppCompact" );

	// Get number of elements
	size_t numElems;
	GV_CUDA_SAFE_CALL( cudaMemcpy( &numElems, d_numElementsPtr, sizeof( size_t ), cudaMemcpyDeviceToHost ) );

#else // USE_CUDPP_LIBRARY

	thrust::device_ptr< uint > firstPtr = thrust::device_ptr< uint >( inputList );
	thrust::device_ptr< uint > lastPtr = thrust::device_ptr< uint >( inputList + inputNumElem );

	uint numElems = thrust::copy_if( /*input first*/firstPtr, /*input last*/lastPtr, /*input stencil*/d_TempUpdateMaskList->begin(),
									/*OUTPUT*/d_UpdateCompactList->begin(),
									/*input predicate*/GvCore::not_equal_to_zero< uint >() ) - d_UpdateCompactList->begin();

	GV_CHECK_CUDA_ERROR( "thrust::copy_if" );

#endif // USE_CUDPP_LIBRARY

	CUDAPM_STOP_EVENT( gpucachemgr_createUpdateList_elemsReduction );

	return static_cast< uint >( numElems );
}
