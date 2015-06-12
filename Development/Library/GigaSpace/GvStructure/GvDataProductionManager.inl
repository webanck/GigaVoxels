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

// Cuda SDK
#include <helper_math.h>

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvPerfMon/GvPerformanceMonitor.h"
#include "GvCore/functional_ext.h"
#include "GvCore/GvError.h"
#if USE_CUDPP_LIBRARY
	#include "GvCache/GvCacheManagerResources.h"
#endif

// TEST
#include "GvCore/Array3DGPULinear.h"

// System
#include <cassert>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvStructure
{

/******************************************************************************
 * Constructor
 *
 * @param voltree a pointer to the data structure.
 * @param gpuprod a pointer to the user's producer.
 * @param nodepoolres the 3d size of the node pool.
 * @param brickpoolres the 3d size of the brick pool.
 * @param graphicsInteroperability Graphics interoperabiliy flag to be able to map buffers to graphics interoperability mode
 ******************************************************************************/
template< typename TDataStructure >
GvDataProductionManager< TDataStructure >
::GvDataProductionManager( TDataStructure* pDataStructure, uint3 nodepoolres, uint3 brickpoolres, uint graphicsInteroperability )
:	GvStructure::GvIDataProductionManager()
#if USE_CUDPP_LIBRARY
,	_scanPlan( 0 )
,	_d_nbValidRequests( NULL )
,	_d_validRequestMasks( NULL )
#endif
,	_producers()
,	_leafNodes( NULL )
,	_emptyNodeVolume( NULL )
,	_nbLeafNodes( 0 )
,	_nbNodes( 0 )
,	_hasTreeDataStructureMonitoring( false )
,	_isProductionTimeLimited( false )
,	_lastProductionTimed( false )
,	_productionTimeLimit( 10.f )
,	_totalNodesProductionTime( 0.f )
,	_totalBrickProductionTime( 0.f )
,	_totalProducedBricks( 0u )
,	_totalProducedNodes( 0u )
{
	// Reference on a data structure
	_dataStructure = pDataStructure;

	// linearize the resolution
	_nodePoolRes = make_uint3( nodepoolres.x * nodepoolres.y * nodepoolres.z, 1, 1 );
	_brickPoolRes = brickpoolres;

	// Cache managers creation : nodes and bricks
	_nodesCacheManager = new NodesCacheManager( _nodePoolRes, _dataStructure->_childArray, graphicsInteroperability );
	_bricksCacheManager = new BricksCacheManager( _brickPoolRes, _dataStructure->_dataArray, graphicsInteroperability );

	//@todo The creation of the localization arrays should be moved here, not in the data structure (this is cache implementation details/features)

	// Node cache manager inititlization
	_nodesCacheManager->_pageTable->locCodeArray = _dataStructure->_localizationCodeArray;
	_nodesCacheManager->_pageTable->locDepthArray = _dataStructure->_localizationDepthArray;
	_nodesCacheManager->_pageTable->getKernel().childArray = _dataStructure->_childArray->getDeviceArray();
	_nodesCacheManager->_pageTable->getKernel().locCodeArray = _dataStructure->_localizationCodeArray->getDeviceArray();
	_nodesCacheManager->_pageTable->getKernel().locDepthArray = _dataStructure->_localizationDepthArray->getDeviceArray();
	_nodesCacheManager->_totalNumLoads = 2;
	_nodesCacheManager->_lastNumLoads = 1;

	// Data cache manager inititlization
	_bricksCacheManager->_pageTable->locCodeArray = _dataStructure->_localizationCodeArray;
	_bricksCacheManager->_pageTable->locDepthArray = _dataStructure->_localizationDepthArray;
	_bricksCacheManager->_pageTable->getKernel().childArray = _dataStructure->_childArray->getDeviceArray();
	_bricksCacheManager->_pageTable->getKernel().dataArray = _dataStructure->_dataArray->getDeviceArray();
	_bricksCacheManager->_pageTable->getKernel().locCodeArray = _dataStructure->_localizationCodeArray->getDeviceArray();
	_bricksCacheManager->_pageTable->getKernel().locDepthArray = _dataStructure->_localizationDepthArray->getDeviceArray();
	_bricksCacheManager->_totalNumLoads = 0;
	_bricksCacheManager->_lastNumLoads = 0;

	// Request buffers initialization
	_updateBufferArray = new GvCore::Array3DGPULinear< uint >( _nodePoolRes, graphicsInteroperability );
	_updateBufferCompactList = new thrust::device_vector< uint >( _nodePoolRes.x * _nodePoolRes.y * _nodePoolRes.z );

	_totalNumBricksLoaded = 0;

	// Device-side cache manager initialization
	_dataProductionManagerKernel._updateBufferArray = this->_updateBufferArray->getDeviceArray();
	_dataProductionManagerKernel._nodeCacheManager = this->_nodesCacheManager->getKernelObject();
	_dataProductionManagerKernel._brickCacheManager = this->_bricksCacheManager->getKernelObject();

	// Initialize max number of requests the cache has to handle
	_maxNbNodeSubdivisions = 5000;
	_maxNbBrickLoads = 3000;
	this->_nbNodeSubdivisionRequests = 0;
	this->_nbBrickLoadRequests = 0;

#if USE_CUDPP_LIBRARY

	// cudpp stream compaction parameters
	uint cudppNbElements = nodepoolres.x * nodepoolres.y * nodepoolres.z;
	_scanPlan = GvCache::GvCacheManagerResources::getScanplan( cudppNbElements );

#ifndef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_DATAPRODUCTIONMANAGER
	GV_CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_nbValidRequests, sizeof( size_t ) ) );
#else
	#ifndef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_WITH_COMBINED_COPY
	GV_CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_nbValidRequests, sizeof( size_t ) ) );
	#else
	GV_CUDA_SAFE_CALL( cudaMalloc( (void**) &_d_nbValidRequests, 3 * sizeof( size_t ) ) );
	_nodesCacheManager->_d_nbValidRequests = _d_nbValidRequests;
	_bricksCacheManager->_d_nbValidRequests = _d_nbValidRequests;
	#endif
#endif

	_d_validRequestMasks = new GvCore::Array3DGPULinear< uint >( _nodePoolRes, graphicsInteroperability );

#endif

	// TO DO : do lazy evaluation => ONLY ALLOCATE WHEN REQUESTED AND USED + free memory just after ?
	_leafNodes = new thrust::device_vector< uint >( _nodePoolRes.x * _nodePoolRes.y * _nodePoolRes.z );
	_emptyNodeVolume = new thrust::device_vector< float >( _nodePoolRes.x * _nodePoolRes.y * _nodePoolRes.z );

	cudaEventCreate( &_startProductionNodes );
	cudaEventCreate( &_stopProductionNodes );
	cudaEventCreate( &_startProductionBricks );
	cudaEventCreate( &_stopProductionBricks );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TDataStructure >
GvDataProductionManager< TDataStructure >
::~GvDataProductionManager()
{
	// Delete cache manager (nodes and bricks)
	delete _nodesCacheManager;
	delete _bricksCacheManager;

	delete _updateBufferArray;
	delete _updateBufferCompactList;

#if USE_CUDPP_LIBRARY
	GV_CUDA_SAFE_CALL( cudaFree( _d_nbValidRequests ) );
	delete _d_validRequestMasks;
#endif

	delete _leafNodes;
	delete _emptyNodeVolume;

	cudaEventDestroy( _startProductionNodes );
	cudaEventDestroy( _stopProductionNodes );
	cudaEventDestroy( _startProductionBricks );
	cudaEventDestroy( _stopProductionBricks );
}

/******************************************************************************
 * This method is called before the rendering process. We just clear the request buffer.
 ******************************************************************************/
template< typename TDataStructure >
void GvDataProductionManager< TDataStructure >
::preRenderPass()
{
	CUDAPM_START_EVENT( gpucache_preRenderPass );

	// Clear subdiv pool
// #ifndef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_DATAPRODUCTIONMANAGER
	_updateBufferArray->fill( 0 );
// #else
// 	_updateBufferArray->fillAsync( 0 );	// TO DO : with a kernel instead of cudaMemSet(), copy engine could overlap
// #endif

	// Number of requests cache has handled
	//_nbNodeSubdivisionRequests = 0;
	//_nbBrickLoadRequests = 0;

#if CUDAPERFMON_CACHE_INFO==1
	_nodesCacheManager->_d_CacheStateBufferArray->fill( 0 );
	_nodesCacheManager->_numPagesUsed = 0;
	_nodesCacheManager->_numPagesWrited = 0;

	_bricksCacheManager->_d_CacheStateBufferArray->fill( 0 );
	_bricksCacheManager->_numPagesUsed = 0;
	_bricksCacheManager->_numPagesWrited = 0;
#endif

	CUDAPM_STOP_EVENT( gpucache_preRenderPass );
}

/******************************************************************************
 * This method is called after the rendering process. She's responsible for processing requests.
 *
 * @return the number of requests processed.
 *
 * @todo Check whether or not the inversion call of updateTimeStamps() with manageUpdates() has side effects
 ******************************************************************************/
template< typename TDataStructure >
uint GvDataProductionManager< TDataStructure >
::handleRequests()
{
	// _lastProductionTimed = false;
	// Measure the time taken by the last production.
	// if ( _lastProductionTimed
	// 		&& ( _nbBrickLoadRequests != 0 || _nbNodeSubdivisionRequests != 0 ) )
	// {
	// 	cudaEventSynchronize( _stopProductionBricks );
	// 	float lastProductionNodesTime, lastProductionBricksTime;
	//
	// 	cudaEventElapsedTime( &lastProductionNodesTime, _startProductionNodes, _stopProductionNodes );
	// 	cudaEventElapsedTime( &lastProductionBricksTime, _startProductionBricks, _stopProductionBricks );
	//
	// 	// Don't take too low number of requests into account (in this cases, the additional
	// 	// costs of launching the kernel, compacting the array... is greater than the
	// 	// brick/node production time)
	// 	if ( _nbNodeSubdivisionRequests > 63 )
	// 	{
	// 		_totalProducedNodes += _nbNodeSubdivisionRequests;
	// 		_totalNodesProductionTime += lastProductionNodesTime;
	// 	}
	//
	// 	if ( _nbBrickLoadRequests > 63 )
	// 	{
	// 		_totalProducedBricks += _nbBrickLoadRequests;
	// 		_totalBrickProductionTime += lastProductionBricksTime;
	// 	}
	//
	// 	// Update the vector of statistics.
	// 	struct GsProductionStatistics stats;
	// 	//stats._frameId = ??? TODO
	// 	stats._nNodes = _nbNodeSubdivisionRequests;
	// 	stats._nodesProductionTime = lastProductionNodesTime;
	// 	stats._nBricks = _nbBrickLoadRequests;
	// 	stats._bricksProductionTime = lastProductionBricksTime;
	// 	_productionStatistics.push_back( stats );
	// }
	// _isProductionTimeLimited should not be used inside this function since it can be changed
	// by the user at anytime and we need to have a constant value throughout the whole function.
	// _lastProductionTimed = _isProductionTimeLimited;

	// TO DO
	// Check whether or not the inversion call of updateTimeStamps() with manageUpdates() has side effects

	// Generate the requests buffer
	//
	// Collect and compact update informations for both nodes and bricks
	/*CUDAPM_START_EVENT( dataProduction_manageRequests );
	uint nbRequests = manageUpdates();
	CUDAPM_STOP_EVENT( dataProduction_manageRequests );*/

	// Stop post-render pass if no request
	//if ( nbRequests > 0 )
	//{
		// Update time stamps
		//
		// - TO DO : the update time stamps could be done in parallel for nodes and bricks using streams
		CUDAPM_START_EVENT( cache_updateTimestamps );
		updateTimeStamps();
		CUDAPM_STOP_EVENT( cache_updateTimestamps );

		// Manage requests
		//
		// - TO DO : if updateTimestamps is before, this task could also be done in parallel using streams
		CUDAPM_START_EVENT( dataProduction_manageRequests );
		uint nbRequests = manageUpdates();
		uint totalRequests = nbRequests;
		// std::cout << "Total requests: " << totalRequests << std::endl;
		CUDAPM_STOP_EVENT( dataProduction_manageRequests );

// #ifdef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_DATAPRODUCTIONMANAGER
// 		// Get number of elements
// 		// BEWARE : synchronization to avoid an expensive final call to cudaDeviceSynchronize()
// 	#ifndef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_WITH_COMBINED_COPY
// 		size_t nbElementsTemp;
// 		_nodesCacheManager->updateTimeStampsCopy( _intraFramePass );
// 		_bricksCacheManager->updateTimeStampsCopy( _intraFramePass );
// 		GV_CUDA_SAFE_CALL( cudaMemcpy( &nbElementsTemp, _d_nbValidRequests, sizeof( size_t ), cudaMemcpyDeviceToHost ) );
// 		nbRequests = static_cast< uint >( nbElementsTemp );
// 	#else
// 		size_t nbElementsTemp[ 3 ];
// 		GV_CUDA_SAFE_CALL( cudaMemcpy( nbElementsTemp, _d_nbValidRequests, 3 * sizeof( size_t ), cudaMemcpyDeviceToHost ) );
// 		nbRequests = static_cast< uint >( nbElementsTemp[ 0 ] );
// 		// BEWARE : in nodes/bricks managers, real value should be _numElemsNotUsed = (uint)numElemsNotUsedST + inactiveNumElems
// 		_nodesCacheManager->_numElemsNotUsedST = static_cast< uint >( nbElementsTemp[ 1 ] );
// 		_bricksCacheManager->_numElemsNotUsedST = static_cast< uint >( nbElementsTemp[ 2 ] );
// 	#endif
// 		// Launch final "stream compaction" steps for "used" elements
// 		this->_numNodeTilesNotInUse = _nodesCacheManager->updateTimeStampsFinal( _intraFramePass );
// 		this->_numBricksNotInUse = _bricksCacheManager->updateTimeStampsFinal( _intraFramePass );
// #endif

		// Handle requests :
// #ifndef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_PRODUCER

		// [ 1 ] - Handle the "subdivide nodes" requests

		// Limit production according to the time limit.
		// First, do as if all the requests were node subdivisions
		// if ( _lastProductionTimed )
		// {
		// 	if ( _totalProducedNodes != 0u )
		// 	{
		// 		nbRequests = min(
		// 				nbRequests,
		// 				static_cast< uint >( _productionTimeLimit * _totalProducedNodes / _totalNodesProductionTime ) );
		// 	}
		// 	cudaEventRecord( _startProductionNodes );
		// }

		CUDAPM_START_EVENT( producer_nodes );
		//uint numSubDiv = manageSubDivisions( nbRequests );
		_nbNodeSubdivisionRequests = manageSubDivisions( nbRequests );
		// std::cout << "Sub requests done: " << _nbNodeSubdivisionRequests << std::endl;
		CUDAPM_STOP_EVENT( producer_nodes );

		// if ( _lastProductionTimed )
		// {
		// 	cudaEventRecord( _stopProductionNodes );
		// 	cudaEventRecord( _startProductionBricks );
		// }

		//  [ 2 ] - Handle the "load/produce bricks" requests

		// Now, we know how many requests are node and how many are bricks, we can limit
		// the number of bricks requests according to the number of node requests performed.
		//Warning! The requests are not exclusives: a request can be of Load & Subdivision??
		uint nbBricks = nbRequests/* - _nbNodeSubdivisionRequests*/;
		// if ( _lastProductionTimed && _totalProducedNodes != 0 && _totalProducedBricks != 0 )
		// {
		// 	// Evaluate how much time will be left after nodes subdivision
		// 	float remainingTime = _productionTimeLimit - _nbNodeSubdivisionRequests * _totalNodesProductionTime / _totalProducedNodes;
		// 	// Limit the number of request to fit in the remaining time
		// 	nbBricks = min(
		// 			nbBricks,
		// 			static_cast< uint >( remainingTime * _totalProducedBricks / _totalBrickProductionTime ) );
		// }

		CUDAPM_START_EVENT( producer_bricks );
		if ( nbBricks > 0 )
		{
			_nbBrickLoadRequests = manageDataLoadGPUProd( nbBricks );
		}
		CUDAPM_STOP_EVENT( producer_bricks );
		// std::cout << "Load requests done: " << _nbBrickLoadRequests << std::endl;
		// std::cout << "Requests done: " << _nbNodeSubdivisionRequests + _nbBrickLoadRequests << std::endl;
		// std::cout << std::endl;
// #else
// 		if ( nbRequests > 0 ) {
// 			produceData( nbRequests );
// 		}
// #endif
	//}

	// Tree data structure monitoring
	if ( _hasTreeDataStructureMonitoring )
	{
		// TEST
		dim3 blockSize( 128, 1, 1 );
		dim3 gridSize( ( _nodePoolRes.x * _nodePoolRes.y * _nodePoolRes.z ) / 128 + 1, 1, 1 );
		GVKernel_TrackLeafNodes< typename TDataStructure::VolTreeKernelType ><<< gridSize, blockSize >>>( _dataStructure->volumeTreeKernel, _nodesCacheManager->_pageTable->getKernel()/*page table*/, ( _nodePoolRes.x * _nodePoolRes.y * _nodePoolRes.z )/*nb nodes*/, _dataStructure->getMaxDepth()/*max depth*/, thrust::raw_pointer_cast( &( *this->_leafNodes )[ 0 ] ), thrust::raw_pointer_cast( &( *this->_emptyNodeVolume )[ 0 ] ) );
		_nbLeafNodes = thrust::reduce( (*_leafNodes).begin(), (*_leafNodes).end(), static_cast< unsigned int >( 0 ), thrust::plus< unsigned int >() );
		const float _emptyVolume = thrust::reduce( (*_emptyNodeVolume).begin(), (*_emptyNodeVolume).end(), static_cast< float >( 0.f ), thrust::plus< float >() );
		//std::cout << "------------------------------------------------" << _nbLeafNodes << std::endl;
		//std::cout << "Volume of empty nodes : " << ( _emptyVolume * 100.f ) << std::endl;
		//std::cout << "------------------------------------------------" << _nbLeafNodes << std::endl;
		//std::cout << "TOTAL number of leaf nodes : " << _nbLeafNodes << std::endl;
		GVKernel_TrackNodes< typename TDataStructure::VolTreeKernelType ><<< gridSize, blockSize >>>( _dataStructure->volumeTreeKernel, _nodesCacheManager->_pageTable->getKernel()/*page table*/, ( _nodePoolRes.x * _nodePoolRes.y * _nodePoolRes.z )/*nb nodes*/, _dataStructure->getMaxDepth()/*max depth*/, thrust::raw_pointer_cast( &( *this->_leafNodes )[ 0 ] ), thrust::raw_pointer_cast( &( *this->_emptyNodeVolume )[ 0 ] ) );
		_nbNodes = thrust::reduce( (*_leafNodes).begin(), (*_leafNodes).end(), static_cast< unsigned int >( 0 ), thrust::plus< unsigned int >() );
		//std::cout << "TOTAL number of nodes : " << _nbNodes << std::endl;
	}

	// if ( _lastProductionTimed && nbRequests > 0 )
	// {
	// 	cudaEventRecord( _stopProductionBricks );
	// }

	return nbRequests;
}

/******************************************************************************
 * This method destroy the current N-tree and clear the caches.
 ******************************************************************************/
template< typename TDataStructure >
void GvDataProductionManager< TDataStructure >
::clearCache()
{
	// Launch Kernel
	dim3 blockSize( 32, 1, 1 );
	dim3 gridSize( 1, 1, 1 );
	// This clears node pool child and brick 1st nodetile after root node
	ClearVolTreeRoot<<< gridSize, blockSize >>>( _dataStructure->volumeTreeKernel, NodeTileRes::getNumElements() );

	GV_CHECK_CUDA_ERROR( "ClearVolTreeRoot" );

	// Reset nodes cache manager
	_nodesCacheManager->clearCache();
	_nodesCacheManager->_totalNumLoads = 2;
	_nodesCacheManager->_lastNumLoads = 1;

	// Reset bricks cache manager
	_bricksCacheManager->clearCache();
	_bricksCacheManager->_totalNumLoads = 0;
	_bricksCacheManager->_lastNumLoads = 0;
}

/******************************************************************************
 * Get the associated device-side object
 *
 * @return The device-side object
 ******************************************************************************/
template< typename TDataStructure >
inline GvDataProductionManager< TDataStructure >
::DataProductionManagerKernelType GvDataProductionManager< TDataStructure >::getKernelObject() const
{
	return _dataProductionManagerKernel;
}

/******************************************************************************
 * Get the update buffer
 *
 * @return The update buffer
 ******************************************************************************/
template< typename TDataStructure >
inline GvCore::Array3DGPULinear< uint >* GvDataProductionManager< TDataStructure >
::getUpdateBuffer() const
{
	return _updateBufferArray;
}

/******************************************************************************
 * Get the nodes cache manager
 *
 * @return the nodes cache manager
 ******************************************************************************/
template< typename TDataStructure >
inline const GvDataProductionManager< TDataStructure >::NodesCacheManager*
GvDataProductionManager< TDataStructure >::getNodesCacheManager() const
{
	return _nodesCacheManager;
}

/******************************************************************************
 * Get the bricks cache manager
 *
 * @return the bricks cache manager
 ******************************************************************************/
template< typename TDataStructure >
inline const GvDataProductionManager< TDataStructure >::BricksCacheManager*
GvDataProductionManager< TDataStructure >::getBricksCacheManager() const
{
	return _bricksCacheManager;
}

/******************************************************************************
 * Get the nodes cache manager
 *
 * @return the nodes cache manager
 ******************************************************************************/
template< typename TDataStructure >
inline GvDataProductionManager< TDataStructure >::NodesCacheManager*
GvDataProductionManager< TDataStructure >::editNodesCacheManager()
{
	return _nodesCacheManager;
}

/******************************************************************************
 * Get the bricks cache manager
 *
 * @return the bricks cache manager
 ******************************************************************************/
template< typename TDataStructure >
inline GvDataProductionManager< TDataStructure >::BricksCacheManager*
GvDataProductionManager< TDataStructure >::editBricksCacheManager()
{
	return _bricksCacheManager;
}

/******************************************************************************
 * ...
 ******************************************************************************/
//template< typename TDataStructure, typename GPUProducer, typename NodeTileRes, typename BrickFullRes >
//void VolTreeGPUCache< TDataStructure, GPUProducer/*, NodeTileRes, BrickFullRes*/ >::updateSymbols()
//{
//	CUDAPM_START_EVENT(gpucache_updateSymbols);
//
//	_nodesCacheManager->updateSymbols();
//	_bricksCacheManager->updateSymbols();
//
//	CUDAPM_STOP_EVENT(gpucache_updateSymbols);
//
//	_useBrickUsageOptim = true;
//	_intraFramePass = false;
//}

/******************************************************************************
 * Update time stamps
 ******************************************************************************/
template< typename TDataStructure >
void GvDataProductionManager< TDataStructure >
::updateTimeStamps()
{
	// Ask nodes cache manager to update time stamps
	CUDAPM_START_EVENT(cache_updateTimestamps_dataStructure);
// #ifndef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_DATAPRODUCTIONMANAGER
	this->_numNodeTilesNotInUse = _nodesCacheManager->updateTimeStamps( _intraFramePass );
// #else
// 	_nodesCacheManager->updateTimeStamps( _intraFramePass );
// #endif
	CUDAPM_STOP_EVENT(cache_updateTimestamps_dataStructure);

// #if USE_BRICK_USAGE_OPTIM
// 	// New optimisation for brick usage flag
// 	// Buggee !
//
// 	/*if ( _useBrickUsageOptim && !_intraFramePass )*/
// 	{
// 		//uint numNodeTiles=this->_numNodeTilesNotInUse;
// 		uint numNodeTileUsed = _nodesCacheManager->getNumElements() - this->_numNodeTilesNotInUse;
//
// 		if ( numNodeTileUsed > 0 )
// 		{
// 			// Launch Kernel
// 			dim3 blockSize( 64, 1, 1 );
// 			uint numBlocks = iDivUp( numNodeTileUsed, blockSize.x );
// 			dim3 gridSize = dim3( std::min( numBlocks, 65535U ), iDivUp( numBlocks, 65535U ), 1 );
// 			UpdateBrickUsageFromNodes<<<gridSize, blockSize, 0>>>( numNodeTileUsed,
// 				thrust::raw_pointer_cast( &(*(_nodesCacheManager->getTimeStampsElemAddressList() ) )[ this->_numNodeTilesNotInUse ] ),
// 				this->_dataStructure->volumeTreeKernel, this->getKernelObject() );
//
// 			GV_CHECK_CUDA_ERROR( "UpdateBrickUsageFromNodes" );
// 		}
// 	}
// #endif

	// Ask bricks cache manager to update time stamps
	CUDAPM_START_EVENT(cache_updateTimestamps_bricks);
// #ifndef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_DATAPRODUCTIONMANAGER
	this->_numBricksNotInUse = _bricksCacheManager->updateTimeStamps( _intraFramePass );
// #else
// 	_bricksCacheManager->updateTimeStamps( _intraFramePass );
// #endif
	CUDAPM_STOP_EVENT(cache_updateTimestamps_bricks);
}

/******************************************************************************
 * This method gather all requests by compacting the list.
 *
 * @return The number of elements in the requests list
 ******************************************************************************/
template <typename TDataStructure>
uint GvDataProductionManager<TDataStructure>::manageUpdates() {
	uint totalNbElements = _nodePoolRes.x * _nodePoolRes.y * _nodePoolRes.z;

	uint nbElements = 0;

	// Optimisation test for case where the cache is not full
	if ( _nodesCacheManager->_totalNumLoads < _nodesCacheManager->getNumElements() )
	{
		totalNbElements = ( _nodesCacheManager->_totalNumLoads ) * NodeTileRes::getNumElements();
		//std::cout << "totalNbElements : " << totalNbElements << "/" << nodePoolRes.x * nodePoolRes.y * nodePoolRes.z << "\n";
	}

	CUDAPM_START_EVENT( dataProduction_manageRequests_elemsReduction );

	// Fill the buffer used to store node addresses updates with subdivision or load requests
#if USE_CUDPP_LIBRARY

	// Set kernel execution configuration
	dim3 blockSize( 64, 1, 1 );
	uint nbBlocks = iDivUp( totalNbElements, blockSize.x );
	dim3 gridSize = dim3( std::min( nbBlocks, 65535U ) , iDivUp( nbBlocks, 65535U ), 1 );

	// This kernel creates the usage mask list of used and non used elements (in current rendering pass) in a single pass
	//
	// Note : Generate an error with CUDA 3.2
	GvKernel_PreProcessRequests<<< gridSize, blockSize, 0 >>>( /*input*/_updateBufferArray->getPointer(), /*output*/_d_validRequestMasks->getPointer(), /*input*/totalNbElements );
	GV_CHECK_CUDA_ERROR( "GvKernel_PreProcessRequests" );

	// TO DO
	//
	// Optimization ?
	// ...
	//
	// Check if, like in Thrust, we could use directly use _updateBufferArray as a predicate (i.e. masks)
	// - does cudpp requires an array of "1"/"0" or "0"/"!0" like in thrust with GvCore::not_equal_to_zero< uint >()
	// - if yes, the GvKernel_PreProcessRequests kernel call could be avoided and _updateBufferArray used as input mask in cudppCompact
	// - ok, cudpp, only check for value > 0 and not == 1, so it could be tested, just check if a speed can occur

	// Given an array d_in and an array of 1/0 flags in deviceValid, returns a compacted array in d_out of corresponding only the "valid" values from d_in.
	//
	// Takes as input an array of elements in GPU memory (d_in) and an equal-sized unsigned int array in GPU memory (deviceValid) that indicate which of those input elements are valid.
	// The output is a packed array, in GPU memory, of only those elements marked as valid.
	//
	// Internally, uses cudppScan.
	CUDPPResult result = cudppCompact(
		/*handle to CUDPPCompactPlan*/_scanPlan,
		/* OUT : compacted output */thrust::raw_pointer_cast( &(*_updateBufferCompactList)[ 0 ] ),
		/* OUT :  number of elements valid flags in the d_isValid input array */_d_nbValidRequests,
		/* input to compact */_updateBufferArray->getPointer(),
		/* which elements in input are valid */_d_validRequestMasks->getPointer(),
		/* nb of elements in input */totalNbElements
	);
	GV_CHECK_CUDA_ERROR( "KERNEL manageUpdates::cudppCompact" );
	assert( result == CUDPP_SUCCESS );

// #ifndef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_DATAPRODUCTIONMANAGER
	// Get number of elements
	size_t nbElementsTemp;
	GV_CUDA_SAFE_CALL(cudaMemcpy(
		&nbElementsTemp,
		_d_nbValidRequests,
		sizeof(size_t),
		cudaMemcpyDeviceToHost
	));
	nbElements = static_cast< uint >( nbElementsTemp );
// #endif

#else // USE_CUDPP_LIBRARY

	nbElements = thrust::copy_if(
		/*first input*/thrust::device_ptr< uint >( _updateBufferArray->getPointer( 0 ) ),
		/*last input*/thrust::device_ptr< uint >( _updateBufferArray->getPointer( 0 ) ) + totalNbElements,
		/*output*/_updateBufferCompactList->begin(),
		/*predicate*/GvCore::not_equal_to_zero< uint >()
	) - _updateBufferCompactList->begin();

#endif // USE_CUDPP_LIBRARY

	CUDAPM_STOP_EVENT( dataProduction_manageRequests_elemsReduction );
	GV_CHECK_CUDA_ERROR( "manageUpdates" );

	return nbElements;
}

/******************************************************************************
 * This method handle the subdivisions requests.
 *
 * @param numUpdateElems the number of requests available in the buffer (of any kind).
 *
 * @return the number of subidivision requests processed.
 ******************************************************************************/
template< typename TDataStructure >
uint GvDataProductionManager<TDataStructure>::manageSubDivisions(uint numUpdateElems) {
	// Global buffer of requests of used elements only
	uint* updateCompactList = thrust::raw_pointer_cast( &(*_updateBufferCompactList)[ 0 ] );

	// Number of nodes to process
	uint numValidNodes = ( _nodesCacheManager->_totalNumLoads ) * NodeTileRes::getNumElements();

	// This will ask nodes producer to subdivide nodes
	assert( _producers.size() > 0 );
	assert( _producers[ 0 ] != NULL );
	ProducerType* producer = _producers[ 0 ];
	return _nodesCacheManager->genericWrite(
		updateCompactList,
		numUpdateElems,
		/*type of request to handle*/DataProductionManagerKernelType::VTC_REQUEST_SUBDIV, _maxNbNodeSubdivisions,
		numValidNodes,
		_dataStructure->_nodePool,
		producer
	);
}

/******************************************************************************
 * This method handle the load requests.
 *
 * @param numUpdateElems the number of requests available in the buffer (of any kind).
 *
 * @return the number of load requests processed.
 ******************************************************************************/
template< typename TDataStructure >
uint GvDataProductionManager<TDataStructure>::manageDataLoadGPUProd(uint numUpdateElems) {
	// Global buffer of requests of used elements only
	uint* updateCompactList = thrust::raw_pointer_cast( &(*_updateBufferCompactList)[ 0 ] );

	// Number of bricks to process
	uint numValidNodes = ( _nodesCacheManager->_totalNumLoads ) * NodeTileRes::getNumElements();

	// This will ask bricks producer to load/produce data
	assert( _producers.size() > 0 );
	assert( _producers[ 0 ] != NULL );
	ProducerType* producer = _producers[ 0 ];
	return _bricksCacheManager->genericWrite(
		updateCompactList,
		numUpdateElems,
		/*type of request to handle*/DataProductionManagerKernelType::VTC_REQUEST_LOAD,
		_maxNbBrickLoads,
		numValidNodes,
		_dataStructure->_dataPool,
		_producers[ 0 ]
	);
}

// #ifdef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_PRODUCER
// /******************************************************************************
//  * This method handle the subdivisions/loads requests.
//  *
//  * @param numUpdateElems the number of requests available in the buffer (of any kind).
//  ******************************************************************************/
// template <typename TDataStructure>
// void GvDataProductionManager<TDataStructure>::produceData(uint numUpdateElems) {
// 	// _nbNodeSubdivisionRequests = manageSubDivisions( numUpdateElems );
//
// 	// Global buffer of requests of used elements only
// 	uint *updateCompactList = thrust::raw_pointer_cast(&(*_updateBufferCompactList)[0]);
//
// 	// Number of nodes to process
// 	uint numValidNodes = _nodesCacheManager->_totalNumLoads * NodeTileRes::getNumElements();
//
// 	// This will ask nodes producer to subdivide nodes
// 	assert( _producers.size() > 0 );
// 	assert( _producers[ 0 ] != NULL );
// 	ProducerType* producer = _producers[ 0 ];
//
// 	_nodesCacheManager->genericWrite(
// 		updateCompactList,
// 		numUpdateElems,
// 		/*type of request to handle*/DataProductionManagerKernelType::VTC_REQUEST_SUBDIV,
// 		_maxNbNodeSubdivisions,
// 		numValidNodes,
// 		_dataStructure->_nodePool,
// 		producer
// 	);
//
// 	_bricksCacheManager->genericWrite(
// 		updateCompactList,
// 		numUpdateElems,
// 		/*type of request to handle*/DataProductionManagerKernelType::VTC_REQUEST_LOAD,
// 		_maxNbBrickLoads,
// 		numValidNodes,
// 		_dataStructure->_dataPool,
// 		producer
// 	);
//
// 	// Get number of elements
// 	size_t numElems[ 2 ];
// 	GV_CUDA_SAFE_CALL( cudaMemcpy( &numElems, _d_nbValidRequests + 1, 2 * sizeof( size_t ), cudaMemcpyDeviceToHost ) );
//
// 	// Limit production according to the time limit.
// 	// First, consider all the request are node subdivision
// 	// if ( _lastProductionTimed )
// 	// {
// 	// 	if ( _totalProducedNodes != 0u )
// 	// 	{
// 	// 		numElems[ 0 ] = min(
// 	// 				static_cast< uint >( numElems[ 0 ] ),
// 	// 				max( 100,
// 	// 					(uint)( _productionTimeLimit * _totalProducedNodes / _totalNodesProductionTime ) ) );
// 	// 	}
// 	// 	cudaEventRecord( _startProductionNodes );
// 	// }
//
// 	// Subdivide
// 	_nbNodeSubdivisionRequests = _nodesCacheManager->genericWriteAsync(
// 		updateCompactList,
// 		numUpdateElems,
// 		/*type of request to handle*/DataProductionManagerKernelType::VTC_REQUEST_SUBDIV,
// 		_maxNbNodeSubdivisions,
// 		numValidNodes,
// 		_dataStructure->_nodePool,
// 		producer,
// 		static_cast<uint>(numElems[0])
// 	);
//
// 	// if ( _lastProductionTimed )
// 	// {
// 	// 	cudaEventRecord( _stopProductionNodes );
// 	// 	cudaEventRecord( _startProductionBricks );
// 	// }
//
// 	if ( _nbNodeSubdivisionRequests < numUpdateElems )
// 	{
// 		// if ( _lastProductionTimed && _totalProducedNodes != 0 && _totalProducedBricks != 0 )
// 		// {
// 		// 	// Evaluate how much time will be left after nodes subdivision
// 		// 	float remainingTime = _productionTimeLimit - _nbNodeSubdivisionRequests * _totalNodesProductionTime / _totalProducedNodes;
// 		// 	// Limit the number of request to fit in the remaining time
// 		// 	numElems[ 1 ] = min(
// 		// 			static_cast< uint >( numElems[ 1 ] ),
// 		// 			max( 100,
// 		// 				(uint)( remainingTime * _totalProducedBricks / _totalBrickProductionTime ) ) );
// 		// }
//
// 		_nbBrickLoadRequests = _bricksCacheManager->genericWriteAsync(
// 			updateCompactList,
// 			numUpdateElems,
// 			/*type of request to handle*/DataProductionManagerKernelType::VTC_REQUEST_LOAD,
// 			_maxNbBrickLoads,
// 			numValidNodes,
// 			_dataStructure->_dataPool,
// 			producer,
// 			static_cast<uint>(numElems[1])
// 		);
// 	}
// }
// #endif

/******************************************************************************
 * Get the max number of requests of node subdivisions.
 *
 * @return the max number of requests
 ******************************************************************************/
template< typename TDataStructure >
uint GvDataProductionManager< TDataStructure >
::getMaxNbNodeSubdivisions() const
{
	return _maxNbNodeSubdivisions;
}

/******************************************************************************
 * Set the max number of requests of node subdivisions.
 *
 * @param pValue the max number of requests
 ******************************************************************************/
template< typename TDataStructure >
void GvDataProductionManager< TDataStructure >
::setMaxNbNodeSubdivisions( uint pValue )
{
	_maxNbNodeSubdivisions = pValue;
}

/******************************************************************************
 * Get the max number of requests of brick of voxel loads.
 *
 * @return the max number of requests
 ******************************************************************************/
template< typename TDataStructure >
uint GvDataProductionManager< TDataStructure >
::getMaxNbBrickLoads() const
{
	return _maxNbBrickLoads;
}

/******************************************************************************
 * Set the max number of requests of brick of voxel loads.
 *
 * @param pValue the max number of requests
 ******************************************************************************/
template< typename TDataStructure >
void GvDataProductionManager< TDataStructure >
::setMaxNbBrickLoads( uint pValue )
{
	_maxNbBrickLoads = pValue;
}

/******************************************************************************
 * Get the number of requests of node subdivisions the cache has handled.
 *
 * @return the number of requests
 ******************************************************************************/
template< typename TDataStructure >
unsigned int GvDataProductionManager< TDataStructure >
::getNbNodeSubdivisionRequests() const
{
	return this->_nbNodeSubdivisionRequests;
}

/******************************************************************************
 * Get the number of requests of brick of voxel loads the cache has handled.
 *
 * @return the number of requests
 ******************************************************************************/
template< typename TDataStructure >
unsigned int GvDataProductionManager< TDataStructure >
::getNbBrickLoadRequests() const
{
	return this->_nbBrickLoadRequests;
}

/******************************************************************************
 * Add a producer
 *
 * @param pProducer the producer to add
 ******************************************************************************/
template< typename TDataStructure >
void GvDataProductionManager< TDataStructure >
::addProducer( ProducerType* pProducer )
{
	assert( pProducer != NULL );

	// TO DO
	// ...
	_producers.push_back( pProducer );
}

/******************************************************************************
 * Remove a producer
 *
 * @param pProducer the producer to remove
 ******************************************************************************/
template< typename TDataStructure >
void GvDataProductionManager< TDataStructure >
::removeProducer( ProducerType* pProducer )
{
	assert( pProducer != NULL );

	// TO DO
	// ...
	assert( false );
}

/******************************************************************************
 * Get the flag telling wheter or not tree data dtructure monitoring is activated
 *
 * @return the flag telling wheter or not tree data dtructure monitoring is activated
 ******************************************************************************/
template< typename TDataStructure >
inline bool GvDataProductionManager< TDataStructure >
::hasTreeDataStructureMonitoring() const
{
	return _hasTreeDataStructureMonitoring;
}

/******************************************************************************
 * Set the flag telling wheter or not tree data dtructure monitoring is activated
 *
 * @param pFlag the flag telling wheter or not tree data dtructure monitoring is activated
 ******************************************************************************/
template< typename TDataStructure >
void GvDataProductionManager< TDataStructure >
::setTreeDataStructureMonitoring( bool pFlag )
{
	_hasTreeDataStructureMonitoring = pFlag;
}

/******************************************************************************
 * Get the flag telling wheter or not cache has exceeded its capacity
 *
 * @return flag telling wheter or not cache has exceeded its capacity
 ******************************************************************************/
template< typename TDataStructure >
inline bool GvDataProductionManager< TDataStructure >
::hasCacheExceededCapacity() const
{
	assert( _nodesCacheManager != NULL );
	assert( _bricksCacheManager != NULL );

	return ( _nodesCacheManager->hasExceededCapacity() || _bricksCacheManager->hasExceededCapacity() );
}

/******************************************************************************
 * This method is called to serialize an object
 *
 * @param pStream the stream where to write
 ******************************************************************************/
template< typename TDataStructure >
inline void GvDataProductionManager< TDataStructure >
::write( std::ostream& pStream ) const
{
	// Node cache
	assert( _nodesCacheManager != NULL );
	if ( _nodesCacheManager != NULL )
	{
		_nodesCacheManager->write( pStream );
	}

	// Data cache
	assert( _bricksCacheManager != NULL );
	if ( _bricksCacheManager != NULL )
	{
		_bricksCacheManager->write( pStream );
	}
}

/******************************************************************************
 * This method is called deserialize an object
 *
 * @param pStream the stream from which to read
 ******************************************************************************/
template< typename TDataStructure >
inline void GvDataProductionManager< TDataStructure >
::read( std::istream& pStream )
{
	// Node cache
	_nodesCacheManager->read( pStream );

	// Data cache
	_bricksCacheManager->read( pStream );
}

/******************************************************************************
 * Get the flag telling whether or not the production time limit is activated.
 *
 * @return the flag telling whether or not the production time limit is activated.
 ******************************************************************************/
template< typename TDataStructure >
bool GvDataProductionManager< TDataStructure >::isProductionTimeLimited() const
{
	return _isProductionTimeLimited;
}

/******************************************************************************
 * Set or unset the flag used to tell whether or not the production time is limited.
 *
 * @param pFlag the flag value.
 ******************************************************************************/
template< typename TDataStructure >
void GvDataProductionManager< TDataStructure >::useProductionTimeLimit( bool pFlag )
{
	_isProductionTimeLimited = pFlag;
}

/******************************************************************************
 * Get the time limit actually in use.
 *
 * @return the time limit.
 ******************************************************************************/
template< typename TDataStructure >
float GvDataProductionManager< TDataStructure >::getProductionTimeLimit() const
{
	return _productionTimeLimit;
}

/******************************************************************************
 * Set the time limit for the production.
 *
 * @param pTime the time limit (in ms).
 ******************************************************************************/
template< typename TDataStructure >
void GvDataProductionManager< TDataStructure >::setProductionTimeLimit( float pTime )
{
	_productionTimeLimit = pTime;
}

} // namespace GvStructure
