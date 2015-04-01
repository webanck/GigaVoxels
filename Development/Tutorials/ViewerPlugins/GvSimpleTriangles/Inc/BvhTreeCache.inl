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

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 *
 * @param bvhTree BVH tree
 * @param gpuprod producer
 * @param voltreepoolres ...
 * @param nodetileres nodetile resolution
 * @param brickpoolres brick pool resolution
 * @param brickRes brick resolution
 ******************************************************************************/
template< typename BvhTreeType >
BvhTreeCache< BvhTreeType >
::BvhTreeCache( BvhTreeType* bvhTree, uint3 voltreepoolres, uint3 nodetileres, uint3 brickpoolres, uint3 brickRes )
{
	_bvhTree		= bvhTree;

	_nodePoolRes	= make_uint3( voltreepoolres.x * voltreepoolres.y * voltreepoolres.z, 1, 1);
	_brickPoolRes	= brickpoolres;

	// Node cache initialization
	nodesCacheManager = new NodesCacheManager( _nodePoolRes, nodetileres );
	//nodesCacheManager->setProvider( gpuprod );

	// Brick cache initialization
	bricksCacheManager = new BricksCacheManager( _brickPoolRes, brickRes );
	//bricksCacheManager->setProvider( gpuprod );

	// Request buffer initialization
	d_UpdateBufferArray			= new GvCore::Array3DGPULinear< uint >( _nodePoolRes );
	d_UpdateBufferCompactList	= new thrust::device_vector< uint >( _nodePoolRes.x * _nodePoolRes.y * _nodePoolRes.z );

	totalNumBricksLoaded = 0;
}

/******************************************************************************
 * Pre-render pass
 ******************************************************************************/
template< typename BvhTreeType >
void BvhTreeCache< BvhTreeType >
::preRenderPass()
{
	CUDAPM_START_EVENT( gpucache_preRenderPass );

	// Clear subdiv pool
	d_UpdateBufferArray->fill( 0 );

	updateSymbols();

	CUDAPM_STOP_EVENT( gpucache_preRenderPass );
}

/******************************************************************************
 * Post-render pass
 ******************************************************************************/
template< typename BvhTreeType >
uint BvhTreeCache< BvhTreeType >
::handleRequests()
{
	//updateSymbols();

	CUDAPM_START_EVENT( cache_updateTimestamps );
	updateTimeStamps();
	CUDAPM_STOP_EVENT( cache_updateTimestamps );

	updateSymbols();

	// Collect and compact update informations for both octree and bricks
	CUDAPM_START_EVENT( dataProduction_manageRequests );
	uint numUpdateElems = manageUpdates();
	CUDAPM_STOP_EVENT( dataProduction_manageRequests );

	// Manage the node subdivision requests 
	CUDAPM_START_EVENT( producer_nodes );
	uint numSubDiv = manageSubDivisions( numUpdateElems );
	CUDAPM_STOP_EVENT( producer_nodes );
	//std::cout << "numSubDiv: "<< numSubDiv << "\n";

	// Manage the brick load/produce requests
	CUDAPM_START_EVENT( producer_bricks );
	uint numBrickLoad = 0;
	if ( numSubDiv < numUpdateElems )
	{
		numBrickLoad = manageDataLoadGPUProd( numUpdateElems );
	}
	CUDAPM_STOP_EVENT( producer_bricks );

	//std::cout << "Cache num elems updated: " << numSubDiv + numBrickLoad <<"\n";

	return numSubDiv + numBrickLoad;
}

/******************************************************************************
 * Update all needed symbols in constant memory
 ******************************************************************************/
template< typename BvhTreeType >
void BvhTreeCache< BvhTreeType >
::updateSymbols()
{
	CUDAPM_START_EVENT( gpucache_updateSymbols );

	// Update node cache manager's symbols in constant memory
	nodesCacheManager->updateSymbols();

	// Update brick manager's symbols in constant memory
	bricksCacheManager->updateSymbols();

	// Copy node tile's time stamp buffer
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_VTC_NTTimeStampArray,
		( &nodesCacheManager->getdTimeStampArray()->getDeviceArray() ),
		sizeof( nodesCacheManager->getdTimeStampArray()->getDeviceArray() ), 0, cudaMemcpyHostToDevice ) );

	// Copy brick's time stamp buffer
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_VTC_BTimeStampArray,
		( &bricksCacheManager->getdTimeStampArray()->getDeviceArray() ),
		sizeof( bricksCacheManager->getdTimeStampArray()->getDeviceArray() ), 0, cudaMemcpyHostToDevice ) );

	// Unified
	//
	// Copy request buffer
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_VTC_UpdateBufferArray,
		( &(d_UpdateBufferArray->getDeviceArray()) ),
		sizeof( d_UpdateBufferArray->getDeviceArray() ), 0, cudaMemcpyHostToDevice ) );

	CUDAPM_STOP_EVENT( gpucache_updateSymbols );
}

/******************************************************************************
 * Update time stamps
 ******************************************************************************/
template< typename BvhTreeType >
void BvhTreeCache< BvhTreeType >
::updateTimeStamps()
{
	CUDAPM_START_EVENT( cache_updateTimestamps_dataStructure );
	nodesCacheManager->updateSymbols();
	numNodeTilesNotInUse = nodesCacheManager->updateTimeStamps();
	CUDAPM_STOP_EVENT( cache_updateTimestamps_dataStructure );

	CUDAPM_START_EVENT( cache_updateTimestamps_bricks );
	bricksCacheManager->updateSymbols();
	numBricksNotInUse = bricksCacheManager->updateTimeStamps();
	CUDAPM_STOP_EVENT( cache_updateTimestamps_bricks );
}

/******************************************************************************
 * Manage updates
 *
 * @return ...
 ******************************************************************************/
template< typename BvhTreeType >
uint BvhTreeCache< BvhTreeType >
::manageUpdates()
{
	uint totalNumElems = _nodePoolRes.x * _nodePoolRes.y * _nodePoolRes.z;

	uint numElems = 0;

	CUDAPM_START_EVENT( dataProduction_manageRequests_elemsReduction );

	// Copy current requests and return their total number
	numElems = thrust::copy_if(
		/*input first element*/thrust::device_ptr< uint >( d_UpdateBufferArray->getPointer( 0 ) ),
		/*input last element*/thrust::device_ptr< uint >( d_UpdateBufferArray->getPointer( 0 ) ) + totalNumElems,
		/*output result*/d_UpdateBufferCompactList->begin(), /*predicate*/GvCore::not_equal_to_zero< uint >() ) - d_UpdateBufferCompactList->begin();

	CUDAPM_STOP_EVENT( dataProduction_manageRequests_elemsReduction );

	return numElems;
}

/******************************************************************************
 * Manage the node subdivision requests
 *
 * @param pNumUpdateElems number of elements to process
 *
 * @return ...
 ******************************************************************************/
template< typename BvhTreeType >
uint BvhTreeCache< BvhTreeType >
::manageSubDivisions( uint pNumUpdateElems )
{
	// Buffer of requests
	uint* updateCompactList = thrust::raw_pointer_cast( &(*d_UpdateBufferCompactList)[ 0 ] );
	// numValidNodes = ( nodesCacheManager->totalNumLoads ) * NodeTileRes::getNumElements();

	// Ask node cache maanger to handle "node subdivision" requests
	assert( _producers.size() > 0 );
	assert( _producers[ 0 ] != NULL );
	ProducerType* producer = _producers[ 0 ];
	return nodesCacheManager->genericWrite( updateCompactList, pNumUpdateElems,
							/*mask of node subdivision request*/0x40000000U, 5000, 0xffffffffU/*numValidNodes*/, _bvhTree->_nodePool, producer );
}

/******************************************************************************
 * Manage the brick load/produce requests
 *
 * @param pNumUpdateElems number of elements to process
 *
 * @return ...
 ******************************************************************************/
// Suppose that the subdivision set the node type
template< typename BvhTreeType >
uint BvhTreeCache< BvhTreeType >
::manageDataLoadGPUProd( uint pNumUpdateElems )
{
	// Buffer of requests
	uint* updateCompactList = thrust::raw_pointer_cast( &(*d_UpdateBufferCompactList)[ 0 ] );
	//uint numValidNodes = ( nodesCacheManager->totalNumLoads ) * NodeTileRes::getNumElements();

	// Ask brick cache maanger to handle "brick load/produce" requests
	assert( _producers.size() > 0 );
	assert( _producers[ 0 ] != NULL );
	ProducerType* producer = _producers[ 0 ];
	return bricksCacheManager->genericWrite( updateCompactList, pNumUpdateElems,
		/*mask of brick load/production request*/0x80000000U, 5000, 0xffffffffU/*numValidNodes*/, _bvhTree->_dataPool, producer );
}

/******************************************************************************
 * Add a producer
 *
 * @param pProducer the producer to add
 ******************************************************************************/
template< typename BvhTreeType >
void BvhTreeCache< BvhTreeType >
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
template< typename BvhTreeType >
void BvhTreeCache< BvhTreeType >
::removeProducer( ProducerType* pProducer )
{
	assert( pProducer != NULL );

	// TO DO
	// ...
	assert( false );
}
