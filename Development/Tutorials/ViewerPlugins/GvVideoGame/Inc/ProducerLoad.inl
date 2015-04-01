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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GvError.h>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 *
 * @param gpuCacheSize gpu cache size
 * @param nodesCacheSize nodes cache size
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
ProducerLoad< DataTList, NodeRes, BrickRes, BorderSize >
::ProducerLoad( size_t gpuCacheSize, size_t nodesCacheSize )
{
	// Resolution of one brick, without borders
	uint3 brickRes = BrickRes::get();
	// Resolution of one brick, including borders
	uint3 brickResWithBorder = brickRes + make_uint3( BorderSize * 2 );
	// Number of voxels per brick
	//size_t nbVoxelsPerBrick = brickResWithBorder.x * brickResWithBorder.y * brickResWithBorder.z;
	// Number of bytes used by one voxel (=sum of the size of each channel_
	size_t voxelSize = GvCore::DataTotalChannelSize< DataTList >::value;

	// size_t brickSize = voxelSize * nbVoxelsPerBrick;
	size_t brickSize = voxelSize * KernelProducerType::BrickVoxelAlignment;
	this->_nbMaxRequests = gpuCacheSize / brickSize;
	this->_bufferNbVoxels = gpuCacheSize / voxelSize;

	// Allocate caches in mappable pinned memory
	_channelsCachesPool	= new DataCachePool( make_uint3( this->_bufferNbVoxels, 1, 1 ), 2 );

	// Localization info initialization (code and depth)
	// This is the ones that producer will have to produce
	_requestListDepth = new GvCore::GvLocalizationInfo::DepthType[ _nbMaxRequests ];
	_requestListLoc = new GvCore::GvLocalizationInfo::CodeType[ _nbMaxRequests ];
	// DEVICE temporary buffers used to retrieve localization info.
	// Data will then be copied in the previous HOST buffers ( _requestListDepth and _requestListLoc) 
	d_TempLocalizationCodeList	= new thrust::device_vector< GvCore::GvLocalizationInfo::CodeType >( nodesCacheSize );
	d_TempLocalizationDepthList	= new thrust::device_vector< GvCore::GvLocalizationInfo::DepthType >( nodesCacheSize );

	// TODO fix maxRequestNumber * 8
	_h_nodesBuffer = new GvCore::Array3D< uint >( dim3( _nbMaxRequests * 8, 1, 1 ), 2 ); // Allocated mappable pinned memory // TODO : check this size limit

	// Check error
	GV_CHECK_CUDA_ERROR( "GPUVoxelProducerLoadDynamic:GPUVoxelProducerLoadDynamic : end" );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
ProducerLoad< DataTList, NodeRes, BrickRes, BorderSize >
::~ProducerLoad()
{
	// TO DO : move the data loader deletion in the producer class
	delete _dataLoader;

	delete _channelsCachesPool;
	delete _requestListDepth;
	delete _requestListLoc;
	delete d_TempLocalizationCodeList;
	delete d_TempLocalizationDepthList;
}

/******************************************************************************
 * Attach a producer to a data channel.
 *
 * @param srcProducer producer
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
void ProducerLoad< DataTList, NodeRes, BrickRes, BorderSize >
::attachProducer( GvUtils::GvIDataLoader< DataTList >* srcProducer )
{
	// Initialize producer
	//srcProducer->setRegionResolution( BrickRes::get() + make_uint3( 2 * BorderSize ) );	// seem to be not used anymore

	// Compute max depth based on channel 0 producer
	float3 featureSize = srcProducer->getFeaturesSize();
	float minFeatureSize = mincc( featureSize.x, mincc( featureSize.y, featureSize.z ) );
	if ( minFeatureSize > 0.0f )
	{
		uint maxres = static_cast< uint >( ceilf( 1.0f / minFeatureSize ) );

		// Compute the octree level corresponding to a given grid resolution.
		_maxDepth = getResolutionLevel( make_uint3( maxres ) );
	}
	else
	{
		_maxDepth = 9;	// Limitation from nodes hashes
	}

	// Store a reference on the producer
	_dataLoader = srcProducer;
}

/******************************************************************************
 * This method is called by the cache manager when you have to produce data for a given pool.
 * Implement the produceData method for the channel 0 (nodes)
 *
 * @param pNumElems the number of elements you have to produce.
 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
 * @param pGpuPool the pool for which we need to produce elements.
 * @param pPageTable the page table associated to the pool
 * @param Loki::Int2Type< 0 > corresponds to the index of the node pool
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
template< typename ElementRes, typename GPUPoolType, typename PageTableType >
inline void ProducerLoad< DataTList, NodeRes, BrickRes, BorderSize >
::produceData( uint numElems,
				thrust::device_vector< uint >* nodesAddressCompactList,
				thrust::device_vector< uint >* elemAddressCompactList,
				GPUPoolType& gpuPool,
				PageTableType pageTable,
				Loki::Int2Type< 0 > )
{
	// Initialize the device-side producer (with the node pool and the brick pool)
	_kernelProducer.init( _maxDepth, _h_nodesBuffer->getDeviceArray(), _channelsCachesPool->getKernelPool() );
	GvCore::GvIProviderKernel< 0, KernelProducerType > kernelProvider( _kernelProducer );

	// Define kernel block size
	// 1D block (warp size)
	dim3 blockSize( 32, 1, 1 );

	// Retrieve localization info
	GvCore::GvLocalizationInfo::CodeType* locCodeList = thrust::raw_pointer_cast( &(*d_TempLocalizationCodeList)[ 0 ] );
	GvCore::GvLocalizationInfo::DepthType* locDepthList = thrust::raw_pointer_cast( &(*d_TempLocalizationDepthList)[ 0 ] );

	// Retrieve elements address lists
	uint* nodesAddressList = thrust::raw_pointer_cast( &(*nodesAddressCompactList)[ 0 ] );
	uint* elemAddressList = thrust::raw_pointer_cast( &(*elemAddressCompactList)[ 0 ] );

	// Iterate through elements (i.e. nodes)
	while ( numElems > 0 )
	{
		// Prevent too workload
		uint numRequests = mincc( numElems, _nbMaxRequests );

		// Create localization info lists of the node elements to produce (code and depth)
		//
		// Resulting lists are written into the two following buffers :
		// - d_TempLocalizationCodeList
		// - d_TempLocalizationDepthList
		pageTable->createLocalizationLists( numRequests, nodesAddressList, d_TempLocalizationCodeList, d_TempLocalizationDepthList );

		// For each node of the lists, thanks to its localization info,
		// an oracle will determine the type the associated 3D region of space
		// (i.e. max depth reached, containing data, etc...)
		//
		// Node info are then written 
		preLoadManagementNodes( numRequests, locDepthList, locCodeList );

		// Call cache helper to write into cache
		//
		// This will then call the associated DEVICE-side producer
		// whose goal is to update the
		_cacheHelper.genericWriteIntoCache< ElementRes >( numRequests, nodesAddressList, elemAddressList, gpuPool, kernelProvider, pageTable, blockSize );

		// Update loop variables
		numElems			-= numRequests;
		nodesAddressList	+= numRequests;
		elemAddressList		+= numRequests;
	}
}

/******************************************************************************
 * This method is called by the cache manager when you have to produce data for a given pool.
 * Implement the produceData method for the channel 1 (bricks)
 *
 * @param pNumElems the number of elements you have to produce.
 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
 * @param pGpuPool the pool for which we need to produce elements.
 * @param pPageTable the page table associated to the pool
 * @param Loki::Int2Type< 1 > corresponds to the index of the brick pool
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
template< typename ElementRes, typename GPUPoolType, typename PageTableType >
inline void ProducerLoad< DataTList, NodeRes, BrickRes, BorderSize >
::produceData( uint numElems,
				thrust::device_vector< uint > *nodesAddressCompactList,
				thrust::device_vector< uint > *elemAddressCompactList,
				GPUPoolType& gpuPool,
				PageTableType pageTable,
				Loki::Int2Type< 1 > )
{
	// Initialize the device-side producer (with the node pool and the brick pool)
	_kernelProducer.init( _maxDepth, _h_nodesBuffer->getDeviceArray(), _channelsCachesPool->getKernelPool() );
	GvCore::GvIProviderKernel< 1, KernelProducerType > kernelProvider( _kernelProducer );

	// Define kernel block size
	dim3 blockSize( 16, 8, 1 );

	// Retrieve localization info
	GvCore::GvLocalizationInfo::CodeType* locCodeList = thrust::raw_pointer_cast( &(*d_TempLocalizationCodeList)[ 0 ] );
	GvCore::GvLocalizationInfo::DepthType* locDepthList = thrust::raw_pointer_cast( &(*d_TempLocalizationDepthList)[ 0 ] );

	// Retrieve elements address lists
	uint* nodesAddressList = thrust::raw_pointer_cast( &(*nodesAddressCompactList)[ 0 ] );
	uint* elemAddressList = thrust::raw_pointer_cast( &(*elemAddressCompactList)[ 0 ] );

	// Iterate through elements
	while ( numElems > 0 )
	{
		uint numRequests = mincc( numElems, _nbMaxRequests );

		// Create localization lists (code and depth)
		pageTable->createLocalizationLists( numRequests, nodesAddressList, d_TempLocalizationCodeList, d_TempLocalizationDepthList );

		// For each brick of the lists, thanks to its localization info,
		// retrieve the associated brick located in this region of space,
		// and load its data from HOST disk (or retrieve data from HOST cache).
		//
		// Voxels data are then written on the DEVICE
		preLoadManagementData( numRequests, locDepthList, locCodeList );

		// Call cache helper to write into cache
		_cacheHelper.genericWriteIntoCache< ElementRes >( numRequests, nodesAddressList, elemAddressList, gpuPool, kernelProvider, pageTable, blockSize );

		// Update loop variables
		numElems			-= numRequests;
		nodesAddressList	+= numRequests;
		elemAddressList		+= numRequests;
	}
}

/******************************************************************************
 * Prepare nodes info for GPU download.
 * Takes a device pointer to the request lists containing depth and localization of the nodes.
 *
 * @param numElements number of elements to process
 * @param d_requestListDepth list of localization depths on the DEVICE
 * @param d_requestListLoc list of localization codes on the DEVICE
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
inline void ProducerLoad< DataTList, NodeRes, BrickRes, BorderSize >
::preLoadManagementNodes( uint numElements, GvCore::GvLocalizationInfo::DepthType* d_requestListDepth, GvCore::GvLocalizationInfo::CodeType* d_requestListLoc )
{
	assert( numElements <= _nbMaxRequests );

	// TODO: use cudaMemcpyAsync
	//
	// Fetch on HOST the updated localization info list (code and depth) of all requested elements
	CUDAPM_START_EVENT( gpuProdDynamic_preLoadMgtNodes_fetchRequestList )
	cudaMemcpy( _requestListDepth, d_requestListDepth, numElements * sizeof( GvCore::GvLocalizationInfo::DepthType ), cudaMemcpyDeviceToHost );
	cudaMemcpy( _requestListLoc, d_requestListLoc, numElements * sizeof( GvCore::GvLocalizationInfo::CodeType ), cudaMemcpyDeviceToHost );
	GV_CHECK_CUDA_ERROR( "preLoadManagementNodes : cudaMemcpy" );
	CUDAPM_STOP_EVENT( gpuProdDynamic_preLoadMgtNodes_fetchRequestList )

	CUDAPM_START_EVENT( gpuProdDynamic_preLoadMgtNodes_dataLoad )

	// Nodes constantness is based on the producer of the channel 0
	typedef GvUtils::GvIDataLoader< DataTList > ProducerType;
	ProducerType* producer = _dataLoader;
	if ( producer )
	{
		uint3 brickResWithBorder = BrickRes::get() + make_uint3( 2 * BorderSize );

		// Iterate through elements (i.e. node tiles)
		for ( uint i = 0; i < numElements; ++i )
		{
			// Get localization info of current element
			GvCore::GvLocalizationDepth::ValueType locDepthElem = this->_requestListDepth[ i ].get();// + 1;
			GvCore::GvLocalizationCode::ValueType locCodeElem = this->_requestListLoc[ i ].get();

			// Localization code of childs is at the next level
			uint locDepth = locDepthElem + 1;//+2;

			// Compute sub nodes info
			// Iterate through each node of the current node tile
			uint3 subNodeOffset;
			uint subNodeOffsetIndex = 0;
			for ( subNodeOffset.z = 0; subNodeOffset.z < NodeRes::z; ++subNodeOffset.z )
			{
				for ( subNodeOffset.y = 0; subNodeOffset.y < NodeRes::y; ++subNodeOffset.y )
				{
					for ( subNodeOffset.x = 0; subNodeOffset.x < NodeRes::x; ++subNodeOffset.x )
					{
						uint3 locCode = locCodeElem * NodeRes::get() + subNodeOffset;

						// Convert localization info to a region of space
						float3 regionPos;
						float3 regionSize;
						this->getRegionFromLocalization( locDepth, locCode * BrickRes::get(), regionPos, regionSize );
						float3 borderSize = this->getBorderSize( locDepth );

						// Retrieve the node located in this region of space,
						// and get its information (i.e. address containing its data type region).
						uint encodedNodeInfo = producer->getRegionInfoNew( regionPos, regionSize );

						// Constant values are terminal
						if ( ( encodedNodeInfo & GV_VTBA_BRICK_FLAG ) == 0 )
						{
							encodedNodeInfo |= GV_VTBA_TERMINAL_FLAG;
						}

						// If we reached the maximal depth, set the terminal flag
						if ( locDepth >= _maxDepth )
						{
							encodedNodeInfo |= GV_VTBA_TERMINAL_FLAG;
						}

						// Write produced data
						_h_nodesBuffer->get( i * static_cast< uint >( NodeRes::getNumElements() ) + subNodeOffsetIndex ) = encodedNodeInfo;

						// Increment sub node offset
						subNodeOffsetIndex++;
					}
				}
			}
		}
	}

	CUDAPM_STOP_EVENT( gpuProdDynamic_preLoadMgtNodes_dataLoad )
}

/******************************************************************************
 * Prepare date for GPU download.
 * Takes a device pointer to the request lists containing depth and localization of the nodes.
 *
 * @param numElements ...
 * @param d_requestListDepth ...
 * @param d_requestListLoc ...
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
inline void ProducerLoad< DataTList, NodeRes, BrickRes, BorderSize >
::preLoadManagementData( uint numElements, GvCore::GvLocalizationInfo::DepthType* d_requestListDepth, GvCore::GvLocalizationInfo::CodeType* d_requestListLoc )
{
	assert( numElements <= _nbMaxRequests );

	// TODO: use cudaMemcpyAsync
	//
	// Fetch on HOST the updated localization info list (code and depth) of all requested elements
	CUDAPM_START_EVENT( gpuProdDynamic_preLoadMgtData_fetchRequestList )
	cudaMemcpy( _requestListDepth, d_requestListDepth, numElements * sizeof( GvCore::GvLocalizationInfo::DepthType ), cudaMemcpyDeviceToHost );
	cudaMemcpy( _requestListLoc, d_requestListLoc, numElements * sizeof( GvCore::GvLocalizationInfo::CodeType ), cudaMemcpyDeviceToHost );
	GV_CHECK_CUDA_ERROR( "preLoadManagementData : cudaMemcpy" );
	CUDAPM_STOP_EVENT( gpuProdDynamic_preLoadMgtData_fetchRequestList )

	CUDAPM_START_EVENT( gpuProdDynamic_preLoadMgtData_dataLoad )
	
	typedef GvUtils::GvIDataLoader< DataTList > ProducerType;
	ProducerType* producer = _dataLoader;
	if ( producer )
	{
		// Compute real brick resolution (i.e. with borders)
		uint3 brickResWithBorder = BrickRes::get() + make_uint3( 2 * BorderSize );

		CUDAPM_START_EVENT( gpuProdDynamic_preLoadMgtData_dataLoad_elemLoop )

		// Iterate through elements (i.e. brick of voxels)
		for ( uint i = 0; i < numElements; ++i )
		{
			// XXX: Fixed depth offset
			uint locDepth = _requestListDepth[ i ].get();// + 1;
			uint3 locCode = _requestListLoc[ i ].get();

			// Convert localization info to a region of space
			float3 regionPos;
			float3 regionSize;
			getRegionFromLocalization( locDepth, locCode * BrickRes::get(), regionPos, regionSize );
			float3 borderSize = getBorderSize( locDepth );

			// Uses statically computed alignment
			uint brickOffset = KernelProducerType::BrickVoxelAlignment;

			// Retrieve the node and associated brick located in this region of space,
			// and depending of its type, if it contains data, load it.
			producer->getRegion( regionPos, regionSize, _channelsCachesPool, brickOffset * i );
		}

		CUDAPM_STOP_EVENT( gpuProdDynamic_preLoadMgtData_dataLoad_elemLoop )
	}

	CUDAPM_STOP_EVENT( gpuProdDynamic_preLoadMgtData_dataLoad )
}

/******************************************************************************
 * Compute the resolution of a given octree level.
 *
 * @param level the given level
 *
 * @return the resolution at the given level
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
inline uint3 ProducerLoad< DataTList, NodeRes, BrickRes, BorderSize >
::getLevelResolution( uint level )
{
	return make_uint3( 1 << level ) * BrickRes::get();
}

/******************************************************************************
 * Compute the octree level corresponding to a given grid resolution.
 *
 * @param resol the given resolution
 *
 * @return the level at the given resolution
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
inline uint ProducerLoad< DataTList, NodeRes, BrickRes, BorderSize >
::getResolutionLevel( uint3 resol )
{
	uint3 brickGridResol = resol / BrickRes::get();

	uint maxBrickGridResol = maxcc( brickGridResol.x, maxcc( brickGridResol.y, brickGridResol.z ) );

	return cclog2( maxBrickGridResol );
}

/******************************************************************************
 * Get the region corresponding to a given localization info (depth and code)
 *
 * @param depth the given localization depth
 * @param locCode the given localization code
 * @param regionPos the returned region position
 * @param regionSize the returned region size
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
inline void ProducerLoad< DataTList, NodeRes, BrickRes, BorderSize >
::getRegionFromLocalization( uint depth, const uint3& locCode, float3& regionPos, float3& regionSize )
{
	// Compute the resolution of a given octree level.
	uint3 levelResolution = getLevelResolution( depth );
	// std::cout << "Level res: " << levelResolution << "\n";

	// Retrieve region position
	regionPos = make_float3( locCode ) / make_float3( levelResolution );

	// Retrieve region size
	regionSize = make_float3( BrickRes::get() ) / make_float3( levelResolution );
}

/******************************************************************************
 * Get the border size at a specified depth
 *
 * @param depth the given depth
 *
 * @return the border size
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
inline float3 ProducerLoad< DataTList, NodeRes, BrickRes, BorderSize >
::getBorderSize( uint pDepth ) 
{
	// Compute the resolution of a given octree level.
	uint3 levelRes = getLevelResolution( pDepth );

	return make_float3( BorderSize ) / make_float3( levelRes );
}
