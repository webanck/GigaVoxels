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

#ifndef _PRODUCERLOAD_H_
#define _PRODUCERLOAD_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GvIProvider.h>
#include <GvCore/GvIProviderKernel.h>
#include <GvCache/GvCacheHelper.h>
#include <GvCore/Array3D.h>
#include <GvCore/Array3DGPULinear.h>
#include <GvCore/GPUPool.h>

// Project
#include "VolumeProducer.h"
#include "ProducerLoad.hcu"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class ProducerLoad
 *
 * @brief The ProducerLoad class provides...
 *
 * ...
 */
template< typename TDataTList, typename TNodeRes, typename TBrickRes, uint TBorderSize >
class ProducerLoad
	: public GvCore::GvIProvider< 0, ProducerLoad< TDataTList, TNodeRes, TBrickRes, TBorderSize > >
	, public GvCore::GvIProvider< 1, ProducerLoad< TDataTList, TNodeRes, TBrickRes, TBorderSize > >
{

	/**
	 * Type definition of a host pool made of a Array3D and a data type list
	 */
	typedef GvCore::GPUPoolHost< GvCore::Array3D, TDataTList > DataCachePool;

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition of a brick full resolution (including borders)
	 */
	typedef GvCore::StaticRes3D
	<
		TBrickRes::x + 2 * TBorderSize,
		TBrickRes::y + 2 * TBorderSize,
		TBrickRes::z + 2 * TBorderSize
	> BrickFullRes;

	/**
	 * Type definition of the associated device-side producer
	 */
	typedef ProducerLoadKernel
	<
		TDataTList, TNodeRes, BrickFullRes,
		typename DataCachePool::KernelPoolType
	> KernelProducerType;

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructeur
	 *
	 * @param pGpuCacheSize ...
	 * @param pNodesCacheSize ...
	 */
	ProducerLoad( size_t pGpuCacheSize, size_t pNodesCacheSize )
	{
		// Localization Info
		_d_TempLocalizationCodeList	= new thrust::device_vector< GvCore::GvLocalizationInfo::CodeType >( pNodesCacheSize );
		_d_TempLocalizationDepthList = new thrust::device_vector< GvCore::GvLocalizationInfo::DepthType >( pNodesCacheSize );

		// Resolution of one brick, without borders
		uint3 brickRes = TBrickRes::get();
		// Resolution of one brick, including borders
		uint3 brickResWithBorder = brickRes + make_uint3( TBorderSize * 2 );
		// Number of voxels per brick
		//size_t nbVoxelsPerBrick = brickResWithBorder.x * brickResWithBorder.y * brickResWithBorder.z;
		// Number of bytes used by one voxel (i.e sum of the size of each channel_)
		size_t voxelSize = GvCore::DataTotalChannelSize< TDataTList >::value;

		//size_t brickSize = voxelSize * nbVoxelsPerBrick;
		size_t brickSize = voxelSize * KernelProducerType::BrickVoxelAlignment;
		this->_nbMaxRequests = pGpuCacheSize / brickSize;
		this->_bufferNbVoxels = pGpuCacheSize / voxelSize;

		// Allocate caches in mappable pinned memory
		_channelsCachesPool	= new DataCachePool( make_uint3( this->_bufferNbVoxels, 1, 1 ), 2 );

		// Request buffers
		_requestListDepth = new GvCore::GvLocalizationInfo::DepthType[ _nbMaxRequests ];
		_requestListLoc = new GvCore::GvLocalizationInfo::CodeType[ _nbMaxRequests ];

		/// TODO : fix maxRequestNumber * 8
		_h_nodesBuffer = new GvCore::Array3D< uint >( dim3( _nbMaxRequests * 8, 1, 1 ), 2 ); // Allocated mappable pinned memory // TODO: check this size limit

		GV_CHECK_CUDA_ERROR( "GPUVoxelProducerLoadDynamic:GPUVoxelProducerLoadDynamic : end" );
	}

	/**
	 * Attach a producer to a data channel.
	 *
	 * @param pSrcProducer data producer
	 */
	void attachProducer( VolumeProducer< TDataTList >* pSrcProducer )
	{
		// Initialize producer
		pSrcProducer->setRegionResolution( TBrickRes::get() + make_uint3( 2 * TBorderSize ) );

		// Compute max depth based on channel 0 producer
		float3 featureSize = pSrcProducer->getFeaturesSize();
		float minFeatureSize = mincc( featureSize.x, mincc( featureSize.y, featureSize.z ) );

		if ( minFeatureSize > 0.0f )
		{
			uint maxres = static_cast< uint >( ceilf( 1.0f / minFeatureSize ) );
			_maxDepth = getResolutionLevel( make_uint3( maxres ) );
		}
		else
		{
			_maxDepth = 9;	// Limitation from nodes hashes
		}

		_volumeProducer = pSrcProducer;
	}

	/**
	 * This method is called by the cache manager when you have to produce data for a given pool.
	 *
	 * @param pNumElems the number of elements you have to produce.
	 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
	 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
	 * @param pGpuPool the pool for which we need to produce elements.
	 * @param pPageTable the page table associated to the pool
	 */
	template< typename TElementRes, typename TGPUPoolType, typename TPageTableType >
	inline void produceData( uint pNumElems,
		thrust::device_vector< uint >* pNodesAddressCompactList,
		thrust::device_vector< uint >* pElemAddressCompactList,
		TGPUPoolType& pGpuPool, TPageTableType pPageTable, Loki::Int2Type< 0 > )
	{
		_kernelProducer.init( _maxDepth, _h_nodesBuffer->getDeviceArray(), _channelsCachesPool->getKernelPool() );

		GvCore::GvIProviderKernel< 0, KernelProducerType > kernelProvider( _kernelProducer );

		dim3 blockSize( 32, 1, 1 );

		GvCore::GvLocalizationInfo::CodeType* locCodeList = thrust::raw_pointer_cast( &(*_d_TempLocalizationCodeList)[0] );
		GvCore::GvLocalizationInfo::DepthType* locDepthList = thrust::raw_pointer_cast( &(*_d_TempLocalizationDepthList)[0] );

		uint* nodesAddressList = thrust::raw_pointer_cast( &(*pNodesAddressCompactList)[0] );
		uint* elemAddressList = thrust::raw_pointer_cast( &(*pElemAddressCompactList)[0] );

		while ( pNumElems > 0 )
		{
			uint numRequests = mincc( pNumElems, _nbMaxRequests );

			// Create localization lists
			pPageTable->createLocalizationLists( numRequests, nodesAddressList,
												_d_TempLocalizationCodeList, _d_TempLocalizationDepthList );

			preLoadManagementNodes( numRequests, locDepthList, locCodeList );

			// Write into cache
			_cacheHelper.genericWriteIntoCache< TElementRes >( numRequests, nodesAddressList,
												elemAddressList, pGpuPool, kernelProvider, pPageTable, blockSize );

			// Update elements
			pNumElems			-= numRequests;
			nodesAddressList	+= numRequests;
			elemAddressList		+= numRequests;
		}
	}

	/**
	 * This method is called by the cache manager when you have to produce data for a given pool.
	 *
	 * @param pNumElems the number of elements you have to produce.
	 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
	 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
	 * @param pGpuPool the pool for which we need to produce elements.
	 * @param pPageTable the page table associated to the pool
	 */
	template< typename TElementRes, typename TGPUPoolType, typename TPageTableType >
	inline void produceData( uint pNumElems,
		thrust::device_vector< uint >* pNodesAddressCompactList,
		thrust::device_vector< uint >* pElemAddressCompactList,
		TGPUPoolType& pGpuPool, TPageTableType pPageTable, Loki::Int2Type< 1 > )
	{
		_kernelProducer.init( _maxDepth, _h_nodesBuffer->getDeviceArray(), _channelsCachesPool->getKernelPool() );

		GvCore::GvIProviderKernel< 1, KernelProducerType > kernelProvider( _kernelProducer );

		dim3 blockSize( 16, 8, 1 );

		GvCore::GvLocalizationInfo::CodeType* locCodeList = thrust::raw_pointer_cast( &(*_d_TempLocalizationCodeList)[0] );
		GvCore::GvLocalizationInfo::DepthType* locDepthList = thrust::raw_pointer_cast( &(*_d_TempLocalizationDepthList)[0] );

		uint* nodesAddressList = thrust::raw_pointer_cast( &(*pNodesAddressCompactList)[0] );
		uint* elemAddressList = thrust::raw_pointer_cast( &(*pElemAddressCompactList)[0] );

		while ( pNumElems > 0 )
		{
			uint numRequests = mincc( pNumElems, _nbMaxRequests );

			// Create localization lists
			pPageTable->createLocalizationLists( numRequests, nodesAddressList,
				_d_TempLocalizationCodeList, _d_TempLocalizationDepthList );

			preLoadManagementData( numRequests, locDepthList, locCodeList );

			// Write into cache
			_cacheHelper.genericWriteIntoCache< TElementRes >( numRequests, nodesAddressList,
				elemAddressList, pGpuPool, kernelProvider, pPageTable, blockSize );

			// Update elements
			pNumElems			-= numRequests;
			nodesAddressList	+= numRequests;
			elemAddressList		+= numRequests;
		}
	}

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Number of voxels in the transfer buffer
	 */
	size_t _bufferNbVoxels;
	/**
	 * Maximum NUMBER of requests allowed
	 */
	size_t _nbMaxRequests;

	/**
	 * Max depth
	 */
	uint _maxDepth;

	/**
	 * Request buffers. Suppose integer types.
	 */
	GvCore::GvLocalizationInfo::DepthType* _requestListDepth; 
	GvCore::GvLocalizationInfo::CodeType* _requestListLoc;

	/**
	 * Indices cache.
	 * Will be accessed through zero-copy.
	 */
	GvCore::Array3D< uint >* _h_nodesBuffer;

	/**
	 * Channels caches pool
	 */
	 DataCachePool* _channelsCachesPool;

	/**
	 * Channels producers pool
	 */
	//ProducersPool* _volumeProducersPool;
	VolumeProducer< TDataTList >* _volumeProducer;

	/**
	 * Cache helper
	 */
	GvCache::GvCacheHelper _cacheHelper;

	/**
	 * Device-side associated producer
	 */
	KernelProducerType _kernelProducer;

	thrust::device_vector< GvCore::GvLocalizationInfo::CodeType >* _d_TempLocalizationCodeList;
	thrust::device_vector< GvCore::GvLocalizationInfo::DepthType >* _d_TempLocalizationDepthList;
	
	/******************************** METHODS *********************************/

	/**
	 * Prepare nodes info for GPU download.
	 * Takes a device pointer to the request lists containing depth and localization of the nodes.
	 *
	 * @param pNumElements number of elements to process
	 * @param pd_requestListDepth localization depth
	 * @param pd_requestListLoc localization code
	 */
	void preLoadManagementNodes( uint pNumElements,
								GvCore::GvLocalizationInfo::DepthType* pd_requestListDepth,
								GvCore::GvLocalizationInfo::CodeType* pd_requestListLoc )
	{
		// Check limits
		assert( pNumElements <= _nbMaxRequests );

		// TODO: use cudaMemcpyAsync
		CUDAPM_START_EVENT( gpuProdDynamic_preLoadMgtNodes_fetchRequestList )
		cudaMemcpy( _requestListDepth, pd_requestListDepth, pNumElements * sizeof( GvCore::GvLocalizationInfo::DepthType ), cudaMemcpyDeviceToHost );
		cudaMemcpy( _requestListLoc, pd_requestListLoc, pNumElements * sizeof( GvCore::GvLocalizationInfo::CodeType ), cudaMemcpyDeviceToHost );
		GV_CHECK_CUDA_ERROR( "preLoadManagementNodes : cudaMemcpy" );
		CUDAPM_STOP_EVENT( gpuProdDynamic_preLoadMgtNodes_fetchRequestList )

		CUDAPM_START_EVENT(gpuProdDynamic_preLoadMgtNodes_dataLoad)

		// Nodes constantness is based on the producer of the channel 0
		typedef VolumeProducer< TDataTList > ProducerType;
		ProducerType* producer = _volumeProducer;
		if ( producer )
		{
			uint3 brickResWithBorder = TBrickRes::get() + make_uint3( 2 * TBorderSize );

			for ( uint i = 0; i < pNumElements; ++i )
			{
				GvCore::GvLocalizationDepth::ValueType locDepthElem = this->_requestListDepth[ i ].get();// + 1;
				GvCore::GvLocalizationCode::ValueType locCodeElem = this->_requestListLoc[ i ].get();

				uint locDepth = locDepthElem + 1;//+2;

				uint subNodeOffsetIndex = 0;

				// Compute sub nodes info
				uint3 subNodeOffset;
				for ( subNodeOffset.z = 0; subNodeOffset.z < TNodeRes::z; ++subNodeOffset.z )
				{
					for ( subNodeOffset.y = 0; subNodeOffset.y < TNodeRes::y; ++subNodeOffset.y )
					{
						for ( subNodeOffset.x = 0; subNodeOffset.x < TNodeRes::x; ++subNodeOffset.x )
						{
							uint3 locCode = locCodeElem * TNodeRes::get() + subNodeOffset;

							// Convert localization to a region
							float3 regionPos;
							float3 regionSize;
							this->getRegionFromLocalization( locDepth, locCode * TBrickRes::get(), regionPos, regionSize );
							float3 borderSize = this->getBorderSize( locDepth );

							// Get the region node
							uint encodedNodeInfo = producer->getRegionInfoNew( regionPos, regionSize );

							// Constant values are terminal
							if ( ( encodedNodeInfo & 0x40000000 ) == 0 )
							{
								encodedNodeInfo |= 0x80000000;
							}

							// If we reached the maximal depth, set the terminal flag
							if ( locDepth >= _maxDepth )
							{
								encodedNodeInfo |= 0x80000000;
							}

							_h_nodesBuffer->get( i * (uint)TNodeRes::getNumElements() + subNodeOffsetIndex ) = encodedNodeInfo;

							// Increment sub node offset
							subNodeOffsetIndex++;
						}
					}
				}
			}
		}

		CUDAPM_STOP_EVENT( gpuProdDynamic_preLoadMgtNodes_dataLoad )
	}

	/**
	 * Prepare date for GPU download.
	 * Takes a device pointer to the request lists containing depth and localization of the nodes.
	 */
	void preLoadManagementData( uint pNumElements,
								GvCore::GvLocalizationInfo::DepthType* pd_requestListDepth,
								GvCore::GvLocalizationInfo::CodeType* pd_requestListLoc )
	{
		// Check limits
		assert( pNumElements <= _nbMaxRequests );

		// TODO: use cudaMemcpyAsync
		CUDAPM_START_EVENT( gpuProdDynamic_preLoadMgtData_fetchRequestList )
		cudaMemcpy(_requestListDepth, pd_requestListDepth, pNumElements * sizeof( GvCore::GvLocalizationInfo::DepthType ), cudaMemcpyDeviceToHost );
		cudaMemcpy(_requestListLoc, pd_requestListLoc, pNumElements * sizeof( GvCore::GvLocalizationInfo::CodeType ), cudaMemcpyDeviceToHost );
		GV_CHECK_CUDA_ERROR( "preLoadManagementData : cudaMemcpy" );
		CUDAPM_STOP_EVENT( gpuProdDynamic_preLoadMgtData_fetchRequestList )

		CUDAPM_START_EVENT( gpuProdDynamic_preLoadMgtData_dataLoad )
		
		typedef VolumeProducer< TDataTList > ProducerType;
		ProducerType* producer = _volumeProducer;
		if ( producer )
		{
			uint3 brickResWithBorder = TBrickRes::get() + make_uint3( 2 * TBorderSize );

			CUDAPM_START_EVENT( gpuProdDynamic_preLoadMgtData_dataLoad_elemLoop )

			for ( uint i = 0; i < pNumElements; ++i )
			{
				// XXX: Fixed depth offset
				uint locDepth = _requestListDepth[ i ].get();// + 1;
				uint3 locCode = _requestListLoc[ i ].get();

				float3 regionPos;
				float3 regionSize;
				getRegionFromLocalization( locDepth, locCode * TBrickRes::get(), regionPos, regionSize );

				float3 borderSize = getBorderSize( locDepth );

				// Uses statically computed alignment
				uint brickOffset = KernelProducerType::BrickVoxelAlignment;

				producer->getRegion( regionPos, regionSize, _channelsCachesPool, brickOffset * i );
			}

			CUDAPM_STOP_EVENT( gpuProdDynamic_preLoadMgtData_dataLoad_elemLoop )
		}

		CUDAPM_STOP_EVENT( gpuProdDynamic_preLoadMgtData_dataLoad )
	}

	/**
	 * Compute the resolution of a given octree level.
	 *
	 * @param pLevel the level
	 *
	 * @return the resolution
	 */
	uint3 getLevelResolution( uint pLevel )
	{
		return make_uint3( 1 << pLevel ) * TBrickRes::get();
	}

	/**
	 * Compute the octree level corresponding to a given grid resolution.
	 *
	 * @param pResol the resolution
	 *
	 * @return the level
	 */
	uint getResolutionLevel( uint3 pResol )
	{
		uint3 brickGridResol = pResol / TBrickRes::get();

		uint maxBrickGridResol = maxcc( brickGridResol.x, maxcc( brickGridResol.y, brickGridResol.z ) );

		return cclog2( maxBrickGridResol );
	}

	/**
	 * Get region (position and size) from localization
	 *
	 * @param pDepth depth
	 * @param pLocCode localization
	 * @param pRegionPos the resulting region's position
	 * @param pRegionSize the resulting region's size
	 */
	void getRegionFromLocalization( uint pDepth, const uint3& pLocCode, float3& pRegionPos, float3& pRegionSize )
	{
		uint3 levelRes = getLevelResolution( pDepth );
		//std::cout<<"Level res: "<<levelRes<<"\n";
		pRegionPos = make_float3( pLocCode ) / make_float3( levelRes );
		pRegionSize = make_float3( TBrickRes::get() ) / make_float3( levelRes );
	}

	/**
	 * Get border size given a depth
	 *
	 * @param pDepth depth
	 *
	 * @return the border size
	 */
	float3 getBorderSize( uint pDepth )
	{
		uint3 levelRes = getLevelResolution( pDepth );
		return make_float3( TBorderSize ) / make_float3( levelRes );
	}

};

#endif // !_FRACTALPRODUCER_H_
