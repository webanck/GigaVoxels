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

#ifndef _PRODUCER_H_
#define _PRODUCER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Gigavoxels
#include <GvCore/GvIProvider.h>
#include <GvCore/GvIProviderKernel.h>
#include <GvCache/GvCacheHelper.h>

// Project
#include "ProducerKernel.h"

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
 * @class Producer
 *
 * @brief The Producer class provides the mecanisms to produce data
 * following data requests emitted during N-tree traversal (rendering).
 *
 * It is the main user entry point to produce data from CPU, for instance,
 * loading data from disk or procedurally generating data.
 *
 * This class is implements the mandatory functions of the GvIProvider base class.
 */
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
class Producer
	: public GvCore::GvIProvider< 0, Producer< DataTList, NodeRes, BrickRes, BorderSize > >
	, public GvCore::GvIProvider< 1, Producer< DataTList, NodeRes, BrickRes, BorderSize > >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Typedef the kernel part of the producer
	 */
	typedef ProducerKernel< DataTList, NodeRes, BrickRes, BorderSize > KernelProducerType;

	/**
	 * This pool will contains an array for each voxel's field
	 */
	typedef GvCore::GPUPoolHost< GvCore::Array3D, DataTList > BricksPool;
	
	/******************************* ATTRIBUTES *******************************/
	
	/**
	 * Defines the maximum number of requests we can handle in one pass
	 */
	static const uint nbMaxRequests = 128;

	/******************************** METHODS *********************************/

	/**
	 * Constructor.
	 * Initialize all buffers.
	 */
	Producer();
	
	/**
	 * Implement the produceData method for the channel 0 (nodes).
	 * This method is called by the cache manager when you have to produce data for a given pool.
	 *
	 * @param numElems the number of elements you have to produce.
	 * @param nodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
	 * @param elemAddressCompactList a list containing numElems addresses where you need to store the result.
	 * @param gpuPool the pool for which we need to produce elements.
	 * @param pageTable the page table associated to the pool
	 * @param Loki::Int2Type< 0 > id of the channel
	 */
	template< typename ElementRes, typename GPUPoolType, typename PageTableType >
	inline void produceData( uint numElems,
		thrust::device_vector< uint >* nodesAddressCompactList,
		thrust::device_vector< uint >* elemAddressCompactList,
		GPUPoolType& gpuPool, PageTableType pageTable, Loki::Int2Type< 0 > );

	/**
	 * Implement the produceData method for the channel 1 (bricks)
	 * This method is called by the cache manager when you have to produce data for a given pool.
	 *
	 * @param numElems the number of elements you have to produce.
	 * @param nodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
	 * @param elemAddressCompactList a list containing numElems addresses where you need to store the result.
	 * @param gpuPool the pool for which we need to produce elements.
	 * @param pageTable the page table associated to the pool
	 * @param Loki::Int2Type< 0 > id of the channel
	 */
	template< typename ElementRes, typename GPUPoolType, typename PageTableType >
	inline void produceData( uint numElems,
		thrust::device_vector< uint >* nodesAddressCompactList,
		thrust::device_vector< uint >* elemAddressCompactList,
		GPUPoolType& gpuPool, PageTableType pageTable, Loki::Int2Type< 1 > );

	/**
	 * ...
	 */
	bool hasBrickDrawOneSlice() const;
	void setBrickDrawOneSlice( bool pFlag );
	void setBrickPresenceFlags( unsigned int pBrickPresenceFlags[][ 8 ][ 8 ] );
	void clearCache();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Presence flags of points inside a brick
	 */
	unsigned int _presenceFlags[ 8 ][ 8 ][ 8 ];

	/**
	 * ...
	 */
	bool _hasBrickDrawOneSlice;

	/******************************** METHODS *********************************/

	/**
	 * Test if a point is in the unit sphere centered at [0,0,0]
	 *
	 * @param pPoint the point to test
	 *
	 * @return a flag to tell wheter or not the point is in the sphere
	 */
	inline bool isInSphere( const float3& pPoint ) const;

	/**
	 * Produce nodes
	 *
	 * @param numElements ...
	 * @param requestListCodePtr ...
	 * @param requestListDepthPtr ...
	 */
	inline void produceNodes( uint numElements, GvCore::GvLocalizationInfo::CodeType* requestListCodePtr, GvCore::GvLocalizationInfo::DepthType* requestListDepthPtr );

	/**
	 * Produce bricks
	 *
	 * @param numElements ...
	 * @param requestListCodePtr ...
	 * @param requestListDepthPtr ...
	 */
	inline void produceBricks( uint numElements, GvCore::GvLocalizationInfo::CodeType* requestListCodePtr, GvCore::GvLocalizationInfo::DepthType* requestListDepthPtr );

	/**
	 * Helpers
	 *
	 * @param level ...
	 *
	 * @return ...
	 */
	inline uint3 getLevelResolution( uint level );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Cache helper used to write data into cache
	 */
	GvCache::GvCacheHelper cacheHelper;

	/**
	 * Associated device-side producer
	 */
	KernelProducerType kernelProducer;

	/**
	 * ...
	 */
	GvCore::GvLocalizationInfo::CodeType* requestListCode;

	/**
	 * ...
	 */
	GvCore::GvLocalizationInfo::DepthType* requestListDepth;

	/**
	 * ...
	 */
	GvCore::Array3D< uint >* nodesBuffer;

	/**
	 * ...
	 */
	BricksPool* bricksPool;

	/**
	 * ...
	 */
	thrust::device_vector< GvCore::GvLocalizationInfo::CodeType >* requestListCodeDevice;

	/**
	 * ...
	 */
	thrust::device_vector< GvCore::GvLocalizationInfo::DepthType >* requestListDepthDevice;

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "Producer.inl"

#endif // !_PRODUCER_H_
