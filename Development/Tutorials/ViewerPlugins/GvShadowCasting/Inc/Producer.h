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

// GigaVoxels
#include <GvUtils/GvSimpleHostProducer.h>
#include <GvCore/Array3D.h>
#include <GvCore/Array3DGPULinear.h>
#include <GvCore/GPUPool.h>
#include <GvUtils/GvIDataLoader.h>

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
 *
 * @param DataTList Data type list
 * @param NodeRes Node tile resolution
 * @param BrickRes Brick resolution
 * @param BorderSize Border size of bricks
 */
template< typename TDataStructureType, typename TDataProductionManager >
class Producer : public GvUtils::GvSimpleHostProducer< ProducerKernel< TDataStructureType >, TDataStructureType, TDataProductionManager >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition of the inherited parent class
	 */
	typedef GvUtils::GvSimpleHostProducer< ProducerKernel< TDataStructureType >, TDataStructureType, TDataProductionManager > ParentClassType;

	/**
	 * Type definition of the node tile resolution
	 */
	typedef typename TDataStructureType::NodeTileResolution NodeRes;

	/**
	 * Type definition of the brick resolution
	 */
	typedef typename TDataStructureType::BrickResolution BrickRes;

	/**
	 * Enumeration to define the brick border size
	 */
	enum
	{
		BorderSize = TDataStructureType::BrickBorderSize
	};

	/**
	 * Linear representation of a node tile
	 */
	//typedef typename TDataProductionManager::NodeTileResLinear NodeTileResLinear;
	typedef typename ParentClassType::NodeTileResLinear NodeTileResLinear;

	/**
	 * Type definition of the full brick resolution (i.e. with border)
	 */
	typedef typename TDataProductionManager::BrickFullRes BrickFullRes;
	//typedef typename ParentClassType::BrickFullRes BrickFullRes;

	/**
	 * Defines the data type list
	 */
	typedef typename TDataStructureType::DataTypeList DataTList;

	/**
	 * Type definition of a data cache pool
	 */
	typedef GvCore::GPUPoolHost< GvCore::Array3D, DataTList > DataCachePool;

	/**
	 * Typedef the kernel part of the producer
	 */
	typedef ProducerKernel< TDataStructureType > KernelProducerType;
	//typedef typename ParentClassType::KernelProducer KernelProducerType;

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param gpuCacheSize gpu cache size
	 * @param nodesCacheSize nodes cache size
	 */
	Producer( size_t gpuCacheSize, size_t nodesCacheSize );

	/**
	 * Destructor
	 */
	virtual ~Producer();

	/**
	 * Initialize
	 *
	 * @param pDataStructure data structure
	 * @param pDataProductionManager data production manager
	 */
	virtual void initialize( TDataStructureType* pDataStructure, TDataProductionManager* pDataProductionManager );

	/**
	 * Finalize
	 */
	virtual void finalize();
	
	/**
	 * This method is called by the cache manager when you have to produce data for a given pool.
	 * Implement the produceData method for the channel 0 (nodes)
	 *
	 * @param pNumElems the number of elements you have to produce.
	 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
	 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
	 * @param pGpuPool the pool for which we need to produce elements.
	 * @param pPageTable the page table associated to the pool
	 * @param Loki::Int2Type< 0 > corresponds to the index of the node pool
	 */
	inline virtual void produceData( uint pNumElems,
									thrust::device_vector< uint >* pNodesAddressCompactList,
									thrust::device_vector< uint >* pElemAddressCompactList,
									Loki::Int2Type< 0 > );
	
	/**
	 * This method is called by the cache manager when you have to produce data for a given pool.
	 * Implement the produceData method for the channel 1 (bricks)
	 *
	 * @param pNumElems the number of elements you have to produce.
	 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
	 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
	 * @param pGpuPool the pool for which we need to produce elements.
	 * @param pPageTable the page table associated to the pool
	 * @param Loki::Int2Type< 1 > corresponds to the index of the brick pool
	 */
	inline virtual void produceData( uint pNumElems,
									thrust::device_vector< uint >* pNodesAddressCompactList,
									thrust::device_vector< uint >* pElemAddressCompactList,
									Loki::Int2Type< 1 > );

	/**
	 * Attach a producer to a data channel.
	 *
	 * @param srcProducer producer
	 */
	void attachProducer( GvUtils::GvIDataLoader< DataTList >* srcProducer );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

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
	 * Localization depth's list of nodes that producer has to produce
	 *
	 * Requests buffer. (note : suppose integer types)
	 */
	GvCore::GvLocalizationInfo::DepthType* _requestListDepth; 

	/**
	 * Localization code's list of nodes that producer has to produce
	 *
	 * Requests buffer. (note : suppose integer types)
	 */
	GvCore::GvLocalizationInfo::CodeType* _requestListLoc;

	/**
	 * Indices cache.
	 * Will be accessed through zero-copy.
	 *
	 * HOST producer store a buffer with nodes address that is used on its associated DEVICE-side object
	 * It corresponds to the childAddress of an GvStructure::OctreeNode.
	 */
	GvCore::Array3D< uint >* _h_nodesBuffer;

	/**
	 * Channels caches pool
	 *
	 * This is where all data reside for each channel (color, normal, etc...)
	 * HOST producer store a brick pool with voxel data that is used on its associated DEVICE-side object
	 */
	DataCachePool* _channelsCachesPool;

	/**
	 * Channels producers pool
	 */
	GvUtils::GvIDataLoader< DataTList >* _dataLoader;

	/******************************** METHODS *********************************/

	/**
	 * Prepare nodes info for GPU download.
	 * Takes a device pointer to the request lists containing depth and localization of the nodes.
	 *
	 * @param numElements number of elements
	 * @param d_requestListDepth ...
	 * @param d_requestListLoc ...
	 */
	inline void preLoadManagementNodes( uint numElements, GvCore::GvLocalizationInfo::DepthType* d_requestListDepth, GvCore::GvLocalizationInfo::CodeType* d_requestListLoc );

	/**
	 * Prepare date for GPU download.
	 * Takes a device pointer to the request lists containing depth and localization of the nodes.
	 *
	 * @param numElements ...
	 * @param d_requestListDepth ...
	 * @param d_requestListLoc ...
	 */
	inline void preLoadManagementData( uint numElements, GvCore::GvLocalizationInfo::DepthType* d_requestListDepth, GvCore::GvLocalizationInfo::CodeType* d_requestListLoc );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Helper buffer used to retrieve the list of localization code that the producer has to produce.
	 *
	 * Data in this temporary buffer is then copied in the HOST producer's _requestListLoc member.
	 */
	thrust::device_vector< GvCore::GvLocalizationInfo::CodeType >* d_TempLocalizationCodeList;

	/**
	 * Helper buffer used to retrieve the list of localization depth that the producer has to produce.
	 *
	 * Data in this temporary buffer is then copied in the HOST producer's _requestListDepth member.
	 */
	thrust::device_vector< GvCore::GvLocalizationInfo::DepthType >* d_TempLocalizationDepthList;

	/******************************** METHODS *********************************/

	/**
	 * Compute the resolution of a given octree level.
	 *
	 * @param level the given level
	 *
	 * @return the resolution at the given level
	 */
	inline uint3 getLevelResolution( uint level );

	/**
	 * Compute the octree level corresponding to a given grid resolution.
	 *
	 * @param resol the given resolution
	 *
	 * @return the level at the given resolution
	 */
	inline uint getResolutionLevel( uint3 resol );

	/**
	 * Get the region corresponding to a given localization info (depth and code)
	 *
	 * @param depth the given localization depth
	 * @param locCode the given localization code
	 * @param regionPos the returned region position
	 * @param regionSize the returned region size
	 */
	inline void getRegionFromLocalization( uint depth, const uint3& locCode, float3& regionPos, float3& regionSize );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "Producer.inl"

#endif // !_PRODUCER_H_
