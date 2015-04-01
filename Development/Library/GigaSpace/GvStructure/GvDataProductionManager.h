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

#ifndef _GV_DATA_PRODUCTION_MANAGER_H_
#define _GV_DATA_PRODUCTION_MANAGER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

// System
#include <iostream>

// Cuda
#include <vector_types.h>

// Thrust
#include <thrust/device_vector.h>
#include <thrust/copy.h>

// GigaSpace
#include "GvStructure/GvIDataProductionManager.h"
#include "GvCore/StaticRes3D.h"
#include "GvCore/Array3D.h"
#include "GvCore/Array3DGPULinear.h"
#include "GvCore/RendererTypes.h"
#include "GvCore/GPUPool.h"
#include "GvCore/StaticRes3D.h"
#include "GvCore/GvPageTable.h"
#include "GvCore/GvIProvider.h"
#include "GvRendering/GvRendererHelpersKernel.h"
#include "GvCore/GPUVoxelProducer.h"
#include "GvCore/GvLocalizationInfo.h"
#include "GvCache/GvCacheManager.h"
//#include "GvCache/GvNodeCacheManager.h"
#include "GvPerfMon/GvPerformanceMonitor.h"
#include "GvStructure/GvVolumeTreeAddressType.h"
#include "GvStructure/GvDataProductionManagerKernel.h"

#if USE_CUDPP_LIBRARY
	// cudpp
	#include <cudpp.h>
#endif

// STL
#include <vector>

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

namespace GvStructure
{

/**
 * @struct GsProductionStatistics
 *
 * @brief The GsProductionStatistics struct provides storage for production statistics.
 *
 * Production management can be monitored by storing statistics
 * with the help of the GsProductionStatistics struct.
 */
struct GsProductionStatistics
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Time (or index of pass)
	 */
	uint _frameId;

	/**
	  * Number of nodes
	 */
	uint _nNodes;

	/**
	 * Time to produce nodes
	 */
	float _nodesProductionTime;

	/**
	 * Number of bricks
	 */
	uint _nBricks;

	/**
	 * Time to produce bricks
	 */
	float _bricksProductionTime;

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

	/******************************** METHODS *********************************/

};

/**
 * @class GvDataProductionManager
 *The GvDataProductionManager class provides the concept of cache.
 * @brief The GvDataProductionManager class provides the concept of cache.
 *
 * This class implements the cache mechanism for the VolumeTree data structure.
 * As device memory is limited, it is used to store the least recently used element in memory.
 * It is responsible to handle the data requests list generated during the rendering process.
 * (ray-tracing - N-tree traversal).
 * Request are then sent to producer to load or produced data on the host or on the device.
 *
 * @param TDataStructure The volume tree data structure (nodes and bricks)
 * @param ProducerType The associated user producer (host or device)
 */
template< typename TDataStructure >
class GvDataProductionManager : public GvStructure::GvIDataProductionManager
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition of the node tile resolution
	 */
	typedef typename TDataStructure::NodeTileResolution NodeTileRes;

	/**
	 * Type definition of the full brick resolution (i.e. with border)
	 */
	typedef typename TDataStructure::FullBrickResolution BrickFullRes;

	/**
	 * Linear representation of a node tile
	 */
	typedef GvCore::StaticRes3D< NodeTileRes::numElements, 1, 1 > NodeTileResLinear;

	/**
	 * Defines the types used to store the localization infos
	 */
	typedef GvCore::Array3DGPULinear< GvCore::GvLocalizationInfo::CodeType > LocCodeArrayType;
	typedef GvCore::Array3DGPULinear< GvCore::GvLocalizationInfo::DepthType > LocDepthArrayType;

	// FIXME: StaticRes3D. Need to move the "linearization" of the resolution
	// into the GPUCache so we have the correct values
	/**
	 * Type definition for nodes page table
	 */
	typedef GvCore::PageTableNodes
	<
		NodeTileRes, NodeTileResLinear,
		VolTreeNodeAddress,	GvCore::Array3DKernelLinear< uint >,
			LocCodeArrayType, LocDepthArrayType
	>
	NodePageTableType;

	/**
	 * Type definition for bricks page table
	 */
	typedef GvCore::PageTableBricks
	<
		NodeTileRes,
		VolTreeNodeAddress, GvCore::Array3DKernelLinear< uint >,
		VolTreeBrickAddress, GvCore::Array3DKernelLinear< uint >,
		LocCodeArrayType, LocDepthArrayType
	>
	BrickPageTableType;

	/**
	 * Type definition for the nodes cache manager
	 */
	typedef GvCache::GvCacheManager
	<
		0, NodeTileResLinear, VolTreeNodeAddress, GvCore::Array3DGPULinear< uint >, NodePageTableType
	>
	NodesCacheManager;
	//typedef GvCache::GvNodeCacheManager< TDataStructure > NodesCacheManager;

	/**
	 * Type definition for the bricks cache manager
	 */
	typedef GvCache::GvCacheManager
	<
		1, BrickFullRes, VolTreeBrickAddress, GvCore::Array3DGPULinear< uint >, BrickPageTableType
	>
	BricksCacheManager;

	/**
	 * Type definition for the associated device-side object
	 */
	typedef GvDataProductionManagerKernel
	<
		NodeTileResLinear, BrickFullRes, VolTreeNodeAddress, VolTreeBrickAddress
	>
	DataProductionManagerKernelType;

	/**
	 * Type definition of producers
	 */
	typedef GvCore::GvIProvider ProducerType;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Use brick usage optimization flag
	 *
	 * @todo Describe its use
	 */
	bool _useBrickUsageOptim;

	/**
	 * Intra frame pass flag
	 *
	 * @todo Describe its use
	 */
	bool _intraFramePass;

	/**
	 * Leaf node tracker
	 */
	thrust::device_vector< unsigned int >* _leafNodes;
	thrust::device_vector< float >* _emptyNodeVolume;
	unsigned int _nbLeafNodes;
	unsigned int _nbNodes;
	bool _isRealTimeTreeDataStructureMonitoringEnabled;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param voltree a pointer to the data structure.
	 * @param gpuprod a pointer to the user's producer.
	 * @param nodepoolres the 3d size of the node pool.
	 * @param brickpoolres the 3d size of the brick pool.
	 * @param graphicsInteroperability Graphics interoperability flag to be able to map buffers to graphics interoperability mode
	 */
	GvDataProductionManager( TDataStructure* pDataStructure, uint3 nodepoolres, uint3 brickpoolres, uint graphicsInteroperability = 0 );

	/**
	 * Destructor
	 */
	virtual ~GvDataProductionManager();

	/**
	 * This method is called before the rendering process. We just clear the request buffer.
	 */
	virtual void preRenderPass();

	/**
	 * This method is called after the rendering process. She's responsible for processing requests.
	 *
	 * @return the number of requests processed.
	 *
	 * @todo Check whether or not the inversion call of updateTimeStamps() with manageUpdates() has side effects
	 */
	virtual uint handleRequests();

	/**
	 * This method destroy the current N-tree and clear the caches.
	 */
	virtual void clearCache();

	/**
	 * Get the associated device-side object
	 *
	 * @return The device-side object
	 */
	inline DataProductionManagerKernelType getKernelObject() const;

	/**
	 * Get the update buffer
	 *
	 * @return The update buffer
	 */
	inline GvCore::Array3DGPULinear< uint >* getUpdateBuffer() const;

	/**
	 * Get the nodes cache manager
	 *
	 * @return the nodes cache manager
	 */
	inline const NodesCacheManager* getNodesCacheManager() const;

	/**
	 * Get the bricks cache manager
	 *
	 * @return the bricks cache manager
	 */
	inline const BricksCacheManager* getBricksCacheManager() const;

	/**
	 * Get the nodes cache manager
	 *
	 * @return the nodes cache manager
	 */
	inline NodesCacheManager* editNodesCacheManager();

	/**
	 * Get the bricks cache manager
	 *
	 * @return the bricks cache manager
	 */
	inline BricksCacheManager* editBricksCacheManager();

	/**
	 * Get the max number of requests of node subdivisions the cache has to handle.
	 *
	 * @return the max number of requests
	 */
	inline uint getMaxNbNodeSubdivisions() const;

	/**
	 * Set the max number of requests of node subdivisions the cache has to handle.
	 *
	 * @param pValue the max number of requests
	 */
	void setMaxNbNodeSubdivisions( uint pValue );

	/**
	 * Get the max number of requests of brick of voxel loads  the cache has to handle.
	 *
	 * @return the max number of requests
	 */
	inline uint getMaxNbBrickLoads() const;

	/**
	 * Set the max number of requests of brick of voxel loads the cache has to handle.
	 *
	 * @param pValue the max number of requests
	 */
	void setMaxNbBrickLoads( uint pValue );

	/**
	 * Get the number of requests of node subdivisions the cache has handled.
	 *
	 * @return the number of requests
	 */
	unsigned int getNbNodeSubdivisionRequests() const;

	/**
	 * Get the number of requests of brick of voxel loads the cache has handled.
	 *
	 * @return the number of requests
	 */
	unsigned int getNbBrickLoadRequests() const;

	/**
	 * Add a producer
	 *
	 * @param pProducer the producer to add
	 */
	void addProducer( ProducerType* pProducer );

	/**
	 * Remove a producer
	 *
	 * @param pProducer the producer to remove
	 */
	void removeProducer( ProducerType* pProducer );

	/**
	 * Get the flag telling whether or not tree data structure monitoring is activated
	 *
	 * @return the flag telling whether or not tree data structure monitoring is activated
	 */
	inline bool hasTreeDataStructureMonitoring() const;

	/**
	 * Set the flag telling whether or not tree data structure monitoring is activated
	 *
	 * @param pFlag the flag telling whether or not tree data structure monitoring is activated
	 */
	void setTreeDataStructureMonitoring( bool pFlag );

	/**
	 * Get the flag telling whether or not cache has exceeded its capacity
	 *
	 * @return flag telling whether or not cache has exceeded its capacity
	 */
	bool hasCacheExceededCapacity() const;

	/**
	 * This method is called to serialize an object
	 *
	 * @param pStream the stream where to write
	 */
	virtual void write( std::ostream& pStream ) const;

	/**
	 * This method is called deserialize an object
	 *
	 * @param pStream the stream from which to read
	 */
	virtual void read( std::istream& pStream );

	/**
	 * Get the flag telling whether or not the production time limit is activated.
	 *
	 * @return the flag telling whether or not the production time limit is activated.
	 */
	bool isProductionTimeLimited() const;

	/**
	 * Set or unset the flag used to tell whether or not the production time is limited.
	 *
	 * @param pFlag the flag value.
	 */
	void useProductionTimeLimit( bool pFlag );

	/**
	 * Get the time limit actually in use.
	 *
	 * @return the time limit.
	 */
	float getProductionTimeLimit() const;

	/**
	 * Set the time limit for the production.
	 *
	 * @param pTime the time limit (in ms).
	 */
	void setProductionTimeLimit( float pTime );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	///**
	// * Nodes page table
	// */
	//NodePageTableType* _nodesPageTable;		// Seemed to be not used any more ?

	///**
	// * Bricks page table
	// */
	//BrickPageTableType* _bricksPageTable;	// Seemed to be not used any more ?

	/**
	 * Volume tree data structure
	 */
	TDataStructure* _dataStructure;

	/**
	 * Node pool resolution
	 */
	uint3 _nodePoolRes;

	/**
	 * Brick pool resolution
	 */
	uint3 _brickPoolRes;

	/**
	 * The associated device-side object
	 */
	DataProductionManagerKernelType _dataProductionManagerKernel;

	/**
	 * Update buffer array
	 *
	 * Buffer used to store node addresses updated with node subdivisions and/or load requests
	 */
	GvCore::Array3DGPULinear< uint >* _updateBufferArray;

	/**
	 * Update buffer compact list
	 *
	 * Buffer resulting from the "_updateBufferArray stream compaction" to only keep nodes associated to a request
	 */
	thrust::device_vector< uint >* _updateBufferCompactList;

	/**
	 * Number of node tiles not in use
	 */
	uint _numNodeTilesNotInUse;

	/**
	 * Number of bricks not in used
	 */
	uint _numBricksNotInUse;

	/**
	 * Total number of loaded bricks
	 */
	uint _totalNumBricksLoaded;				// Seemed to be not used any more ?

	/**
	 * Nodes cache manager
	 */
	NodesCacheManager* _nodesCacheManager;

	/**
	 * Bricks cache manager
	 */
	BricksCacheManager* _bricksCacheManager;

	/**
	 * Maximum number of subdivision requests the cache has to handle
	 */
	uint _maxNbNodeSubdivisions;

	/**
	 *  Maximum number of load requests the cache has to handle
	 */
	uint _maxNbBrickLoads;

	/**
	 * Number of subdivision requests the cache has handled
	 */
	uint _nbNodeSubdivisionRequests;

	/**
	 *  Number of load requests the cache has handled
	 */
	uint _nbBrickLoadRequests;

#if USE_CUDPP_LIBRARY
	/**
	 * CUDPP stream compaction parameters to process the requests buffer
	 */
	CUDPPHandle _scanPlan;
	size_t* _d_nbValidRequests;
	GvCore::Array3DGPULinear< uint >* _d_validRequestMasks;
#endif

	/**
	 * List of producers
	 */
	std::vector< ProducerType* > _producers;

	/**
	 * Flag to tell whether or not tree data structure monitoring is activated
	 */
	bool _hasTreeDataStructureMonitoring;

	/**
	 * Events used to measure the production time.
	 */
	cudaEvent_t _startProductionNodes, _stopProductionNodes, _stopProductionBricks, _startProductionBricks;

	/**
	 * Total production time for the bricks/nodes since the start of GigaVoxels.
	 */
	float _totalNodesProductionTime, _totalBrickProductionTime;

	/**
	 * Number of bricks/nodes produced since the start of GigaVoxels.
	 */
	uint _totalProducedBricks, _totalProducedNodes;

	/**
	 * Vector containing statistics about production.
	 */
	std::vector< GsProductionStatistics > _productionStatistics;

	/**
	 * Limit of time we are allowed to spend during production.
	 * This is not an hard limit, the ProductionManager will try to limit the number
	 * of requests according to the mean production time it observed so far.
	 */
	float _productionTimeLimit;

	/**
	 * Flag indicating whether or not the production time is limited.
	 */
	bool _isProductionTimeLimited;
	
	/**
	 * Flag indicating whether or not the last production was timed.
	 * It is used to now if the cudaEvent were correctly initialised.
	 */
	bool _lastProductionTimed;

	/******************************** METHODS *********************************/

	/**
	 * Update time stamps
	 */
	virtual void updateTimeStamps();

	/**
	 * This method gather all requests by compacting the list.
	 *
	 * @return The number of elements in the requests list
	 */
	virtual uint manageUpdates();

	/**
	 * This method handle the subdivisions requests.
	 *
	 * @param numUpdateElems the number of requests available in the buffer (of any kind).
	 *
	 * @return the number of subdivision requests processed.
	 */
	virtual uint manageSubDivisions( uint numUpdateElems );

	/**
	 * This method handle the load requests.
	 *
	 * @param numUpdateElems the number of requests available in the buffer (of any kind).
	 *
	 * @return the number of load requests processed.
	 */
	virtual uint manageDataLoadGPUProd( uint numUpdateElems );

#ifdef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_PRODUCER
	/**
	 * ...
	 */
	virtual void produceData( uint numUpdateElems );
#endif

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvDataProductionManager( const GvDataProductionManager& );

	/**
	 * Copy operator forbidden.
	 */
	GvDataProductionManager& operator=( const GvDataProductionManager& );

};

} // namespace GvStructure

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvDataProductionManager.inl"

#endif
