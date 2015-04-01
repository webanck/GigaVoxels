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

#ifndef _GV_NODE_CACHE_MANAGER_H_
#define _GV_NODE_CACHE_MANAGER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvPerfMon/GvPerformanceMonitor.h"

// CUDA
#include <vector_types.h>

// CUDA SDK
#include <helper_math.h>

// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// GigaVoxels
#include "GvCache/GvCacheManagerKernel.h"
#include "GvCore/Array3D.h"
#include "GvCore/Array3DGPULinear.h"
#include "GvCore/functional_ext.h"
#include "GvCache/GvCacheManagerResources.h"
#include "GvCore/GvISerializable.h"
#include "GvStructure/GvVolumeTreeAddressType.h"
#include "GvCore/StaticRes3D.h"
#include "GvCore/GvPageTable.h"
#include "GvCore/GvLocalizationInfo.h"
#include "GvCore/Array3DKernelLinear.h"

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

namespace GvCache
{

/** 
 * @class GvNodeCacheManager
 *
 * @brief The GvNodeCacheManager class provides mecanisms to handle a cache on device (i.e. GPU)
 *
 * @ingroup GvCache
 *
 * This class is used to manage a cache on the GPU.
 * It is based on a LRU mecanism (Least Recently Used) to get temporal coherency in data.
 *
 * Aide PARAMETRES TEMPLATES :
 * dans VolumeTreeCache.h :
 * - PageTableType == PageTableNodes< ... Array3DKernelLinear< uint > ... > ou PageTableBricks< ... >
 * - GPUProviderType == IProvider< 1, GPUProducer > ou bien avec 0
 *
 * @todo add "virtual" to specific methods
 */
template< typename TDataStructure >
class GvNodeCacheManager : public GvCore::GvISerializable
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
	 * Linear representation of a node tile
	 */
	typedef GvCore::StaticRes3D< NodeTileRes::numElements, 1, 1 > ElementRes;

	/**
	 * Defines the types used to store the localization infos
	 */
	typedef GvCore::Array3DGPULinear< GvCore::GvLocalizationInfo::CodeType > LocCodeArrayType;
	typedef GvCore::Array3DGPULinear< GvCore::GvLocalizationInfo::DepthType > LocDepthArrayType;

	/**
	 * Type of element address
	 */
	typedef GvStructure::VolTreeNodeAddress AddressType;

	// FIXME: StaticRes3D. Need to move the "linearization" of the resolution
	// into the GPUCache so we have the correct values
	/**
	 * Type definition for nodes page table
	 */
	typedef GvCore::PageTableNodes
	<
		NodeTileRes, ElementRes,
		AddressType, GvCore::Array3DKernelLinear< uint >,
			LocCodeArrayType, LocDepthArrayType
	>
	PageTableType;
	
	/**
	 * ... 
	 */
	typedef GvCore::Array3DGPULinear< uint > PageTableArrayType;

	/**
	 * Type definition for the GPU side associated object
	 *
	 * @todo pass this parameter as a template parameter in order to be able to overload this component easily
	 */
	typedef GvCacheManagerKernel< ElementRes, AddressType > KernelType;

	/**
	 * Cache policy
	 */
	enum ECachePolicy
	{
		eDefaultPolicy = 0,
		ePreventReplacingUsedElementsPolicy = 1,
		eSmoothLoadingPolicy = 1 << 1,
		eAllPolicies = ( 1 << 2 ) - 1
	};

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Page table
	 */
	PageTableType* _pageTable;

	/**
	 * Internal counters
	 */
	uint _totalNumLoads;
	uint _lastNumLoads;
	uint _numElemsNotUsed;

	/**
	 * In the GigaSpace engine, Cache management requires to :
	 * - protect "null" reference (element address)
	 * - root nodes in the data structure (i.e. octree, etc...)
	 * So, each array managed by the Cache needs to take care of these particular elements.
	 *
	 * Note : still a bug when too much loading - TODO: check this
	 */
	static const uint _cNbLockedElements;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param pCachesize size of the cache
	 * @param pPageTableArray the array of elements that the cache has to managed
	 * @param pGraphicsInteroperability a flag used to map buffers to OpenGL graphics library
	 */
	GvNodeCacheManager( const uint3& pCachesize, PageTableArrayType* pPageTableArray, uint pGraphicsInteroperability = 0 );

	/**
	 * Destructor
	 */
	virtual ~GvNodeCacheManager();

	/**
	 * Get the number of elements managed by the cache.
	 *
	 * @return the number of elements managed by the cache
	 */
	uint getNumElements() const;

	/**
	 * Get the associated device side object
	 *
	 * @return the associated device side object
	 */
	KernelType getKernelObject();

	/**
	 * Clear the cache
	 */
	void clearCache();

	/**
	 * Update symbols
	 * (variables in constant memory)
	 */
	void updateSymbols();

	/**
	 * Update the list of available elements according to their timestamps.
	 * Unused and recycled elements will be placed first.
	 *
	 * @param manageUpdatesOnly ...
	 *
	 * @return the number of available elements
	 */
	uint updateTimeStamps( bool manageUpdatesOnly );
	
	/**
	 * Main method to launch the production of nodes or bricks 
	 *
	 * @param updateList global buffer of requests of used elements only (node subdivision an brick lod/produce)
	 * @param numUpdateElems ...
	 * @param updateMask Type of request to handle (node subdivision or brick load/produce)
	 * @param maxNumElems Max number of elements to process
	 * @param numValidNodes ...
	 * @param gpuPool pool used to write new produced data inside (node pool or data pool)
	 *
	 * @return the number of produced elements
	 */
	template< typename GPUPoolType, typename TProducerType >
	uint genericWrite( uint* updateList, uint numUpdateElems, uint updateMask,
							uint maxNumElems, uint numValidNodes, const GPUPoolType& gpuPool, TProducerType* pProducer );

	/**
	 * Set the cache policy
	 *
	 * @param pPolicy the cache policy
	 */
	void setPolicy( ECachePolicy pPolicy );

	/**
	 * Get the cache policy
	 *
	 * @return the cache policy
	 */
	ECachePolicy getPolicy() const;

	/**
	 * Get the number of elements managed by the cache.
	 *
	 * @return the number of elements managed by the cache
	 */
	uint getNbUnusedElements() const;

	/**
	 * Get the timestamp list of the cache.
	 * There is as many timestamps as elements in the cache.
	 */
	GvCore::Array3DGPULinear< uint >* getTimeStampList() const;

	/**
	 * Get the sorted list of cache elements, least recently used first.
	 * There is as many timestamps as elements in the cache.
	 */
	thrust::device_vector< uint >* getElementList() const;

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

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Cache size
	 */
	uint3 _cacheSize;

	/**
	 * Cache size for elements
	 */
	uint3 _elemsCacheSize;

	/**
	 * Number of managed elements
	 */
	uint _numElements;

	/**
	 * Cache policy
	 */
	ECachePolicy _policy;

	/**
	 * Timestamp buffer.
	 *
	 * It attaches a 32-bit integer timestamp to each element (node tile or brick) of the pool.
	 * Timestamp corresponds to the time of the current rendering pass.
	 */
	GvCore::Array3DGPULinear< uint >* _d_TimeStampArray;

	/**
	 * This list contains all elements addresses, sorted correctly so the unused one
	 * are at the beginning.
	 */
	thrust::device_vector< uint >* _d_elemAddressList;
	thrust::device_vector< uint >* _d_elemAddressListTmp;	// tmp buffer

	/**
	 * List of elements (with their requests) to process (each element is unique due to compaction processing)
	 */
	thrust::device_vector< uint >* _d_UpdateCompactList;
	thrust::device_vector< uint >* _d_TempUpdateMaskList; // the buffer of masks of valid requests

	/**
	 * Temporary buffers used to store resulting mask list of used and non-used elements
	 * during the current rendering frame
	 */
	thrust::device_vector< uint >* _d_TempMaskList;
	thrust::device_vector< uint >* _d_TempMaskList2; // for cudpp approach

	/**
	 * Reference on the node pool's "child array" or "data array"
	 */
	PageTableArrayType* _d_pageTableArray;

	/**
	 * The associated device side object
	 */
	KernelType _d_cacheManagerKernel;

	/**
	 * CUDPP
	 */
	size_t* _d_numElementsPtr;
	CUDPPHandle _scanplan;

	/******************************** METHODS *********************************/

	/**
	 * Create the "update" list of a given type.
	 *
	 * "Update list" is the list of elements and their associated requests of a given type (node subdivision or brick load/produce)
	 *
	 * @param inputList Buffer of node addresses and their associated requests. First two bits of their 32 bit addresses stores the type of request
	 * @param inputNumElem Number of elements to process
	 * @param testFlag type of request (node subdivision or brick load/produce)
	 *
	 * @return the number of requests of given type
	 */
	uint createUpdateList( uint* inputList, uint inputNumElem, uint testFlag );

	/**
	 * Invalidate elements
	 *
	 * Timestamps are reset to 1 and node addresses to 0 (but not the 2 first flags)
	 *
	 * @param numElems ...
	 * @param numValidPageTableSlots ...
	 */
	void invalidateElements( uint numElems, int numValidPageTableSlots = -1 );

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
	GvNodeCacheManager( const GvNodeCacheManager& );

	/**
	 * Copy operator forbidden.
	 */
	GvNodeCacheManager& operator=( const GvNodeCacheManager& );

};

} // namespace GvCache

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvNodeCacheManager.inl"

#endif
