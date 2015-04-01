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

#ifndef _GV_SIMPLE_HOST_PRODUCER_H_
#define _GV_SIMPLE_HOST_PRODUCER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/GvProvider.h"
#include "GvCache/GvCacheHelper.h"

// Thrust
#include <thrust/device_vector.h>

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

namespace GvUtils
{

//-----------------------------------------------------------------------------
// IDEA : for each HOST class, add a last parameter as its device-side associated class with a default value !!!
// This way, its easier to override and customize the class.
//
// http://www.generic-programming.org/languages/cpp/techniques.php
//-----------------------------------------------------------------------------

/**
 * @class GvSimpleHostProducer
 *
 * @brief The GvSimpleHostProducer class provides the mecanisms to produce data
 * following data requests emitted during N-tree traversal (rendering).
 *
 * It is the main user entry point to produce data from CPU, for instance,
 * loading data from disk or procedurally generating data.
 *
 * This class is implements the mandatory functions of the GvIProvider base class.
 */
template< typename TKernelProducerType, typename TDataStructureType, typename TDataProductionManager >
class GvSimpleHostProducer : public GvCore::GvProvider< TDataStructureType, TDataProductionManager >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Typedef the associated device-side producer
	 */
	typedef TKernelProducerType KernelProducer;

	///**
	// * Typedef the associated device-side producer
	// */
	//typedef TDataStructureType DataStructure;

	/**
	 * Type definition of the node page table
	 */
	typedef typename TDataProductionManager::NodePageTableType NodePageTableType;

	/**
	 * Type definition of the node page table
	 */
	typedef typename TDataProductionManager::BrickPageTableType DataPageTableType;

	/**
	 * Linear representation of a node tile
	 */
	typedef typename TDataProductionManager::NodeTileResLinear NodeTileResLinear;

	/**
	 * Type definition of the full brick resolution (i.e. with border)
	 */
	typedef typename TDataProductionManager::BrickFullRes BrickFullRes;

	/**
	 * Type definition of the node pool type
	 */
	typedef typename TDataStructureType::NodePoolType NodePoolType;

	/**
	 * Type definition of the data pool type
	 */
	typedef typename TDataStructureType::DataPoolType DataPoolType;

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvSimpleHostProducer();

	/**
	 * Destructor
	 */
	virtual ~GvSimpleHostProducer();

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

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Associated device-side producer
	 */
	KernelProducer _kernelProducer;

	/**
	 * Cache helper used to write data into cache
	 */
	GvCache::GvCacheHelper _cacheHelper;

	/**
	 * Type definition of the node page table
	 */
	NodePageTableType* _nodePageTable;

	/**
	 * Type definition of the node page table
	 */
	DataPageTableType* _dataPageTable;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvSimpleHostProducer( const GvSimpleHostProducer& );

	/**
	 * Copy operator forbidden.
	 */
	GvSimpleHostProducer& operator=( const GvSimpleHostProducer& );

};

} // namespace GvUtils

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvSimpleHostProducer.inl"

#endif // !_GV_SIMPLE_HOST_PRODUCER_H_
