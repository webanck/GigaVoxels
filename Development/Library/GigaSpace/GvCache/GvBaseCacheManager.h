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

#ifndef _GV_BASE_CACHE_MANAGER_H_
#define _GV_BASE_CACHE_MANAGER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCache/GvICacheManager.h"
#include "GvCore/Array3DGPULinear.h"
#include "GvCore/vector_types_ext.h"

// cudpp
#include <cudpp.h>

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
 * @class GvBaseCacheManager
 *
 * @brief The GvBaseCacheManager class provides the mecanisms to manage elements in a cache.
 *
 * @ingroup GvCache
 *
 * This class is the base class for all host cache manager.
 *
 * It is the main user entry point to manage elements in a cache.
 */
class GIGASPACE_EXPORT GvBaseCacheManager : public GvICacheManager
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

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

	/******************************** METHODS *********************************/

	/**
	 * Destructor
	 */
	virtual ~GvBaseCacheManager();

	/**
	 * Finalize
	 */
	virtual void finalize();

	/**
	 * Get the number of elements managed by the cache.
	 *
	 * @return the number of elements managed by the cache
	 */
	uint getNbElements() const;

	/**
	 * Clear the cache
	 */
	virtual void clear();

	/**
	 * Update symbols
	 * (variables in constant memory)
	 */
	virtual void updateSymbols();

	/**
	 * Update the list of available elements according to their timestamps.
	 * Unused and recycled elements will be placed first.
	 *
	 * @param manageUpdatesOnly ...
	 *
	 * @return the number of available elements
	 */
	virtual uint updateTimeStamps( bool pManageUpdatesOnly );
	
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
	GvCore::Array3DGPULinear< uint >* getTimestamps() const;

	/**
	 * Get the sorted list of cache elements, least recently used first.
	 * There is as many timestamps as elements in the cache.
	 */
	GvCore::Array3DGPULinear< uint >* getElements() const;

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
	uint _nbElements;

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
	GvCore::Array3DGPULinear< uint >* _timestamps;

	/**
	 * This list contains all elements addresses, sorted correctly so the unused one
	 * are at the beginning.
	 */
	GvCore::Array3DGPULinear< uint >* _elementAddresses;
	GvCore::Array3DGPULinear< uint >* _elementAddressesTmp;	// tmp buffer

	/**
	 * List of elements (with their requests) to process (each element is unique due to compaction processing)
	 */
	GvCore::Array3DGPULinear< uint >* _requests;
	GvCore::Array3DGPULinear< uint >* _requestMasksTmp; // the buffer of masks of valid requests

	/**
	 * Temporary buffers used to store resulting mask list of used and non-used elements
	 * during the current rendering frame
	 */
	GvCore::Array3DGPULinear< uint >* _ununsedElementMasks;
	GvCore::Array3DGPULinear< uint >* _usedElementMasks;

	/**
	 * CUDPP
	 */
	size_t* _d_numElementsPtr;
	CUDPPHandle _scanplan;
	
	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvBaseCacheManager();

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
	GvBaseCacheManager( const GvBaseCacheManager& );

	/**
	 * Copy operator forbidden.
	 */
	GvBaseCacheManager& operator=( const GvBaseCacheManager& );

};

} // namespace GvCache

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvBaseCacheManager.inl"

#endif // !_GV_BASE_CACHE_MANAGER_H_
