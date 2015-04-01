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

#ifndef _GV_I_PROVIDER_H_
#define _GV_I_PROVIDER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/vector_types_ext.h"

// Loki
#include <loki/TypeManip.h>

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

// TO DO
//
// For Fabrice, passing localization info is not good for the API. This is implementation details...

namespace GvCore
{

/** 
 * @class GvIProvider
 *
 * @brief The GvIProvider class provides the mecanisms to produce data
 * following data requests emitted during N-tree traversal (rendering).
 *
 * @ingroup GvCore
 *
 * This class is the base class for all host producers.
 *
 * It is the main user entry point to produce data from CPU, for instance,
 * loading data from disk or procedurally generating data.
 */
class GIGASPACE_EXPORT GvIProvider
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Destructor
	 */
	virtual ~GvIProvider();

	/**
	 * Finalize
	 */
	virtual void finalize();

	/**
	 * This method is called by the cache manager when you have to produce data for a given pool.
	 *
	 * @param pNumElems the number of elements you have to produce.
	 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
	 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
	 */
	virtual void produceData( uint pNumElems,
										thrust::device_vector< uint >* pNodesAddressCompactList,
										thrust::device_vector< uint >* pElemAddressCompactList,
										Loki::Int2Type< 0 > ) = 0;

	/**
	 * This method is called by the cache manager when you have to produce data for a given pool.
	 *
	 * @param pNumElems the number of elements you have to produce.
	 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
	 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
	 */
	virtual void produceData( uint pNumElems,
										thrust::device_vector< uint >* pNodesAddressCompactList,
										thrust::device_vector< uint >* pElemAddressCompactList,
										Loki::Int2Type< 1 > ) = 0;

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvIProvider();

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
	GvIProvider( const GvIProvider& );

	/**
	 * Copy operator forbidden.
	 */
	GvIProvider& operator=( const GvIProvider& );

};

} // namespace GvCore

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvIProvider.inl"

#endif // !_GV_I_PROVIDER_H_
