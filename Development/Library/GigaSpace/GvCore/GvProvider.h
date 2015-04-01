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

#ifndef _GV_PROVIDER_H_
#define _GV_PROVIDER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/GvIProvider.h"

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
 * @class GvProvider
 *
 * @brief The GvProvider class provides the mecanisms to produce data
 * following data requests emitted during N-tree traversal (rendering).
 *
 * @ingroup GvCore
 *
 * This class is the base class for all host producers.
 *
 * It is the main user entry point to produce data from CPU, for instance,
 * loading data from disk or procedurally generating data.
 */
template< typename TDataStructure, typename TDataProductionManager >
class GvProvider : public GvIProvider
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition of the data structure
	 */
	typedef TDataStructure DataStructureType;

	/**
	 * Type definition of the data structure
	 */
	typedef TDataProductionManager DataProductionManagerType;

	/**
	 * Type definition of the node pool type
	 */
	typedef typename TDataStructure::NodePoolType NodePoolType;

	/**
	 * Type definition of the data pool type
	 */
	typedef typename TDataStructure::DataPoolType DataPoolType;

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Destructor
	 */
	virtual ~GvProvider();

	/**
	 * Initialize
	 *
	 * @param pDataStructure data structure
	 * @param pDataProductionManager data production manager
	 */
	virtual void initialize( TDataStructure* pDataStructure, TDataProductionManager* pDataProductionManager );

	/**
	 * Finalize
	 */
	virtual void finalize();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Reference on the data structure
	 */
	TDataStructure* _dataStructure;

	/**
	 * Reference on the data production manager
	 */
	TDataProductionManager* _dataProductionManager;

	/**
	 * Reference on the node pool
	 */
	NodePoolType* _nodePool;

	/**
	 * Reference on the data pool
	 */
	DataPoolType* _dataPool;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvProvider();

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
	GvProvider( const GvProvider& );

	/**
	 * Copy operator forbidden.
	 */
	GvProvider& operator=( const GvProvider& );

};

} // namespace GvCore

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvProvider.inl"

#endif // !_GV_PROVIDER_H_
