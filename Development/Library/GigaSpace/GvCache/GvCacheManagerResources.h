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

#ifndef _GV_CACHE_MANAGER_RESOURCES_H_
#define _GV_CACHE_MANAGER_RESOURCES_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Gigavoxels
#include "GvCore/GvCoreConfig.h"

// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#if USE_CUDPP_LIBRARY
// cudpp
#include <cudpp.h>
#endif

// CUDA
#include <helper_math.h>

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
 * @class GvCacheManagerResources
 *
 * @brief The GvCacheManagerResources class provides...
 *
 * @ingroup GvCache
 *
 * ...
 */
class GIGASPACE_EXPORT GvCacheManagerResources
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Initialize
	 */
	static void initialize();

	/**
	 * Finalize
	 */
	static void finalize();

	/**
	 * Get the temp usage mask1
	 *
	 * @param pSize ...
	 *
	 * @return ...
	 */
	static thrust::device_vector< uint >* getTempUsageMask1( size_t pSize );

	/**
	 * Get the temp usage mask2
	 *
	 * @param pSize ...
	 *
	 * @return ...
	 */
	static thrust::device_vector< uint >* getTempUsageMask2( size_t pSize );

#if USE_CUDPP_LIBRARY
	/**
	 * Get a CUDPP plan given a number of elements to be processed.
	 *
	 * @param pSize The maximum number of elements to be processed
	 *
	 * @return a handle on the plan
	 */
	static CUDPPHandle getScanplan( uint pSize );

	/**
	 * Get a CUDPP plan given a number of elements to be processed.
	 *
	 * @param pSize The maximum number of elements to be processed
	 *
	 * @return a handle on the plan
	 */
	static CUDPPHandle getCudppLibrary();
#endif

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Temp usage mask1
	 */
	static thrust::device_vector< uint >* _d_tempUsageMask1;

	/**
	 * Temp usage mask2
	 */
	static thrust::device_vector< uint >* _d_tempUsageMask2;

#if USE_CUDPP_LIBRARY
	/**
	 * Handle on an instance of the CUDPP library.
	 */
	static CUDPPHandle _cudppLibrary;

	/**
	 * Scan plan.
	 * A plan is a data structure containing state and intermediate storage space that CUDPP uses to execute algorithms on data.
	 */
	static CUDPPHandle _scanplan;

	/**
	 * Scan plan size, i.e. the maximum number of elements to be processed
	 */
	static uint _scanPlanSize;
#endif

};

} //namespace GvCache

#endif
