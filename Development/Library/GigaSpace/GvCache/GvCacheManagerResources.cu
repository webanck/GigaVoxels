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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// System
#include <stdio.h>

// Gigavoxels
#include "GvConfig.h"
#include "GvCache/GvCacheManagerResources.h"

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvCache;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * ...
 */
thrust::device_vector< uint >* GvCacheManagerResources::_d_tempUsageMask1 = NULL;
thrust::device_vector< uint >* GvCacheManagerResources::_d_tempUsageMask2 = NULL;

#if USE_CUDPP_LIBRARY
	/**
	 * ...
	 */
	CUDPPHandle GvCacheManagerResources::_scanplan = 0;
	CUDPPHandle GvCacheManagerResources::_cudppLibrary = 0;

	/**
	 * ...
	 */
	uint GvCacheManagerResources::_scanPlanSize = 0;
#endif

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Initialize
 ******************************************************************************/
void GvCacheManagerResources::initialize()
{
}

/******************************************************************************
 * Finalize
 ******************************************************************************/
void GvCacheManagerResources::finalize()
{
	delete _d_tempUsageMask1;
	_d_tempUsageMask1 = NULL;

	delete _d_tempUsageMask2;
	_d_tempUsageMask2 = NULL;

#if USE_CUDPP_LIBRARY
	CUDPPResult result = cudppDestroyPlan( _scanplan );
	if ( CUDPP_SUCCESS != result )
	{
		printf( "Error destroying CUDPPPlan\n" );

		exit( -1 );	// TO DO : remove/replace this exit()...
	}
#endif
}

/******************************************************************************
 * Get the temp usage mask1
 *
 * @param pSize ...
 *
 * @return ...
 ******************************************************************************/
thrust::device_vector< uint >* GvCacheManagerResources::getTempUsageMask1( size_t pSize )
{
	if ( ! _d_tempUsageMask1 )
	{
		_d_tempUsageMask1 = new thrust::device_vector< uint >( pSize );
	}
	else if ( _d_tempUsageMask1->size() < pSize )
	{
		_d_tempUsageMask1->resize( pSize );
	}

	return _d_tempUsageMask1;
}

/******************************************************************************
 * Get the temp usage mask2
 *
 * @param pSize ...
 *
 * @return ...
 ******************************************************************************/
thrust::device_vector< uint >* GvCacheManagerResources::getTempUsageMask2( size_t pSize )
{
	if ( ! _d_tempUsageMask2 )
	{
		_d_tempUsageMask2 = new thrust::device_vector< uint >( pSize );
	}
	else if ( _d_tempUsageMask2->size() < pSize )
	{
		_d_tempUsageMask2->resize( pSize );
	}

	return _d_tempUsageMask2;
}

#if USE_CUDPP_LIBRARY
/******************************************************************************
 * Get a CUDPP plan given a number of elements to be processed.
 *
 * @param pSize The maximum number of elements to be processed
 *
 * @return a handle on the plan
 ******************************************************************************/
CUDPPHandle GvCacheManagerResources::getScanplan( uint pSize )
{
	if ( _scanPlanSize < pSize )
	{
		if ( _scanPlanSize > 0 )
		{
			if ( cudppDestroyPlan( _scanplan ) != CUDPP_SUCCESS )
			{
				printf( "warning: cudppDestroyPlan() failed!\n ");
			}
			
			if ( cudppDestroy( _cudppLibrary ) != CUDPP_SUCCESS )
			{
				printf( "warning: cudppDestroy() failed!\n" );
			}

			_scanPlanSize = 0;
		}

		// Creates an instance of the CUDPP library, and returns a handle.
		cudppCreate( &_cudppLibrary );

		// Create a CUDPP plan.
		//
		// A plan is a data structure containing state and intermediate storage space that CUDPP uses to execute algorithms on data.
		// A plan is created by passing to cudppPlan() a CUDPPConfiguration that specifies the algorithm, operator, datatype, and options.
		// The size of the data must also be passed to cudppPlan(), in the numElements, numRows, and rowPitch arguments.
		// These sizes are used to allocate internal storage space at the time the plan is created.
		// The CUDPP planner may use the sizes, options, and information about the present hardware to choose optimal settings.
		// Note that numElements is the maximum size of the array to be processed with this plan.
		// That means that a plan may be re-used to process (for example, to sort or scan) smaller arrays.
		//
		// ---- configuration struct specifying algorithm and options 
		CUDPPConfiguration config;
		config.op = CUDPP_ADD;
		config.datatype = CUDPP_UINT;
		config.algorithm = CUDPP_COMPACT;
		config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
		// ---- pointer to an opaque handle to the internal plan
		_scanplan = 0;
		// ---- create the CUDPP plan.
		CUDPPResult result = cudppPlan( _cudppLibrary, &_scanplan,
										config,
										pSize,		// The maximum number of elements to be processed
										1,			// The number of rows (for 2D operations) to be processed
										0 );		// The pitch of the rows of input data, in elements

		if ( CUDPP_SUCCESS != result )
		{
			printf( "Error creating CUDPPPlan\n" );
			exit( -1 );			// TO DO : remove this exit and use exception ?
		}

		_scanPlanSize = pSize;
	}

	return _scanplan;
}
/******************************************************************************
 * Get a CUDPP plan given a number of elements to be processed.
 *
 * @param pSize The maximum number of elements to be processed
 *
 * @return a handle on the plan
 ******************************************************************************/
CUDPPHandle GvCacheManagerResources::getCudppLibrary()
{
	return _cudppLibrary;
}
#endif
