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

#ifndef _BVH_TREE_CACHE_HELPER_H_
#define _BVH_TREE_CACHE_HELPER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Gigavoxels
#include "BvhTreeCacheHelper.hcu"

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
 * @class GPUCacheHelper
 *
 * @brief The GPUCacheHelper class provides...
 *
 * ...
 */
class BvhTreeCacheHelper
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * This method is a helper for writing into the cache.
	 *
	 * @param pNumElements The number of elements we need to produce and write.
	 * @param pNodesAddressList The numElements nodes concerned by the production.
	 * @param pElemAddressList The numElements addresses of the new elements.
	 * @param pGpuPool The pool where we will write the produced elements.
	 * @param pGpuProvider The provider called for the production.
	 * @param pPageTable 
	 * @param pBlockSize The user defined blockSize used to launch the kernel.
	 */
	template< typename ElementRes, typename BvhTreeType, typename PoolType, typename ProviderType, typename PageTableType >
	void genericWriteIntoCache( uint pNumElements, uint* pNodesAddressList, uint* pElemAddressList,
								const BvhTreeType& bvhTree, const PoolType& pGpuPool, const ProviderType& pGpuProvider,
								const PageTableType& pPageTable, const dim3& pBlockSize )
	{
		// Define kernel grid size
		dim3 gridSize( std::min( pNumElements, 65535U ), iDivUp( pNumElements, 65535U ), 1 );

		// Launch kernel to produce data on device (node subdivision or data load/production)
		GenericWriteIntoCache< ElementRes >
			<<< gridSize, pBlockSize >>>
			( pNumElements, pNodesAddressList, pElemAddressList, bvhTree->getKernelObject(), pGpuPool->getKernelPool(), pGpuProvider, pPageTable/*pPageTable->getKernel()*/ );

		CUT_CHECK_ERROR( "GenericWriteIntoCache" );
	}

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

};

#endif // !_BVH_TREE_CACHE_HELPER_H_
