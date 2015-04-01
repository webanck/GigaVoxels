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

#ifndef _GV_CACHE_HELPER_H_
#define _GV_CACHE_HELPER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaSpace
#include "GvCore/GvCoreConfig.h"
#include "GvCore/vector_types_ext.h"

// Cuda
#include <vector_types.h>

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
 * @class GvCacheHelper
 *
 * @brief The GvCacheHelper class encapsulates calls to kernel launch to produce data on device
 *
 * @ingroup GvCache
 */
class GIGASPACE_EXPORT GvCacheHelper
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
	template< typename ElementRes, typename GPUPoolType, typename GPUProviderType, typename PageTableType >
	inline void genericWriteIntoCache( uint pNumElements, uint* pNodesAddressList, uint* pElemAddressList,
										const GPUPoolType& pGpuPool, const GPUProviderType& pGpuProvider,
										const PageTableType& pPageTable, const dim3& pBlockSize );

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

} //namespace GvCache

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvCacheHelper.inl"

#endif // !GVCACHEHELPER_H
