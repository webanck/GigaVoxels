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

// GigaVoxels
#include "GvCore/GvError.h"

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvCache
{

/******************************************************************************
 * This method is a helper for writing into the cache.
 *
 * @param pNumElements The number of elements we need to produce and write.
 * @param pNodesAddressList The numElements nodes concerned by the production.
 * @param pElemAddressList The numElements addresses of the new elements.
 * @param pGpuPool The pool where we will write the produced elements.
 * @param pGpuProvider The provider called for the production.
 * @param pPageTable 
 * @param pBlockSize The user defined blockSize used to launch the kernel.
 ******************************************************************************/
template< typename ElementRes, typename GPUPoolType, typename GPUProviderType, typename PageTableType >
inline void GvCacheHelper::genericWriteIntoCache( const uint pNumElements, uint* pNodesAddressList, uint* pElemAddressList,
												  const GPUPoolType& pGpuPool, const GPUProviderType& pGpuProvider,
												  const PageTableType& pPageTable, const dim3& pBlockSize )
{
	// Define kernel grid size
	dim3 gridSize( std::min( pNumElements, 65535U ), iDivUp( pNumElements, 65535U ), 1 );

	// Launch kernel
	GvKernel_genericWriteIntoCache< ElementRes >
		<<< gridSize, pBlockSize >>>
		( pNumElements, pNodesAddressList, pElemAddressList, pGpuPool->getKernelPool(), pGpuProvider, pPageTable->getKernel() );

	GV_CHECK_CUDA_ERROR( "GvKernel_genericWriteIntoCache" );
}

} // namespace GvCache
