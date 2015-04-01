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

#ifndef _GV_CACHE_HELPER_KERNEL_H_
#define _GV_CACHE_HELPER_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaSpace
#include "GvCore/GvCoreConfig.h"
#include "GvStructure/GvVolumeTreeAddressType.h"
#include "GvCore/vector_types_ext.h"

/******************************************************************************
 ***************************** KERNEL DEFINITION ******************************
 ******************************************************************************/

namespace GvCache
{

/******************************************************************************
 * KERNEL GvKernel_genericWriteIntoCache
 *
 * This method is a helper for writing into the cache.
 *
 * @param pNumElems The number of elements we need to produce and write.
 * @param pNodesAddressList buffer of element addresses that producer(s) has to process (subdivide or produce/load)
 * @param pElemAddressList buffer of available element addresses in cache where producer(s) can write
 * @param pGpuPool The pool where we will write the produced elements.
 * @param pGpuProvider The provider called for the production.
 * @param pPageTable page table used to retrieve localization information from element addresses
 ******************************************************************************/
template< typename TElementRes, typename TGPUPoolType, typename TGPUProviderType, typename TPageTableType >
__global__
// TO DO : Prolifing / Optimization
// - use "Launch Bounds" feature to profile / optimize code
// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
void GvKernel_genericWriteIntoCache( const uint pNumElems, uint* pNodesAddressList, uint* pElemAddressList,
						    TGPUPoolType pGpuPool, TGPUProviderType pGpuProvider, TPageTableType pPageTable )
{
	// Retrieve global indexes
	const uint elemNum = blockIdx.x;
	const uint processID = threadIdx.x + __uimul( threadIdx.y, blockDim.x ) + __uimul( threadIdx.z, __uimul( blockDim.x, blockDim.y ) );

	// Check bound
	if ( elemNum < pNumElems )
	{
		// Clean the syntax a bit
		typedef typename TPageTableType::ElemAddressType ElemAddressType;

		// Shared Memory declaration
		__shared__ uint nodeAddress;
		__shared__ ElemAddressType elemAddress;
		__shared__ GvCore::GvLocalizationInfo parentLocInfo;

		// Retrieve node address and its localisation info along with new element address
		//
		// Done by only one thread of the kernel
		if ( processID == 0 )
		{
			// Compute node address
			const uint nodeAddressEnc = pNodesAddressList[ elemNum ];
			nodeAddress = GvStructure::VolTreeNodeAddress::unpackAddress( nodeAddressEnc ).x;

			// Compute element address
			const uint elemIndexEnc = pElemAddressList[ elemNum ];
			const ElemAddressType elemIndex = TPageTableType::ElemType::unpackAddress( elemIndexEnc );
			elemAddress = elemIndex * TElementRes::get(); // convert into node address             ===> NOTE : for bricks, the resolution holds the border !!!

			// Get the localization of the current element
			//parentLocInfo = pPageTable.getLocalizationInfo( elemNum );
			parentLocInfo = pPageTable.getLocalizationInfo( nodeAddress );
		}

		// Thread Synchronization
		__syncthreads();

		// Produce data
#ifndef GV_USE_PRODUCTION_OPTIMIZATION_INTERNAL
		// Shared Memory declaration
		__shared__ uint producerFeedback;	// Shared Memory declaration
		producerFeedback = pGpuProvider.produceData( pGpuPool, elemNum, processID, elemAddress, parentLocInfo ); // TODO <= This can't work
		// Thread Synchronization
		__syncthreads();
#else
		// Optimization
		// - remove this synchonization for brick production
		// - let user-defined synchronization barriers in the producer directly
		uint producerFeedback = pGpuProvider.produceData( pGpuPool, elemNum, processID, elemAddress, parentLocInfo );
#endif

		// Note : for "nodes", producerFeedback is un-un-used for the moment

		// Update page table (who manages localisation information)
		//
		// Done by only one thread of the kernel
		if ( processID == 0 )
		{
			pPageTable.setPointer( nodeAddress, elemAddress, producerFeedback );
		}
	}
}

/******************************************************************************
 * KERNEL GvKernel_genericWriteIntoCache_NoSynchronization
 *
 * This method is a helper for writing into the cache.
 *
 * @param pNumElems The number of elements we need to produce and write.
 * @param pNodesAddressList buffer of element addresses that producer(s) has to process (subdivide or produce/load)
 * @param pElemAddressList buffer of available element addresses in cache where producer(s) can write
 * @param pGpuPool The pool where we will write the produced elements.
 * @param pGpuProvider The provider called for the production.
 * @param pPageTable page table used to retrieve localization information from element addresses
 ******************************************************************************/
template< typename TElementRes, typename TGPUPoolType, typename TGPUProviderType, typename TPageTableType >
__global__
// TO DO : Prolifing / Optimization
// - use "Launch Bounds" feature to profile / optimize code
// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
void GvKernel_genericWriteIntoCache_NoSynchronization( const uint pNumElems, uint* pNodesAddressList, uint* pElemAddressList,
						    TGPUPoolType pGpuPool, TGPUProviderType pGpuProvider, TPageTableType pPageTable )
{
	// Retrieve global indexes
	const uint elemNum = blockIdx.x;
	const uint processID = threadIdx.x + __uimul( threadIdx.y, blockDim.x ) + __uimul( threadIdx.z, __uimul( blockDim.x, blockDim.y ) );

	// Check bound
	if ( elemNum < pNumElems )
	{
		// Clean the syntax a bit
		typedef typename TPageTableType::ElemAddressType ElemAddressType;

		// Shared Memory declaration
		/*__shared__*/ uint nodeAddress;
		__shared__ ElemAddressType elemAddress;
		__shared__ GvCore::GvLocalizationInfo parentLocInfo;

		// Retrieve node address and its localisation info along with new element address
		//
		// Done by only one thread of the kernel
		if ( processID == 0 )
		{
			// Compute node address
			const uint nodeAddressEnc = pNodesAddressList[ elemNum ];
			nodeAddress = GvStructure::VolTreeNodeAddress::unpackAddress( nodeAddressEnc ).x;

			// Compute element address
			const uint elemIndexEnc = pElemAddressList[ elemNum ];
			const ElemAddressType elemIndex = TPageTableType::ElemType::unpackAddress( elemIndexEnc );
			elemAddress = elemIndex * TElementRes::get(); // convert into node address             ===> NOTE : for bricks, the resolution holds the border !!!

			// Get the localization of the current element
			//parentLocInfo = pPageTable.getLocalizationInfo( elemNum );
			parentLocInfo = pPageTable.getLocalizationInfo( nodeAddress );
		}

		// Produce data
		uint producerFeedback = pGpuProvider.produceData( pGpuPool, elemNum, processID, elemAddress, parentLocInfo );

		// Update page table (who manages localisation information)
		//
		// Done by only one thread of the kernel
		if ( processID == 0 )
		{
			pPageTable.setPointer( nodeAddress, elemAddress, producerFeedback );
		}
	}
}

} // namespace GvCache

#endif
