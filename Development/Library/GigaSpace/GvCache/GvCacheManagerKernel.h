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

#ifndef _GV_CACHE_MANAGER_KERNEL_H_
#define _GV_CACHE_MANAGER_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <vector_types.h>

// Gigavoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/Array3DKernelLinear.h"
#include "GvRendering/GvRendererContext.h"
#include "GvStructure/GvNode.h"

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
 * @struct GvCacheManagerKernel
 *
 * @brief The GvCacheManagerKernel class provides mecanisms to update usage information of elements
 *
 * @ingroup GvCache
 *
 * GPU side object used to update timestamp usage information of an element (node tile or brick)
 */
template< class ElementRes, class AddressType >
struct GvCacheManagerKernel
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Timestamp buffer.
	 * It holds usage information of elements
	 */
	GvCore::Array3DKernelLinear< uint > _timeStampArray;

	/******************************** METHODS *********************************/

	/**
	 * Update timestamp usage information of an element (node tile or brick)
	 * with current time (i.e. current rendering pass)
	 * given its address in its corresponding pool (node or brick).
	 *
	 * @param pElemAddress The address of the element for which we want to update usage information
	 */
	__device__
	__forceinline__ void setElementUsage( uint pElemAddress );

	/**
	 * Update timestamp usage information of an element (node tile or brick)
	 * with current time (i.e. current rendering pass)
	 * given its address in its corresponding pool (node or brick).
	 *
	 * @param pElemAddress The address of the element on which we want to update usage information
	 */
	__device__
	__forceinline__ void setElementUsage( uint3 pElemAddress );

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

} // namespace GvCache

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvCacheManagerKernel.inl"

/******************************************************************************
 ***************************** KERNEL DEFINITION ******************************
 ******************************************************************************/

namespace GvCache
{

	/******************************************************************************
	 * CacheManagerFlagTimeStampsSP kernel
	 *
	 * This kernel creates the usage mask list of used and non used elements (in current rendering pass) in a single pass
	 *
	 * @param pCacheManager Cache manager
	 * @param pNumElem Number of elememts to process
	 * @param pTimeStampsElemAddressList Timestamp buffer list
	 * @param pTempMaskList Resulting temporary mask list of non-used elements
	 * @param pTempMaskList2 Resulting temporary mask list of used elements
	 ******************************************************************************/
	template< class ElementRes, class AddressType >
	__global__
	// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
	void CacheManagerFlagTimeStampsSP( GvCacheManagerKernel< ElementRes, AddressType > pCacheManager,
									  const uint pNumElem, const uint* pTimeStampsElemAddressList, uint* pTempMaskList, uint* pTempMaskList2 )
	{
		// Retrieve global data index
		const uint lineSize = __uimul( blockDim.x, gridDim.x );
		const uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );

		// Check bounds
		if ( elem < pNumElem )
		{
			// Retrieve element processed by current thread
			const uint elemAddressEnc = pTimeStampsElemAddressList[ elem ];

			// Unpack its address
			const uint3 elemAddress = AddressType::unpackAddress( elemAddressEnc );

			// Generate an error
			if ( pCacheManager._timeStampArray.get( elemAddress ) == k_currentTime )
			{
				pTempMaskList[ elem ] = 0;
				pTempMaskList2[ elem ] = 1;
			}
			else
			{
				pTempMaskList[ elem ] = 1;
				pTempMaskList2[ elem ] = 0;
			}

			/*pTempMaskList[ elem ] = 0;
			pTempMaskList2[ elem ] = 1;*/
		}
	}

	/******************************************************************************
	 * InitElemAddressList kernel
	 *
	 * @param addressList
	 * @param numElems
	 * @param elemsCacheRes
	 ******************************************************************************/
	template< typename AddressType >
	__global__
	// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
	void InitElemAddressList( uint* addressList, uint numElems, uint3 elemsCacheRes )
	{
		uint lineSize = __uimul( blockDim.x, gridDim.x );
		uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );

		if ( elem < numElems )
		{
			uint3 pos;
			pos.x = elem % elemsCacheRes.x;
			pos.y = ( elem / elemsCacheRes.x ) % elemsCacheRes.y;
			pos.z = ( elem / ( elemsCacheRes.x * elemsCacheRes.y ) );

			addressList[ elem - 1 ] = AddressType::packAddress( pos );
		}
	}

/******************************************************************************
 * CacheManagerFlagInvalidations KERNEL
 *
 * Reset the time stamp info of given elements to 1.
 *
 * @param pCacheManager cache manager
 * @param pNumElems number of elements to process
 * @param pSortedElemAddressList input list of elements to process (sorted list with unused elements before used ones)
 ******************************************************************************/
template< class ElementRes, class AddressType >
__global__
// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
void CacheManagerFlagInvalidations( GvCacheManagerKernel< ElementRes, AddressType > pCacheManager,
								   const uint pNumElems, const uint* pSortedElemAddressList )
{
	// Retrieve global index
	const uint lineSize = __uimul( blockDim.x, gridDim.x );
	const uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );

	// Check bounds
	if ( elem < pNumElems )
	{
		// Retrieve element address processed by current thread
		const uint elemAddressEnc = pSortedElemAddressList[ elem ];
		const uint3 elemAddress = AddressType::unpackAddress( elemAddressEnc );

		// Update cache manager element (1 is set to reset timestamp)
		pCacheManager._timeStampArray.set( elemAddress, 1 );
	}
}

/******************************************************************************
 * CacheManagerInvalidatePointers KERNEL
 *
 * Reset all node addresses in the cache to NULL (i.e 0).
 * Only the first 30 bits of address are set to 0, not the 2 first flags.
 *
 * @param pCacheManager cache manager
 * @param pNumElems number of elements to process
 * @param pPageTable page table associated to the cache manager from which elements will be processed
 ******************************************************************************/
template< class ElementRes, class AddressType, class PageTableKernelArrayType >
__global__
// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
void CacheManagerInvalidatePointers( GvCacheManagerKernel< ElementRes, AddressType > pCacheManager,
									const uint pNumElems, PageTableKernelArrayType pPageTable )
{
	// Retrieve global data index
	const uint lineSize = __uimul( blockDim.x, gridDim.x );
	const uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );

	// Check bounds
	if ( elem < pNumElems )
	{
		const uint elementPointer = pPageTable.get( elem );

		// TODO: allow constant values !

		if ( ! AddressType::isNull( elementPointer ) )
		{
			const uint3 elemAddress = AddressType::unpackAddress( elementPointer );
			const uint3 elemCacheSlot = ( elemAddress / ElementRes::get() );

			if ( pCacheManager._timeStampArray.get( elemCacheSlot ) == 1 )
			{
				// Reset the 30 first bits of address to 0, and let the 2 first the same
				pPageTable.set( elem, elementPointer & ~(AddressType::packedMask) );
			}
		}
	}
}

	/******************************************************************************
	 * CacheManagerCreateUpdateMask kernel
	 *
	 * Given a flag indicating a type of request (subdivision or load), and the updated global buffer of requests,
	 * it fills a resulting mask buffer.
	 * In fact, this mask indicates, for each element of the usage list, if it was used in the current rendering pass
	 *
	 * @param pNumElem Number of elements to process
	 * @param pUpdateList Buffer of node addresses updated with subdivision or load requests.
	 * @param pResMask List of resulting usage mask
	 * @param pFlag Request flag : either node subdivision or data load/produce
	 ******************************************************************************/
	__global__
	// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
	void CacheManagerCreateUpdateMask( const uint pNumElem, const uint* pUpdateList, uint* pResMask, const uint pFlag )
	{
		// Retrieve global data index
		const uint lineSize = __uimul( blockDim.x, gridDim.x );
		const uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );

		// Out of bound check
		if ( elem < pNumElem )
		{
			// Retrieve
			const uint elemVal = pUpdateList[ elem ];

			// Compare element value with flag one and write mask value
			if ( elemVal & pFlag )
			{
				pResMask[ elem ] = 1;
			}
			else
			{
				pResMask[ elem ] = 0;
			}
		}
	}

	// Optim
	/******************************************************************************
	 * UpdateBrickUsageFromNodes kernel
	 *
	 * @param numElem
	 * @param nodeTilesAddressList
	 * @param volumeTree
	 * @param gpuCache
	 ******************************************************************************/
	template< class VolTreeKernel, class GPUCacheType >
	__global__
	// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
	void UpdateBrickUsageFromNodes( uint numElem, uint *nodeTilesAddressList, VolTreeKernel volumeTree, GPUCacheType gpuCache )
	{
		uint lineSize = __uimul( blockDim.x, gridDim.x );
		uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );

		if ( elem < numElem )
		{
			uint nodeTileAddressEnc = nodeTilesAddressList[ elem ];
			uint nodeTileAddressNode = nodeTileAddressEnc * VolTreeKernel::NodeResolution::getNumElements();

			for ( uint i = 0; i < VolTreeKernel::NodeResolution::getNumElements(); ++i )
			{
				GvStructure::GvNode node;
				volumeTree.fetchNode( node, nodeTileAddressNode, i );

				if ( node.hasBrick() )
				{
					gpuCache._brickCacheManager.setElementUsage( node.getBrickAddress() );
					//setBrickUsage<VolTreeKernel>(node.getBrickAddress());
				}
			}
		}
	}

	// FIXME: Move this to another place!
	/******************************************************************************
	 * UpdateBrickUsageFromNodes kernel
	 *
	 * @param syntheticBuffer
	 * @param numElems
	 * @param lruElemAddressList
	 * @param elemsCacheSize
	 ******************************************************************************/
	template< typename ElementRes, typename AddressType >
	__global__
	// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
	void SyntheticInfo_Update_DataWrite( uchar4* syntheticBuffer, uint numElems, uint* lruElemAddressList, uint3 elemsCacheSize )
	{
		uint lineSize = __uimul( blockDim.x, gridDim.x );
		uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );

		if ( elem < numElems )
		{
			uint pageIdxEnc = lruElemAddressList[ elem ];
			uint3 pageIdx = AddressType::unpackAddress( pageIdxEnc );
			uint syntheticIdx = pageIdx.x + pageIdx.y * elemsCacheSize.x + pageIdx.z * elemsCacheSize.x * elemsCacheSize.y;
			syntheticBuffer[ syntheticIdx ].w = 1;
		}
	}

	/******************************************************************************
	 * UpdateBrickUsageFromNodes kernel
	 *
	 * @param syntheticBuffer
	 * @param numPageUsed
	 * @param usedPageList
	 * @param elemsCacheSize
	 ******************************************************************************/
	template< typename ElementRes, typename AddressType >
	__global__
	// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
	void SyntheticInfo_Update_PageUsed( uchar4* syntheticBuffer, uint numPageUsed, uint* usedPageList, uint3 elemsCacheSize )
	{
		uint lineSize = __uimul( blockDim.x, gridDim.x );
		uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );

		if ( elem < numPageUsed )
		{
			uint pageIdxEnc = usedPageList[ elem ];
			uint3 pageIdx = AddressType::unpackAddress( pageIdxEnc );
			uint syntheticIdx = pageIdx.x + pageIdx.y * elemsCacheSize.x + pageIdx.z * elemsCacheSize.x * elemsCacheSize.y;
			syntheticBuffer[ syntheticIdx ].x = 1;
		}
	}

} // namespace GvCache

#endif // !_GV_CACHE_MANAGER_KERNEL_H_
