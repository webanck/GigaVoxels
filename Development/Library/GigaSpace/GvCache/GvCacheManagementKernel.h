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

#ifndef _GV_CACHE_MANAGEMENT_KERNEL_H_
#define _GV_CACHE_MANAGEMENT_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/vector_types_ext.h"
#include "GvStructure/GvVolumeTreeAddressType.h"
#include "GvRendering/GvRendererContext.h"

// Cuda
#include <host_defines.h>

/******************************************************************************
 ***************************** KERNEL DEFINITION ******************************
 ******************************************************************************/

// NOTE :
// can't use a GvCacheManagementKernel.cu file, it does not work cause "constant" is not the same in different compilation units...

namespace GvCache
{

/******************************************************************************
 * GvKernel_NodeCacheManagerFlagTimeStamps kernel
 *
 * This kernel creates the usage mask list of used and non used elements (in current rendering pass) in a single pass
 *
 * @param pCacheManager Cache manager
 * @param pNumElem Number of elememts to process
 * @param pTimeStampsElemAddressList Timestamp buffer list
 * @param pTempMaskList Resulting temporary mask list of non-used elements
 * @param pTempMaskList2 Resulting temporary mask list of used elements
 ******************************************************************************/
__global__
GIGASPACE_EXPORT void GvKernel_NodeCacheManagerFlagTimeStamps( const unsigned int pNbElements
											, const unsigned int* /*__restrict__*/ pSortedElements, const unsigned int* /*__restrict__*/ pTimestamps
											, unsigned int* /*__restrict__*/ pUnusedElementMasks, unsigned int* /*__restrict__*/ pUsedElementMasks );
///**
// * ...
// */
//__global__
///*GIGASPACE_EXPORT*/ void GvKernel_DataCacheManagerFlagTimeStamps( const unsigned int pNbElements
//											, const unsigned int* /*__restrict__*/ pSortedElements, const unsigned int* /*__restrict__*/ pTimestamps
//											, unsigned int* /*__restrict__*/ pUnusedElementMasks, unsigned int* /*__restrict__*/ pUsedElementMasks
//											, const unsigned int pResolution, const unsigned int pPitch );

} // namespace GvCache

namespace GvCache
{

/******************************************************************************
 * GvKernel_NodeCacheManagerFlagTimeStamps kernel
 *
 * This kernel creates the usage mask list of used and non used elements (in current rendering pass) in a single pass
 *
 * @param pCacheManager Cache manager
 * @param pNumElem Number of elememts to process
 * @param pTimeStampsElemAddressList Timestamp buffer list
 * @param pTempMaskList Resulting temporary mask list of non-used elements
 * @param pTempMaskList2 Resulting temporary mask list of used elements
 ******************************************************************************/
__global__
// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
void GvKernel_NodeCacheManagerFlagTimeStamps( const unsigned int pNbElements
											, const unsigned int* /*__restrict__*/ pSortedElements, const unsigned int* /*__restrict__*/ pTimestamps
											, unsigned int* /*__restrict__*/ pUnusedElementMasks, unsigned int* /*__restrict__*/ pUsedElementMasks )
{
	// Retrieve global data index
	const unsigned int lineSize = __uimul( blockDim.x, gridDim.x );
	const unsigned int index = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );

	// Check bounds
	if ( index < pNbElements )
	{
		// Retrieve element processed by current thread
		const unsigned int elementIndex = pSortedElements[ index ];

		// Check element's timestamp and set associated masks accordingly
		if ( pTimestamps[ elementIndex ] == k_currentTime )
		{
			pUnusedElementMasks[ index ] = 0;
			pUsedElementMasks[ index ] = 1;
		}
		else
		{
			pUnusedElementMasks[ index ] = 1;
			pUsedElementMasks[ index ] = 0;
		}
	}
}

///**
// * ...
// */
//__global__
//void GvKernel_DataCacheManagerFlagTimeStamps( const unsigned int pNbElements
//											, const unsigned int* /*__restrict__*/ pSortedElements, const unsigned int* /*__restrict__*/ pTimestamps
//											, unsigned int* /*__restrict__*/ pUnusedElementMasks, unsigned int* /*__restrict__*/ pUsedElementMasks
//											, const unsigned int pResolution, const unsigned int pPitch )
//{
//	// Retrieve global data index
//	const unsigned int lineSize = __uimul( blockDim.x, gridDim.x );
//	const unsigned int index = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );
//
//	/*__shared__ unsigned int resolution;
//	if ( threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 )
//	{
//		resolution = pResolution;
//	}
//	__syncthreads();*/
//
//	// Check bounds
//	if ( index < pNbElements )
//	{
//		// Retrieve element processed by current thread
//		const unsigned int elementPackedAddress = pSortedElements[ index ];
//
//		// Unpack its address
//		const uint3 elemAddress = GvStructure::VolTreeBrickAddress::unpackAddress( elementPackedAddress );
//
//		// Retrieve element processed by current thread
//		const unsigned int elementIndex = elemAddress.x + __uimul( elemAddress.y, /*resolution*/pResolution ) + __uimul( elemAddress.z, /*resolution*/pPitch );
//	//	const unsigned int elementIndex = ( elemAddress.z * 28/*resolution*/ * 4/*sizeof( unsigned int )*/ ) + ( ( elemAddress.y * 28/*resolution*/ ) + elemAddress.x );
//
//		// Check element's timestamp and set associated masks accordingly
//		if ( pTimestamps[ elementIndex ] == k_currentTime )
//		{
//			pUnusedElementMasks[ index ] = 0;
//			pUsedElementMasks[ index ] = 1;
//		}
//		else
//		{
//			pUnusedElementMasks[ index ] = 1;
//			pUsedElementMasks[ index ] = 0;
//		}
//	}
//}

} // namespace GvCache

#endif // !_GV_CACHE_MANAGEMENT_KERNEL_H_
