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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvError.h"

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvCore
{

/******************************************************************************
 * Create localization info list (code and depth)
 *
 * Given a list of N nodes, it retrieves their localization info (code + depth)
 *
 * @param pNumElems number of elements to process (i.e. nodes)
 * @param pNodesAddressCompactList a list of nodes from which to retrieve localization info
 * @param pResLocCodeList the resulting localization code array of all requested elements
 * @param pResLocDepthList the resulting localization depth array of all requested elements
 ******************************************************************************/
template< typename NodeTileRes, typename LocCodeArrayType, typename LocDepthArrayType >
inline void PageTable< NodeTileRes, LocCodeArrayType, LocDepthArrayType >
::createLocalizationLists( uint pNumElems, uint* pNodesAddressCompactList,
							thrust::device_vector< GvLocalizationInfo::CodeType >* pResLocCodeList,
							thrust::device_vector< GvLocalizationInfo::DepthType >* pResLocDepthList )
{
	// Set kernel execution configuration
	dim3 blockSize( 64, 1, 1 );
	uint numBlocks = iDivUp( pNumElems, blockSize.x );
	dim3 gridSize = dim3( std::min( numBlocks, 65535U ), iDivUp( numBlocks, 65535U ), 1 );

	// Launch kernel
	//
	// Create the lists containing localization and depth of each element.
	CreateLocalizationLists< NodeTileRes >
			<<< gridSize, blockSize, 0 >>>( /*in*/pNumElems, /*in*/pNodesAddressCompactList,
											/*in*/locCodeArray->getPointer(), /*in*/locDepthArray->getPointer(),
											/*out*/thrust::raw_pointer_cast( &( *pResLocCodeList )[ 0 ] ), 
											/*out*/thrust::raw_pointer_cast( &( *pResLocDepthList )[ 0 ] ) );

	GV_CHECK_CUDA_ERROR( "CreateLocalizationLists" );
}

} // namespace GvCore

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvCore
{

/******************************************************************************
 * Get the associated device-side object
 *
 * @return the associated device-side object
 ******************************************************************************/
template< typename NodeTileRes, typename ElementRes, typename AddressType, typename KernelArrayType, typename LocCodeArrayType, typename LocDepthArrayType >
inline typename PageTableNodes< NodeTileRes, ElementRes, AddressType, KernelArrayType, LocCodeArrayType, LocDepthArrayType >::KernelType& PageTableNodes< NodeTileRes, ElementRes, AddressType, KernelArrayType, LocCodeArrayType, LocDepthArrayType >
::getKernel()
{
	return pageTableKernel;
}
	
} // namespace GvCore

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvCore
{

/******************************************************************************
 * Get the associated device-side object
 *
 * @return the associated device-side object
 ******************************************************************************/
template< typename NodeTileRes, typename ChildAddressType, typename ChildKernelArrayType, typename DataAddressType, typename DataKernelArrayType, typename LocCodeArrayType, typename LocDepthArrayType >
inline typename PageTableBricks< NodeTileRes, ChildAddressType, ChildKernelArrayType, DataAddressType, DataKernelArrayType, LocCodeArrayType, LocDepthArrayType >::KernelType& PageTableBricks< NodeTileRes, ChildAddressType, ChildKernelArrayType, DataAddressType, DataKernelArrayType, LocCodeArrayType, LocDepthArrayType >
::getKernel()
{
	return pageTableKernel;
}
	
} // namespace GvCore
