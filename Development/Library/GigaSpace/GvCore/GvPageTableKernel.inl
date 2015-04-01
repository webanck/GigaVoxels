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
#include "GvStructure/GvNode.h"

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

// TO DO : attention ce KERNEL n'est pas dans un namespace...
/******************************************************************************
 * KERNEL : CreateLocalizationLists
 *
 * Extract localization informations associated with a list of given elements.
 *
 * Special version for gpuProducerLoadCache that transform loccodes to usable ones (warning, loose one bit of depth !)
 *
 * @param pNbElements number of elements to process
 * @param pLoadAddressList List of input node addresses
 * @param pLocCodeList List of localization code coming from the main page table of the data structure and referenced by cache managers (nodes and bricks)
 * @param pLocDepthList List of localization depth coming from the main page table of the data structure and referenced by cache managers (nodes and bricks)
 * @param pResLocCodeList Resulting output localization code list
 * @param pResLocDepthList Resulting output localization depth list
 ******************************************************************************/
template< class NodeTileRes >
__global__
void CreateLocalizationLists( const uint pNbElements, const uint* pLoadAddressList, 
							 const GvCore::GvLocalizationInfo::CodeType* pLocCodeList, const GvCore::GvLocalizationInfo::DepthType* pLocDepthList,
							 GvCore::GvLocalizationInfo::CodeType* pResLocCodeList, GvCore::GvLocalizationInfo::DepthType* pResLocDepthList )
{
	// Retrieve global indexes
	const uint lineSize = __uimul( blockDim.x, gridDim.x );
	const uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );

	// Check bounds
	if ( elem < pNbElements )
	{
		// Retrieve node address and its localisation info
		// along with new element address

		// Retrieve node address
		const uint nodeAddressEnc = pLoadAddressList[ elem ];
		const uint3 nodeAddress = GvStructure::GvNode::unpackNodeAddress( nodeAddressEnc );

		// Retrieve its "node tile" address
		const uint nodeTileAddress = nodeAddress.x / NodeTileRes::getNumElements();

		// Retrieve its "localization info"
		const GvCore::GvLocalizationInfo::CodeType *tileLocCodeEnc = &pLocCodeList[ nodeTileAddress ];

		// Linear offset of the node in the node tile
		const uint linearOffset = nodeAddress.x - ( nodeTileAddress * NodeTileRes::getNumElements() );
		// 3D offset of the node in the node tile
		const uint3 nodeOffset = NodeTileRes::toFloat3( linearOffset );

		const GvCore::GvLocalizationInfo::CodeType nodeLocCodeEnc = tileLocCodeEnc->addLevel< NodeTileRes >( nodeOffset );

		// Write localization info
		pResLocDepthList[ elem ] = pLocDepthList[ nodeTileAddress ];
		pResLocCodeList[ elem ] = nodeLocCodeEnc;
	}
}

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvCore
{

/******************************************************************************
 * This method returns the LocalizationInfo structure associated with the given
 * node address.
 *
 * @param nodeAddress ...
 *
 * @return ...
 ******************************************************************************/
template< typename Derived, typename AddressType, typename KernelArrayType >
__device__
__forceinline__ GvLocalizationInfo PageTableKernel< Derived, AddressType, KernelArrayType >
::getLocalizationInfo( uint nodeAddress ) const
{
	return static_cast< const Derived* >( this )->getLocalizationInfoImpl( nodeAddress );
}

/******************************************************************************
 * This method should...
 *
 * @param elemAddress ...
 * @param elemPointer ...
 * @param flag ...
 ******************************************************************************/
template< typename Derived, typename AddressType, typename KernelArrayType >
__device__
__forceinline__ void PageTableKernel< Derived, AddressType, KernelArrayType >
::setPointer( uint elemAddress, uint3 elemPointer, uint flag )
{
	static_cast< Derived* >( this )->setPointerImpl( elemAddress, elemPointer, flag );
}

} // namespace GvCore

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvCore
{

/******************************************************************************
 * Return the localization info of a node in the node pool
 *
 * @param nodeAddress Address of the node in the node pool
 *
 * @return The localization info of the node
 ******************************************************************************/
template< typename NodeTileRes, typename ElementRes, typename AddressType, typename KernelArrayType, typename LocCodeArrayType, typename LocDepthArrayType >
__device__
__forceinline__ GvLocalizationInfo PageTableNodesKernel< NodeTileRes, ElementRes, AddressType, KernelArrayType, LocCodeArrayType, LocDepthArrayType >
::getLocalizationInfoImpl( uint nodeAddress ) const
{
	// Compute the address of the current node tile (and its offset in the node tile)
	uint nodeTileIndex = nodeAddress / NodeTileRes::getNumElements();
	uint nodeTileAddress = nodeTileIndex * NodeTileRes::getNumElements();
	uint nodeTileOffset = nodeAddress - nodeTileAddress;

	// Compute the node offset (in 3D, in the node tile)
	uint3 nodeOffset = NodeTileRes::toFloat3( nodeTileOffset );

	// Fetch associated localization infos
	GvLocalizationInfo::CodeType parentLocCode = locCodeArray.get( nodeTileIndex );
	GvLocalizationInfo::DepthType parentLocDepth = locDepthArray.get( nodeTileIndex );

	// Localization info initialization
	GvLocalizationInfo locInfo;
	locInfo.locCode = parentLocCode.addLevel< NodeTileRes >( nodeOffset );
	locInfo.locDepth = parentLocDepth;

	return locInfo;
}

/******************************************************************************
 * ...
 *
 * @param elemAddress ...
 * @param elemPointer ...
 * @param flags ...
 ******************************************************************************/
template< typename NodeTileRes, typename ElementRes, typename AddressType, typename KernelArrayType, typename LocCodeArrayType, typename LocDepthArrayType >
__device__
__forceinline__ void PageTableNodesKernel< NodeTileRes, ElementRes, AddressType, KernelArrayType, LocCodeArrayType, LocDepthArrayType >
::setPointerImpl( uint elemAddress, ElemAddressType elemPointer, uint flags )
{
	ElemPackedAddressType packedChildAddress	= childArray.get( elemAddress );
	ElemPackedAddressType packedAddress			= AddressType::packAddress( elemPointer );

	// Update node tile's pointer
	childArray.set( elemAddress,
		( packedChildAddress & 0x40000000 ) | ( packedAddress & 0x3FFFFFFF ) );

	// Compute the address of the current node tile
	uint nodeTileIndex = elemAddress / NodeTileRes::getNumElements();
	uint nodeTileAddress = nodeTileIndex * NodeTileRes::getNumElements();
	uint nodeTileOffset = elemAddress - nodeTileAddress;

	// Compute the node offset
	uint3 nodeOffset = NodeTileRes::toFloat3( nodeTileOffset );

	// Fetch associated localization infos
	GvLocalizationInfo::CodeType parentLocCode = locCodeArray.get( nodeTileIndex );
	GvLocalizationInfo::DepthType parentLocDepth = locDepthArray.get( nodeTileIndex );

	// Compute the address of the new node tile
	uint newNodeTileIndex = elemPointer.x / ElementRes::getNumElements();
	//uint newNodeTileAddress = newNodeTileIndex * ElementRes::getNumElements();	// --> semble ne pas être utilisé ?

	// Update associated localization infos
	GvLocalizationInfo::CodeType newLocCode = parentLocCode.addLevel< NodeTileRes >( nodeOffset );
	GvLocalizationInfo::DepthType newLocDepth = parentLocDepth.addLevel();

	locCodeArray.set( newNodeTileIndex, newLocCode );
	locDepthArray.set( newNodeTileIndex, newLocDepth );
}

} // namespace GvCore

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvCore
{

/******************************************************************************
 * Return the localization info of a node in the node pool
 *
 * @param nodeAddress Address of the node in the node pool
 *
 * @return The localization info of the node
 ******************************************************************************/
template< typename NodeTileRes, typename ChildAddressType, typename ChildKernelArrayType, typename DataAddressType, typename DataKernelArrayType, typename LocCodeArrayType, typename LocDepthArrayType >
__device__
__forceinline__ GvLocalizationInfo PageTableBricksKernel< NodeTileRes, ChildAddressType, ChildKernelArrayType, DataAddressType, DataKernelArrayType, LocCodeArrayType, LocDepthArrayType >
::getLocalizationInfoImpl( uint nodeAddress ) const
{
	// Compute the address of the current node tile (and its offset in the node tile)
	uint nodeTileIndex = nodeAddress / NodeTileRes::getNumElements();
	uint nodeTileAddress = nodeTileIndex * NodeTileRes::getNumElements();
	uint nodeTileOffset = nodeAddress - nodeTileAddress;

	// Compute the node offset (in 3D, in the node tile)
	uint3 nodeOffset = NodeTileRes::toFloat3( nodeTileOffset );

	// Fetch associated localization infos
	GvLocalizationInfo::CodeType parentLocCode = locCodeArray.get( nodeTileIndex );
	GvLocalizationInfo::DepthType parentLocDepth = locDepthArray.get( nodeTileIndex );

	// Localization info initialization
	GvLocalizationInfo locInfo;
	locInfo.locCode = parentLocCode.addLevel< NodeTileRes >( nodeOffset );
	locInfo.locDepth = parentLocDepth;

	return locInfo;
}

/******************************************************************************
 * ...
 *
 * @param ...
 * @param ...
 * @param flags this vlaue is retrieves from Producer::produceData< 1 > methods)
 ******************************************************************************/
template< typename NodeTileRes, typename ChildAddressType, typename ChildKernelArrayType, typename DataAddressType, typename DataKernelArrayType, typename LocCodeArrayType, typename LocDepthArrayType >
__device__
__forceinline__ void PageTableBricksKernel< NodeTileRes, ChildAddressType, ChildKernelArrayType, DataAddressType, DataKernelArrayType, LocCodeArrayType, LocDepthArrayType >
::setPointerImpl( uint elemAddress, ElemAddressType elemPointer, uint flags )
{
	// XXX: Should be removed
	ElemAddressType brickPointer = elemPointer + make_uint3( 1 ); // Warning: fixed border size !	=> QUESTION ??

	PackedChildAddressType packedChildAddress	= childArray.get( elemAddress );
	ElemPackedAddressType packedBrickAddress	= DataAddressType::packAddress( brickPointer );

	// We store brick
	packedChildAddress |= 0x40000000;

	// Check flags value and modify address accordingly.
	// If flags is greater than 0, it means that the node containing the brick is terminal
	if ( flags > 0 )
	{
		// If flags equals 2, it means that the brick is empty
		if ( flags == 2 )
		{
			// Empty brick flag
			packedBrickAddress = 0;
			packedChildAddress &= 0xBFFFFFFF;
		}

		// Terminal flag
		packedChildAddress |= 0x80000000;
	}

	childArray.set( elemAddress, packedChildAddress );
	dataArray.set( elemAddress, packedBrickAddress );
}

} // namespace GvCore
