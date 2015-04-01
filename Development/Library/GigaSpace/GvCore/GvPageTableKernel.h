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

#ifndef _GV_PAGE_TABLE_KERNEL_H_
#define _GV_PAGE_TABLE_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/GvLocalizationInfo.h"

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
							 GvCore::GvLocalizationInfo::CodeType* pResLocCodeList, GvCore::GvLocalizationInfo::DepthType* pResLocDepthList );

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{

/** 
 * @struct PageTableKernel
 *
 * @brief The PageTableKernel struct provides...
 *
 * @ingroup GvCore
 *
 * This is the base class for all gpu page table implementations.
 */
template< typename Derived, typename AddressType, typename KernelArrayType >
struct PageTableKernel
{
	
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * This method returns the LocalizationInfo structure associated with the given
	 * node address.
	 *
	 * @param nodeAddress ...
	 *
	 * @return ...
	 */
	__device__
	__forceinline__ GvLocalizationInfo getLocalizationInfo( uint nodeAddress ) const;

	/**
	 * This method should...
	 *
	 * @param elemAddress ...
	 * @param elemPointer ...
	 * @param flag ...
	 */
	__device__
	__forceinline__ void setPointer( uint elemAddress, uint3 elemPointer, uint flag = 0 );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} // namespace GvCore

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{

/** 
 * @struct PageTableNodesKernel
 *
 * @brief The PageTableNodesKernel struct provides...
 *
 * @ingroup GvCore
 *
 * ...
 */
template
<
	typename NodeTileRes, typename ElementRes,
	typename AddressType, typename KernelArrayType,
	typename LocCodeArrayType, typename LocDepthArrayType
>
struct PageTableNodesKernel : public PageTableKernel< PageTableNodesKernel< NodeTileRes, ElementRes, AddressType, KernelArrayType, LocCodeArrayType, LocDepthArrayType >,
	AddressType, KernelArrayType >
{
	
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	typedef AddressType								ElemType;
	typedef typename AddressType::AddressType		ElemAddressType;
	typedef typename AddressType::PackedAddressType	ElemPackedAddressType;

	/******************************* ATTRIBUTES *******************************/

	KernelArrayType		childArray;
	LocCodeArrayType	locCodeArray;
	LocDepthArrayType	locDepthArray;

	/******************************** METHODS *********************************/

	// FIXME: Move into the parent class
	/**
	 * Return the localization info of a node in the node pool
	 *
	 * @param nodeAddress Address of the node in the node pool
	 *
	 * @return The localization info of the node
	 */
	__device__
	__forceinline__ GvLocalizationInfo getLocalizationInfoImpl( uint nodeAddress ) const;

	/**
	 * ...
	 *
	 * @param elemAddress ...
	 * @param elemPointer ...
	 * @param flags ...
	 */
	__device__
	__forceinline__ void setPointerImpl( uint elemAddress, ElemAddressType elemPointer, uint flags = 0 );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} // namespace GvCore

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{

/** 
 * @struct PageTableBricksKernel
 *
 * @brief The PageTableBricksKernel struct provides...
 *
 * @ingroup GvCore
 *
 * ...
 */
template
<
	typename NodeTileRes,
	typename ChildAddressType, typename ChildKernelArrayType,
	typename DataAddressType, typename DataKernelArrayType,
	typename LocCodeArrayType, typename LocDepthArrayType
>
struct PageTableBricksKernel : public PageTableKernel< PageTableBricksKernel< NodeTileRes, ChildAddressType, ChildKernelArrayType,
	DataAddressType, DataKernelArrayType, LocCodeArrayType, LocDepthArrayType >, DataAddressType, DataKernelArrayType >
{
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	typedef DataAddressType									ElemType;
	typedef typename DataAddressType::AddressType			ElemAddressType;
	typedef typename DataAddressType::PackedAddressType		ElemPackedAddressType;

	//typedef typename ChildAddressType::AddressType			UnpackedChildAddressType;
	typedef typename ChildAddressType::PackedAddressType	PackedChildAddressType;

	/******************************* ATTRIBUTES *******************************/

	ChildKernelArrayType	childArray;
	DataKernelArrayType		dataArray;
	LocCodeArrayType		locCodeArray;
	LocDepthArrayType		locDepthArray;

	/******************************** METHODS *********************************/

	// FIXME: Move into the parent class
	/**
	 * Return the localization info of a node in the node pool
	 *
	 * @param nodeAddress Address of the node in the node pool
	 *
	 * @return The localization info of the node
	 */
	__device__
	__forceinline__ GvLocalizationInfo getLocalizationInfoImpl( uint nodeAddress ) const;

	/**
	 * ...
	 *
	 * @param ...
	 * @param ...
	 * @param flags this vlaue is retrieves from Producer::produceData< 1 > methods)
	 */
	__device__
	__forceinline__ void setPointerImpl( uint elemAddress, ElemAddressType elemPointer, uint flags = 0 );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} // namespace GvCore

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvPageTableKernel.inl"

#endif // !_GV_PAGE_TABLE_KERNEL_H_
