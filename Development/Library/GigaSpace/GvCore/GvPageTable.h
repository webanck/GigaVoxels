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

#ifndef _GV_PAGE_TABLE_H_
#define _GV_PAGE_TABLE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/GvPageTableKernel.h"

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

namespace GvCore
{

/** 
 * @struct PageTable
 *
 * @brief The PageTable struct provides localisation information of all elements of the data structure
 *
 * @ingroup GvCore
 *
 * It is used to retrieve localization info (code + depth), .i.e 3D world position of associated node's region in space,
 * from nodes stored in the Cache Management System.
 *
 * @param NodeTileRes Node tile resolution
 * @param LocCodeArrayType Type of array storing localization code (ex : Array3DGPULinear< LocalizationInfo::CodeType >)
 * @param LocDepthArrayType Type of array storing localization depth (ex : Array3DGPULinear< LocalizationInfo::DepthType >)
 */
template< typename NodeTileRes, typename LocCodeArrayType, typename LocDepthArrayType >
struct PageTable
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Localization code array
	 *
	 * Global localization codes of all nodes of the data structure (for the moment, its a reference)
	 */
	LocCodeArrayType* locCodeArray;

	/**
	 * Localization depth array
	 *
	 * Global localization codes of all nodes of the data structure (for the moment, its a reference)
	 */
	LocDepthArrayType* locDepthArray;

	/******************************** METHODS *********************************/

	/**
	 * Create localization info list (code and depth)
	 *
	 * Given a list of N nodes, it retrieves their localization info (code + depth)
	 *
	 * @param pNumElems number of elements to process (i.e. nodes)
	 * @param pNodesAddressCompactList a list of nodes from which to retrieve localization info
	 * @param pResLocCodeList the resulting localization code array of all requested elements
	 * @param pResLocDepthList the resulting localization depth array of all requested elements
	 */
	inline void createLocalizationLists( uint pNumElems, uint* pNodesAddressCompactList,
										thrust::device_vector< GvLocalizationInfo::CodeType >* pResLocCodeList,
										thrust::device_vector< GvLocalizationInfo::DepthType >* pResLocDepthList );

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
 * @struct PageTableNodes
 *
 * @brief The PageTableNodes struct provides a page table specialized to handle a node pool
 *
 * @ingroup GvCore
 *
 * ...
 *
 * TIPS : extract from file VolumeTreeCache.h :
 * typedef PageTableNodes< NodeTileRes, NodeTileResLinear,
 *		VolTreeNodeAddress,	Array3DKernelLinear< uint >,
 *		LocCodeArrayType, LocDepthArrayType > NodePageTableType;
 */
template
<
	typename NodeTileRes, typename ElementRes,
	typename AddressType, typename KernelArrayType,
	typename LocCodeArrayType, typename LocDepthArrayType
>
struct PageTableNodes : public PageTable< NodeTileRes, LocCodeArrayType, LocDepthArrayType >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition for the associated device-side object
	 */
	typedef PageTableNodesKernel
	<
		NodeTileRes, ElementRes,
		AddressType, KernelArrayType,
		typename LocCodeArrayType::KernelArrayType,
		typename LocDepthArrayType::KernelArrayType
	>
	KernelType;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * The associated device-side object
	 */
	KernelType pageTableKernel;

	/******************************** METHODS *********************************/

	/**
	 * Get the associated device-side object
	 *
	 * @return the associated device-side object
	 */
	inline KernelType& getKernel();
		
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
 * @struct PageTableBricks
 *
 * @brief The PageTableBricks struct provides a page table specialized to handle a brick pool (i.e data)
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
struct PageTableBricks : public PageTable< NodeTileRes, LocCodeArrayType, LocDepthArrayType >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition for the associated device-side object
	 */
	typedef PageTableBricksKernel
	<
		NodeTileRes,
		ChildAddressType, ChildKernelArrayType,
		DataAddressType, DataKernelArrayType,
		typename LocCodeArrayType::KernelArrayType,
		typename LocDepthArrayType::KernelArrayType
	>
	KernelType;
	
	/******************************* ATTRIBUTES *******************************/

	/**
	 * The associated device-side object
	 */
	KernelType pageTableKernel;

	/******************************** METHODS *********************************/

	/**
	 * Get the associated device-side object
	 *
	 * @return the associated device-side object
	 */
	inline KernelType& getKernel();

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

#include "GvPageTable.inl"

#endif // !_GV_PAGE_TABLE_H_
