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

#ifndef _GV_ADDRESS_TYPE_H_
#define _GV_ADDRESS_TYPE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

#define GV_VTBA_BRICK_FLAG		0x40000000U
#define GV_VTBA_TERMINAL_FLAG	0x80000000U

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvStructure
{

/**
 * A structure to manipulate a VolumeTree node address.
 */
struct VolTreeNodeAddress
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/
	
	/**
	 * Defines the type of a node address.
	 */
	typedef uint3 AddressType;

	/**
	 * Defines the type of a packed node address.
	 */
	typedef uint PackedAddressType;

	/**
	 * ...
	 */
	enum
	{
		packedMask = 0x3FFFFFFF
	};

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * This function packs the given address.
	 *
	 * @param address an address.
	 *
	 * @return the packed representation of the address.
	 */
	__host__ __device__
	inline static PackedAddressType packAddress( const uint3& address );

	/**
	 * ...
	 */
	__host__ __device__
	inline static PackedAddressType packAddress( uint a );

	/**
	 * This function unpacks the given address.
	 *
	 * @param address the packed address.
	 *
	 * @return the unpacked representation of the address.
	 */
	__host__ __device__
	inline static uint3 unpackAddress( PackedAddressType address );

	/**
	 * This function returns true if the address is the null address.
	 */
	__host__ __device__
	inline static bool isNull( uint pa );	

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

} // namespace GvStructure

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvStructure
{

//! a structure to manipulate a VolumeTree brick address.
struct VolTreeBrickAddress
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	/**
	 * Defines the type of a brick address.
	 */
	typedef uint3 AddressType;

	/**
	 * Defines the type of a packed brick address.
	 */
	typedef uint PackedAddressType;

	/**
	 * ...
	 */
	enum
	{
		packedMask = 0x3FFFFFFF
	};

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * This function packs the given address.
	 *
	 * @param address an address.
	 *
	 * @return the packed representation of the address.
	 */
	__host__ __device__
	static PackedAddressType packAddress( const uint3& address );

	/**
	 * This function unpacks the given address.
	 *
	 * @param address the packed address.
	 *
	 * @return the unpacked representation of the address.
	 */
	__host__ __device__
	static uint3 unpackAddress( PackedAddressType address );

	/**
	 * This function returns true if the address is the null address.
	 */
	__host__ __device__
	static bool isNull( uint pa );

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

} // namespace GvStructure

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvVolumeTreeAddressType.inl"

#endif
