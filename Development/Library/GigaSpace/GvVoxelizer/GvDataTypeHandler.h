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

#ifndef _GV_DATA_TYPE_HANDLER_H_
#define _GV_DATA_TYPE_HANDLER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

// System
#include <string>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvVoxelizer
{

/** 
 * @class GvDataTypeHandler
 *
 * @brief The GvDataTypeHandler class provides methods to deal with data type
 * used with GigaVoxels.
 *
 * It handles memory allocation and memory localization of data in buffers.
 */
class GIGASPACE_EXPORT GvDataTypeHandler
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Voxel data type enumeration that can be handled during voxelization
	 */
	typedef enum
	{
		gvUCHAR,
		gvUCHAR4,
		gvUSHORT,
		gvFLOAT,
		gvFLOAT4
	}
	VoxelDataType;

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
     * Retrieve the number of bytes representing a given data type
	 *
	 * @param pDataType a data type (i.e. uchar4, float, float4, etc...)
	 *
	 * @return the number of bytes representing the data type
     */
	static unsigned int canalByteSize( VoxelDataType pDataType );

	/**
     * Allocate memory associated to a number of elements of a given data type
	 *
	 * @param pDataType a data type (i.e. uchar4, float, float4, etc...)
	 * @param pNbElements the number of elements to allocate
	 *
	 * @return a pointer on the allocated memory space
     */
	static void* allocateVoxels( VoxelDataType pDataType, unsigned int pNbElements );

	/**
     * Retrieve the address of an element of a given data type in an associated buffer
	 *
	 * Note : this is used to retrieve voxel addresses in brick data buffers.
	 *
	 * @param pDataType a data type (i.e. uchar4, float, float4, etc...)
	 * @param pDataBuffer a buffer associated to elements of the given data type
	 * @param pElementPosition the position of the element in the associated buffer
	 *
	 * @return the address of the element in the buffer
     */
	static void* getAddress( VoxelDataType pDataType, void* pDataBuffer, unsigned int pElementPosition );

	/**
     * Retrieve the name representing a given data type
	 *
	 * @param pDataType a data type (i.e. uchar4, float, float4, etc...)
	 *
	 * @return the name of the data type
     */
	static std::string getTypeName( VoxelDataType pDataType );

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

}

#endif
