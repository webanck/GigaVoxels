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

#include "GvVoxelizer/GvDataTypeHandler.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// System 
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvVoxelizer
using namespace GvVoxelizer;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Retrieve the number of bytes representing a given data type
 *
 * @param pDataType a data type (i.e. uchar4, float, float4, etc...)
 *
 * @return the number of bytes representing the data type
 ******************************************************************************/
unsigned int GvDataTypeHandler::canalByteSize( VoxelDataType pDataType )
{
	unsigned int result = 0;

	switch ( pDataType )
	{
		case gvUCHAR:
			result = sizeof( unsigned char );
			break;

		case gvUCHAR4:
			result = 4 * sizeof( unsigned char );
			break;

		case gvUSHORT:
			result = sizeof( unsigned short );
			break;

		case gvFLOAT:
			result = sizeof( float );
			break;

		case gvFLOAT4:
			result = 4 * sizeof( float );
			break;

		default:
			// TO DO
			// Handle error
			assert( false );
			break;
	}

	return result;
}

/******************************************************************************
 * Allocate memory associated to a number of elements of a given data type
 *
 * @param pDataType a data type (i.e. uchar4, float, float4, etc...)
 * @param pNbElements the number of elements to allocate
 *
 * @return a pointer on the allocated memory space
 ******************************************************************************/
void* GvDataTypeHandler::allocateVoxels( VoxelDataType pDataType, unsigned int pNbElements )
{
	void* result = NULL;

	switch ( pDataType )
	{
//		case gvUCHAR4:
//			result = new unsigned char[ 4 * pNbElements ];
//			break;
//
//		case gvFLOAT:
//			result = new float[ pNbElements ];
//			break;
//
//		case gvFLOAT4:
//			result = new float[ 4 * pNbElements ];
//			break;


				case gvUCHAR:
                        result = operator new( sizeof(unsigned char) * pNbElements );
                        break;

                case gvUCHAR4:
                        result = operator new( sizeof(unsigned char) * 4 * pNbElements );
                        break;

				case gvUSHORT:
                        result = operator new( sizeof(unsigned short) * pNbElements );
                        break;

                case gvFLOAT:
                        result = operator new( sizeof(float) * pNbElements );
                        break;

                case gvFLOAT4:
                        result = operator new( sizeof(float) * 4 * pNbElements );
                        break;

		default:
			// TO DO
			// Handle error
			assert( false );
			break;
	}

	return result;
}

/******************************************************************************
 * Retrieve the address of an element of a given data type in an associated buffer
 *
 * Note : this is used to retrieve voxel addresses in brick data buffers.
 *
 * @param pDataType a data type (i.e. uchar4, float, float4, etc...)
 * @param pDataBuffer a buffer associated to elements of the given data type
 * @param pElementPosition the position of the element in the associated buffer
 *
 * @return the address of the element in the buffer
 ******************************************************************************/
void* GvDataTypeHandler::getAddress( VoxelDataType pDataType, void* pDataBuffer, unsigned int pElementPosition )
{
	void* result = NULL;

	switch ( pDataType )
	{
		case gvUCHAR:
			result = &( static_cast< unsigned char* >( pDataBuffer )[ pElementPosition ] );
			break;

		case gvUCHAR4:
			result = &( static_cast< unsigned char* >( pDataBuffer )[ 4 * pElementPosition ] );
			break;

		case gvUSHORT:
			result = &( static_cast< unsigned short* >( pDataBuffer )[ pElementPosition ] );
			break;

		case gvFLOAT:
			result = &(static_cast< float* >( pDataBuffer )[ pElementPosition ]);
			break;

		case gvFLOAT4:
			result = &(static_cast< float* >( pDataBuffer )[ 4 * pElementPosition ] );
			break;

		default:
			// TO DO
			// Handle error
			assert( false );
			break;
	}

	return result;
}

/******************************************************************************
 * Retrieve the name representing a given data type
 *
 * @param pDataType a data type (i.e. uchar4, float, float4, etc...)
 *
 * @return the name of the data type
 ******************************************************************************/
std::string GvDataTypeHandler::getTypeName( VoxelDataType pDataType )
{
	std::string result;

	switch( pDataType )
	{
		case gvUCHAR:
			result = std::string( "uchar" );
			break;

		case gvUCHAR4:
			result = std::string( "uchar4" );
			break;

		case gvUSHORT:
			result = std::string( "ushort" );
			break;

		case gvFLOAT:
			result = std::string( "float" );
			break;

		case gvFLOAT4:
			result = std::string( "float4" );
			break;

		default:
			// TO DO
			// Handle error
			assert( false );
			break;
	}

	return result;
}
