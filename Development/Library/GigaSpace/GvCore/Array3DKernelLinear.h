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

#ifndef _ARRAY3DKERNELLINEAR_H_
#define _ARRAY3DKERNELLINEAR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Gigavoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/vector_types_ext.h"

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
 * @class Array3DKernelLinear
 *
 * @brief The Array3DKernelLinear class provides an interface to manipulate
 * arrays on device (i.e. GPU).
 *
 * @ingroup GvCore
 *
 * Device-side class interface to 1D, 2D or 3D array located in device memory.
 * Internally, it does not take ownership of data but references a 1D array.
 * Users can map their multi-dimensions host arrays on this 1D device memory
 * buffer by accessing it with 1D, 2D, or 3D indexes.
 *
 * This is a device-side helper class used by host array. It is associated
 * to the following arrays : 
 * - Array3D
 * - Array3DGPULinear
 *
 * @param T type of the array (uint, int2, float3, etc...)
 *
 * @see Array3D, Array3DGPULinear
 */
template< typename T >
class Array3DKernelLinear
{
	
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Inititialize.
	 *
	 * @param data pointer on data
	 * @param res resolution
	 * @param pitch pitch
	 */
	void init( T* pData, const uint3& pRes, size_t pPitch );

	/**
	 * Get the resolution.
	 *
	 * @return the resolution
	 */
	__device__
	__forceinline__ uint3 getResolution() const;

	/**
	 * Get the memory size.
	 *
	 * @return the memory size
	 */
	__device__
	size_t getMemorySize() const;

	/**
	 * Get the value at a given 1D address.
	 *
	 * @param pAddress a 1D address
	 *
	 * @return the value at the given address
	 */
	__device__
	/*const*/ T get( uint pAddress ) const;

	/**
	 * Get the value at a given 2D position.
	 *
	 * @param pPosition a 2D position
	 *
	 * @return the value at the given position
	 */
	__device__
	/*const*/ T get( const uint2& pPosition ) const;

	/**
	 * Get the value at a given 3D position.
	 *
	 * @param pPosition a 3D position
	 *
	 * @return the value at the given position
	 */
	__device__
	/*const*/ T get( const uint3& pPosition ) const;

	/**
	 * Get the value at a given 1D address in a safe way.
	 * Bounds are checked and address is modified if needed (as a clamp).
	 *
	 * @param pAddress a 1D address
	 *
	 * @return the value at the given address
	 */
	__device__
	/*const*/ T getSafe( uint pAddress ) const;

	/**
	 * Get the value at a given 3D position in a safe way.
	 * Bounds are checked and position is modified if needed (as a clamp).
	 *
	 * @param pPosition a 3D position
	 *
	 * @return the value at the given position
	 */
	__device__
	/*const*/ T getSafe( uint3 pPosition ) const;

	/**
	 * Get a pointer on data at a given 1D address.
	 *
	 * @param pAddress a 1D address
	 *
	 * @return the pointer at the given address
	 */
	__device__
	T* getPointer( uint pAddress = 0 );

	/**
	 * Set the value at a given 1D address in the data array.
	 *
	 * @param pAddress a 1D address
	 * @param pVal a value
	 */
	__device__
	void set( const uint pAddress, T val );

	/**
	 * Set the value at a given 2D position in the data array.
	 *
	 * @param pPosition a 2D position
	 * @param pVal a value
	 */
	__device__
	void set( const uint2& pPosition, T val );

	/**
	 * Set the value at a given 3D position in the data array.
	 *
	 * @param pPosition a 3D position
	 * @param pVal a value
	 */
	__device__
	void set( const uint3& pPosition, T val );

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

	/**
	 * Pointer on array data
	 */
	T* _data;

	/**
	 * Array resolution
	 */
	uint3 _resolution;

	/**
	 * Pitch in bytes
	 */
	size_t _pitch;

	/**
	 * Pitch in elements
	 */
	size_t _pitchxy;

	/******************************** METHODS *********************************/

	/**
	 * Helper function used to get the corresponding index array at a given
	 * 3D position in a safe way.
	 * Position is checked and modified if needed (as a clamp).
	 *
	 * @param pPosition a 3D position
	 *
	 * @return the corresponding index array at the given 3D position
	 */
	__device__
	__forceinline__ uint3 getSecureIndex( uint3 pPosition ) const;

	/**
	 * Helper function used to get the offset in the 1D linear data array
	 * given a 2D position.
	 *
	 * @param pPosition a 2D position
	 *
	 * @return the corresponding offset in the 1D linear data array
	 */
	__device__
	__forceinline__ uint getOffset( const uint2& pPosition ) const;

	/**
	 * Helper function used to get the offset in the 1D linear data array
	 * given a 3D position.
	 *
	 * @param pPosition a 3D position
	 *
	 * @return the corresponding offset in the 1D linear data array
	 */
	__device__
	__forceinline__ uint getOffset( const uint3& pPosition ) const;

};

} // namespace GvCore

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "Array3DKernelLinear.inl"

/******************************************************************************
 ************************** INSTANTIATION SECTION *****************************
 ******************************************************************************/

namespace GvCore
{
	//GIGASPACE_TEMPLATE_EXPORT template class GIGASPACE_EXPORT Array3DKernelLinear< uint >;
	//GIGASPACE_TEMPLATE_EXPORT template class GIGASPACE_EXPORT Array3DKernelLinear< float >;
}

#endif // !_ARRAY3DKERNELLINEAR_H_
