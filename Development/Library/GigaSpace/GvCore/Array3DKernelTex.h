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

#ifndef ARRAY3DKERNEL_H
#define ARRAY3DKERNEL_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <cuda.h>
#include <cuda_runtime.h>

// GigaVoxels
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

// TO DO : est-ce que ces deux "forward declaration" sont utiles ?
// Gigavoxels
namespace GvCore
{
	template< typename T > class Array3DGPULinear;
	template< typename T > class Array3DGPUTex;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{

	/** 
	 * @class Array3DKernelTex
	 *
	 * @brief The Array3DKernelTex class provides...
	 *
	 * @ingroup GvCore
	 *
	 * Kernel interface to 3D Array located in GPU texture memory.
	 */
	template< typename T >
	class Array3DKernelTex
	{

		/**************************************************************************
		 ***************************** PUBLIC SECTION *****************************
		 **************************************************************************/

	public:

		/******************************* ATTRIBUTES *******************************/

		/******************************** METHODS *********************************/

		/**
		 * Get the value at given position
		 *
		 * @param position position
		 *
		 * @return the value at given position
		 */
		template< uint channel >
		__device__
		__forceinline__ T get( const uint3& position ) const;

		/**
		 * Set the value at given position
		 *
		 * @param position position
		 * @param val the value to write
		 */
		template< uint channel >
		__device__
		__forceinline__ void set( const uint3& position, T val );

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

} //namespace GvCore

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "Array3DKernelTex.inl"

#endif // !ARRAY3DKERNEL_H
