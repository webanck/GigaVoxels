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

#ifndef GVSTATICRES3D_H
#define GVSTATICRES3D_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/GvMath.h"
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
	 * @struct StaticRes3D
	 *
	 * @brief The StaticRes3D struct provides the concept of a 3D resolution.
	 *
	 * @ingroup GvCore
	 *
	 * This is notably used to define space resolution/extent of node tiles and bricks of voxels.
	 *
	 * @note All members are computed at compile-time.
	 */
	template< uint Trx, uint Try, uint Trz >
	struct StaticRes3D
	{

		/**************************************************************************
		 ***************************** PUBLIC SECTION *****************************
		 **************************************************************************/

		/****************************** INNER TYPES *******************************/

		/**
		 * Resolution in each axis.
		 */
		enum
		{
			x = Trx,
			y = Try,
			z = Trz
		};

		/**
		 * Total number of elements.
		 */
		enum
		{
			numElements = x * y * z
		};

		/**
		 * Precomputed log2() value of each axis resolution.
		 */
		enum
		{
			xLog2 = Log2< Trx >::value,
			yLog2 = Log2< Try >::value,
			zLog2 = Log2< Trz >::value
		};

		/**
		 * Precomputed min resolution.
		 */
		enum
		{
			minRes = Min< Min< x, y >::value, z >::value
		};

		/**
		 * Precomputed max resolution
		 */
		enum
		{
			maxRes = Max< Max< x, y >::value, z >::value
		};

		/**
		 * Precomputed boolean value to specify if each axis resolution is a power of two.
		 */
		enum
		{
			xIsPOT = ( x & ( x - 1 ) ) == 0,
			yIsPOT = ( y & ( y - 1 ) ) == 0,
			zIsPOT = ( z & ( z - 1 ) ) == 0
		};

		/******************************* ATTRIBUTES *******************************/

		/******************************** METHODS *********************************/

		/**
		 * Return the resolution as a uint3.
		 *
		 * @return the resolution
		 */
		__host__ __device__
		static uint3 get();

		/**
		 * Return the resolution as a float3.
		 *
		 * @return the resolution
		 */
		__host__ __device__
		static float3 getFloat3();

		/**
		 * Return the number of elements
		 *
		 * @return the number of elements
		 */
		__device__ __host__
		static uint getNumElements();

		//__host__ __device__
		//static uint getNumElementsLog2();

		/**
		 * Return the log2(resolution) as an uint3.
		 *
		 * @return the log2(resolution)
		 */
		__host__ __device__
		static uint3 getLog2();

		/**
		 * Convert a three-dimensionnal value to a linear value.
		 *
		 * @param pValue The 3D value to convert
		 *
		 * @return the 1D linearized converted value
		 */
		__host__ __device__
		static uint toFloat1( uint3 pValue );

		/**
		 * Convert a linear value to a three-dimensionnal value.
		 *
		 * @param pValue The 1D value to convert
		 *
		 * @return the 3D converted value
		 */
		__host__ __device__
		static uint3 toFloat3( uint pValue );

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

} //namespace GvCore

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{

	/** 
	 * @struct StaticRes1D
	 *
	 * @brief The StaticRes1D struct provides the concept of a uniform 3D resolution.
	 *
	 * @ingroup GvCore
	 *
	 * This is a specialization of a StaticRes3D 3D resolution where each dimension is equal.
	 * This is notably used to define space resolution/extent of node tiles and bricks of voxels.
	 *
	 * @note All members are computed at compile-time.
	 *
	 * @todo Not sure this is the best way to do it.
	 */
	template< uint Tr >
	struct StaticRes1D : StaticRes3D< Tr, Tr, Tr >
	{

		/**************************************************************************
		 ***************************** PUBLIC SECTION *****************************
		 **************************************************************************/

		/****************************** INNER TYPES *******************************/

		/******************************* ATTRIBUTES *******************************/

		/******************************** METHODS *********************************/

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

} //namespace GvCore

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "StaticRes3D.inl"

#endif
