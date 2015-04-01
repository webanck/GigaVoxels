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

#ifndef GVMATH_H
#define GVMATH_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

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

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{

	/** 
	 * @struct Log2
	 *
	 * @brief The Log2 struct provides...
	 *
	 * @ingroup GvCore
	 *
	 * ...
	 *
	 * @note All is done at compile-time.
	 */
	template< uint TValue >
	struct Log2
	{
		/**
		 * ...
		 */
		enum
		{
			value = Log2< ( TValue >> 1 ) >::value + 1
		};
	};

	/**
	 * Log2 struct specialization
	 *
	 * @note All is done at compile-time.
	 */
	template<>
	struct Log2< 1 >
	{
		/**
		 * ...
		 */
		enum
		{
			value = 0
		};
	};

	/**
	 * Log2 struct specialization
	 *
	 * @note All is done at compile-time.
	 *
	 * @todo Need a Log2<> template which round-up the results.
	 */
	template<>
	struct Log2< 3 >
	{
		/**
		 * ...
		 */
		enum
		{
			value = 2
		};
	};

} // namespace GvCore

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{

	/** 
	 * @struct Max
	 *
	 * @brief The Max struct provides...
	 *
	 * @ingroup GvCore
	 *
	 * ...
	 *
	 * @note All is done at compile-time.
	 */
	template< int Ta, int Tb >
	struct Max
	{
		/**
		 * ...
		 */
		enum
		{
			value = Ta > Tb ? Ta : Tb
		};
	};

} //namespace GvCore

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{

	/** 
	 * @struct Min
	 *
	 * @brief The Min struct provides...
	 *
	 * @ingroup GvCore
	 *
	 * ...
	 *
	 * @note All is done at compile-time.
	 */
	template< int Ta, int Tb >
	struct Min
	{
		/**
		 * ...
		 */
		enum
		{
			value = Ta < Tb ? Ta : Tb
		};
	};

} //namespace GvCore

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{

	/** 
	 * @struct Min
	 *
	 * @brief The Min struct provides...
	 *
	 * @ingroup GvCore
	 *
	 * ...
	 *
	 * @note All is done at compile-time.
	 */
	template< int Ta, int Tb >
	struct IDivUp
	{
		/**
		 * ...
		 */
		enum
		{
			value = ( Ta % Tb != 0 ) ? ( Ta / Tb + 1 ) : ( Ta / Tb )
		};
	};

} //namespace GvCore

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#endif
