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

#ifndef GVFUNCTIONAL_EXT_H
#define GVFUNCTIONAL_EXT_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

// Cuda
#include <host_defines.h>

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
	 * @struct equal_to_zero
	 *
	 * @brief The equal_to_zero struct provides ...
	 *
	 * @ingroup GvCore
	 *
	 * ...
	 */
	template< typename T >
	struct equal_to_zero
	{
		
		/**
		 * ...
		 */
		__host__ __device__
		inline bool operator()( const T& lhs )
		{
			return lhs == T( 0 );
		}

	};

} // namespace GvCore

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{

	/** 
	 * @struct not_equal_to_zero
	 *
	 * @brief The not_equal_to_zero struct provides ...
	 *
	 * @ingroup GvCore
	 *
	 * ...
	 */
	template< typename T >
	struct not_equal_to_zero
	{
		
		/**
		 * ...
		 */
		__host__ __device__
		inline bool operator()( const T& lhs )
		{
			return lhs != T( 0 );
		}

	};

} // namespace GvCore

#endif // !GVFUNCTIONAL_EXT_H
