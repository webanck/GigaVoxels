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

#ifndef _TYPEHELPERS_H_
#define _TYPEHELPERS_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/vector_types_ext.h"

// Cuda
#include <vector_types.h>

// System
#include <cassert>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * MACRO definition.
 * Used to convert a type to a string.
 */
#define DECLARE_TYPE_STRING( TType ) \
	template<> \
	const char* typeToString< TType >() \
	{ \
		return #TType; \
	}

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
	 * Convert a type to a string
	 *
	 * @return the type as a string
	 */
	template< typename TType >
	const char* typeToString()
	{
		bool Unsupported_Type = false;
		assert( Unsupported_Type );
		return "<unsupported-type>";
	}

	// Template specialization of the typeToString() method

	// Char types
	DECLARE_TYPE_STRING( char )
	DECLARE_TYPE_STRING( char2 )
	DECLARE_TYPE_STRING( char4 )

	// Unsigned char types
	DECLARE_TYPE_STRING( uchar )
	DECLARE_TYPE_STRING( uchar2 )
	DECLARE_TYPE_STRING( uchar4 )

	// Short types
	DECLARE_TYPE_STRING( short )
	DECLARE_TYPE_STRING( short2 )
	DECLARE_TYPE_STRING( short4 )

	// Unsigned short types
	DECLARE_TYPE_STRING( ushort )
	DECLARE_TYPE_STRING( ushort2 )
	DECLARE_TYPE_STRING( ushort4 )

	// Half types
	//DECLARE_TYPE_STRING( half )
	DECLARE_TYPE_STRING( half2 )
	DECLARE_TYPE_STRING( half4 )

	// Float types
	DECLARE_TYPE_STRING( float )
	DECLARE_TYPE_STRING( float2 )
	DECLARE_TYPE_STRING( float4 )

}

#endif // !_TYPEHELPERS_H_
