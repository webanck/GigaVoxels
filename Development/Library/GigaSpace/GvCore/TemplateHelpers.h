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

#ifndef _TEMPLATEHELPERS_H_
#define _TEMPLATEHELPERS_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * MACRO.
 */
#define TEMPLATE_INSTANCIATE_CLASS( T1, T2 ) \
	template class T1< T2 >;

/**
 * MACRO.
 */
#define TEMPLATE_INSTANCIATE_CLASS_UCHAR_TYPES( T1 ) \
	TEMPLATE_INSTANCIATE_CLASS( T1, uchar ) \
	TEMPLATE_INSTANCIATE_CLASS( T1, uchar2 ) \
	TEMPLATE_INSTANCIATE_CLASS( T1, uchar3 ) \
	TEMPLATE_INSTANCIATE_CLASS( T1, uchar4 )

/**
 * MACRO.
 */
#define TEMPLATE_INSTANCIATE_CLASS_UINT_TYPES( T1 ) \
	TEMPLATE_INSTANCIATE_CLASS( T1, uint ) \
	TEMPLATE_INSTANCIATE_CLASS( T1, uint2 ) \
	TEMPLATE_INSTANCIATE_CLASS( T1, uint3 ) \
	TEMPLATE_INSTANCIATE_CLASS( T1, uint4 )

/**
 * MACRO.
 */
#define TEMPLATE_INSTANCIATE_CLASS_FLOAT_TYPES(T1) \
	TEMPLATE_INSTANCIATE_CLASS( T1, float ) \
	TEMPLATE_INSTANCIATE_CLASS( T1, float2 ) \
	TEMPLATE_INSTANCIATE_CLASS( T1, float3 ) \
	TEMPLATE_INSTANCIATE_CLASS( T1, float4 )

/**
 * MACRO.
 */
#define TEMPLATE_INSTANCIATE_CLASS_TYPES( T1 ) \
	TEMPLATE_INSTANCIATE_CLASS_UCHAR_TYPES( T1 ) \
	TEMPLATE_INSTANCIATE_CLASS_UINT_TYPES( T1 ) \
	TEMPLATE_INSTANCIATE_CLASS_FLOAT_TYPES( T1 )

#endif // !_TEMPLATEHELPERS_H_

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/
