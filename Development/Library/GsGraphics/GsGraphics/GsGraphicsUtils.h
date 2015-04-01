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

#ifndef _GS_GRAPHICS_UTILS_H_
#define _GS_GRAPHICS_UTILS_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaSpace
#include "GsGraphics/GsGraphicsCoreConfig.h"

// OpenGL
#include <GL/glew.h>

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

namespace GsGraphics
{
	
/** 
 * @class GsGraphicsCore
 *
 * @brief The GsGraphicsCore class provides an interface for accessing OpenGL properties.
 *
 * @ingroup GsGraphics
 *
 * It holds OpenGL properties.
 */
class GSGRAPHICS_EXPORT GsGraphicsUtils
{

	/**************************************************************************
	 ***************************** FRIEND SECTION *****************************
	 **************************************************************************/

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/
		
	/**
	 * Helper functrion to copy GL matrix from GLdouble[ 16 ] to float[ 16 ] array
	 * - ex : modelview or projection
	 *
	 * @param pDestinationMatrix
	 * @param pSourceMatrix
	 */
	static void copyGLMatrix( float* pDestinationMatrix, GLdouble* pSourceMatrix );

	/**
	 * Helper functrion to copy a pair  of GL matrices from GLdouble[ 16 ] to float[ 16 ] array
	 * - ex : modelview and projection
	 *
	 * @param pDestinationMatrix1
	 * @param pSourceMatrix1
	 * @param pDestinationMatrix2
	 * @param pSourceMatrix2
	 */
	static void copyGLMatrices( float* pDestinationMatrix1, GLdouble* pSourceMatrix1,
								float* pDestinationMatrix2, GLdouble* pSourceMatrix2 );
		
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

} // namespace GsGraphics

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#endif // !_GS_GRAPHICS_UTILS_H_
