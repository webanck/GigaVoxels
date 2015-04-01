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

#ifndef _GS_GRAPHICS_CORE_H_
#define _GS_GRAPHICS_CORE_H_

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

//#define glDispatchComputeEXT GLEW_GET_FUN(__glewMemoryBarrierEXT)
//
//typedef void (GLAPIENTRY *PFNGLDISPATCHCOMPUTEEXTPROC)(GLuint num_groups_x,  GLuint num_groups_y,  GLuint num_groups_z);
//#define glDispatchComputeEXT GLEW_GET_FUN(__glewMemoryBarrierEXT)
//GLEW_FUN_EXPORT PFNGLMEMORYBARRIEREXTPROC __glewMemoryBarrierEXT;

/**
 * Compute shader
 */
//typedef void (GLAPIENTRY *PFNGLDISPATCHCOMPUTEEXTPROC)( GLuint num_groups_x, GLuint num_groups_y, GLuint num_groups_z );
//PFNGLDISPATCHCOMPUTEEXTPROC glDispatchComputeEXT;

/**
 * Compute shader
 */
#ifndef GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS
#define GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS 0x90EB
#endif
#ifndef GL_MAX_COMPUTE_WORK_GROUP_COUNT
#define GL_MAX_COMPUTE_WORK_GROUP_COUNT 0x91BE
#endif
#ifndef GL_MAX_COMPUTE_WORK_GROUP_SIZE
#define GL_MAX_COMPUTE_WORK_GROUP_SIZE 0x91BF
#endif

/**
 * OpenGL
 */
//typedef void (GLAPIENTRY * PFNGLTEXSTORAGE2DPROC) (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height);
//PFNGLTEXSTORAGE2DPROC glTexStorage2D;

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
class GSGRAPHICS_EXPORT GsGraphicsCore
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
	 * Constructor
	 */
	GsGraphicsCore();

	/**
	 * Destructor
	 */
	~GsGraphicsCore();

	/**
	 * Print information about the device
	 */
	static void printInfo();
		
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

#endif // !_GS_GRAPHICS_CORE_H_

