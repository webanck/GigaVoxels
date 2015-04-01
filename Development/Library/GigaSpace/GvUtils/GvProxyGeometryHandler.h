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

#ifndef _GV_PROXY_GEOMETRY_HANDLER_H_
#define _GV_PROXY_GEOMETRY_HANDLER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

// OpenGL
#include <GL/glew.h>

// CUDA
#include <driver_types.h>

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

namespace GvUtils
{

/** 
 * @class GvProxyGeometryHandler
 *
 * @brief The GvProxyGeometryHandler class provides interface to proxy geometry
 *
 * Proxy geometries are used to provided an approximation or a lower resolution
 * of a model.
 */
class GIGASPACE_EXPORT GvProxyGeometryHandler
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * CUDA graphics resource
	 */
	struct cudaGraphicsResource* _d_vertices;	// TEST : à remettre en protected

	/**
	 *
	 */
	unsigned int _nbPoints;

	/**
	 * Vertex array
	 */
	GLuint _vertexBuffer;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvProxyGeometryHandler();

	/**
	 * Destructor
	 */
	 virtual ~GvProxyGeometryHandler();

	 /**
	  * Initialize
	  *
	  * @return a flag to tell wheter or not it succeeds.
	  */
	 bool initialize();

	  /**
	  * Finalize
	  *
	  * @return a flag to tell wheter or not it succeeds.
	  */
	 bool finalize();

	 /**
	  * Render
	  */
	 void render();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	///**
	// * Vertex array
	// */
	//GLuint _vertexBuffer;

	/**
	 * Index array
	 */
	GLuint _indexBuffer;

	///**
	// * CUDA graphics resource
	// */
	//struct cudaGraphicsResource* _d_vertices;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} // namespace GvUtils

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

//#include "GvProxyGeometryHandler.inl"

#endif
