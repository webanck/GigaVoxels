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

#ifndef _GV_RAY_MAP_H_
#define _GV_RAY_MAP_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

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

// GigaVoxels
namespace GvRendering
{
	class GvGraphicsResource;
}
namespace GsGraphics
{
	class GsShaderProgram;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvUtils
{

/** 
 * @class GvRayMap
 *
 * @brief The GvRayMap class provides interface to handle a ray map.
 *
 * Ray map is a container of ray initialized for the rendering phase.
 */
class GIGASPACE_EXPORT GvRayMap
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Ray map type
	 */
	enum RayMapType
	{
		eClassical,
		eFishEye,
		eReflectionMap,
		eRefractionMap
	};

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvRayMap();

	/**
	 * Destructor
	 */
	virtual ~GvRayMap();

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
	 * Set the ray map dimensions
	 *
	 * @param pWidth width
	 * @param pHeight height
	 */
	void setResolution( unsigned int pWidth, unsigned int pHeight );

	/**
	 * ...
	 */
	bool createShaderProgram( const char* pFileNameVS, const char* pFileNameFS );

	/**
	 * Render
	 */
	void render();

	/**
	 * Get the associated graphics resource
	 *
	 * @return the associated graphics resource
	 */
	GvRendering::GvGraphicsResource* getGraphicsResource();

	/**
	 * Get the shader program
	 *
	 * @return the shader program
	 */
	const GsGraphics::GsShaderProgram* getShaderProgram() const;

	/**
	 * Edit the shader program
	 *
	 * @return the shader program
	 */
	GsGraphics::GsShaderProgram* editShaderProgram();

	/**
	 * Get the ray map type
	 *
	 * @return the ray map type
	 */
	RayMapType getRayMapType() const;

	/**
	 * Set the ray map type
	 *
	 * @param pValue the ray map type
	 */
	void setRayMapType( RayMapType pValue );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Ray map type
	 */
	RayMapType _rayMapType;

	/**
	 * OpenGL ray map buffer
	 */
	GLuint _rayMap;

	/**
	 * Associated graphics resource
	 */
	GvRendering::GvGraphicsResource* _graphicsResource;

	/**
	 * Flag to tell wheter or not the associated instance is initialized
	 */
	 bool _isInitialized;

	/**
	 * Ray map generator's GLSL shader program
	 */
	GsGraphics::GsShaderProgram* _shaderProgram;

	/**
	 * Frame width
	 */
	unsigned int _width;

	/**
	 * Frame height
	 */
	unsigned int _height;

	/**
	 * Frame buffer object
	 */
	GLuint _frameBuffer;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} // namespace GvUtils

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

//#include "GvRayMap.inl"

#endif
