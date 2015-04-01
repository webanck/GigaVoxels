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

#ifndef _GV_IMAGE_DOWNSCALING_H_
#define _GV_IMAGE_DOWNSCALING_H_

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

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvUtils
{

/** 
 * @class GvImageDownscaling
 *
 * @brief The GvImageDownscaling class provides interface to downscale image
 * during rendering.
 *
 * To improve performance, this class reduce the number of fragments
 * by rendering the image into an offscreen render target with lower resolution
 * (for example, with half the resolution x and y) and zoom up the image
 * when displaying it on the screen.
 */
class GIGASPACE_EXPORT GvImageDownscaling
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvImageDownscaling();

	/**
	 * Destructor
	 */
	virtual ~GvImageDownscaling();

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
	 * Get the buffer's width
	 *
	 * @return the buffer's width
	 */
	unsigned int getWidth() const;

	/**
	 * Get the buffer's height
	 *
	 * @return the buffer's height
	 */
	unsigned int getHeight() const;

	/**
	 * Set the buffer's resolution
	 *
	 * @param pWidth width
	 * @param pHeight height
	 */
	void setResolution( unsigned int pWidth, unsigned int pHeight );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Buffer width
	 */
	unsigned int _width;

	/**
	 * Buffer height
	 */
	unsigned int _height;

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

//#include "GvImageDownscaling.inl"

#endif
