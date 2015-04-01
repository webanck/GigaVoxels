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

#ifndef _GV_COMMON_GRAPHICS_PASS_H_
#define _GV_COMMON_GRAPHICS_PASS_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

// OpenGL
#include <GL/glew.h>

// Cuda
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
 * @class GvCommonGraphicsPass
 *
 * @brief The GvCommonGraphicsPass class provides interface to
 *
 * Some resources from OpenGL may be mapped into the address space of CUDA,
 * either to enable CUDA to read data written by OpenGL, or to enable CUDA
 * to write data for consumption by OpenGL.
 */
class GIGASPACE_EXPORT GvCommonGraphicsPass
{

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
	GvCommonGraphicsPass();

	/**
	 * Destructor
	 */
	 virtual ~GvCommonGraphicsPass();

	/**
	 * Initiliaze
	 */
	virtual void initialize();

	/**
	 * Finalize
	 */
	virtual void finalize();

	/**
	 * Reset
	 *
	 * @param pWidth width
	 * @param pHeight height
	 */
	virtual void reset();

	/**
	 * Get the color texture
	 *
	 * @return the color texture
	 */
	GLuint getColorTexture() const;

	/**
	 * Get the color render buffer
	 *
	 * @return the color render buffer
	 */
	GLuint getColorRenderBuffer() const;

	/**
	 * Get the depth buffer
	 *
	 * @return the depth buffer
	 */
	GLuint getDepthBuffer() const;

	/**
	 * Get the depth texture
	 *
	 * @return the depth texture
	 */
	GLuint getDepthTexture() const;

	/**
	 * Get the framebuffer object
	 *
	 * @return the framebuffer object
	 */
	GLuint getFrameBuffer() const;

	/**
	 * Get the buffer width
	 *
	 * @return the buffer width
	 */
	int getBufferWidth() const;

	/**
	 * Set the buffer width
	 *
	 * @param the buffer width
	 */
	void setBufferWidth( int pWidth );

	/**
	 * Get the height
	 *
	 * @return the height
	 */
	int getBufferHeight() const;

	/**
	 * Set the buffer height
	 *
	 * @param the buffer height
	 */
	void setBufferHeight( int pHeight );

	/**
	 * Set the buffer size
	 *
	 * @param the buffer size
	 */
	void setBufferSize( int pWidth, int pHeight );

	/**
	 * Tell wheter or not the pipeline uses image downscaling.
	 *
	 * @return the flag telling wheter or not the pipeline uses image downscaling
	 */
	bool hasImageDownscaling() const;

	/**
	 * Set the flag telling wheter or not the pipeline uses image downscaling
	 *
	 * @param pFlag the flag telling wheter or not the pipeline uses image downscaling
	 */
	void setImageDownscaling( bool pFlag );

	/**
	 * Get the image downscaling width
	 *
	 * @return the image downscaling width 
	 */
	int getImageDownscalingWidth() const;

	/**
	 * Get the image downscaling height
	 *
	 * @return the image downscaling width
	 */
	int getImageDownscalingHeight() const;

	/**
	 * Set the image downscaling width
	 *
	 * @param pValue the image downscaling width 
	 */
	void setImageDownscalingWidth( int pValue );

	/**
	 * Set the image downscaling height
	 *
	 * @param pValue the image downscaling height 
	 */
	void setImageDownscalingHeight( int pValue );

	/**
	 * Set the image downscaling size
	 *
	 * @param pWidth the image downscaling size 
	 * @param pHeight the image downscaling size 
	 */
	void setImageDownscalingSize( int pWidth, int pHeight );

	/**
	 * Get the type
	 *
	 * @return the type
	 */
	unsigned int getType() const;

	/**
	 * Set the type
	 *
	 * @param  pValue the type
	 */
	void setType( unsigned int pValue );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Color texture
	 */
	GLuint _colorTexture;

	/**
	 * Color render buffer
	 */
	GLuint _colorRenderBuffer;

	/**
	 * Depth texture
	 */
	GLuint _depthTexture;

	/**
	 * Depth buffer
	 */
	GLuint _depthBuffer;

	/**
	 * Frame buffer
	 */
	GLuint _frameBuffer;

	/**
	 * Internal graphics buffer's width
	 */
	int _width;

	/**
	 * Internal graphics buffer's height
	 */
	int _height;

	/**
	 * Flag telling wheter or not the pipeline uses image downscaling
	 */
	bool _hasImageDownscaling;

	/**
	 * Image downscaling width
	 */
	int _imageDownscalingWidth;

	/**
	 * Image downscaling height
	 */
	int _imageDownscalingHeight;

	/**
	 * Type
	 */
	unsigned int _type;

	/******************************** METHODS *********************************/

	/**
	 * Initiliaze buffers
	 *
	 * @param pWidth width
	 * @param pHeight height
	 */
	virtual void initializeBuffers();

	/**
	 * Finalize buffers
	 */
	virtual void finalizeBuffers();

	/**
	 * Reset buffers
	 *
	 * @param pWidth width
	 * @param pHeight height
	 */
	virtual void resetBuffers();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvCommonGraphicsPass( const GvCommonGraphicsPass& );

	/**
	 * Copy operator forbidden.
	 */
	GvCommonGraphicsPass& operator=( const GvCommonGraphicsPass& );

};

} // namespace GvUtils

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvCommonGraphicsPass.inl"

#endif
