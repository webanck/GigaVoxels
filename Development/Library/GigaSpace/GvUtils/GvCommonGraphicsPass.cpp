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

#include "GvUtils/GvCommonGraphicsPass.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvError.h"

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvUtils;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvCommonGraphicsPass::GvCommonGraphicsPass()
:	_colorTexture( 0 )
,	_colorRenderBuffer( 0 )
,	_depthTexture( 0 )
,	_depthBuffer( 0 )
,	_frameBuffer( 0 )
,	_width( 1 )
,	_height( 1 )
,	_hasImageDownscaling( false )
,	_imageDownscalingWidth( 512 )
,	_imageDownscalingHeight( 512 )
,	_type( 0 )
{
}

/******************************************************************************
 * Desconstructor
 ******************************************************************************/
GvCommonGraphicsPass::~GvCommonGraphicsPass()
{
	// Finalize
	finalize();
}

/******************************************************************************
 * Initiliaze
 *
 * @param pWidth width
 * @param pHeight height
 ******************************************************************************/
void GvCommonGraphicsPass::initialize()
{
	// Initialize buffers
	initializeBuffers();
}

/******************************************************************************
 * Finalize
 ******************************************************************************/
void GvCommonGraphicsPass::finalize()
{
	// Finalize buffers
	finalizeBuffers();
}

/******************************************************************************
 * Reset
 *
 * @param pWidth width
 * @param pHeight height
 ******************************************************************************/
void GvCommonGraphicsPass::reset()
{
	// Reset buffers
	resetBuffers();
}

/******************************************************************************
 * Initiliaze buffers
 *
 * @param pWidth width
 * @param pHeight height
 ******************************************************************************/
void GvCommonGraphicsPass::initializeBuffers()
{
	// Handle image downscaling if activated
	int bufferWidth = _width;
	int bufferHeight = _height;
	if ( _hasImageDownscaling )
	{
		bufferWidth = _imageDownscalingWidth;
		bufferHeight = _imageDownscalingHeight;
	}

	// [ 1 ] - initialize buffer used to read/write color
	if ( _type == 0 )
	{
		// Create a texture that will be used to display the output color buffer data
		// coming from the GigaVoxels volume rendering pipeline.
		// Texture will be filled with data coming from previous color PBO.
		// A full-screen quad will be used.
		glGenTextures( 1, &_colorTexture );
		glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _colorTexture );
		glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
		glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
		glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
		glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
		glTexImage2D( GL_TEXTURE_RECTANGLE_EXT, 0, GL_RGBA8, bufferWidth, bufferHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL );
		glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
		GV_CHECK_GL_ERROR();
	}
	else
	{	// Create a render buffer that will be used to display the output color buffer data
		// coming from the GigaVoxels volume rendering pipeline.
		// Render buffer will be filled with data coming from previous color PBO.
		// A full-screen quad will be used.
		glGenRenderbuffers( 1, &_colorRenderBuffer );
		glBindRenderbuffer( GL_RENDERBUFFER, _colorRenderBuffer );
		glRenderbufferStorage( GL_RENDERBUFFER, GL_RGBA8, bufferWidth, bufferHeight );
		glBindRenderbuffer( GL_RENDERBUFFER, 0 );
		GV_CHECK_GL_ERROR();
	}

	// [ 2 ] - initialize buffers used to read/write depth

	// Create a Pixel Buffer Object that will be used to read depth buffer data
	// coming from the default OpenGL framebuffer.
	// This graphics resource will be mapped in the GigaVoxels CUDA memory.
	glGenBuffers( 1, &_depthBuffer );
	glBindBuffer( GL_PIXEL_PACK_BUFFER, _depthBuffer );
	glBufferData( GL_PIXEL_PACK_BUFFER, bufferWidth * bufferHeight * sizeof( GLuint ), NULL, GL_DYNAMIC_DRAW );
	glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
	GV_CHECK_GL_ERROR();

	// Create a texture that will be used to display the output depth buffer data
	// coming from the GigaVoxels volume rendering pipeline.
	// Texture will be filled with data coming from previous depth PBO.
	// A full-screen quad will be used.
	glGenTextures( 1, &_depthTexture );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _depthTexture );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexImage2D( GL_TEXTURE_RECTANGLE_EXT, 0, GL_DEPTH24_STENCIL8_EXT, bufferWidth, bufferHeight, 0, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, NULL );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
	GV_CHECK_GL_ERROR();

	// [ 3 ] - initialize framebuffer used to read/write color and depth

	// Create a Frame Buffer Object that will be used to read/write color and depth buffer data
	// coming from the default OpenGL framebuffer.
	// This graphics resource will be mapped in the GigaVoxels CUDA memory.
	glGenFramebuffers( 1, &_frameBuffer );
	glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );
	if ( _type == 0 )
	{
		glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_EXT, _colorTexture, 0 );
	}
	else
	{
		glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, _colorRenderBuffer );
	}
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, _depthTexture, 0 );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, _depthTexture, 0 );
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );
	GV_CHECK_GL_ERROR();
}

/******************************************************************************
 * FinalizeBuffers
 ******************************************************************************/
void GvCommonGraphicsPass::finalizeBuffers()
{
	// Delete OpenGL depth buffer
	if ( _depthBuffer )
	{
		glDeleteBuffers( 1, &_depthBuffer );
	}

	if ( _depthTexture )
	{
		glDeleteTextures( 1, &_depthTexture );
	}

	//if ( _type == 0 )
	//{
		// Delete OpenGL color and depth textures
		if ( _colorTexture )
		{
			glDeleteTextures( 1, &_colorTexture );
		}
	//}
	//else
	//{
		// Delete OpenGL color render buffer
		if ( _colorRenderBuffer )
		{
			glDeleteRenderbuffers( 1, &_colorRenderBuffer );
		}
	//}

	// Delete OpenGL framebuffer
	if ( _frameBuffer )
	{
		glDeleteFramebuffers( 1, &_frameBuffer );
	}
}

/******************************************************************************
 * Reset buffers
 *
 * @param pWidth width
 * @param pHeight height
 ******************************************************************************/
void GvCommonGraphicsPass::resetBuffers()
{	
	// Finalize buffers
	finalizeBuffers();

	// Initialize buffers
	initializeBuffers();
}

/******************************************************************************
 * Set the buffer width
 *
 * @param the buffer width
 ******************************************************************************/
void GvCommonGraphicsPass::setBufferWidth( int pWidth )
{
	if ( _width != pWidth )
	{
		_width = pWidth;
	}
}

/******************************************************************************
 * Set the buffer height
 *
 * @param the buffer height
 ******************************************************************************/
void GvCommonGraphicsPass::setBufferHeight( int pHeight )
{
	if ( _height != pHeight )
	{
		_height = pHeight;
	}
}

/******************************************************************************
 * Set the buffer size
 *
 * @param the buffer size
 ******************************************************************************/
void GvCommonGraphicsPass::setBufferSize( int pWidth, int pHeight )
{
	if ( ( _width != pWidth ) || ( _height != pHeight ) )
	{
		_width = pWidth;
		_height = pHeight;
	}
}

/******************************************************************************
 * Set the flag telling wheter or not the pipeline uses image downscaling
 *
 * @param pFlag the flag telling wheter or not the pipeline uses image downscaling
 ******************************************************************************/
void GvCommonGraphicsPass::setImageDownscaling( bool pFlag )
{
	_hasImageDownscaling = pFlag;
}

/******************************************************************************
 * Set the image downscaling width
 *
 * @param pValue the image downscaling width 
 ******************************************************************************/
void GvCommonGraphicsPass::setImageDownscalingWidth( int pValue )
{
	if ( _imageDownscalingWidth != pValue )
	{
		_imageDownscalingWidth = pValue;
	}
}

/******************************************************************************
 * Set the image downscaling height
 *
 * @param pValue the image downscaling height 
 ******************************************************************************/
void GvCommonGraphicsPass::setImageDownscalingHeight( int pValue )
{
	if ( _imageDownscalingHeight != pValue )
	{
		_imageDownscalingHeight = pValue;
	}
}

/******************************************************************************
 * Set the image downscaling size
 *
 * @param pWidth the image downscaling size 
 * @param pHeight the image downscaling size 
 ******************************************************************************/
void GvCommonGraphicsPass::setImageDownscalingSize( int pWidth, int pHeight )
{
	if ( ( _imageDownscalingWidth != pWidth ) || ( _imageDownscalingHeight != pHeight ) )
	{
		_imageDownscalingWidth = pWidth;
		_imageDownscalingHeight = pHeight;
	}
}

/******************************************************************************
 * Set the type
 *
 * @param  pValue the type
 ******************************************************************************/
void GvCommonGraphicsPass::setType( unsigned int pValue )
{
	if ( _type != pValue )
	{
		_type = pValue;

		// Reset buffers
		resetBuffers();
	}
}
