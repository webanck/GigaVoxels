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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvUtils
{

/******************************************************************************
 * Get the depth buffer
 *
 * @return the depth buffer
 ******************************************************************************/
inline GLuint GvCommonGraphicsPass::getDepthBuffer() const
{
	return _depthBuffer;
}

/******************************************************************************
 * Get the color texture
 *
 * @return the color texture
 ******************************************************************************/
inline GLuint GvCommonGraphicsPass::getColorTexture() const
{
	return _colorTexture;
}

/******************************************************************************
 * Get the color render buffer
 *
 * @return the color render buffer
 ******************************************************************************/
inline GLuint GvCommonGraphicsPass::getColorRenderBuffer() const
{
	return _colorRenderBuffer;
}

/******************************************************************************
 * Get the depth texture
 *
 * @return the depth texture
 ******************************************************************************/
inline GLuint GvCommonGraphicsPass::getDepthTexture() const
{
	return _depthTexture;
}

/******************************************************************************
 * Get the framebuffer object
 *
 * @return the framebuffer object
 ******************************************************************************/
inline GLuint GvCommonGraphicsPass::getFrameBuffer() const
{
	return _frameBuffer;
}

/******************************************************************************
 * Get the width
 *
 * @return the width
 ******************************************************************************/
inline int GvCommonGraphicsPass::getBufferWidth() const
{
	return _width;
}

/******************************************************************************
 * Get the height
 *
 * @return the height
 ******************************************************************************/
inline int GvCommonGraphicsPass::getBufferHeight() const
{
	return _height;
}

/******************************************************************************
 * Tell wheter or not the pipeline uses image downscaling.
 *
 * @return the flag telling wheter or not the pipeline uses image downscaling
 ******************************************************************************/
inline bool GvCommonGraphicsPass::hasImageDownscaling() const
{
	return _hasImageDownscaling;
}

/******************************************************************************
 * Get the image downscaling width
 *
 * @return the image downscaling width 
 ******************************************************************************/
inline int GvCommonGraphicsPass::getImageDownscalingWidth() const
{
	return _imageDownscalingWidth;
}

/******************************************************************************
 * Get the image downscaling height
 *
 * @return the image downscaling width
 ******************************************************************************/
inline int GvCommonGraphicsPass::getImageDownscalingHeight() const
{
	return _imageDownscalingHeight;
}

/******************************************************************************
 * Get the type
 *
 * @return the type
 ******************************************************************************/
inline unsigned int GvCommonGraphicsPass::getType() const
{
	return _type;
}

} // namespace GvUtils
