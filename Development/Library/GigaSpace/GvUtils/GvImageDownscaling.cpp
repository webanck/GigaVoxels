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

#include "GvUtils/GvImageDownscaling.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvRendering/GvGraphicsResource.h"
#include "GvCore/GvError.h"
#include "GvUtils/GvShaderManager.h"

// CUDA
#include <driver_types.h>

// System
#include <cassert>
#include <cstddef>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvUtils;
using namespace GvRendering;

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
GvImageDownscaling::GvImageDownscaling()
:	_width( 0 )
,	_height( 0 )
{
}

/******************************************************************************
 * Desconstructor
 ******************************************************************************/
GvImageDownscaling::~GvImageDownscaling()
{
	finalize();
}

/******************************************************************************
 * Initialize
 *
 * @return a flag to tell wheter or not it succeeds.
 ******************************************************************************/
bool GvImageDownscaling::initialize()
{
	return false;
}

/******************************************************************************
 * Finalize
 *
 * @return a flag to tell wheter or not it succeeds.
 ******************************************************************************/
bool GvImageDownscaling::finalize()
{
	return false;
}

/******************************************************************************
 * Get the buffer's width
 *
 * @return the buffer's width
 ******************************************************************************/
unsigned int GvImageDownscaling::getWidth() const
{
	return _width;
}

/******************************************************************************
 * Get the buffer's height
 *
 * @return the buffer's height
 ******************************************************************************/
unsigned int GvImageDownscaling::getHeight() const
{
	return _height;
}

/******************************************************************************
 * Set the buffer's resolution
 *
 * @param pWidth width
 * @param pHeight height
 ******************************************************************************/
void GvImageDownscaling::setResolution( unsigned int pWidth, unsigned int pHeight )
{
	assert( ! ( pWidth == 0 || pHeight == 0 ) );

	_width = pWidth;
	_height= pHeight;
}
