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

#include "PipelineGLUTWindowInterface.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

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
PipelineGLUTWindowInterface::PipelineGLUTWindowInterface()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
PipelineGLUTWindowInterface::~PipelineGLUTWindowInterface()
{
}

/******************************************************************************
 * Initialize
 *
 * @return Flag to tell wheter or not it succeded
 ******************************************************************************/
bool PipelineGLUTWindowInterface::initialize()
{
	return true;
}

/******************************************************************************
 * Finalize
 *
 * @return Flag to tell wheter or not it succeded
 ******************************************************************************/
bool PipelineGLUTWindowInterface::finalize()
{
	return true;
}

/******************************************************************************
 * Display callback
 ******************************************************************************/
void PipelineGLUTWindowInterface::onDisplayFuncExecuted()
{
}

/******************************************************************************
 * Reshape callback
 *
 * @param pWidth The new window width in pixels
 * @param pHeight The new window height in pixels
 ******************************************************************************/
void PipelineGLUTWindowInterface::onReshapeFuncExecuted( int pWidth, int pHeight )
{
}

/******************************************************************************
 * Keyboard callback
 *
 * @param pKey ASCII character of the pressed key
 * @param pX Mouse location in window relative coordinates when the key was pressed
 * @param pY Mouse location in window relative coordinates when the key was pressed
 ******************************************************************************/
void PipelineGLUTWindowInterface::onKeyboardFuncExecuted( unsigned char pKey, int pX, int pY )
{
}

/******************************************************************************
 * Mouse callback
 *
 * @param pButton The button parameter is one of left, middle or right.
 * @param pState The state parameter indicates whether the callback was due to a release or press respectively.
 * @param pX Mouse location in window relative coordinates when the mouse button state changed
 * @param pY Mouse location in window relative coordinates when the mouse button state changed
 ******************************************************************************/
void PipelineGLUTWindowInterface::onMouseFuncExecuted( int pButton, int pState, int pX, int pY )
{
}

/******************************************************************************
 * Idle callback
 ******************************************************************************/
void PipelineGLUTWindowInterface::onIdleFuncExecuted()
{
}
