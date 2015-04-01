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

#ifndef PIPELINEGLUTWINDOWINTERFACE_H
#define PIPELINEGLUTWINDOWINTERFACE_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

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

/** 
 * @class PipelineGLUTWindowInterface
 *
 * @brief The PipelineGLUTWindowInterface class provides an interface to handle
 * integration with the GLUT library.
 *
 * It provides mapping of standard callbacks called by the GLUT library.
 */
class PipelineGLUTWindowInterface
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
	PipelineGLUTWindowInterface();

	/**
	 * Destructor
	 */
	virtual ~PipelineGLUTWindowInterface();
	
	/**
	 * Initialize
	 *
	 * @return Flag to tell wheter or not it succeded
	 */
	virtual bool initialize();

	/**
	 * Finalize
	 *
	 * @return Flag to tell wheter or not it succeded
	 */
	virtual bool finalize();
	
	/**
	 * Display callback
	 */
	virtual void onDisplayFuncExecuted() = 0;

	/**
	 * Reshape callback
	 *
	 * @param pWidth The new window width in pixels
	 * @param pHeight The new window height in pixels
	 */
	virtual void onReshapeFuncExecuted( int pWidth, int pHeight );

	/**
	 * Keyboard callback
	 *
	 * @param pKey ASCII character of the pressed key
	 * @param pX Mouse location in window relative coordinates when the key was pressed
	 * @param pY Mouse location in window relative coordinates when the key was pressed
	 */
	virtual void onKeyboardFuncExecuted( unsigned char pKey, int pX, int pY );

	/**
	 * Mouse callback
	 *
	 * @param pButton The button parameter is one of left, middle or right.
	 * @param pState The state parameter indicates whether the callback was due to a release or press respectively.
	 * @param pX Mouse location in window relative coordinates when the mouse button state changed
	 * @param pY Mouse location in window relative coordinates when the mouse button state changed
	 */
	virtual void onMouseFuncExecuted( int pButton, int pState, int pX, int pY );

	/**
	 * Idle callback
	 */
	virtual void onIdleFuncExecuted();
	
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

#endif // !PIPELINEGLUTWINDOWINTERFACE_H
