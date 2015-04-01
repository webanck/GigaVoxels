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

#ifndef _SAMPLE_VIEWER_H_
#define _SAMPLE_VIEWER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// OpenGL
#include <GL/glew.h>

// QGLViewer
#include <QGLViewer/qglviewer.h>
#include <QKeyEvent>

// Project
#include "SampleCore.h"

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
 * @class SampleViewer
 *
 * @brief The SampleViewer class provides a viewer widget for rendering.
 *
 * It holds a GigaVoxels pipeline.
 */
class SampleViewer : public QGLViewer
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
	SampleViewer();

	/**
	 * Destructor
	 */
	virtual ~SampleViewer();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Initialize the viewer
	 */
	virtual void init();

	/**
	 * Draw function called each frame
	 */
	virtual void draw();

	/**
	 * Resize GL event handler
	 *
	 * @param pWidth the new width
	 * @param pHeight the new height
	 */
	virtual void resizeGL( int pWidth, int pHeight );

	/**
	 * Get the viewer size hint
	 *
	 * @return the viewer size hint
	 */
	virtual QSize sizeHint() const;

	/**
	 * Key press event handler
	 *
	 * @param pEvent the event
	 */
	virtual void keyPressEvent( QKeyEvent* pEvent );

	/**
	 * Mouse press event handler
	 *
	 * @param pEvent the event
	 */
	virtual void mousePressEvent( QMouseEvent* e );

	/**
	 * Mouse move event handler
	 *
	 * @param pEvent the event
	 */
	virtual void mouseMoveEvent( QMouseEvent* e );

	/**
	 * Mouse release event handler
	 *
	 * @param pEvent the event
	 */
	virtual void mouseReleaseEvent( QMouseEvent* e );

	/**
	 * Set light
	 *
	 * @param theta ...
	 * @param phi ...
	 */
	void setLight( float theta, float phi );

	/**
	 * Draw light
	 */
	void drawLight() const;

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	bool mMoveLight;
	bool mControlLight;
	float mLight[7]; // (x,y,z,theta,phi, xpos, ypos) 
	SampleCore *mSampleCore;
	//qglviewer::ManipulatedFrame* mLight1;

	/******************************** METHODS *********************************/

};

#endif // !_SAMPLE_VIEWER_H_
