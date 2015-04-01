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

#ifndef _SAMPLEVIEWER_H_
#define _SAMPLEVIEWER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// OpenGL
#include <GL/glew.h>

// Qt
#include <QGLViewer/qglviewer.h>
#include <QKeyEvent>

// Project
#include "Parameters.h"
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

// Qtfe
class Qtfe;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class SampleViewer
 *
 * @brief The SampleViewer class provides...
 *
 * ...
 */
class SampleViewer : public QGLViewer
{

	Q_OBJECT

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
	~SampleViewer();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Transfer function editor
	 */
	Qtfe* _transferFunctionEditor;
	
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
	 * @param width the new width
	 * @param height the new height
	 */
	virtual void resizeGL( int width, int height );

	/**
	 * Get the viewer size hint
	 *
	 * @return the viewer size hint
	 */
	virtual QSize sizeHint() const;

	/**
	 * Key press event handler
	 *
	 * @param e the event
	 */
	virtual void keyPressEvent( QKeyEvent* e );

	/**
	 * Mouse press event handler
	 *
	 * @param e the event
	 */
	virtual void mousePressEvent( QMouseEvent* e );

	/**
	 * Mouse move event handler
	 *
	 * @param e the event
	 */
	virtual void mouseMoveEvent( QMouseEvent* e );

	/**
	 * Mouse release event handler
	 *
	 * @param e the event
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

	/********************************* SLOTS **********************************/

protected slots:

	/**
	 * Slot called when at least one canal changed
	 */
	void onFunctionChanged();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	SampleCore* _sampleCore;

	///**
	// * Light
	// */
	//qglviewer::ManipulatedFrame* _light1;

	/**
	 * ...
	 */
	bool _moveLight;

	/**
	 * ...
	 */
	bool _controlLight;

	/**
	 * ...
	 */
	float _light[ 7 ]; // ( x, y, z, theta, phi, xpos, ypos )

	/*
	 *
	 */
	float _noiseParameters[2];

	/******************************** METHODS *********************************/

};

#endif // !_SAMPLEVIEWER_H_
