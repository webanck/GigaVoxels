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

#ifndef _SAMPLE_VIEWER_H_
#define _SAMPLE_VIEWER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// OpenGL
#include <GL/glew.h>

// Qt
#include <QGLViewer/qglviewer.h>
#include <QKeyEvent>

// Simple Sphere
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

// QGLViewer
namespace qglviewer
{
	class ManipulatedFrame;
}

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
	// Qt Macro
	Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

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
	 * Get the flag to tell wheter or not light manipulation is activated
	 *
	 * @return the light manipulation flag
	 */
	bool getLightManipulation() const;

	/**
	 * Set the flag to tell wheter or not light manipulation is activated
	 *
	 * @param pFlag the light manipulation flag
	 */
	void setLightManipulation( bool pFlag );

	/********************************* SLOTS **********************************/

protected slots:

	/**
	 * Slot called when the light ManipulatedFrame has been modified
	 */
	void onLightFrameManipulated();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Core class containing the GigaVoxels pipeline
	 */
	SampleCore* _sampleCore;

	/**
	 * QGLViewer Manipulated Frame used to draw and manipulate a light in the 3D view
	 */
	qglviewer::ManipulatedFrame* _light1;

	/**
	 * Flag to tell wheter or not light manipulation is activated
	 */
	bool _lightManipulation;

	/******************************** METHODS *********************************/

};

#endif // !_SAMPLE_VIEWER_H_
