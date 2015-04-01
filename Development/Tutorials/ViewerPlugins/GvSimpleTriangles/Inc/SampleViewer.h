/*
 * Copyright (C) 2011 Fabrice Neyret <Fabrice.Neyret@imag.fr>
 * Copyright (C) 2011 Cyril Crassin <Cyril.Crassin@icare3d.org>
 * Copyright (C) 2011 Morgan Armand <morgan.armand@gmail.com>
 *
 * This file is part of Gigavoxels.
 *
 * Gigavoxels is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Gigavoxels is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Gigavoxels.  If not, see <http://www.gnu.org/licenses/>.
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
	 * ...
	 */
	virtual void init();

	/**
	 * ...
	 */
	virtual void draw();

	/**
	 * ...
	 *
	 * @param width ...
	 * @param height ...
	 */
	virtual void resizeGL( int width, int height );

	/**
	 * ...
	 *
	 * @return ...
	 */
	virtual QSize sizeHint() const;

	/**
	 * ...
	 *
	 * @param e ...
	 */
	virtual void keyPressEvent( QKeyEvent* e );
	
	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	SampleCore* mSampleCore;

	/**
	 * ...
	 */
	qglviewer::ManipulatedFrame* mLight1;

	/******************************** METHODS *********************************/

};

#endif // !_SAMPLEVIEWER_H_
