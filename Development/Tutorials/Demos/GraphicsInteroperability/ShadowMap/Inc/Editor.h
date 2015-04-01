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

#ifndef _EDITOR_H_
#define _EDITOR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GL
#include <GL/glew.h>

// QGViewer
#include <QGLViewer/qglviewer.h>

// Qt
#include <QWidget>

// Project
#include "UI_GvQEditor.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// Project
class ShadowMap;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class GvvCacheEditor
 *
 * @brief The GvvCacheEditor class provides ...
 *
 * ...
 */
class Editor : public QWidget, public Ui::GvQEditor
{

	// Qt Macro
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
	 *
	 * pParent ...
	 * pFlags ...
	 */
	Editor( QWidget* pParent = NULL, Qt::WindowFlags pFlags = 0 );

	/**
	 * Destructor
	 */
	virtual ~Editor();

	/**
	 * Initialize this editor with the specified GigaVoxels pipeline
	 *
	 * @param pPipeline ...
	 */
	void initialize( ShadowMap* pShadowMap, qglviewer::ManipulatedFrame* pLight );

	///**
	// * Populates this editor with the specified GigaVoxels pipeline
	// *
	// * @param pPipeline ...
	// */
	//void populate( ShadowMap* ShadowMap );

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
	
	/**
	 * Shadow map
	 */
	ShadowMap* _shadowMap;

	/**
	 * QGLViewer Manipulated Frame used to draw and manipulate a light in the 3D view
	 */
	qglviewer::ManipulatedFrame* _light;
	
	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	Editor( const Editor& );

	/**
	 * Copy operator forbidden.
	 */
	Editor& operator=( const Editor& );

	/********************************* SLOTS **********************************/

private slots:

	/**
	 * Slot called when value has changed
	 */
	void on__cameraEyeXDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__cameraEyeYDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__cameraEyeZDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__cameraCenterXDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__cameraCenterYDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__cameraCenterZDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__cameraUpXDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__cameraUpYDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__cameraUpZDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__cameraFovYDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__cameraAspectDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__cameraZNearDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__cameraZFarDoubleSpinBox_valueChanged( double pValue );
	
	/**
	 * Slot called when value has changed
	 */
	void on__materialAmbientXDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__materialAmbientYDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__materialAmbientZDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__materialDiffuseXDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__materialDiffuseYDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__materialDiffuseZDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__materialSpecularXDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__materialSpecularYDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__materialSpecularZDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__materialShininessDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__lightPositionXDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__lightPositionYDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__lightPositionZDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__lightIntensityXDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__lightIntensityYDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__lightIntensityZDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__lightCenterXDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__lightCenterYDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__lightCenterZDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__lightUpXDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__lightUpYDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__lightUpZDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__lightFovYDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__lightAspectDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__lightZNearDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when value has changed
	 */
	void on__lightZFarDoubleSpinBox_valueChanged( double pValue );

};

#endif
