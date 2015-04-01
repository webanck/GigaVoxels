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

#ifndef _GVV_TRANSFORMATION_EDITOR_H_
#define _GVV_TRANSFORMATION_EDITOR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "GvvSectionEditor.h"
#include "UI_GvvQTransformationEditor.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GvViewer
namespace GvViewerCore
{
	class GvvPipelineInterface;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * @class GvvTransformationEditor
 *
 * @brief The GvvTransformationEditor class provides ...
 *
 * ...
 */
class GVVIEWERGUI_EXPORT GvvTransformationEditor : public GvvSectionEditor, public Ui::GvvQTransformationEditor
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
	 * Default constructor
	 */
	GvvTransformationEditor( QWidget* pParent = 0, Qt::WindowFlags pFlags = 0 );

	/**
	 * Destructor
	 */
	virtual ~GvvTransformationEditor();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * Populates this editor with the specified browsable
	 *
	 * @param pBrowsable specifies the browsable to be edited
	 */
	virtual void populate( GvViewerCore::GvvBrowsable* pBrowsable );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

private slots:

	/**
	 * Slot called when max depth value has changed
	 */
	void on__xTranslationSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__yTranslationSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__zTranslationSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__xRotationSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__yRotationSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__zRotationSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__angleRotationSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__uniformScaleSpinBox_valueChanged( double pValue );
		
};

} // namespace GvViewerGui

#endif
