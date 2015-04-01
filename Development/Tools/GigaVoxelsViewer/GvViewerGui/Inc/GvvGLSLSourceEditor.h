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

#ifndef _GVV_SHADER_SOURCE_EDITOR_
#define _GVV_SHADER_SOURCE_EDITOR_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "UI_GvQGLSLSourceEditor.h"

// Qt
#include <QWidget>

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
 * ...
 */
class GVVIEWERGUI_EXPORT GvvGLSLSourceEditor : public QWidget
{
	// Qt macro
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
	GvvGLSLSourceEditor( QWidget* pParent );

	/**
	 * ...
	 *
	 * @param ...
	 */
	void populate( GvViewerCore::GvvPipelineInterface* pPipeline );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Current pipeline
	 */
	GvViewerCore::GvvPipelineInterface* _pipeline;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Ui designer class
	 */
	Ui::GvQGLSLSourceEditor _ui;

	/******************************** METHODS *********************************/

private slots:

	/**
	 * Light action
	 */
	void onReload();

	/**
	 * Open action
	 */
	void onApply();

	/**
	 * Slot called when current page index has changed
	 *
	 * @param pIndex ...
	 */
	void on_tabWidget_currentChanged( int pIndex );

	/**
	 * Slot called when apply button has been released
	 */
	void on__applyButton_released();
	
};

} // namespace GvViewerGui

#endif
