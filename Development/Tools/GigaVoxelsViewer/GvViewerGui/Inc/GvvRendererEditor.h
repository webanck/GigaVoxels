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

#ifndef _GVV_RENDERER_EDITOR_H_
#define _GVV_RENDERER_EDITOR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "GvvSectionEditor.h"
#include "UI_GvvQRendererEditor.h"

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
 * @class GvvCacheEditor
 *
 * @brief The GvvCacheEditor class provides ...
 *
 * ...
 */
class GVVIEWERGUI_EXPORT GvvRendererEditor : public GvvSectionEditor, public Ui::GvvQRendererEditor
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
	GvvRendererEditor( QWidget *parent = 0, Qt::WindowFlags flags = 0 );

	/**
	 * Destructor
	 */
	virtual ~GvvRendererEditor();

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
	void on__maxDepthSpinBox_valueChanged( int i );

	/**
	 * Slot called when cache policy value has changed (dynamic update)
	 */
	void on__dynamicUpdateCheckBox_toggled( bool pChecked );

	/**
	 * Slot called when the renderer request priority strategy has changed
	 */
	void on__priorityOnBricksRadioButton_toggled( bool pChecked );

	// ---- Viewport / Graphics buffer size ----

	/**
	 * Slot called when image downscaling mode value has changed
	 */
	void on__viewportOffscreenSizeGroupBox_toggled( bool pChecked );

	/**
	 * Slot called when graphics buffer width value has changed
	 */
	void on__graphicsBufferWidthSpinBox_valueChanged( int pValue );

	/**
	 * Slot called when graphics buffer height value has changed
	 */
	void on__graphicsBufferHeightSpinBox_valueChanged( int pValue );

	/**
	 * Slot called when the viewer has been resized
	 *
	 * @param pWidth new viewer width
	 * @param pHeight new viewr height
	 */
	void onViewerResized( int pWidth, int pHeight );

	// ---- Time budget monitoring ----

	/**
	 * Slot called when time budget monitoring state value has changed
	 */
	void on__timeBudgetParametersGroupBox_toggled( bool pChecked );
	
	/**
	 * Slot called when time budget value has changed
	 */
	void on__timeBudgetSpinBox_valueChanged( int pValue );

};

} // namespace GvViewerGui

#endif
