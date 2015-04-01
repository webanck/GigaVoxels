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

#ifndef _GVV_CACHE_EDITOR_H_
#define _GVV_CACHE_EDITOR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "GvvSectionEditor.h"
#include "UI_GvvQCacheEditor.h"

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
class GVVIEWERGUI_EXPORT GvvCacheEditor : public GvvSectionEditor, public Ui::GvvQCacheEditor
{
	// Qt Macro
	Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	///**
	// * ...
	// *
	// * @param pParent ...
	// * @param pBrowsable ...
	// *
	// * @return ...
	// */
	//static GvvEditor* create( QWidget* pParent, GvViewerCore::GvvBrowsable* pBrowsable );

	///**
	// * Populate the widget with a pipeline
	// *
	// * @param pPipeline the pipeline
	// */
	//void populate( GvViewerCore::GvvPipelineInterface* pPipeline );

	/**
	 * Default constructor
	 */
	GvvCacheEditor( QWidget *parent = 0, Qt::WindowFlags flags = 0 );

	/**
	 * Destructor
	 */
	virtual ~GvvCacheEditor();

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
	 * Slot called when custom cache policy value has changed
	 */
	void on__preventReplacingUsedElementsCachePolicyCheckBox_toggled( bool pChecked );

	/**
	 * Slot called when custom cache policy value has changed
	 */
	void on__smoothLoadingCachePolicyGroupBox_toggled( bool pChecked );

	/**
	 * Slot called when number of node subdivision value has changed
	 */
	void on__nbSubdivisionsSpinBox_valueChanged( int i );

	/**
	  * Slot called when number of brick loads value has changed
	 */
	void on__nbLoadsSpinBox_valueChanged( int i );

	/**
	 * Slot called when custom cache policy value has changed
	 */
	void on__timeLimitGroupBox_toggled( bool pChecked );

	/**
	  * Slot called when the time limit value has changed
	 */
	void on__timeLimitDoubleSpinBox_valueChanged( double pValue );

};

} // namespace GvViewerGui

#endif
