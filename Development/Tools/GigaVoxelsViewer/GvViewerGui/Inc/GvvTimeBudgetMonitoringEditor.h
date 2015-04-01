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

#ifndef _GVV_TIME_BUDGET_MONITORING_EDITOR_H_
#define _GVV_TIME_BUDGET_MONITORING_EDITOR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "UI_GvvQTimeBudgetView.h"

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
namespace GvViewerGui
{
	class GvvTimeBudgetPlotView;
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
class GVVIEWERGUI_EXPORT GvvTimeBudgetMonitoringEditor : public QWidget, public Ui::GvvQTimeBudgetView
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
	GvvTimeBudgetMonitoringEditor( QWidget* pParent = NULL, Qt::WindowFlags pFlags = 0 );

	/**
	 * Destructor
	 */
	virtual ~GvvTimeBudgetMonitoringEditor();

	/**
	 * Populates this editor with the specified GigaVoxels pipeline
	 *
	 * @param pPipeline specifies the pipeline to be edited
	 */
	void populate( GvViewerCore::GvvPipelineInterface* pPipeline );

	/**
	 * Draw the specified curve
	 *
	 * @param pCurve specifies the curve to be drawn
	 */
	void onCurveChanged( unsigned int pFrame, float pFrameTime );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Plot view
	 */
	GvvTimeBudgetPlotView* _plotView;

	/**
	 * GigaVoxels pipeline
	 */
	GvViewerCore::GvvPipelineInterface* _pipeline;

	/******************************** METHODS *********************************/
	
	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvvTimeBudgetMonitoringEditor( const GvvTimeBudgetMonitoringEditor& );

	/**
	 * Copy operator forbidden.
	 */
	GvvTimeBudgetMonitoringEditor& operator=( const GvvTimeBudgetMonitoringEditor& );

	/********************************* SLOTS **********************************/

private slots:

	/**
	 * Slot called when time budget parameters group box state has changed
	 */
//	void on__timeBudgetParametersGroupBox_toggled( bool pChecked );

	/**
	 * Slot called when user requested time budget value has changed
	 */
//	void on__timeBudgetSpinBox_valueChanged( int pValue );

};

} // namespace GvViewerGui

#endif
