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

#ifndef _GVV_CACHE_USAGE_VIEW_H_
#define _GVV_CACHE_USAGE_VIEW_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "UI_GvvQCacheUsageWidget.h"

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

// Qwt
class QwtThermo;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * @class GvvCacheUsageView
 *
 * @brief The GvvCacheUsageView class provides ...
 *
 * @ingroup GvViewerGui
 *
 * This class is used to ...
 */
class GVVIEWERGUI_EXPORT GvvCacheUsageView : public QWidget, public Ui::GvvQCacheUsageWidget
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
	 * Constructs a plot widget
	 */
	GvvCacheUsageView( QWidget* pParent, const char* pName = 0 );

	/**
	 * Default destructor
	 */
	virtual ~GvvCacheUsageView();

	/**
	 * Update view
	 *
	 * @param pNodeCacheUsage node cache usage (%)
	 * @param pBrickCacheUsage brick cache usage (%)
	 */
	void update( const GvViewerCore::GvvPipelineInterface* pPipeline );

	/********************************* SLOTS **********************************/

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************** TYPEDEFS ********************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Nodes cache usage
	 */
	QwtThermo* _nodesCacheUsage;

	/**
	 * Bricks cache usage
	 */
	QwtThermo* _bricksCacheUsage;

	/**
	 * Tree node sparseness
	 */
	QwtThermo* _treeNodeSparseness;

	/**
	 * Tree brick sparseness
	 */
	QwtThermo* _treeBrickSparseness;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************** TYPEDEFS ********************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

private slots:

	/**
	 * Slot called when tree data structure monitoring's state has changed
	 */
	void on__treeMonitoringGroupBox_toggled( bool pChecked );

};

} // namespace GvViewerGui

#endif
