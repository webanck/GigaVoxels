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

#ifndef _GVV_PLOT_VIEW_H_
#define _GVV_PLOT_VIEW_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"

// Qwt
#include <qwt_plot.h>

// Qt
#include <QColor>
#include <QObject>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// Qwt
class QwtPlotCurve;
class QwtPlotMarker;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * @class GvvPlotView
 *
 * @brief The GvvPlotView class provides ...
 *
 * @ingroup GvViewerGui
 *
 * This class is used to ...
 */
class GVVIEWERGUI_EXPORT GvvPlotView : public QwtPlot
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
	GvvPlotView( QWidget* pParent, const char* pName = 0 );

	/**
	 * Default destructor
	 */
	virtual ~GvvPlotView();

	/**
	 * Draw the specified curve
	 *
	 * @param pCurve specifies the curve to be drawn
	 */
	void onCurveChanged( unsigned int pFrame, unsigned int pNodeValue, unsigned int pBrickValue, unsigned int pUnusedNodeValue, unsigned int pUnusedBrickValue );

	/********************************* SLOTS **********************************/

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************** TYPEDEFS ********************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Node curve
	 */
	QwtPlotCurve* _nodeCurve;
	double _xDataNode[ 1000 ];
	double _yDataNode[ 1000 ];
	QwtPlotMarker* _nodeMarker;

	/**
	 * Brick curve
	 */
	QwtPlotCurve* _brickCurve;
	double _xDataBrick[ 1000 ];
	double _yDataBrick[ 1000 ];
	QwtPlotMarker* _brickMarker;

	///**
	// * Unused nodes
	// */
	//QwtPlotCurve* _unusedNodesCurve;
	//double _xDataUnusedNodes[ 1000 ];
	//double _yDataUnusedNodes[ 1000 ];
	//QwtPlotMarker* _unusedNodesMarker;

	///**
	// * Unused bricks
	// */
	//QwtPlotCurve* _unusedBricksCurve;
	//double _xDataUnusedBricks[ 1000 ];
	//double _yDataUnusedBricks[ 1000 ];
	//QwtPlotMarker* _unusedBricksMarker;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************** TYPEDEFS ********************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} // namespace GvViewerGui

#endif
