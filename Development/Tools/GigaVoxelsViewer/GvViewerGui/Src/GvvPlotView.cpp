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

#include "GvvPlotView.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qt
#include <QPainter>
#include <QPicture>
#include <QFile>
#include <QFont>

// Qwt
#include <qwt_plot_picker.h>
#include <qwt_plot_layout.h>
#include <qwt_dyngrid_layout.h>
#include <qwt_painter.h>
#include <qwt_plot_grid.h>
#include <qwt_plot_panner.h>
#include <qwt_plot_zoomer.h>
#include <qwt_symbol.h>
#include <qwt_legend.h>
#include <qwt_legend_item.h>
//#include <qwt_legend_item.h> for Qwt 6.1
#include <qwt_plot_curve.h>
#include <qwt_scale_engine.h>
#include <qwt_scale_widget.h>
#include <qwt_text_label.h>
#include <qwt_plot_marker.h>

#include <cstring>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// XGraph
using namespace GvViewerGui;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Default constructor
 ******************************************************************************/
GvvPlotView::GvvPlotView( QWidget* pParent, const char* pName ) 
:	QwtPlot( QwtText( pName ), pParent )
,	_nodeCurve( NULL )
,	_nodeMarker( NULL )
,	_brickCurve( NULL )
,	_brickMarker( NULL )
//,	_unusedNodesCurve( NULL )
//,	_unusedBricksCurve( NULL )
{
	//** Setups background color
	setCanvasBackground( QColor( "white" ) );

	//** Setups the panner
    QwtPlotPanner* panner = new QwtPlotPanner( canvas() );
    panner->setMouseButton( Qt::MidButton );
    panner->setEnabled( true );

	//** Setups the grid
	QwtPlotGrid* grid = new QwtPlotGrid;
    grid->attach( this );
	grid->enableX( true );
	grid->enableY( true );
	grid->setMajPen( QPen( QBrush( QColor( Qt::black ) ), 1.f, Qt::DotLine ) );
	//grid->setMajorPen( QPen( QBrush( QColor( Qt::black ) ), 1.f, Qt::DotLine ) );
	grid->enableXMin( true );
	grid->enableYMin( false );
	grid->setMinPen( QPen( QBrush( QColor( Qt::black ) ), 1.f, Qt::DotLine ) );
	//grid->setMinorPen( QPen( QBrush( QColor( Qt::black ) ), 1.f, Qt::DotLine ) );

	//** Setups the zoomer
	QwtPlotZoomer* zoommer = new QwtPlotZoomer( QwtPlot::xBottom, QwtPlot::yLeft, canvas() );
	zoommer->setMaxStackDepth( 4 );
	//zoommer->setSelectionFlags( QwtPicker::DragSelection | QwtPicker::CornerToCorner );
	zoommer->setTrackerMode( QwtPicker::AlwaysOff );
	zoommer->setMousePattern( QwtEventPattern::MouseSelect2, Qt::RightButton, Qt::ControlModifier );
	zoommer->setMousePattern( QwtEventPattern::MouseSelect3, Qt::RightButton );
	zoommer->setRubberBand( QwtPicker::RectRubberBand );
	zoommer->setRubberBandPen( QColor( Qt::green ) );   
	zoommer->setEnabled( true );
	zoommer->zoom( 0 );

	QwtPlotPicker* picker = new QwtPlotPicker( QwtPlot::xBottom, QwtPlot::yLeft,
								QwtPlotPicker::CrossRubberBand, 
								QwtPicker::AlwaysOn, 
								canvas() );
	if ( picker != NULL )
	{
		picker->setRubberBandPen( QColor( Qt::green ) );
		picker->setRubberBand( QwtPicker::CrossRubberBand );
		picker->setTrackerPen( QColor( Qt::black ) );
	}

	_nodeMarker = new QwtPlotMarker();
	_nodeMarker->attach( this );
	QwtSymbol* nodeSymbol = new QwtSymbol( QwtSymbol::Diamond, QBrush( QColor( Qt::red ) ), QPen(), QSize( 15, 15 ) );
	_nodeMarker->setSymbol( nodeSymbol );

	_brickMarker = new QwtPlotMarker();
	_brickMarker->attach( this );
	QwtSymbol* brickSymbol = new QwtSymbol( QwtSymbol::Ellipse, QBrush( QColor( Qt::green ) ), QPen(), QSize( 15, 15 ) );
	_brickMarker->setSymbol( brickSymbol );

	//** Setups the legend
	QwtLegend* legend = new QwtLegend();
	legend->setItemMode( QwtLegend::CheckableItem );
	//legend->setDefaultItemMode( QwtLegendData::Checkable );
	insertLegend( legend, QwtPlot::RightLegend );

	//** Sets auto replot
	setAutoReplot( true );

	//** Disables the autodelete mode
	setAutoDelete( true );

	_nodeCurve = new QwtPlotCurve( "Nodes Subdivisions" );
	_nodeCurve->attach( this );
	_nodeCurve->setRenderHint( QwtPlotItem::RenderAntialiased );
	_nodeCurve->setPen( QPen( QColor( Qt::red ), 1.0f, Qt::SolidLine ) );

	_brickCurve = new QwtPlotCurve( "Bricks Production" );
	_brickCurve->attach( this );
	_brickCurve->setRenderHint( QwtPlotItem::RenderAntialiased );
	_brickCurve->setPen( QPen( QColor( Qt::green ), 1.0f, Qt::SolidLine ) );

	//_unusedNodesCurve = new QwtPlotCurve( "Unused Nodes" );
	//_unusedNodesCurve->attach( this );
	//_unusedNodesCurve->setRenderHint( QwtPlotItem::RenderAntialiased );
	//_unusedNodesCurve->setPen( QPen( QColor( Qt::blue ), 1.0f, Qt::DotLine ) );

	//_unusedBricksCurve = new QwtPlotCurve( "Unused Bricks" );
	//_unusedBricksCurve->attach( this );
	//_unusedBricksCurve->setRenderHint( QwtPlotItem::RenderAntialiased );
	//_unusedBricksCurve->setPen( QPen( QColor( Qt::yellow ), 1.0f, Qt::DotLine ) );

	for ( unsigned int i = 0; i < 1000 ; i++ )
	{
		_xDataNode[ i ] = i;
		_yDataNode[ i ] = 0;

		_xDataBrick[ i ] = i;
		_yDataBrick[ i ] = 0;

		//_xDataUnusedNodes[ i ] = i;
		//_yDataUnusedNodes[ i ] = 0;

		//_xDataUnusedBricks[ i ] = i;
		//_yDataUnusedBricks[ i ] = 0;
	}
	_nodeCurve->setRawSamples( _xDataNode, _yDataNode, 1000 );
	_brickCurve->setRawSamples( _xDataBrick, _yDataBrick, 1000 );
	/*_unusedNodesCurve->setRawSamples( _xDataUnusedNodes, _yDataUnusedNodes, 1000 );
	_unusedBricksCurve->setRawSamples( _xDataUnusedBricks, _yDataUnusedBricks, 1000 );*/
}

/******************************************************************************
 * Default destructor
 ******************************************************************************/
GvvPlotView::~GvvPlotView()
{
}

/******************************************************************************
 * Draw the specified curve
 *
 * @param pCurve specifies the curve to be drawn
 ******************************************************************************/
void GvvPlotView::onCurveChanged( unsigned int pFrame, unsigned int pNodeValue, unsigned int pBrickValue, unsigned int pUnusedNodeValue, unsigned int pUnusedBrickValue )
{
	if ( ( pFrame % 1000 ) == 0 )
	{
		memset( _yDataNode, 0, 1000 *sizeof( double ) );
		memset( _yDataBrick, 0, 1000 *sizeof( double ) );
	//	memset( _yDataUnusedNodes, 0, 1000 *sizeof( double ) );
	//	memset( _yDataUnusedBricks, 0, 1000 *sizeof( double ) );
	}

	const unsigned int indexFrame = pFrame % 1000;

	_yDataNode[ indexFrame ] = pNodeValue;
	_yDataBrick[ indexFrame ] = pBrickValue;
	//_yDataUnusedNodes[ indexFrame ] = pUnusedNodeValue;
//	_yDataUnusedBricks[ indexFrame ] = pUnusedBrickValue;

	_nodeMarker->setXValue( indexFrame );
	_nodeMarker->setYValue( pNodeValue );
	_brickMarker->setXValue( indexFrame );
	_brickMarker->setYValue( pBrickValue );
	/*_unusedNodesMarker->setXValue( indexFrame );
	_unusedNodesMarker->setYValue( pUnusedNodeValue );
	_unusedBricksMarker->setXValue( indexFrame );
	_unusedBricksMarker->setYValue( pUnusedBrickValue );*/

	replot();
}
