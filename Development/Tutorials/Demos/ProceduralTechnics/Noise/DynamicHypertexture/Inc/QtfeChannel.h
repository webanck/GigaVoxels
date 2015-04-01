/************************************************************************

Copyright (C) 2012 Eric Heitz (er.heitz@gmail.com). All rights reserved.

This file is part of Qtfe (Qt Transfer Function Editor).

Qtfe is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 of 
the License, or (at your option) any later version.

Qtfe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Qtfe.  If not, see <http://www.gnu.org/licenses/>.

************************************************************************/

#ifndef _Q_TFE_CHANNEL_
#define _Q_TFE_CHANNEL_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qt
#include <QWidget>
#include <QMouseEvent>
#include <QList>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// Qtfe
class Qtfe;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class QtfeChannel
 *
 * @brief The QtfeChannel class provides the interface of a channel of the main Qtfe editor.
 *
 * A channel models a 2D fonction of one component of a complex transfer function.
 * It is a 1D array of 2D points that will be bind to a specific output slot (R,G,B,A).
 * The list of points are interpolated to draw a curve.
 *
 * Warning : QtfeChannel is not supposed to be used outside Qtfe
 */
class QtfeChannel : public QWidget
{
	// Qt macro
	Q_OBJECT

	/**************************************************************************
     ***************************** FRIEND SECTION *****************************
     **************************************************************************/

	friend class Qtfe;

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/******************************** SIGNALS *********************************/

signals:

	/**
	 * Signal emitted when channel has been modified
	 */
	void channelChanged();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/

	/**
	 * Curve first point
	 */
	QPointF _first;

	/**
	 * Curve last point
	 */
	QPointF _last;

	/**
	 * List of points of associated curve
	 */
	QList< QPointF* > _list;

	/**
	 * Mouse flag to tell wheter or not mouse button has been pressed (i.e. left click)
	 */
	bool _pressed;

	/**
	 * Position of the current selected point if any.
	 * Position is expressed in the curve coordinate system (between [0.0 ; 1.0])
	 */
	QPointF* _selected;

	/**
	 * Helper variable for min "x" position of the current user selected/interacting interval.
	 *
	 * This position is set during mouse press event,
	 * and take the value of the max bound of the currently selected interval (plus an epsillon value).
	 */
	qreal _min;

	/**
	 * Helper variable for max "x" position of the current user selected/interacting interval.
	 *
	 * This position is set during mouse press event,
	 * and take the value of the min bound of the currently selected interval (minus an epsillon value).
	 */
	qreal _max;

	/**
	 * Background image on which curve is displayed
	 */
	QImage* _background;

	/**
	 * Point size used to draw distinguishable points on the curve.
	 */
	static const int _pointSizePixel;

	/**
	 * Width of circle used to display a circle around points near the mouse button.
	 */
	static const int _circleSizePixel;

	/**
	 * Line width used to draw the curve associated to list of points.
	 */
	static const int _lineWidth;
	
	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param pParent parent widget (it must be an instance of Qtfe editor)
	 */
	QtfeChannel( Qtfe* pParent );

	/**
	 * Destructor
	 */
	~QtfeChannel();

	/**
	 * Evaluate the function at given "t" parameter.
	 *
	 * @param t "x" position's parameter (it will be clamped to [ 0.0 ; 1.0 ] if needed) 
	 *
	 * @return the resulting value
	 */
	qreal evalf( qreal x ) const;

	/**
	 * Get the list of points
	 *
	 * return the list of points
	 */
	const QList< QPointF* >& getPoints() const;

	/**
	 * Set the first point value
	 *
	 * @param pValue the given value
	 */
	void setFirstPoint( qreal pValue );

	/**
	 * Set the last point value
	 *
	 * @param pValue the given value
	 */
	void setLastPoint( qreal pValue );

	/**
	 * Insert a point
	 *
	 * @param pPoint the point to add
	 */
	void insertPoint( const QPointF& pPoint );

	/**
	 * Handle the paint event.
	 * Draw all elements (points, interpolated curve and selected point if any)
	 *
	 * @param pEvent the paint event
	 */
	virtual void paintEvent( QPaintEvent* pEvent );

	/**
	 * Handle the mouse press event.
	 *
	 * Left click is used to add points.
	 * Right click is used to remove selected point if any.
	 *
	 * @param pEvent the mouse event
	 */
	virtual void mousePressEvent( QMouseEvent* pEvent );

	/**
	 * Handle the mouse release event.
	 *
	 * @param pEvent the mouse event
	 */
	virtual void mouseReleaseEvent( QMouseEvent* event );

	/**
	 * Handle the mouse move event.
	 *
	 * This is used to move selected point if any,
	 * or display a red circle arround a point of the curve when aproaching it.
	 *
	 * @param pEvent the mouse event
	 */
	virtual void mouseMoveEvent( QMouseEvent* event );

	/**
	 * Handle the leave event.
	 *
	 * @param pEvent the leave event
	 */
	virtual void leaveEvent( QEvent* pEvent );

	/**
	 * Handle the resize event
	 *
	 * @param pEvent the resize event
	 */
	virtual void resizeEvent( QResizeEvent* pEvent );

	/**
	 * Convert a point of the curve in the associated widget coordinates system
	 *
	 * @param pPoint a point of the curve
	 *
	 * @return the point in widget coordinates system
	 */
	QPoint listPos2WidgetPos( const QPointF& pPoint );

	/**
	 * Convert a point of the associated widget coordinates system to a point of the curve
	 *
	 * @param pPoint a point in widget coordinates system
	 *
	 * @return the point in the curve
	 */
	QPointF WidgetPos2listPos( const QPoint& pPoint );

	/**
	 * Helper function for linear interpolation of two values
	 *
	 * @param p0 value of first parameter
	 * @param p1 value of second parameter
	 * @param t weight of the interpolation parameter
	 *
	 * @return the interpolated value
	 */
	static qreal interp1( qreal p0, qreal p1, qreal t );

	/**
	 * Helper function for interpolation
	 *
	 * @param p0 ...
	 * @param v0 ...
	 * @param p1 ...
	 * @param v1 ...
	 * @param t ...
	 *
	 * @return the interpolated value
	 */
	static qreal interp2( qreal p0, qreal v0, qreal p1, qreal v1, qreal t );

	/**
	 * Helper function used to compare two points.
	 * Only x position is used.
	 *
	 * @param pPoint1 first point to compare
	 * @param pPoint1 second point to compare
	 */
	static bool cmp( QPointF* pPoint1, QPointF* pPoint2 );

};
 
#endif
