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

#include "QtfeChannel.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qtfe
#include "Qtfe.h"

// Qt
#include <QPainter>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Point size used to draw distinguishable points on the curve.
 */
const int QtfeChannel::_pointSizePixel = 5;

/**
 * Width of circle used to display a circle around points near the mouse button.
 */
const int QtfeChannel::_circleSizePixel = 9;

/**
 * Line width used to draw the curve associated to list of points.
 */
const int QtfeChannel::_lineWidth = 2;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 *
 * @param pParent parent widget (it must be an instance of Qtfe editor
 ******************************************************************************/
QtfeChannel::QtfeChannel( Qtfe* pParent )
:	QWidget( pParent, pParent->windowFlags() )
,	_first( 0.0, 0.0 )
,	_last( 1.0, 1.0 )
,	_list()
,	_pressed( false )
,	_selected( NULL )
,	_min( 0.0 )
,	_max( 1.0 )
,	_background( NULL )
{
	// Size constraint
	setMinimumHeight( 75 );
	
	// Enable mouse tracking
	this->setMouseTracking( true );
	
	// Set the associated curve's first and last points
	_list.push_back( &_first );
	_list.push_back( &_last );

	// Create a background image to display the associated curve
	_background = new QImage( this->size(), QImage::Format_Mono );
	_background->fill( 1 );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
QtfeChannel::~QtfeChannel()
{
	// Destroy background image
	delete _background;

	// Iterate through list of points and destroy them
	for ( int i = 0; i < _list.size(); ++i )
	{
		if ( _list[ i ] != &_first && _list[ i ] != &_last )
		{
			delete _list[ i ];
		}
	}
}

/******************************************************************************
 * Evaluate the function at given "t" parameter.
 *
 * @param t "x" position's parameter (it will be clamped to [ 0.0 ; 1.0 ] if needed) 
 *
 * @return the resulting value
 ******************************************************************************/
qreal QtfeChannel::evalf( qreal t ) const
{
	// Handle lower bound
	if ( t <= 0.0 )
	{
		return _first.y();
	}

	// Handle upper bound
	if ( t >= 1.0 )
	{
		return _last.y();
	}

	// Iterate trough points to find the interval in which "t" belongs.
	// Then, evaluate the function at "t" by interpolating values.
	for ( int i = 0; i < _list.size() - 1; ++i )
	{
		// Retrieve current and next point "x" positions (i.e current interval)
		qreal x0 = _list[ i ]->x();
		qreal x1 = _list[ i + 1 ]->x();

		// If "t" is not in the current interval, go on to next interval
		if ( t < x0 || t > x1 )
		{
			continue;
		}

		// Retrieve corresponding "y" positions (i.e current interval)
		qreal y0 = _list[ i ]->y();
		qreal y1 = _list[ i + 1 ]->y();

		qreal v0 = ( i > 0 ) ? ( _list[ i + 1 ]->y() - _list[ i - 1 ]->y() ) : ( _list[ 1 ]->y() - _list[ 0 ]->y() );
		qreal v1 = ( i < _list.size() - 2 ) ? ( _list[ i + 2 ]->y() - _list[ i ]->y() ) : ( _list[ _list.size() - 1 ]->y() - _list[ _list.size() - 2 ]->y() );

		// Interpolate values and clamp to [ 0.0 ; 1.0 ]
		qreal res = qMin( qMax( interp2( y0, v0, y1, v1, ( t - x0 ) / ( x1 - x0 ) ), 0.0 ), 1.0 );
		return res;
	}

	// Default value
	return 0.0;
}

/******************************************************************************
 * Get the list of points
 *
 * return the list of points
 ******************************************************************************/
const QList< QPointF* >& QtfeChannel::getPoints() const
{
	return _list;
}

/******************************************************************************
 * Set the first point value
 *
 * @param pValue the given value
 ******************************************************************************/
void QtfeChannel::setFirstPoint( qreal pValue )
{
	// Set the first point data
	_first.setY( qMin( qMax( pValue, 0.0 ), 1.0 ) );

	// Update flag of selected point
	_selected = NULL;

	// Repaint widget
	repaint();

	// Emit signal
	emit channelChanged(); 
}

/******************************************************************************
 * Set the last point value
 *
 * @param pValue the given value
 ******************************************************************************/
void QtfeChannel::setLastPoint( qreal pValue )
{
	// Set the last point data
	_last.setY( qMin( qMax( pValue, 0.0 ), 1.0 ) );

	// Update flag of selected point
	_selected = NULL;

	// Repaint widget
	repaint();

	// Emit signal
	emit channelChanged();
}

/******************************************************************************
 * Insert a point
 *
 * @param pPoint the point to add
 ******************************************************************************/
void QtfeChannel::insertPoint( const QPointF& pPoint )
{
	// Create a new point and store it
	QPointF* point = new QPointF( qMin( qMax( pPoint.x(), 0.0 ), 1.0 ), qMin( qMax( pPoint.y(), 0.0 ), 1.0 ) );
	_list.push_back( point );

	// Update flag of selected point
	_selected = NULL;

	// Sort the list of points
	qSort( _list.begin(), _list.end(), cmp );

	// Repaint widget
	repaint();
	
	// Emit signal
	emit channelChanged();
}

/******************************************************************************
 * Handle the resize event
 *
 * @param pEvent the resize event
 ******************************************************************************/
void QtfeChannel::resizeEvent ( QResizeEvent* pEvent )
{
	// Destroy background image
	delete _background;

	// Create a new background image
	_background = new QImage( pEvent->size(), QImage::Format_Mono );
	_background->fill( 1 );
}

/******************************************************************************
 * Convert a point of the curve in the associated widget coordinates system
 *
 * @param pPoint a point of the curve
 *
 * @return the point in widget coordinates system
 ******************************************************************************/
QPoint QtfeChannel::listPos2WidgetPos( const QPointF& pPoint )
{
	return QPoint( pPoint.x() * width(), height() * ( 1.0 - pPoint.y() ) );
}

/******************************************************************************
 * Convert a point of the associated widget coordinates system to a point of the curve
 *
 * @param pPoint a point in widget coordinates system
 *
 * @return the point in the curve
 ******************************************************************************/
QPointF QtfeChannel::WidgetPos2listPos( const QPoint& pPoint )
{
	return QPointF( pPoint.x() / static_cast< qreal >( width() ), 1.0 - pPoint.y() / static_cast< qreal >( height() ) );
}

/******************************************************************************
 * Handle the paint event.
 * Draw all elements (points, interpolated curve and selected point if any)
 *
 * @param pEvent the paint event
 ******************************************************************************/
void QtfeChannel::paintEvent( QPaintEvent* pEvent )
{
	// Initialize painter
	QPainter painter( this );
	painter.setRenderHint( QPainter::Antialiasing, true );

	// Draw image background
	painter.drawImage( 0, 0, *_background );

	// Draw all points of the curve
	QPen pen( Qt::black, _pointSizePixel, Qt::SolidLine );
	painter.setPen( pen );
	for ( int i = 0; i < _list.size(); ++i )
	{
		painter.drawPoint( listPos2WidgetPos( *_list[ i ] ) );
	}

	// Draw curve (with interpolated points between previous points)
	pen.setWidth( _lineWidth );
	painter.setPen( pen );
	qreal x0 = 0.0;
	qreal y0 = evalf( x0 );
	for ( int p = 1; p < _list.size(); ++p )
	for ( int i = 1;  i <= 10; ++i )
	{
		qreal x1 = interp1( _list[ p - 1 ]->x(), _list[ p ]->x(), i / 10.0 );
		qreal y1 = evalf( x1 );
		painter.drawLine( listPos2WidgetPos( QPointF( x0, y0 ) ), listPos2WidgetPos( QPointF( x1, y1 ) ) );
		x0 = x1;
		y0 = y1;
	}

	// Draw selected point if any
	pen.setColor( Qt::red );
	painter.setPen( pen );
	if ( _selected )
	{
		painter.drawEllipse( listPos2WidgetPos( *_selected ), _circleSizePixel, _circleSizePixel );
	}

	// Call base class paint event handler
	QWidget::paintEvent( pEvent );
}

/******************************************************************************
 * Handle the mouse press event.
 *
 * Left click is used to add points.
 * Right click is used to remove selected point if any.
 *
 * @param pEvent the mouse event
 ******************************************************************************/
void QtfeChannel::mousePressEvent( QMouseEvent* pEvent )
{
	// Convert mouse button position to curve's coordinate system
	QPointF pf = WidgetPos2listPos( pEvent->pos() );

	// Handle left mouse button.
	// Left click is used to add points.
	if ( pEvent->button() == Qt::LeftButton )
	{
		// Update point selection's flag
		_pressed = true;

		// If there is no selected point
		if ( ! _selected )
		{
			// Iterate through list of points
			for ( int i = 1; i < _list.size(); ++i )
			{
				if ( _list[ i - 1 ]->x() < pf.x() && pf.x() < _list[ i ]->x() )
				{
					_min = _list[ i - 1 ]->x() + 0.01;
					_max = _list[ i ]->x() - 0.01;

					break;
				}
			}

			// Create a newly selected point and store it
			_selected = new QPointF( pf );
			_list.push_back( _selected );

			// Sort list of points
			qSort( _list.begin(), _list.end(), cmp );

			// Repaint the associated widget
			this->repaint();

			// Emit signal
			emit channelChanged();
		}
	}

	// Handle right mouse button.
	// Right click is used to remove selected point if any.
	if ( pEvent->button() == Qt::RightButton && _selected )
	{
		if ( _selected != &_first && _selected != &_last )
		{
			// Remove selected point from list of points
			_list.removeOne( _selected );

			// Destroy selected point
			delete _selected;
			_selected = NULL;

			// Repaint the associated widget
			this->repaint();

			// Emit signal
			emit channelChanged();
		}
	}
}

/******************************************************************************
 * Handle the mouse release event.
 *
 * @param pEvent the mouse event
 ******************************************************************************/
void QtfeChannel::mouseReleaseEvent ( QMouseEvent* pEvent )
{	
	// Handle left mouse button.
	if ( pEvent->button() == Qt::LeftButton )
	{
		// Update point selection's flag
		_pressed = false;
	}
}

/******************************************************************************
 * Handle the mouse move event.
 *
 * This is used to move selected point if any,
 * or display a red circle arround a point of the curve when aproaching it.
 *
 * @param pEvent the mouse event
 ******************************************************************************/
void QtfeChannel::mouseMoveEvent( QMouseEvent* pEvent )
{
	// Convert mouse button position to curve's coordinate system
	QPointF pf = WidgetPos2listPos( pEvent->pos() );

	// If mouse button has already been pressed (i.e. left click),
	// move selected point if any.
	if ( _pressed )
	{
		// If a point has already been selected
		if ( _selected )
		{
			// Move selected point (position is clamped)
			if ( _selected != &_first && _selected != &_last )
			{				
				_selected->setX( qMin( qMax( pf.x(), _min ), _max ) );
			}
			_selected->setY( qMin( qMax( pf.y(), 0.0 ), 1.0 ) );

			// Repaint the associated widget
			this->repaint();

			// Emit signal
			emit channelChanged();

			// Exit
			return;
		}	
	}
	else
	{	
		// If no button has been pressed,
		// search if the mouse position is inside the circle
		// surrounding a point of the list of points (of the curve).
		// If so, set it as selected, in order to display a red circle around it.
		
		QPointF* nearest = NULL;

		qreal d_min = _circleSizePixel * _circleSizePixel;
		qreal W = width() * width();
		qreal H = height() * height();
		
		// Iterate through list of points
		for ( int i = 0; i < _list.size(); ++i )
		{
			qreal x = _list[ i ]->x() - pf.x();
			qreal y = _list[ i ]->y() - pf.y();

			qreal d = x * x * W + y * y * H;
			if ( d < d_min )
			{
				nearest = _list[ i ];
				d_min = d;

				// Handle first point
				if ( i == 0 )
				{
					_min = 0.0;
					_max = 0.0;
				}
				// Handle last point
				else if ( i == _list.size() - 1 )
				{
					_min = 1.0;
					_max = 1.0;
				}
				// Handle intermediate points
				else
				{
					_min = _list[ i - 1]->x() + 0.01;
					_max = _list[ i + 1 ]->x() - 0.01;
				}
			}
		}

		// Check if a new selected point point has been found
		if ( nearest != _selected )
		{	
			// Update selected point
			_selected = nearest;

			// Repaint the associated widget
			this->repaint();
		}
	}
}

/******************************************************************************
 * Handle the leave event.
 *
 * @param pEvent the leave event
 ******************************************************************************/
void QtfeChannel::leaveEvent( QEvent* pEvent )
{
	// Reset selected point
	_selected = NULL;

	// Repaint the associated widget
	this->repaint();

	// Call base class leave event handler
	QWidget::leaveEvent( pEvent );
}

/******************************************************************************
 * Helper function for linear interpolation of two values
 *
 * @param p0 value of first parameter
 * @param p1 value of second parameter
 * @param t weight of the interpolation parameter
 *
 * @return the interpolated value
 ******************************************************************************/
qreal QtfeChannel::interp1( qreal p0, qreal p1, qreal t )
{
	return ( 1.0f - t ) * p0 + t * p1;
}

/******************************************************************************
 * Helper function for interpolation
 *
 * @param p0 ...
 * @param v0 ...
 * @param p1 ...
 * @param v1 ...
 * @param t ...
 *
 * @return the interpolated value
 ******************************************************************************/
qreal QtfeChannel::interp2( qreal p0, qreal v0, qreal p1, qreal v1, qreal t )
{
	return t * ( t * ( t * ( 2.0f * p0 + v0 - 2.0f * p1 + v1 ) + ( -3.0f * p0 - 2.0f * v0 + 3.0f * p1 - v1 ) ) + v0 ) + p0;
}

/******************************************************************************
 * Helper function used to compare two points.
 * Only x position is used.
 *
 * @param pPoint1 first point to compare
 * @param pPoint1 second point to compare
 ******************************************************************************/
bool QtfeChannel::cmp( QPointF* pPoint1, QPointF* pPoint2 )
{
	return pPoint1->x() <= pPoint2->x();
}
