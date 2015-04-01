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

#include "QtfeOutput.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qtfe
#include "Qtfe.h"

// Qt
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QSpinBox>
#include <QPainter>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

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
 * Constructor
 *
 * @param pParent parent widget (it must be an instance of Qtfe editor
 ******************************************************************************/
QtfeOutput::QtfeOutput( Qtfe* pParent )
:	QWidget( pParent, pParent->windowFlags() )
,	_parent( pParent )
{
	// Size constraint
	setMinimumHeight( 50 );

	// Initialize output indexes
	_R = -1;
	_B = -1;
	_G = -1;
	_A = -1;

	// Main "output" widget, child of the main Qtfe editor
	_background = new QWidget( this );

	// Red component's slot widget editor
	QLabel* labelR = new QLabel( "R", this );
	_qspinboxR = new QSpinBox( this );
	_qspinboxR->setMinimum( -1 );
	_qspinboxR->setValue( _R );
	QObject::connect( _qspinboxR, SIGNAL( valueChanged( int ) ), this, SLOT( bindChannelToR( int ) ) );

	// Green component's slot widget editor
	QLabel* labelG = new QLabel( "G", this );
	_qspinboxG = new QSpinBox( this );
	_qspinboxG->setMinimum( -1 );
	_qspinboxG->setValue( _G );
	QObject::connect( _qspinboxG, SIGNAL( valueChanged( int ) ), this, SLOT( bindChannelToG( int ) ) );

	// Blue component's slot widget editor
	QLabel* labelB = new QLabel( "B", this );
	_qspinboxB = new QSpinBox( this );
	_qspinboxB->setMinimum( -1 );
	_qspinboxB->setValue( _B );
	QObject::connect( _qspinboxB, SIGNAL( valueChanged( int ) ), this, SLOT( bindChannelToB( int ) ) );

	// Alpha component's slot widget editor
	QLabel* labelA = new QLabel( "A", this );
	_qspinboxA = new QSpinBox( this );
	_qspinboxA->setMinimum( -1 );
	_qspinboxA->setValue( _A );
	QObject::connect( _qspinboxA, SIGNAL( valueChanged( int ) ), this, SLOT( bindChannelToA( int ) ) );

	// Layout used to position previous R,G,B,A widget editors
	//QGridLayout* layoutChannels = new QGridLayout;
	//layoutChannels->addWidget( labelR, 0, 0 );
	//layoutChannels->addWidget( _qspinboxR, 0, 1 );
	//layoutChannels->addWidget( labelG, 0, 2 );
	//layoutChannels->addWidget( _qspinboxG, 0, 3 );
	//layoutChannels->addWidget( labelB, 1, 0 );
	//layoutChannels->addWidget( _qspinboxB, 1, 1 );
	//layoutChannels->addWidget( labelA, 1, 2 );
	//layoutChannels->addWidget( _qspinboxA, 1, 3 );
	QHBoxLayout* layoutChannels = new QHBoxLayout;
	layoutChannels->addWidget( labelR );
	layoutChannels->addWidget( _qspinboxR );
	layoutChannels->addStretch();
	layoutChannels->addWidget( labelG );
	layoutChannels->addWidget( _qspinboxG );
	layoutChannels->addStretch();
	layoutChannels->addWidget( labelB );
	layoutChannels->addWidget( _qspinboxB );
	layoutChannels->addStretch();
	layoutChannels->addWidget( labelA );
	layoutChannels->addWidget( _qspinboxA );

	// Main widget layout used to position the image background next to the previous R,G,B,A widget editors
	//QHBoxLayout* layout = new QHBoxLayout( this );
	QVBoxLayout* layout = new QVBoxLayout( this );
	layout->setContentsMargins(0,0,0,0);
	layout->addWidget( _background, 5 );
	layout->addLayout( layoutChannels );
}

/******************************************************************************
 * Bind a channel to output slot
 *
 * @param pChannelIndex index of channel
 ******************************************************************************/
void QtfeOutput::bindChannelToR( int pChannelIndex )
{
	// Update internal value if valid
	if ( -1 <= pChannelIndex && pChannelIndex < _parent->dim() )
	{
		_R = pChannelIndex;
	}

	// Update associated widget editor
	_qspinboxR->setValue( _R );

	// Repaint the associated widget
	repaint();

	// Emit signal
	emit outputBindingChanged(); 
}

/******************************************************************************
 * Bind a channel to output slot
 *
 * @param pChannelIndex index of channel
 ******************************************************************************/
void QtfeOutput::bindChannelToG( int pChannelIndex )
{
	// Update internal value if valid
	if ( -1 <= pChannelIndex && pChannelIndex < _parent->dim() )
	{
		_G = pChannelIndex;
	}

	// Update associated widget editor
	_qspinboxG->setValue( _G );

	// Repaint the associated widget
	repaint();

	// Emit signal
	emit outputBindingChanged(); 
}

/******************************************************************************
 * Bind a channel to output slot
 *
 * @param pChannelIndex index of channel
 ******************************************************************************/
void QtfeOutput::bindChannelToB( int pChannelIndex )
{
	// Update internal value if valid
	if ( -1 <= pChannelIndex && pChannelIndex < _parent->dim() )
	{
		_B = pChannelIndex;
	}

	// Update associated widget editor
	_qspinboxB->setValue( _B );

	// Repaint the associated widget
	repaint();

	// Emit signal
	emit outputBindingChanged(); 
}

/******************************************************************************
 * Bind a channel to output slot
 *
 * @param pChannelIndex index of channel
 ******************************************************************************/
void QtfeOutput::bindChannelToA( int pChannelIndex )
{
	// Update internal value if valid
	if ( -1 <= pChannelIndex && pChannelIndex < _parent->dim() )
	{
		_A = pChannelIndex;
	}

	// Update associated widget editor
	_qspinboxA->setValue( _A );

	// Repaint the associated widget
	repaint();

	// Emit signal
	emit outputBindingChanged(); 
}

/******************************************************************************
 * Handle the paint event.
 *
 * Draw the curve associated to bound channels.
 * Apha component is used to sculpt the curve,
 * i.e. alpha is the "y" component of the 2D drawn function.
 *	
 * We only display data under the alpha curve.
 *
 * @param pEvent the paint event
 ******************************************************************************/
void QtfeOutput::paintEvent( QPaintEvent* pEvent )
{
	// Initialize painter
	QPainter painter( this );
	painter.setRenderHint( QPainter::Antialiasing, true );
	
	// Create a temporary image in which to draw pixels
	QImage image( _background->size(), QImage::Format_ARGB32 );

	// Iterate through image width's pixels.
	// Apha components are used to draw curve,
	// i.e. alpha is the "y" component of the 2D drawn function.
	//
	// We only display data under the alpha curve.
	for ( int i = 0; i < _background->width(); ++i )
	{
		// Retrieve "x" normalized position in the curve associated to current pixel
		qreal x = i / static_cast< qreal >( _background->width() );

		// Evaluate bound channels at given "x" position
		qreal r = _parent->evalf( _R, x );
		qreal g = _parent->evalf( _G, x );
		qreal b = _parent->evalf( _B, x );
		qreal a = ( -1 < _A && _A < _parent->dim() ) ? _parent->evalf( _A, x ) : 1.0;

		// Convert computed [r,g,b,a] data to Qt color.
		// Note : Qt color are stored in a #AARRGGBB quadruplet equivalent to an "unsigned int".
		unsigned int R = r * 255;
		unsigned int G = g * 255;
		unsigned int B = b * 255;
		unsigned int color = ( R << 16 ) + ( G << 8 ) + B;

		// Determine the height position corresponding to alpha.
		int jmin = ( 1.0 - a ) * _background->height();

		// Iterate through image height's pixels.
		// Image (0,0) is top left, so here color is full transparent.
		for ( int j = 0; j < jmin; ++j )
		{
			image.setPixel( i, j, color );
		}

		// Iterate through image height's pixels.
		// Image (0,0) is top left, so here color is full opaque.
		for ( int j = jmin; j < _background->height(); ++j )
		{
			// Qt color are stored in a #AARRGGBB quadruplet equivalent to an unsigned int.
			// Here, alpha if forced to be opaque.
			image.setPixel( i, j, color + 0xff000000 );
		}
	}

	// Draw the image
	painter.drawImage( _background->pos(), image );

	// Call base class paint event handler
	QWidget::paintEvent( pEvent );
}
