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

#ifndef _Q_TFE_OUTPUT_
#define _Q_TFE_OUTPUT_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

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

// Qt
class QSpinBox;
class QMultiEditor;

// Qtfe
class Qtfe;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class QtfeOutput
 *
 * @brief The QtfeOutput class provides the interface of an output of the main Qtfe editor.
 *
 * An output is made of four components R,G,B,A.
 * Each R,G,B,A values is packed as an #AARRGGBB quadruplet, from wich Alpha component
 * is used to sculpt the associated 2D curve.
 *
 * Warning : QtfeOutput is not supposed to be used outside Qtfe
 */
class QtfeOutput : public QWidget 
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
	 * Signal emitted when an output binding has been modified
	 */
	void outputBindingChanged();

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
	 * Parent widget is the main Qtfe editor.
	 */
	const Qtfe* _parent;

	/**
	 * Red component's slot index
	 */
	int _R;
	
	/**
	 * Green component's slot index
	 */
	int _G;

	/**
	 * Blue component's slot index
	 */
	int _B;

	/**
	 * Alpha component's slot index
	 */
	int _A;

	/**
	 * Widget used to draw the curve associated to this output.
	 * An computed image of the curve is draw all over its area.
	 */
	QWidget* _background;

	/**
	 * Spin box editor assiciated to "red" component
	 */
	QSpinBox* _qspinboxR;

	/**
	 * Spin box editor assiciated to "green" component
	 */
	QSpinBox* _qspinboxG;

	/**
	 * Spin box editor assiciated to "blue" component
	 */
	QSpinBox* _qspinboxB;

	/**
	 * Spin box editor assiciated to "alpha" component
	 */
	QSpinBox* _qspinboxA;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param pParent parent widget (it must be an instance of Qtfe editor)
	 */
	QtfeOutput( Qtfe* pParent );

	/**
	 * Handle the paint event.
	 *
	 * Draw the curve associated to bound channels.
	 * Apha component is used to sculpt the curve,
	 * i.e. alpha is the "y" component of the 2D drawn function.
	 *	
	 * We only display data under the alpha curve.
	 *
	 * @param pEvent the paint event
	 */
	virtual void paintEvent( QPaintEvent* pEvent );

	/********************************* SLOTS **********************************/

private slots:

	/**
	 * Bind a channel to output slot
	 *
	 * @param pChannelIndex index of channel
	 */
	void bindChannelToR( int pChannelIndex );

	/**
	 * Bind a channel to output slot
	 *
	 * @param pChannelIndex index of channel
	 */
	void bindChannelToG( int pChannelIndex );

	/**
	 * Bind a channel to output slot
	 *
	 * @param pChannelIndex index of channel
	 */
	void bindChannelToB( int pChannelIndex );

	/**
	 * Bind a channel to output slot
	 *
	 * @param pChannelIndex index of channel
	 */
	void bindChannelToA( int pChannelIndex );

};

#endif
