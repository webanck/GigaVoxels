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

#ifndef _Q_TFE_
#define _Q_TFE_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qt
#include <QWidget>
#include <QVector>

// Qtfe
#include "QtfeChannel.h"
#include "QtfeOutput.h"
#include "UI_GvQTransferFunctionEditor.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class Qtfe
 *
 * @brief The Qtfe class provides the main interface of create and edit
 * transfer functions.
 *
 * Qtfe stands for Qt Transfer Function Editor.
 * It is made of an editor and channels bound to outputs.
 *
 * Most of the time, user defines four channels associated to red, green, blue 
 * and alpha components and one output that represents the resulting 2D curve
 * sculpted by the alpha component.
 */
class Qtfe : public QWidget, public Ui::GvQTransferFunctionEditor
{
	// Qt macro
	Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param pParent parent widget
	 * @param pFlags window flags
	 */
	Qtfe( QWidget* pParent = 0, Qt::WindowFlags pFlags = 0 );
	
	/**
	 * Destructor
	 */
	~Qtfe();

	/**
	 * Initialize the editor by loading data from file
	 *
	 * @pFilename the file in Qtfe format
	 */
	bool load( const QString& pFilename );

	/**
	 * Add new channels
	 *
	 * @param pNbChannels the number of new channels
	 */
	void addChannels( int pNbChannels );

	/**
	 * Add new RGBA outputs
	 *
	 * @param pNbOuputs the number of new outputs
	 */
	void addOutputs( int n );
	
	/**
	 * Bind a channel to an output (-1 to disable the channel)
	 *
	 * @param pChannelIndex index of channel
	 * @param pOutputIndex index of output
	 */
	void bindChannelToOutputR( int pChannelIndex, int pOutputIndex );

	/**
	 * Bind a channel to an output (-1 to disable the channel)
	 *
	 * @param pChannelIndex index of channel
	 * @param pOutputIndex index of output
	 */
	void bindChannelToOutputG( int pChannelIndex, int pOutputIndex );

	/**
	 * Bind a channel to an output (-1 to disable the channel)
	 *
	 * @param pChannelIndex index of channel
	 * @param pOutputIndex index of output
	 */
	void bindChannelToOutputB( int pChannelIndex, int pOutputIndex );

	/**
	 * Bind a channel to an output (-1 to disable the channel)
	 *
	 * @param pChannelIndex index of channel
	 * @param pOutputIndex index of output
	 */
	void bindChannelToOutputA( int pChannelIndex, int pOutputIndex );
	
	/**
	 * Get the transfer function dimension (channels number)
	 *
	 * @return the nb of channels
	 */
	int dim() const;
	
	/**
	 * Evaluate channel at x in [0.0, 1.0].
	 * Invalid channel or x values return 0.0
	 *
	 * @param pChannelIndex index of channel
	 * @param pXValue x position where to evaluate the function
	 *
	 * @return the function transfer value at given position
	 */
	qreal evalf( int pChannelIndex, qreal pXValue ) const;

	/******************************** SIGNALS *********************************/

signals:

	/**
	 * Signal emitted if at least one channel has changed
	 */
	void functionChanged();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * List of channels
	 */
	QVector< QtfeChannel* > _channels;

	/**
	 * List of outputs
	 */
	QVector< QtfeOutput* > _outputs;

	/**
	 * File name associated to transfer function
	 */
	QString _filename;

	/******************************** METHODS *********************************/

	/********************************* SLOTS **********************************/

protected slots:

	/**
	 * Slot called when the Save button has been clicked.
	 *
	 * Serialize the transfer funtion in file.
	 * Currently, format is XML document.
	 */
	void onSaveButtonClicked();

	/**
	 * Slot called when the Save As button has been clicked.
	 *
	 * Open a user interface dialog to save data in file.
	 * Currently, format is XML document.
	 */
	void onSaveAsButtonClicked();

	/**
	 * Slot called when the Load button has been clicked.
	 *
	 * Open a user interface dialog to load a transfer function.
	 * Currently, format is XML document.
	 */
	void onLoadButtonClicked();

	/**
	 * Slot called when a channel has been modified.
	 */
	void onChannelChanged();

	/**
	 * Slot called when an output binding has been modified.
	 */
	void onOutputBindingChanged();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

};

#endif
