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

#ifndef GVVPREFERENCESDIALOG_H
#define GVVPREFERENCESDIALOG_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "UI_GvvQPreferencesDialog.h"

// Qt
#include <QDialog>

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

class GVVIEWERGUI_EXPORT GvvPreferencesDialog : public QDialog, public Ui::GvvQPreferencesDialog
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
	 * Default constructor.
	 */
	GvvPreferencesDialog( QWidget* pParent = NULL );

	/**
	 * Destructor.
	 */
	virtual ~GvvPreferencesDialog();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/********************************* SLOTS **********************************/
		
protected slots:

	/**
	 * Slot called when 3D window background color tool button is released
	 */
	void on__3DWindowBackgroundColorToolButton_released();

	/**
	 * Slot called when data structure appearance color tool button is released
	 */
	void on__nodeHasBrickTerminalColorToolButton_released();

	/**
	 * Slot called when data structure appearance color tool button is released
	 */
	void on__nodeHasBrickNotTerminalColorToolButton_released();

	/**
	 * Slot called when data structure appearance color tool button is released
	 */
	void on__nodeIsBrickNotInCacheColorToolButton_released();

	/**
	 * Slot called when data structure appearance color tool button is released
	 */
	void on__nodeEmptyOrConstantColorToolButton_released();

	/**
	 * Slot called when data structure appearance check box is toggled
	 */
	void on__nodeHasBrickTerminalCheckBox_toggled( bool pChecked );
	
	/**
	 * Slot called when data structure appearance check box is toggled
	 */
	void on__nodeHasBrickNotTerminalCheckBox_toggled( bool pChecked );

	/**
	 * Slot called when data structure appearance check box is toggled
	 */
	void on__nodeIsBrickNotInCacheCheckBox_toggled( bool pChecked );

	/**
	 * Slot called when data structure appearance check box is toggled
	 */
	void on__nodeEmptyOrConstantCheckBox_toggled( bool pChecked );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/
	
	/**
	 * Copy constructor forbidden.
	 */
	GvvPreferencesDialog( const GvvPreferencesDialog& );

	/**
	 * Copy operator forbidden.
	 */
	GvvPreferencesDialog& operator=( const GvvPreferencesDialog& );

	/********************************* SLOTS **********************************/

};

#endif
