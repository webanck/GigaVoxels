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

#ifndef _GVX_VOXELIZER_DIALOG_H
#define _GVX_VOXELIZER_DIALOG_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "UI_GvxQVoxelizerDialog.h"

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

namespace Gvx
{

/** 
 * @class GvxVoxelizerDialog
 *
 * @brief The GvxVoxelizerDialog class provides a widget to edit settings
 * that will be used during voxelization.
 *
 * It enables users to edit configuration parameters used to voxelize data
 * (i.e. filename, max level of resolution, width of a brick, etc...)
 */
class GvxVoxelizerDialog : public QDialog, public Ui::GvxQVoxelizerDialog
{
	// Qt macro
	Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * 3D model filename
	 */
	QString _fileName;

	/**
	 * Max scene resolution
	 */
	unsigned int _maxResolution;

	/**
	 * Flag to tell wheter or not to generate normals
	 */
	bool _isGenerateNormalsOn;

	/**
	 * Brick width
	 */
	unsigned int _brickWidth;

	/**
	 * Data type
	 */
	unsigned int _dataType;

	/**
	 * Number of filter iteration
	 */
	unsigned int _nbFilterOperation;

	/**
	 * Filter type
	 * 0 = mean
	 * 1 = gaussian
	 * 2 = laplacian
	 */
	unsigned int _filterType;

	/* 
	 * build normal field ? 
	 */
	bool _normals;

	/******************************** METHODS *********************************/

	/**
	 * Default constructor.
	 */
	GvxVoxelizerDialog( QWidget* pParent = NULL );

	/**
	 * Destructor.
	 */
	virtual ~GvxVoxelizerDialog();

	/********************************* SLOTS **********************************/

public slots:

	/**
	 * Override QDialog accept() method
	 * (Hides the modal dialog and sets the result code to Accepted)
	 */
	virtual void accept();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/********************************* SLOTS **********************************/
		
protected slots:

	/**
	 * Slot called when Credits push button is released
	 */
	void on__3DModelToolButton_released();

	/**
	 * Slot called when License push button is released
	 */
	void on__maxResolutionComboBox_currentIndexChanged( const QString& pText );

	/**
	 * Set the Filter type
	 * 0 = mean
	 * 1 = gaussian
	 * 2 = laplacian
	 */
	void on__filterTypeComboBox_currentIndexChanged (int pValue);

	/**
	 * Set the Number of filter iteration
	 */
	void on__filterIterationsSpinBox_valueChanged( int pValue);

	/*
	 * Set whether or not we produce the normal field
	 */
	void on__generateNormalsCheckBox_toggled (bool pChecked);

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/
	
	/**
	 * Copy constructor forbidden.
	 */
	GvxVoxelizerDialog( const GvxVoxelizerDialog& );

	/**
	 * Copy operator forbidden.
	 */
	GvxVoxelizerDialog& operator=( const GvxVoxelizerDialog& );

	/********************************* SLOTS **********************************/

};

}

#endif
