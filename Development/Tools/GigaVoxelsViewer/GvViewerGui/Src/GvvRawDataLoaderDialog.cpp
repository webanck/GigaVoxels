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

#include "GvvRawDataLoaderDialog.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qt
#include <QFileDialog>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
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
 * Default constructor.
 ******************************************************************************/
GvvRawDataLoaderDialog::GvvRawDataLoaderDialog( QWidget* pParent ) 
:	QDialog( pParent )
{
	//** Set the name
	setAccessibleName( qApp->translate( "GvvRawDataLoaderDialog", "RAW Data Loader Dialog" ) );

	//** Initalizes the dialog
	setupUi( this );

	// Custom initialization
	_nbLevelsLineEdit->setText( QString::number( get3DModelResolution() ) );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvRawDataLoaderDialog::~GvvRawDataLoaderDialog()
{
}

/******************************************************************************
 * Slot called when 3D window background color tool button is released
 ******************************************************************************/
void GvvRawDataLoaderDialog::on__3DModelDirectoryToolButton_released()
{
	QString filename = QFileDialog::getOpenFileName( this, "Choose a file", QString( "." ), tr( "RAW data file (*.raw)" ) );
	if ( ! filename.isEmpty() )
	{
		_3DModelFilenameLineEdit->setText( filename );
	}
}

/******************************************************************************
 * Slot called when License push button is released.
 ******************************************************************************/
void GvvRawDataLoaderDialog::on__maxResolutionComboBox_currentIndexChanged( const QString& pText )
{
	unsigned int maxResolution = pText.toUInt() - 1;
	_nbLevelsLineEdit->setText( QString::number( ( 1 << maxResolution ) * 8 ) );
}

/******************************************************************************
 * Get the 3D model filename to load
 *
 * @return the 3D model filename to load
 ******************************************************************************/
QString GvvRawDataLoaderDialog::get3DModelFilename() const
{
	
	return _3DModelFilenameLineEdit->text();
	
}

/******************************************************************************
 * Get the 3D model resolution
 *
 * @return the 3D model resolution
 ******************************************************************************/
unsigned int GvvRawDataLoaderDialog::get3DModelResolution() const
{
	unsigned int maxResolution =  ( 1 << _maxResolutionComboBox->currentIndex() ) * 8;

	return maxResolution;
}
