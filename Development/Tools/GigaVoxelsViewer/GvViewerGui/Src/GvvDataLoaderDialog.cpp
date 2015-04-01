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

#include "GvvDataLoaderDialog.h"

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
GvvDataLoaderDialog::GvvDataLoaderDialog( QWidget* pParent ) 
:	QDialog( pParent )
{
	//** Set the name
	setAccessibleName( qApp->translate( "GvvDataLoaderDialog", "Data Loader Dialog" ) );

	//** Initalizes the dialog
	setupUi( this );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvDataLoaderDialog::~GvvDataLoaderDialog()
{
}

/******************************************************************************
 * Slot called when 3D window background color tool button is released
 ******************************************************************************/
void GvvDataLoaderDialog::on__3DModelToolButton_released()
{
	QString file = QFileDialog::getOpenFileName(this,"Select the XML file describing the 3D model",".","*.xml",NULL/*, QFileDialog::Option::DontUseNativeDialog*/);
	if ( ! file.isEmpty() )
	{
		_3DModelLineEdit->setText(file);

	}
}



/******************************************************************************
 * Get the 3D model filename to load
 *
 * @return the 3D model filename to load
 ******************************************************************************/
QString GvvDataLoaderDialog::get3DModelFilename() const
{
	//QString directory = _3DModelLineEdit->text();
	QString name = _3DModelLineEdit->text();
	

	QString filename = name;//directory + QDir::separator() + name;

	// TO DO
	// Test the existence...

	return filename;
}

/******************************************************************************
 * Get the 3D model resolution
 *
 * @return the 3D model resolution
 ******************************************************************************/
unsigned int GvvDataLoaderDialog::get3DModelResolution() const
{
	//unsigned int maxResolution =  ( 1 << _maxResolutionComboBox->currentIndex() ) * 8;

	return 0;//maxResolution;
}
