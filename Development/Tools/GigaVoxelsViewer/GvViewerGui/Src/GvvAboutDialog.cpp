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

#include "GvvAboutDialog.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvApplication.h"

// Qt
#include <QMessageBox>

// STL
#include <string>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// STL
using namespace std;

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
GvvAboutDialog::GvvAboutDialog( QWidget* pParent ) 
:	QDialog( pParent )
{
	//** Set the name
	setAccessibleName( qApp->translate( "GvvAboutDialog", "About Dialog" ) );

	//** Initalizes the dialog
	setupUi( this );

	//** Populate the dialog
	//mAboutPixmap->setPixmap( GvvEnvironment::get().getSystemFilePath( GvvEnvironment::eAboutScreenFile ) );

	//mVersionLabel->setText( QString::number(GVV_VERSION_MAJOR) + "." +
	//					QString::number(GVV_VERSION_MINOR) );

	//** Populate the plug-ins list
	//** - empty the table
	//mPluginsList->clear();

	//** - layout the table
	//mPluginsList->verticalHeader()->hide();
	//mPluginsList->verticalHeader()->setDefaultSectionSize( 18 );

	//** - set table size
	//mPluginsList->setColumnCount( 1 );
	//mPluginsList->setRowCount( GvvPluginManager::get().getNbPlugins() );

	//** - set table column headers
	//QStringList lLabels;
	//lLabels.append( qApp->translate( "GvvAboutDialog", "Installed plugins" ) );
	//mPluginsList->setHorizontalHeaderLabels( lLabels );
	//mPluginsList->horizontalHeader()->setResizeMode ( QHeaderView::Stretch );

	////** - fill the table
	//for
	//	( int lInd = 0; lInd < GvvPluginManager::get().getNbPlugins(); lInd++ )
	//{
	//	const GvvPlugin* lPlugin = GvvPluginManager::get().getPlugin( lInd );

	//	//** plug-ins name
	//	QTableWidgetItem* lNameItem = new QTableWidgetItem( lPlugin->getName() );
	//	mPluginsList->setItem( lInd, 0, lNameItem );
	//	lNameItem->setFlags( Qt::ItemIsEnabled );
	//}
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvAboutDialog::~GvvAboutDialog()
{
}

/******************************************************************************
 * Slot called when Credits push button is released.
 ******************************************************************************/
void GvvAboutDialog::on__creditsPushButton_released()
{
	QMessageBox::information( this, tr( "Credits" ), tr( "Not yet implemented..." ) );
}

/******************************************************************************
 * Slot called when License push button is released.
 ******************************************************************************/
void GvvAboutDialog::on__licensePushButton_released()
{
	QMessageBox::information( this, tr( "License" ), tr( "Not yet implemented..." ) );
}
