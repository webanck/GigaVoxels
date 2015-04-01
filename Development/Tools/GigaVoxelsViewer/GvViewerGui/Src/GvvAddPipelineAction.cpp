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

#include "GvvAddPipelineAction.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "GvvPluginManager.h"

// Qt
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QProcess>
#include <QDesktopServices>
#include <QUrl>
#include <QMessageBox>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerGui;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/
/**
 * The unique name of the action
 */
const QString GvvAddPipelineAction::cName = "addPipeline";

/**
 * The default text assigned to the action
 */
const char* GvvAddPipelineAction::cDefaultText = QT_TR_NOOP( "Add Pipeline" );

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructs an action dependant of the applications project
 *
 * @param	pFileName	specifies the filename of the manual
 * @param	pText		specifies the descriptive text of this action
 * @param	pIconName	specifies the name of the icon for this action located in the icons application path
 *							Does nothing if the string is empty. A full file path can also be given.
 * @param	pIsToggled	specified if the action is toggled or not
 ******************************************************************************/
GvvAddPipelineAction::GvvAddPipelineAction( const QString& pText, 
										const QString& pIconName,
										bool pIsToggled )
:	GvvAction( GvvApplication::get().getMainWindow(), cName, pText, pIconName, pIsToggled )
{
	//** Sets the status tip
	setStatusTip( qApp->translate( "GvvAddPipelineAction", "Add Pipeline" ) );
	setShortcut( qApp->translate( "GvvAddPipelineAction", "P" ) );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvAddPipelineAction::~GvvAddPipelineAction()
{
}

/******************************************************************************
 * Overwrites the execute action
 ******************************************************************************/
void GvvAddPipelineAction::execute()
{
	QString fileName = QFileDialog::getOpenFileName( GvvApplication::get().getMainWindow(), "Choose a file", QString( "." ), tr( "GigaVoxels Pipeline Files (*.gvp)" ) );
	if ( ! fileName.isEmpty() )
	{
		GvvPluginManager::get().unloadAll();
		GvvPluginManager::get().loadPlugin( fileName.toStdString() );
	}
}
