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

#include "GvvDisplayPerformanceCountersAction.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "Gvv3DWindow.h"
#include "GvvPipelineInterfaceViewer.h"
#include "GvvPipelineInterface.h"

// Qt
#include <QDir>
#include <QFile>
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
const QString GvvDisplayPerformanceCountersAction::cName = "displayPerformanceCounters";

/**
 * The default text assigned to the action
 */
const char* GvvDisplayPerformanceCountersAction::cDefaultText = QT_TR_NOOP( "Display Performance Counters" );

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
GvvDisplayPerformanceCountersAction::GvvDisplayPerformanceCountersAction( const QString& pFileName, 
										const QString& pText, 
										const QString& pIconName,
										bool pIsToggled )
:	GvvAction( GvvApplication::get().getMainWindow(), cName, pText, pIconName, pIsToggled )
{
	//** Sets the status tip
	setStatusTip( qApp->translate( "GvvDisplayPerformanceCountersAction", "Display Performance Counters" ) );
	setShortcut( qApp->translate( "GvvDisplayPerformanceCountersAction", "P" ) );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvDisplayPerformanceCountersAction::~GvvDisplayPerformanceCountersAction()
{
}

/******************************************************************************
 * Overwrites the execute action
 ******************************************************************************/
void GvvDisplayPerformanceCountersAction::execute()
{
	//QMessageBox::information( GvvApplication::get().getMainWindow(), tr( "Display Performance Counters" ), tr( "Not yet implemented..." ) );

	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
	GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
	if ( pipeline != NULL )
	{
		unsigned int performanceCountersMode = 0;
		if ( isChecked() )
		{
			performanceCountersMode = 1;	// Device mode
		}
		pipeline->togglePerfmonDisplay( performanceCountersMode );
	}
	}
}
