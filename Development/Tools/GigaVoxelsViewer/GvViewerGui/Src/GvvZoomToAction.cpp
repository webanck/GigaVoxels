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

#include "GvvZoomToAction.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "Gvv3DWindow.h"
#include "GvvPipelineInterfaceViewer.h"
#include "GvvPipelineInterface.h"
#include "GvvGLSceneInterface.h"
#include "GvvContextManager.h"
#include "GvvBrowsable.h"

// Qt
#include <QDir>
#include <QFile>
#include <QProcess>
#include <QDesktopServices>
#include <QUrl>

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
const QString GvvZoomToAction::cName = "zoomTo";

/**
 * The default text assigned to the action
 */
const char* GvvZoomToAction::cDefaultText = QT_TR_NOOP( "Zoom To" );

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
GvvZoomToAction::GvvZoomToAction( const QString& pFileName, 
										const QString& pText, 
										const QString& pIconName,
										bool pIsToggled )
:	GvvAction( GvvApplication::get().getMainWindow(), cName, pText, pIconName, pIsToggled )
{
	//** Sets the status tip
	setStatusTip( qApp->translate( "GvvZoomToAction", "Zoom To" ) );
	//setShortcut( qApp->translate( "GvvZoomToAction", "Z" ) );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvZoomToAction::~GvvZoomToAction()
{
}

/******************************************************************************
 * Overwrites the execute action
 ******************************************************************************/
void GvvZoomToAction::execute()
{
	GvvBrowsable* browsable = GvvContextManager::get()->editCurrentBrowsable();
	if ( browsable != NULL )
	{
		//** Zoom to element
		GvvPipelineInterface* pipeline = dynamic_cast< GvvPipelineInterface* >( browsable );
		if ( pipeline != NULL )
		{
			// TO DO
			// ...
		}

		//** Zoom to element
		GvvGLSceneInterface* scene = dynamic_cast< GvvGLSceneInterface* >( browsable );
		if ( scene != NULL )
		{
			GvvApplication& application = GvvApplication::get();
			GvvMainWindow* mainWindow = application.getMainWindow();
			Gvv3DWindow* window3D = mainWindow->get3DWindow();
			GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
			if ( pipelineViewer != NULL )
			{
				// Scene bounding box
				qglviewer::Vec bboxMin( scene->_minX, scene->_minY, scene->_minZ );
				qglviewer::Vec bboxMax( scene->_maxX, scene->_maxY, scene->_maxZ );

				// Modify scene radius
				const float sceneRadius = qglviewer::Vec( bboxMax - bboxMin ).norm();
				pipelineViewer->setSceneRadius( sceneRadius );
				
				// Fit to bounding box
				pipelineViewer->camera()->fitBoundingBox( bboxMin, bboxMax );
			}
		}
	}
}
