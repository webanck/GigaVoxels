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

#include "GvvOpenProgrammableShaderEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "GvvContextManager.h"
#include "GvvPipelineInterface.h"
#include "GvvGLSLSourceEditor.h"
#include "GvvContextManager.h"
#include "GvvBrowsable.h"

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
const QString GvvOpenProgrammableShaderEditor::cName = "openProgrammableShaderEditor";

/**
 * The default text assigned to the action
 */
const char* GvvOpenProgrammableShaderEditor::cDefaultText = QT_TR_NOOP( "Open Programmable Shader Editor" );

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
GvvOpenProgrammableShaderEditor::GvvOpenProgrammableShaderEditor( const QString& pFileName, 
										const QString& pText, 
										const QString& pIconName,
										bool pIsToggled )
:	GvvAction( GvvApplication::get().getMainWindow(), cName, pText, pIconName, pIsToggled )
,	GvvContextListener(  )
,	mFileName( pFileName )
{
	//** Sets the status tip
	setStatusTip( qApp->translate( "GvvOpenProgrammableShaderEditor", "Open Programmable Shader Editor" ) );
//	setShortcut( qApp->translate( "GvvOpenProgrammableShaderEditor", "F1" ) );

	//  Disabled by default
	setDisabled( true );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvOpenProgrammableShaderEditor::~GvvOpenProgrammableShaderEditor()
{
}

/******************************************************************************
 * Overwrites the execute action
 ******************************************************************************/
void GvvOpenProgrammableShaderEditor::execute()
{
	if ( GvvApplication::get().getMainWindow()->getGLSLourceEditor() != NULL )
	{
		GvvApplication::get().getMainWindow()->getGLSLourceEditor()->show();

		GvvBrowsable* browsable = GvvContextManager::get()->editCurrentBrowsable();
		if ( browsable != NULL )
		{
			GvvPipelineInterface* pipeline = dynamic_cast< GvvPipelineInterface* >( browsable );
			if ( pipeline != NULL )
			{
				if ( pipeline->hasProgrammableShaders() )
				{
					GvvApplication::get().getMainWindow()->getGLSLourceEditor()->populate( pipeline );
				}
			}
		}
	}
}

/******************************************************************************
 * Updates this action before being shown
 ******************************************************************************/
void GvvOpenProgrammableShaderEditor::onAboutToShow()
{
	GvvBrowsable* browsable = GvvContextManager::get()->editCurrentBrowsable();
	if ( browsable != NULL )
	{
		GvvPipelineInterface* pipeline = dynamic_cast< GvvPipelineInterface* >( browsable );
		if ( pipeline != NULL )
		{
			if ( pipeline->hasProgrammableShaders() )
			{
				setEnabled( true );
			}
		}
	}
}
