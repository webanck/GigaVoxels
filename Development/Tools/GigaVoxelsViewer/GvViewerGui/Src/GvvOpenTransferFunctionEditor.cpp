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

#include "GvvOpenTransferFunctionEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "GvvContextManager.h"
#include "GvvPipelineInterface.h"
#include "GvvTransferFunctionEditor.h"

// Qtfe
#include "Qtfe.h"

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
const QString GvvOpenTransferFunctionEditor::cName = "openTransferFunctionEditor";

/**
 * The default text assigned to the action
 */
const char* GvvOpenTransferFunctionEditor::cDefaultText = QT_TR_NOOP( "Open Transfer Function Editor" );

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
GvvOpenTransferFunctionEditor::GvvOpenTransferFunctionEditor( const QString& pFileName, 
										const QString& pText, 
										const QString& pIconName,
										bool pIsToggled )
:	GvvAction( GvvApplication::get().getMainWindow(), cName, pText, pIconName, pIsToggled )
,	GvvContextListener(  )
,	mFileName( pFileName )
{
	//** Sets the status tip
	setStatusTip( qApp->translate( "GvvOpenTransferFunctionEditor", "Open transfer function editor" ) );
//	setShortcut( qApp->translate( "GvvOpenTransferFunctionEditor", "F1" ) );

	//  Disabled by default
	setDisabled( true );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvOpenTransferFunctionEditor::~GvvOpenTransferFunctionEditor()
{
}

/******************************************************************************
 * Overwrites the execute action
 ******************************************************************************/
void GvvOpenTransferFunctionEditor::execute()
{
	if ( GvvApplication::get().getMainWindow()->getTransferFunctionEditor() != NULL )
	{
		GvvApplication::get().getMainWindow()->getTransferFunctionEditor()->show();
	}
}

/******************************************************************************
 * This slot is called when the current editable changed
 ******************************************************************************/
void GvvOpenTransferFunctionEditor::onCurrentBrowsableChanged()
{
	const GvvPipelineInterface* pipeline = dynamic_cast< const GvvPipelineInterface* >( GvvContextManager::get()->getCurrentBrowsable() );
	setEnabled(  pipeline != NULL );
}
