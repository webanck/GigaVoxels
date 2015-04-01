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

#include "GvvRemoveBrowsableAction.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "GvvContextManager.h"
#include "GvvBrowsable.h"
#include "GvvPipelineInterface.h"
#include "GvvPipelineManager.h"
#include "GvvPluginManager.h"
#include "GvvGLSceneInterface.h"
#include "GvvGLSceneManager.h"

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
const QString GvvRemoveBrowsableAction::cName = "removeBrowsable";

/**
 * The default text assigned to the action
 */
const char* GvvRemoveBrowsableAction::cDefaultText = QT_TR_NOOP( "Remove" );

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 *******'***********************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructs an action.
 *
 * @param	pText specifies the descriptive text of this action
 * @param	pIconName specifies the name of the icon for this action located in the icons application path
 *					Does nothing if the string is empty. A full file path can also be given.
 ******************************************************************************/
GvvRemoveBrowsableAction::GvvRemoveBrowsableAction( const QString& pText, const QString& pIconName )
:	GvvAction( GvvApplication::get().getMainWindow(), cName, pText, pIconName )
{
	setStatusTip( qApp->translate("GvvRemoveBrowsableAction", cDefaultText ) );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvRemoveBrowsableAction::~GvvRemoveBrowsableAction()
{
}

/******************************************************************************
 * Overwrites the execute method
 ******************************************************************************/
void GvvRemoveBrowsableAction::execute()
{
	GvvBrowsable* browsable = GvvContextManager::get()->editCurrentBrowsable();
	if ( browsable != NULL )
	{
		//** Updates the context
		GvvContextManager::get()->setCurrentBrowsable( NULL );

		//** Remove the element
		GvvPipelineInterface* pipeline = dynamic_cast< GvvPipelineInterface* >( browsable );
		if ( pipeline != NULL )
		{
			GvvPipelineManager::get().removePipeline( pipeline );

			// Destroy the pipeline
			// TO DO ----------------------------
			//delete pipeline;
			//pipeline = NULL;
			GvvPluginManager::get().unloadAll();
			// ----------------------------------
		}
		else
		{
			//** Remove the element
			GvvGLSceneInterface* scene = dynamic_cast< GvvGLSceneInterface* >( browsable );
			if ( scene != NULL )
			{
				GvvGLSceneManager::get().removeGLScene( scene );

				// Destroy object
				delete scene;
				scene = NULL;
			}
		}
	}
}
