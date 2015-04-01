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

#include "GvvGLSceneManagerListener.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGLSceneInterface.h"
#include "GvvGLSceneManager.h"

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;

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
 * Constructor
 ******************************************************************************/
GvvGLSceneManagerListener::GvvGLSceneManagerListener()
{
	GvvGLSceneManager::get().registerListener( this );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvGLSceneManagerListener::~GvvGLSceneManagerListener()
{
	GvvGLSceneManager::get().unregisterListener( this );
}

/******************************************************************************
 * Add a scene.
 *
 * @param pScene the scene to add
 ******************************************************************************/
void GvvGLSceneManagerListener::onGLSceneAdded( GvvGLSceneInterface* pScene )
{
}

/******************************************************************************
 * Remove a scene.
 *
 * @param pScene the scene to remove
 ******************************************************************************/
void GvvGLSceneManagerListener::onGLSceneRemoved( GvvGLSceneInterface* pScene )
{
}

/******************************************************************************
 * Remove a scene has been modified.
 *
 * @param pScene the modified scene
 ******************************************************************************/
void GvvGLSceneManagerListener::onGLSceneModified( GvvGLSceneInterface* pScene )
{
}
