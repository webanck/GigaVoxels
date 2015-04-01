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

#ifndef _GVV_GL_SCENE_MANAGER_LISTENER_H_
#define _GVV_GL_SCENE_MANAGER_LISTENER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvCoreConfig.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GvViewer
namespace GvViewerCore
{
	class GvvGLSceneManager;
	class GvvGLSceneInterface;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerCore
{

/**
 * GvPluginManager
 */
class GVVIEWERCORE_EXPORT GvvGLSceneManagerListener
{

	/**************************************************************************
     ***************************** FRIEND SECTION *****************************
     **************************************************************************/

	/**
	 * ...
	 */
	friend class GvvGLSceneManager;

	/**************************************************************************
     ***************************** PUBLIC SECTION *****************************
     **************************************************************************/

public:

    /******************************* INNER TYPES *******************************/

    /******************************* ATTRIBUTES *******************************/

    /******************************** METHODS *********************************/
	
    /**************************************************************************
     **************************** PROTECTED SECTION ***************************
     **************************************************************************/

protected:

    /******************************* INNER TYPES *******************************/

    /******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
     * Constructor
     */
    GvvGLSceneManagerListener();

	/**
     * Destructor
     */
    virtual ~GvvGLSceneManagerListener();

	/**
	 * Add a scene.
	 *
	 * @param pScene the scene to add
	 */
	virtual void onGLSceneAdded( GvvGLSceneInterface* pScene );

	/**
	 * Remove a scene.
	 *
	 * @param pScene the scene to remove
	 */
	virtual void onGLSceneRemoved( GvvGLSceneInterface* pScene );

	/**
	 * Tell that a scene has been modified.
	 *
	 * @param pScene the modified scene
	 */
	virtual void onGLSceneModified( GvvGLSceneInterface* pScene );

    /**************************************************************************
     ***************************** PRIVATE SECTION ****************************
     **************************************************************************/

private:

    /******************************* INNER TYPES *******************************/

    /******************************* ATTRIBUTES *******************************/

    /******************************** METHODS *********************************/

};

} // namespace GvViewerCore

#endif
