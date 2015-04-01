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

#ifndef _GVV_GL_SCENE_MANAGER_H_
#define _GVV_GL_SCENE_MANAGER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvCoreConfig.h"

// STL
#include <vector>
#include <string>

// Assimp
#include <assimp/cimport.h>
#include <assimp/scene.h>

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
	class GvvGLSceneInterface;
	class GvvGLSceneManagerListener;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerCore
{

/**
 * GvPluginManager
 */
class GVVIEWERCORE_EXPORT GvvGLSceneManager
{

	/**************************************************************************
     ***************************** PUBLIC SECTION *****************************
     **************************************************************************/

public:

    /******************************* INNER TYPES *******************************/

    /******************************* ATTRIBUTES *******************************/

    /******************************** METHODS *********************************/

    /**
     * Get the unique instance.
     *
     * @return the unique instance
     */
    static GvvGLSceneManager& get();

	/**
	 * Load 3D scene from file
	 *
	 * @param pFilename file to load
	 *
	 * @return flag telling wheter or not loading has succeded
	 */
	//bool load( const std::string& pFilename );
	const aiScene* load( const std::string& pFilename );

	/**
	 * Add a pipeline.
	 *
	 * @param the pipeline to add
	 */
	void addGLScene( GvvGLSceneInterface* pGLScene );

	/**
	 * Remove a pipeline.
	 *
	 * @param the pipeline to remove
	 */
	void removeGLScene( GvvGLSceneInterface* pGLScene );

	/**
	 * Tell that a pipeline has been modified.
	 *
	 * @param the modified pipeline
	 */
	void setModified( GvvGLSceneInterface* pGLScene );

	/**
	 * Register a listener.
	 *
	 * @param pListener the listener to register
	 */
	void registerListener( GvvGLSceneManagerListener* pListener );

	/**
	 * Unregister a listener.
	 *
	 * @param pListener the listener to unregister
	 */
	void unregisterListener( GvvGLSceneManagerListener* pListener );

   /**************************************************************************
    **************************** PROTECTED SECTION ***************************
    **************************************************************************/

protected:

    /******************************* INNER TYPES *******************************/

    /******************************* ATTRIBUTES *******************************/

	/**
     * List of pipelines
     */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
    std::vector< GvvGLSceneInterface* >_scenes;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
     * List of listeners
     */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
    std::vector< GvvGLSceneManagerListener* > _listeners;
#if defined _MSC_VER
#pragma warning( pop )
#endif
		
	/**
	 * The root structure of the imported data. 
	 * 
	 *  Everything that was imported from the given file can be accessed from here.
	 *  Objects of this class are generally maintained and owned by Assimp, not
	 *  by the caller. You shouldn't want to instance it, nor should you ever try to
	 *  delete a given scene on your own.
	 */
	const aiScene* _scene;
	
	/**
	 * Represents a log stream.
	 * A log stream receives all log messages and streams them _somewhere_.
	 */
	aiLogStream _logStream;

    /******************************** METHODS *********************************/

    /**************************************************************************
     ***************************** PRIVATE SECTION ****************************
     **************************************************************************/

private:

    /******************************* INNER TYPES *******************************/

    /******************************* ATTRIBUTES *******************************/

    /**
     * The unique instance
     */
    static GvvGLSceneManager* msInstance;

    /******************************** METHODS *********************************/

    /**
     * Constructor
     */
    GvvGLSceneManager();

	/**
     * Constructor
     */
    ~GvvGLSceneManager();

};

} // namespace GvViewerCore

#endif
