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

#ifndef GVVPIPELINEMANAGER_H
#define GVVPIPELINEMANAGER_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvCoreConfig.h"

// STL
#include <vector>

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
	class GvvPipelineInterface;
	class GvvPipelineManagerListener;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerCore
{

/**
 * GvPluginManager
 */
class GVVIEWERCORE_EXPORT GvvPipelineManager
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
    static GvvPipelineManager& get();

	/**
	 * Add a pipeline.
	 *
	 * @param the pipeline to add
	 */
	void addPipeline( GvvPipelineInterface* pPipeline );

	/**
	 * Remove a pipeline.
	 *
	 * @param the pipeline to remove
	 */
	void removePipeline( GvvPipelineInterface* pPipeline );

	/**
	 * Tell that a pipeline has been modified.
	 *
	 * @param the modified pipeline
	 */
	void setModified( GvvPipelineInterface* pPipeline );

	/**
	 * Register a listener.
	 *
	 * @param pListener the listener to register
	 */
	void registerListener( GvvPipelineManagerListener* pListener );

	/**
	 * Unregister a listener.
	 *
	 * @param pListener the listener to unregister
	 */
	void unregisterListener( GvvPipelineManagerListener* pListener );

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
    std::vector< GvvPipelineInterface* > mPipelines;
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
    std::vector< GvvPipelineManagerListener* > mListeners;
#if defined _MSC_VER
#pragma warning( pop )
#endif

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
    static GvvPipelineManager* msInstance;

    /******************************** METHODS *********************************/

    /**
     * Constructor
     */
    GvvPipelineManager();

};

} // namespace GvViewerCore

#endif
