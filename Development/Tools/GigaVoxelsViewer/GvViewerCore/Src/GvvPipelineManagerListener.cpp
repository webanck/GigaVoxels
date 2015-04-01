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

#include "GvvPipelineManagerListener.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvPipelineInterface.h"
#include "GvvPipelineManager.h"

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
GvvPipelineManagerListener::GvvPipelineManagerListener()
{
	GvvPipelineManager::get().registerListener( this );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvPipelineManagerListener::~GvvPipelineManagerListener()
{
	GvvPipelineManager::get().unregisterListener( this );
}

/******************************************************************************
 * Add a pipeline.
 *
 * @param the pipeline to add
 ******************************************************************************/
void GvvPipelineManagerListener::onPipelineAdded( GvvPipelineInterface* pPipeline )
{
}

/******************************************************************************
 * Remove a pipeline.
 *
 * @param the pipeline to remove
 ******************************************************************************/
void GvvPipelineManagerListener::onPipelineRemoved( GvvPipelineInterface* pPipeline )
{
}

/******************************************************************************
 * Remove a pipeline has been modified.
 *
 * @param the modified pipeline
 ******************************************************************************/
void GvvPipelineManagerListener::onPipelineModified( GvvPipelineInterface* pPipeline )
{
}
