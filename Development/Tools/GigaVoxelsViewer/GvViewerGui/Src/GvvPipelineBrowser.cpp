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

#include "GvvPipelineBrowser.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvBrowserItem.h"
#include "GvvBrowsable.h"
#include "GvvContextMenu.h"
#include "GvvPipelineInterface.h"
#include "GvvTransferFunctionInterface.h"
#include "GvvMeshInterface.h"
#include "GvvProgrammableShaderInterface.h"

// Qt
#include <QContextMenuEvent>
#include <QTreeWidget>

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerGui;

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
 * Default constructor.
 ******************************************************************************/
GvvPipelineBrowser::GvvPipelineBrowser( QWidget* pParent ) 
:	GvvBrowser( pParent )
,	GvvPipelineManagerListener()
{
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvPipelineBrowser::~GvvPipelineBrowser()
{
}

/******************************************************************************
 * Add a pipeline.
 *
 * @param the pipeline to add
 ******************************************************************************/
void GvvPipelineBrowser::onPipelineAdded( GvvPipelineInterface* pPipeline )
{
	assert( pPipeline != NULL);
	if ( pPipeline != NULL )
	{
		GvvBrowserItem* pipelineItem = createItem( pPipeline );
		addTopLevelItem( pipelineItem );

		// Add transfer function
		//if ( pPipeline->hasTransferFunction() )
		//{
		//	GvvTransferFunctionInterface* transferFunction = pPipeline->getTransferFunction();
		//	if ( transferFunction != NULL )
		//	{
		//		GvvBrowserItem* transferFunctionItem = createItem( transferFunction );
		//		pipelineItem->addChild( transferFunctionItem );
		//	}
		//}

		// Add mesh item
		if ( pPipeline->hasMesh() )
		{
			GvvMeshInterface* mesh = pPipeline->editMesh( 0 );
			if ( mesh != NULL )
			{
				GvvBrowserItem* meshItem = createItem( mesh );
				pipelineItem->addChild( meshItem );

				if ( mesh->hasProgrammableShader() )
				{
					GvvProgrammableShaderInterface* programmableShader = mesh->editProgrammableShader( 0 );
					if ( programmableShader != NULL )
					{
						GvvBrowserItem* programmableShaderItem = createItem( programmableShader );
						meshItem->addChild( programmableShaderItem );
					}
				}
			}
		}

		// Expand item
		expandItem( pipelineItem );
	}
}

/******************************************************************************
 * Add a pipeline.
 *
 * @param the pipeline to add
 ******************************************************************************/
void GvvPipelineBrowser::onPipelineRemoved( GvvPipelineInterface* pPipeline )
{
	// Finds the item assigned to the given browsable
	GvvBrowserItem* item = find( pPipeline );
	if ( item != NULL )
	{
		int index = indexOfTopLevelItem( item );
		takeTopLevelItem( index );
	}
}
