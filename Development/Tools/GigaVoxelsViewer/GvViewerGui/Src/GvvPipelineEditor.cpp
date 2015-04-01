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

#include "GvvPipelineEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvPipelineInterface.h"
#include "GvvCacheEditor.h"
#include "GvvTransformationEditor.h"
#include "GvvRendererEditor.h"

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
 * ...
 *
 * @param pParent ...
 * @param pBrowsable ...
 *
 * @return ...
 ******************************************************************************/
GvvEditor* GvvPipelineEditor::create( QWidget* pParent, GvvBrowsable* pBrowsable )
{
	return new GvvPipelineEditor( pParent );
}

/******************************************************************************
 * Default constructor.
 ******************************************************************************/
GvvPipelineEditor::GvvPipelineEditor( QWidget* pParent, Qt::WindowFlags pFlags )
:	GvvEditor( pParent, pFlags )
,	GvvPipelineManagerListener()
,	_cacheEditor( NULL )
,	_transformationEditor( NULL )
,	_rendererEditor( NULL )
{
	// Data Structure / Cache editor
	_cacheEditor = new GvvCacheEditor( pParent, pFlags );
	_sectionEditors.push_back( _cacheEditor );

	// Renderer editor
	_rendererEditor = new GvvRendererEditor( pParent, pFlags );
	_sectionEditors.push_back( _rendererEditor );

	// Transformation editor
	_transformationEditor = new GvvTransformationEditor( pParent, pFlags );
	_sectionEditors.push_back( _transformationEditor );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvPipelineEditor::~GvvPipelineEditor()
{
}

/******************************************************************************
 * Remove a pipeline has been modified.
 *
 * @param the modified pipeline
 ******************************************************************************/
void GvvPipelineEditor::onPipelineModified( GvvPipelineInterface* pPipeline )
{
	populate( pPipeline );
}
