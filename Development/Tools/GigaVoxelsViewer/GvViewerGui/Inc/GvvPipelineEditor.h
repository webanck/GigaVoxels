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

#ifndef GVVPIPELINEEDITOR_H
#define GVVPIPELINEEDITOR_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "GvvEditor.h"
#include "GvvPipelineManagerListener.h"

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
	class GvvBrowsable;
}

namespace GvViewerGui
{
	class GvvCacheEditor;
	class GvvTransformationEditor;
	class GvvRendererEditor;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * ...
 *
 * @ingroup GvViewerGui
 */
class GVVIEWERGUI_EXPORT GvvPipelineEditor : public GvvEditor, public GvViewerCore::GvvPipelineManagerListener
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * ...
	 *
	 * @param pParent ...
	 * @param pBrowsable ...
	 *
	 * @return ...
	 */
	static GvvEditor* create( QWidget* pParent, GvViewerCore::GvvBrowsable* pBrowsable );

	/**
	 * Destructor.
	 */
	virtual ~GvvPipelineEditor();
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************* ATTRIBUTES *******************************/

	/**
	 * The cache editor
	 */
	GvvCacheEditor* _cacheEditor;

	/**
	 * The transformation editor
	 */
	GvvTransformationEditor* _transformationEditor;

	/**
	 * The renderer editor
	 */
	GvvRendererEditor* _rendererEditor;

	/******************************** METHODS *********************************/

	/**
	 * Default constructor.
	 */
	GvvPipelineEditor( QWidget* pParent = NULL, Qt::WindowFlags pFlags = 0 );
	
	/**
	 * Tell that a pipeline has been modified.
	 *
	 * @param the modified pipeline
	 */
	virtual void onPipelineModified( GvViewerCore::GvvPipelineInterface* pPipeline );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Copy constructor forbidden.
	 */
	GvvPipelineEditor( const GvvPipelineEditor& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvPipelineEditor& operator=( const GvvPipelineEditor& );
	
};

} // namespace GvViewerGui

#endif
