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
#ifndef _GVV_GL_SCENE_BROWSER_H_
#define _GVV_GL_SCENE_BROWSER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "GvvBrowser.h"
#include "GvvGLSceneManagerListener.h"

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
	class GvvGLSceneInterface;
}

namespace GvViewerGui
{
	class GvvContextMenu;
	class GvvBrowserItem;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * This class represents the abstract base class for all browsers. It manages
 * a map of contextual menus.
 *
 * @ingroup GvViewerGui
 */
class GVVIEWERGUI_EXPORT GvvGLSceneBrowser : public GvvBrowser, public GvViewerCore::GvvGLSceneManagerListener
{
	// Qt macro
	Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Default constructor.
	 * 
	 * @param pParent the parent widget
	 */
	GvvGLSceneBrowser( QWidget* pParent );
	
	/**
	 * Destructor.
	 */
	virtual ~GvvGLSceneBrowser();
		
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************** TYPEDEFS ********************************/

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * Add a pipeline.
	 *
	 * @param the pipeline to add
	 */
	virtual void onGLSceneAdded( GvViewerCore::GvvGLSceneInterface* pScene );

	/**
	 * Remove a pipeline.
	 *
	 * @param the pipeline to remove
	 */
	virtual void onGLSceneRemoved( GvViewerCore::GvvGLSceneInterface* pScene );
	
	/********************************** SLOTS **********************************/

protected slots:

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Copy constructor forbidden.
	 */
	GvvGLSceneBrowser( const GvvGLSceneBrowser& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvGLSceneBrowser& operator=( const GvvGLSceneBrowser& );
	
};

} // namespace GvViewerGui

#endif
