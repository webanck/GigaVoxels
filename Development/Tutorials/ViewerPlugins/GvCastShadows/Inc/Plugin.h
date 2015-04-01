/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2014 INRIA <http://www.inria.fr/>
 *
 * Authors : GigaVoxels Team
 *
 * GigaVoxels is distributed under a dual-license scheme.
 * You can obtain a specific license from Inria at gigavoxels-licensing@inria.fr.
 * Otherwise the default license is the GPL version 3.
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

#ifndef _PLUGIN_H_
#define _PLUGIN_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include <GvvPluginInterface.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// Viewer
namespace GvViewerCore
{
    class GvvPluginManager;
}

// Project
class SampleCore;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class CustomEditor
 *
 * @brief The CustomEditor class provides a custom editor to this GigaVoxels
 * pipeline effect.
 *
 * This editor has a static creator function used by the factory class "GvvEditorWindow"
 * to create the associated editor (@see GvvEditorWindow::registerEditorFactory())
 */
class Plugin : public GvViewerCore::GvvPluginInterface
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:
	
	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param pManager the singleton Plugin Manager
	 */
	Plugin( GvViewerCore::GvvPluginManager& pManager );

	/**
     * Destructor
     */
    virtual ~Plugin();

	/**
     * Get the plugin name
	 *
	 * @return the plugin name
     */
    virtual const std::string& getName();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	***************************** PRIVATE SECTION ****************************
	**************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
	
	/**
	 * Reference on the singleton Plugin Manager
	 */
	GvViewerCore::GvvPluginManager& _manager;

	/**
	 * Name
	 */
	std::string _name;

	/**
	 * Export name
	 */
	std::string _exportName;

	/**
	 * Reference on a GigaVoxels pipeline
	 */
	SampleCore* _pipeline;

	/******************************** METHODS *********************************/

	/**
	 * Initialize the plugin
	 */
	void initialize();

	/**
	 * Finalize the plugin
	 */
	void finalize();

};

#endif  // _PLUGIN_H_
