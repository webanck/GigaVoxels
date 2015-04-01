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

#ifndef GVPLUGINMANAGER_H
#define GVPLUGINMANAGER_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Gigavoxels
#include "GvCore/GvCoreConfig.h"

#include <string>
#include <vector>

#ifdef _WIN32
	#include <windows.h>
#else
#endif

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GigaVoxels
namespace GvUtils
{
	class GvPluginInterface;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvUtils
{

/**
 * GvPluginManager
 */
class GIGASPACE_EXPORT GvPluginManager
{
    /**************************************************************************
     ***************************** PUBLIC SECTION *****************************
     **************************************************************************/

public:

    /******************************* INNER TYPES *******************************/

    /******************************* ATTRIBUTES *******************************/

    /******************************** METHODS *********************************/

    /**
     * getInstance()
     *
     * @return
     */
    static GvPluginManager& get();

    /**
     * loadPlugins()
     *
     * @param pDir
     */
    void loadPlugins( const std::string& pDir );

	/**
     * getFilenames()
     *
     * @param pFilename
     */
    bool loadPlugin( const std::string& pFilename );

    /**
     * unloadAll()
     */
    void unloadAll();

    /**
     * getNumPlugins()
     *
     * @return
     */
    size_t getNbPlugins() const;

    /**
     * getPlugin()
     *
     * @param pIndex
     *
     * @return
     */
    GvPluginInterface* getPlugin( int pIndex );

   /**************************************************************************
    **************************** PROTECTED SECTION ***************************
    **************************************************************************/

protected:

    /******************************* INNER TYPES *******************************/

    /******************************* ATTRIBUTES *******************************/

    /******************************** METHODS *********************************/

    /**************************************************************************
     ***************************** PRIVATE SECTION ****************************
     **************************************************************************/

private:

    /******************************* INNER TYPES *******************************/

    /**
     * Structure PluginInfo
     */
    struct PluginInfo
    {
        /**
         * Pointer on the plugin
         */
        GvPluginInterface* mPlugin;

		/**
         * Handle to library
         */
#ifdef _WIN32
		HINSTANCE mHandle;
#else
        void* mHandle;
#endif
    };

    /******************************* ATTRIBUTES *******************************/

    /**
     * The unique instance
     */
    static GvPluginManager* msInstance;

    /**
     * List of plugins
     */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
    std::vector< PluginInfo > mPlugins;
#if defined _MSC_VER
#pragma warning( pop )
#endif

    /******************************** METHODS *********************************/

    /**
     * PluginManager()
     */
    GvPluginManager();

    /**
     * getFilenames()
     *
     * @param pDir
     * @param pFilenames
     */
    void getFilenames( const std::string& pDir, std::vector< std::string >& pFilenames ) const;
	
};

} // namespace GvUtils

namespace GvUtils
{

	/**
	 * Definition of GV_CREATE_PLUGIN as a function's pointer
	 */
	 typedef GvPluginInterface* (* GV_CREATE_PLUGIN)( GvPluginManager& pManager );

} // namespace GvUtils


#endif
