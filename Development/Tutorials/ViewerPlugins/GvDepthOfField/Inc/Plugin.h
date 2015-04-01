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

#ifndef _PLUGIN_H_
#define _PLUGIN_H_

// GvViewer
#include <GvvPluginInterface.h>

namespace GvViewerCore
{
    class GvvPluginManager;
}

class SampleCore;

class Plugin : public GvViewerCore::GvvPluginInterface
{

  public:

    // Constructors and 
    Plugin( GvViewerCore::GvvPluginManager& pManager );

	/**
     * Destructor
     */
    virtual ~Plugin();

    virtual const std::string& getName();
	
  private:

	  GvViewerCore::GvvPluginManager& mManager;

      std::string mName;

      std::string mExportName;

	  SampleCore* _pipeline;

	  void initialize();

	  void finalize();

};

#endif  // _PLUGIN_H_
