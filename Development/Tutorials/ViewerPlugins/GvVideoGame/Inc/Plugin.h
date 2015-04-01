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

#ifndef GVMYPLUGIN_H
#define GVMYPLUGIN_H

#include <GvUtils/GvPluginInterface.h>

namespace GvUtils
{
    class GvPluginManager;
}

class SampleCore;

class GvMyPlugin : public GvUtils::GvPluginInterface
{

  public:

    // Constructors and 
    GvMyPlugin( GvUtils::GvPluginManager& pManager );

	/**
     * Destructor
     */
    virtual ~GvMyPlugin();

    virtual const std::string& getName();
	
  private:

      GvUtils::GvPluginManager& mManager;

      std::string mName;

      std::string mExportName;

	  SampleCore* mPipeline;

	  void initialize();

	  void finalize();

};

#endif  // GVMYPLUGIN_H
