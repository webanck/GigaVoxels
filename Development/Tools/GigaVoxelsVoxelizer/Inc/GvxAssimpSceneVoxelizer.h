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

#ifndef _GVX_ASSIMP_SCENE_VOXELIZER_H_
#define _GVX_ASSIMP_SCENE_VOXELIZER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "GvxSceneVoxelizer.h"

// Assimp
#include <assimp/cimport.h>
#include <assimp/scene.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace Gvx
{

/** 
 * @class GvxAssimpSceneVoxelizer
 *
 * @brief The GvxAssimpSceneVoxelizer class provides an implementation
 * of a scene voxelizer with ASSIMP, the Open Asset Import Library.
 *
 * It is used to load a 3D scene in memory and traverse it to retrieve
 * useful data needed during voxelization (i.e. vertices, faces, normals,
 * materials, textures, etc...)
 */
class GvxAssimpSceneVoxelizer : public GvxSceneVoxelizer
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvxAssimpSceneVoxelizer();

	/**
	 * Destructor
	 */
	virtual ~GvxAssimpSceneVoxelizer();

	/**
	 * Normalize the scene.
	 * It determines the whole scene bounding box and then modifies vertices
	 * to scale the scene.
	 */
	virtual bool normalizeScene();

	/**
	 * Voxelize the scene
	 */
	virtual bool voxelizeScene();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * The root structure of the imported data. 
	 * 
	 *  Everything that was imported from the given file can be accessed from here.
	 *  Objects of this class are generally maintained and owned by Assimp, not
	 *  by the caller. You shouldn't want to instance it, nor should you ever try to
	 *  delete a given scene on your own.
	 */
	aiScene* _scene;
	
	/**
	 * Represents a log stream.
	 * A log stream receives all log messages and streams them _somewhere_.
	 */
	aiLogStream _logStream;
	
	/******************************** METHODS *********************************/

	/**
	 * Load/import the scene
	 */
	virtual bool loadScene();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvxAssimpSceneVoxelizer( const GvxAssimpSceneVoxelizer& );

	/**
	 * Copy operator forbidden.
	 */
	GvxAssimpSceneVoxelizer& operator=( const GvxAssimpSceneVoxelizer& );

};

}

#endif
