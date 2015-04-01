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

#ifndef _GVX_SCENE_VOXELIZER_H_
#define _GVX_SCENE_VOXELIZER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "GvxDataTypeHandler.h"
#include "GvxVoxelizerEngine.h"

// System
#include <string>

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
 * @class GvxSceneVoxelizer
 *
 * @brief The GvxSceneVoxelizer class provides an interface to a voxelize a scene.
 *
 * The main idea is to load a 3D scene in memory and traverse it to retrieve
 * useful data needed during voxelization (i.e. vertices, faces, normals,
 * materials, textures, etc...)
 */
class GvxSceneVoxelizer
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
	GvxSceneVoxelizer();

	/**
	 * Destructor
	 */
	virtual ~GvxSceneVoxelizer();

	/**
	 * The main method called to voxelize a scene.
	 * All settings must have been done previously (i.e. filename, path, etc...)
	 */
	virtual bool launchVoxelizationProcess();

	/**
	 * Get the data file path
	 *
	 * @return the data file path
	 */
	const std::string& getFilePath() const;

	/**
	 * Set the data file path
	 *
	 * @param pFilePath the data file path
	 */
	void setFilePath( const std::string& pFilePath );

	/**
	 * Get the data file name
	 *
	 * @return the data file name
	 */
	const std::string& getFileName() const;

	/**
	 * Set the data file name
	 *
	 * @param pFileName the data file name
	 */
	void setFileName( const std::string& pFileName );

	/**
	 * Get the data file extension
	 *
	 * @return the data file extension
	 */
	const std::string& getFileExtension() const;

	/**
	 * Set the data file extension
	 *
	 * @param pFileExtension the data file extension
	 */
	void setFileExtension( const std::string& pFileExtension );

	/**
	 * Get the max level of resolution
	 *
	 * @return the max level of resolution
	 */
	unsigned int getMaxResolution() const;

	/**
	 * Set the max level of resolution
	 *
	 * @param pValue the max level of resolution
	 */
	void setMaxResolution( unsigned int pValue );

	/**
	 * Tell wheter or not normals generation is activated
	 *
	 * @return a flag telling wheter or not normals generation is activated
	 */
	bool isGenerateNormalsOn() const;

	/**
	 * Set the flag telling wheter or not normals generation is activated
	 *
	 * @param pFlag the flag telling wheter or not normals generation is activated
	 */
	void setGenerateNormalsOn( bool pFlag );

	/**
	 * Get the brick width
	 *
	 * @return the brick width
	 */
	unsigned int getBrickWidth() const;

	/**
	 * Set the brick width
	 *
	 * @param pValue the brick width
	 */
	void setBrickWidth( unsigned int pValue );

	/**
	 * Get the data type of voxels
	 *
	 * @return the data type of voxels
	 */
	GvxDataTypeHandler::VoxelDataType getDataType() const;

	/**
	 * Set the data type of voxels
	 *
	 * @param pType the data type of voxels
	 */
	void setDataType( GvxDataTypeHandler::VoxelDataType pType );

	/**
	 * Set the filter type
	 *
	 * 0 = mean
	 * 1 = gaussian
	 * 2 = laplacian
	 */
	void setFilterType(int filterType);

	/**
	 * Set the number of application of the filter
	 */
	void setFilterIterations(int nbFilterOperation);

	/**
	 * Set whether or not we generate the normal field
	 */
	void setNormals ( bool normals);

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * 3D model file path.
	 * Path must be terminated by the specific Operationg System directory seperator (/, \, //, etc...).
	 */
	std::string _filePath;
	
	/**
	 * 3D model file name
	 */
	std::string _fileName;
	
	/**
	 * 3D model file extension
	 */
	std::string _fileExtension;

	/**
	 * Max scene resolution
	 */
	unsigned int _maxResolution;

	/**
	 * Flag to tell wheter or not to generate normals
	 */
	bool _isGenerateNormalsOn;

	/**
	 * Brick width
	 */
	unsigned int _brickWidth;

	/**
	 * Data type
	 */
	GvxDataTypeHandler::VoxelDataType _dataType;

	/**
	 * Voxelizer engine
	 */
	GvxVoxelizerEngine _voxelizerEngine;
	
	/******************************** METHODS *********************************/

	/**
	 * Load/import the scene
	 */
	virtual bool loadScene();

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

	/**
	 * Apply the mip-mapping algorithmn.
	 * Given a pre-filtered voxel scene at a given level of resolution,
	 * it generates a mip-map pyramid hierarchy of coarser levels (until 0).
	 */
	virtual bool mipmap();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvxSceneVoxelizer( const GvxSceneVoxelizer& );

	/**
	 * Copy operator forbidden.
	 */
	GvxSceneVoxelizer& operator=( const GvxSceneVoxelizer& );

};

}

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvxSceneVoxelizer.inl"

#endif
