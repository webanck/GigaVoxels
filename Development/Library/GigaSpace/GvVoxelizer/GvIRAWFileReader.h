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

#ifndef _GV_I_RAW_FILE_READER_H_
#define _GV_I_RAW_FILE_READER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvVoxelizer/GvDataStructureIOHandler.h"
#include "GvVoxelizer/GvDataTypeHandler.h"

// STL
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

namespace GvVoxelizer
{

/** 
 * @class GvIRAWFileReader
 *
 * @brief The GvIRAWFileReader class provides an implementation
 * of a scene voxelizer with ASSIMP, the Open Asset Import Library.
 *
 * It is used to load a 3D scene in memory and traverse it to retrieve
 * useful data needed during voxelization (i.e. vertices, faces, normals,
 * materials, textures, etc...)
 *
 * TO DO : add support to multi-data type ==> GvDataStructureMipmapGemerator is dependent of type...
 */
class GIGASPACE_EXPORT GvIRAWFileReader
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Enumeration of reading mode
	 */
	enum Mode
	{
		eUndefinedMode,
		eASCII,
		eBinary
	};

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvIRAWFileReader();

	/**
	 * Destructor
	 */
	virtual ~GvIRAWFileReader();

	/**
	 * Load/import the scene
	 */
	virtual bool read();

	/**
	 * 3D model file name
	 */
	const std::string& getFilename() const;

	/**
	 * 3D model file name
	 */
	void setFilename( const std::string& pName );

	/**
	 * Data resolution
	 */
	unsigned int getDataResolution() const;

	/**
	 * Data resolution
	 */
	void setDataResolution( unsigned int pValue );

	/**
	 * Mode (binary or ascii)
	 */
	Mode getMode() const;

	/**
	 * Mode (binary or ascii)
	 */
	void setMode( Mode pMode );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * 3D model file name
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::string _filename;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * Data resolution
	 */
	unsigned int _dataResolution;

	/**
	 * Mode (binary or ascii)
	 */
	Mode _mode;

	/**
	 * File/stream handler.
	 * It ios used to read and/ or write to GigaVoxels files (internal format).
	 */
	GvDataStructureIOHandler* _dataStructureIOHandler;
	
	/******************************** METHODS *********************************/

	/**
	 * Load/import the scene
	 */
	virtual bool readData() = 0;

	/**
	 * Apply the mip-mapping algorithmn.
	 * Given a pre-filtered voxel scene at a given level of resolution,
	 * it generates a mip-map pyramid hierarchy of coarser levels (until 0).
	 */
	virtual bool generateMipmapPyramid();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvIRAWFileReader( const GvIRAWFileReader& );

	/**
	 * Copy operator forbidden.
	 */
	GvIRAWFileReader& operator=( const GvIRAWFileReader& );

};

}

#endif
