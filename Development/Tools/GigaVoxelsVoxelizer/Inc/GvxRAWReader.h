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

#ifndef _GVX_RAW_READER_H_
#define _GVX_RAW_READER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "GvxDataStructureIOHandler.h"
#include "GvxDataTypeHandler.h"

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

namespace Gvx
{

/** 
 * @class GvxRAWReader
 *
 * @brief The GvxRAWReader class provides an implementation
 * of a scene voxelizer with ASSIMP, the Open Asset Import Library.
 *
 * It is used to load a 3D scene in memory and traverse it to retrieve
 * useful data needed during voxelization (i.e. vertices, faces, normals,
 * materials, textures, etc...)
 */
class GvxRAWReader
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
		eASCII,
		eBinary
	};

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvxRAWReader();

	/**
	 * Destructor
	 */
	virtual ~GvxRAWReader();

	/**
	 * Load/import the scene
	 */
	virtual bool read();

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
	GvxDataStructureIOHandler* _dataStructureIOHandler;
	
	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvxRAWReader( const GvxRAWReader& );

	/**
	 * Copy operator forbidden.
	 */
	GvxRAWReader& operator=( const GvxRAWReader& );

};

}

#endif
