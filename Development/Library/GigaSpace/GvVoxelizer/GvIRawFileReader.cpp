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

#include "GvVoxelizer/GvIRAWFileReader.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvVoxelizer/GvDataStructureMipmapGenerator.h"

// STL
#include <iostream>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvVoxelizer
using namespace GvVoxelizer;

// STL
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvIRAWFileReader::GvIRAWFileReader()
:	_filename()
,	_dataResolution( 0 )
,	_mode( eUndefinedMode )
,	_dataStructureIOHandler( NULL )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvIRAWFileReader::~GvIRAWFileReader()
{
}

/******************************************************************************
 * Load/import the scene the scene
 ******************************************************************************/
bool GvIRAWFileReader::read()
{
	bool result = false;

	std::cout << "- [step 1 / 3] - Read data and write vowels..." << std::endl;
	result = readData();	// TO DO : add a boolean return value

	std::cout << "- [step 2 / 3] - Update borders..." << std::endl;
	_dataStructureIOHandler->computeBorders();	// TO DO : add a boolean return value

	std::cout << "- [step 3 / 3] - Mipmap pyramid generation..." << std::endl;
	result = generateMipmapPyramid();

	return result;
}

/******************************************************************************
 * Load/import the scene the scene
 ******************************************************************************/
bool GvIRAWFileReader::readData()
{
	return false;
}

/******************************************************************************
 * Apply the mip-mapping algorithmn.
 * Given a pre-filtered voxel scene at a given level of resolution,
 * it generates a mip-map pyramid hierarchy of coarser levels (until 0).
 ******************************************************************************/
bool GvIRAWFileReader::generateMipmapPyramid()
{
	return GvDataStructureMipmapGenerator::generateMipmapPyramid( getFilename(), getDataResolution() );
}

/******************************************************************************
	 * 3D model file name
 ******************************************************************************/
const std::string& GvIRAWFileReader::getFilename() const
{
	return _filename;
}

/******************************************************************************
	 * 3D model file name
 ******************************************************************************/
void GvIRAWFileReader::setFilename( const std::string& pName )
{
	_filename = pName;
}
	
/******************************************************************************
	 * Data resolution
 ******************************************************************************/
unsigned int GvIRAWFileReader::getDataResolution() const
{
	return _dataResolution;
}

/******************************************************************************
	 * Data resolution
 ******************************************************************************/
void GvIRAWFileReader::setDataResolution( unsigned int pValue )
{
	_dataResolution = pValue;
}

/******************************************************************************
	 * Mode (binary or ascii)
 ******************************************************************************/
GvIRAWFileReader::Mode GvIRAWFileReader::getMode() const
{
	return _mode;
}

/******************************************************************************
	 * Mode (binary or ascii)
 ******************************************************************************/
void GvIRAWFileReader::setMode( Mode pMode )
{
	_mode = pMode;
}
