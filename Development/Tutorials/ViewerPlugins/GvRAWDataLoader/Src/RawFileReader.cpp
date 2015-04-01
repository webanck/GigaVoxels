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

#include "RawFileReader.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvVoxelizer/GvDataStructureIOHandler.h"
#include "GvVoxelizer/GvDataTypeHandler.h"

// System
#include <cassert>
#include <iostream>
#include <fstream>
#include <cmath>

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
RawFileReader::RawFileReader()
:	GvIRAWFileReader()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
RawFileReader::~RawFileReader()
{
}

/******************************************************************************
 * Load/import the scene the scene
 ******************************************************************************/
bool RawFileReader::readData()
{
	bool result = false;

	_mode = eBinary;

	FILE* file = NULL;

	// Create a file/streamer handler to read/write GigaVoxels data
	unsigned int levelOfResolution = static_cast< unsigned int >( log( static_cast< float >( getDataResolution() / 8/*<== if 8 voxels by bricks*/ ) ) / log( static_cast< float >( 2 ) ) );
	unsigned int brickWidth = 8;
	GvDataTypeHandler::VoxelDataType dataType = GvDataTypeHandler::gvUCHAR4;
	_dataStructureIOHandler = new GvDataStructureIOHandler( getFilename(), levelOfResolution, brickWidth, dataType, true );

	std::cout << "- read file : " << getFilename() << std::endl;

	if ( _mode == eBinary )
	{
		file = fopen( getFilename().c_str(), "rb" );
		if ( file != NULL )
		{
			// Set voxel data
			unsigned int voxelPosition[ 3 ];
			unsigned char voxelData[ 4 ];

			for ( unsigned int z = 0; z < _dataResolution; z++ )
			{
				for ( unsigned int y = 0; y < _dataResolution; y++ )
				{
					for ( unsigned int x = 0; x < _dataResolution; x++ )
					{
						// Read voxel data
						unsigned char voxelData_0;
						fread( &voxelData_0, sizeof( unsigned char ), 1, file );
						if ( voxelData_0 != 0 )
						{
							voxelData[ 0 ] = voxelData_0;
							voxelData[ 1 ] = voxelData_0;
							voxelData[ 2 ] = voxelData_0;
							voxelData[ 3 ] = voxelData_0;

							// Voxel position
							voxelPosition[ 0 ] = x;
							voxelPosition[ 1 ] = y;
							voxelPosition[ 2 ] = z;
							
							// Write voxel data (in channel 0)
							_dataStructureIOHandler->setVoxel( voxelPosition, voxelData, 0 );
						}
					}
				}
			}
		}
		else
		{
			assert( false );
		}
	}
	else if ( _mode == eASCII )
	{
		// TO DO
		// Add ASCII mode exemple
		// ...

		/*file = fopen( filename.c_str(), "r" );
		if ( file != NULL )
		{
		}
		else
		{*/
			assert( false );
		//}
	}

	return result;
}
