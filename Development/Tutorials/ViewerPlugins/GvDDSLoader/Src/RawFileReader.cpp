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
#include <cstdlib>

// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

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

typedef struct {
    GLuint dwSize;
    GLuint dwFlags;
    GLuint dwFourCC;
    GLuint dwRGBBitCount;
    GLuint dwRBitMask;
    GLuint dwGBitMask;
    GLuint dwBBitMask;
    GLuint dwABitMask;
} DDS_PIXELFORMAT;

typedef struct {
    GLuint dwMagic;
    GLuint dwSize;
    GLuint dwFlags;
    GLuint dwHeight;
    GLuint dwWidth;
    GLuint dwLinearSize;
    GLuint dwDepth;
    GLuint dwMipMapCount;
    GLuint dwReserved1[11];
    DDS_PIXELFORMAT ddpf;
    GLuint dwCaps;
    GLuint dwCaps2;
    GLuint dwCaps3;
    GLuint dwCaps4;
    GLuint dwReserved2;
} DDS_HEADER;

DDS_HEADER DDS_headers;
DDS_PIXELFORMAT DDS_pixelformat;

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
			fread( &DDS_headers, sizeof( DDS_headers ), 1, file );

			GLuint img_width = DDS_headers.dwWidth;
			GLuint img_height = DDS_headers.dwHeight;
			GLuint img_depth = DDS_headers.dwDepth;

			//unsigned int* imgdata = (unsigned int*)malloc( img_width * img_height * img_depth );
			//fread( imgdata, sizeof( unsigned int ), img_width * img_height * img_depth, file );
			//fclose( file );

			// Set voxel data
			unsigned int voxelPosition[ 3 ];
			unsigned char voxelData[ 4 ];

			//for ( unsigned int z = 0; z < _dataResolution; z++ )
			//{
			//	for ( unsigned int y = 0; y < _dataResolution; y++ )
			//	{
			//		for ( unsigned int x = 0; x < _dataResolution; x++ )
			//		{
			//			// Read voxel data
			//			unsigned int index = x + y * img_width + z * ( img_width * img_height );
			//			if ( imgdata[ index ] != 0 )
			//			{
			//				voxelData[ 0 ] = imgdata[ index ] & 0x00ff0000;
			//				voxelData[ 1 ] = imgdata[ index ] & 0x0000ff00;
			//				voxelData[ 2 ] = imgdata[ index ] & 0x000000ff;
			//				voxelData[ 3 ] = imgdata[ index ] & 0xff000000;

			//				// Voxel position
			//				voxelPosition[ 0 ] = x;
			//				voxelPosition[ 1 ] = y;
			//				voxelPosition[ 2 ] = z;
			//				
			//				// Write voxel data (in channel 0)
			//				_dataStructureIOHandler->setVoxel( voxelPosition, voxelData, 0 );
			//			}
			//		}
			//	}
			//}
			unsigned int* imgdata = static_cast< unsigned int* >( malloc( sizeof( unsigned int ) * img_width * img_height * img_depth ) );
			size_t count = fread( imgdata, sizeof( unsigned int ), img_width * img_height * img_depth, file );
			fclose( file );

			//unsigned char voxelData;
			for ( unsigned int k = 0; k < img_depth; k++ )
			{
				for ( unsigned int j = 0; j < img_height; j++ )
				{
					for ( unsigned int i = 0; i< img_width; i++ )
					{
						//fread( &voxelData, sizeof( unsigned char ), 1, fp );
						int index = i + j * img_width + k * ( img_width * img_height );
						if ( imgdata[ index ] != 0 )
						{
							//int kok = 0;
							//kok++;
							//std::cout << imgdata[ index ] << std::endl;

							////-------------------------------------------------
							//unsigned int temp1 = ( imgdata[ index ] & 0x00ff0000 ) >> 16;
							//unsigned int temp2 = ( imgdata[ index ] & 0x0000ff00 ) >> 8;
							//unsigned int temp3 = imgdata[ index ] & 0x000000ff;
							//unsigned int temp4 = ( imgdata[ index ] & 0xff000000 ) >> 24;
							////std::cout << imgdata[ index ] << std::endl;
							//if ( temp4 != 0 )
							//{
							//	int kok = 0;
							//	kok++;
							//}
							////-------------------------------------------------

							voxelData[ 0 ] = ( imgdata[ index ] & 0x00ff0000 ) >> 16;
							voxelData[ 1 ] = ( imgdata[ index ] & 0x0000ff00 ) >> 8;
							voxelData[ 2 ] = imgdata[ index ] & 0x000000ff;
							voxelData[ 3 ] = ( imgdata[ index ] & 0xff000000 ) >> 24;

								// Voxel position
							voxelPosition[ 0 ] = i;
							voxelPosition[ 1 ] = j;
							voxelPosition[ 2 ] = k;
							
							// Write voxel data (in channel 0)
							_dataStructureIOHandler->setVoxel( voxelPosition, voxelData, 0 );
						}
						//std::cout << voxelData << std::endl;
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
