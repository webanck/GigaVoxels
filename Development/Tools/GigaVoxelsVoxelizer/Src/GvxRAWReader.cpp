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

#include "GvxRAWReader.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// System
#include <cassert>
#include <iostream>
#include <fstream>
#include <cmath>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvVoxelizer
using namespace Gvx;

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
GvxRAWReader::GvxRAWReader()
:	_filePath()
,	_fileName()
,	_fileExtension()
,	_dataResolution( 512 )
,	_mode( eASCII )
,	_dataStructureIOHandler( NULL )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvxRAWReader::~GvxRAWReader()
{
}

/******************************************************************************
 * Apply the mip-mapping algorithmn.
 * Given a pre-filtered voxel scene at a given level of resolution,
 * it generates a mip-map pyramid hierarchy of coarser levels (until 0).
 ******************************************************************************/
void mipmap()
{
	//------------------------------- TEST ---------------------------------------------
	string filename = string( "D:\\Projects\\GigaVoxelsTrunk\\Data\\Voxels\\Raw\\skull.raw" );
	unsigned int dataResolution = 256;
	//unsigned int levelOfResolution = static_cast< unsigned int >( log( static_cast< float >( dataResolution ) ) / log( static_cast< float >( 2 ) ) );
	unsigned int levelOfResolution = 5;
	unsigned int brickWidth = 8;
	GvxDataTypeHandler::VoxelDataType dataType = GvxDataTypeHandler::gvUCHAR4;
	std::vector< GvxDataTypeHandler::VoxelDataType > dataTypes;
	dataTypes.push_back( dataType );
	//------------------------------- TEST ---------------------------------------------

	// The mip-map pyramid hierarchy is built recursively from adjacent levels.
	// Two files/streamers are used :
	// UP is an already pre-filtered scene at resolution [ N ]
	// DOWN is the coarser version to generate at resolution [ N - 1 ]
	//GvxDataStructureIOHandler* dataStructureIOHandlerUP = new GvxDataStructureIOHandler( _fileName, _level, _brickWidth, _dataTypes, false );
	GvxDataStructureIOHandler* dataStructureIOHandlerUP = new GvxDataStructureIOHandler( filename, levelOfResolution, brickWidth, dataTypes, false );
	GvxDataStructureIOHandler* dataStructureIOHandlerDOWN = NULL;

	// Iterate through levels of resolution
	//for ( int level = _level - 1; level >= 0; level-- )
	for ( int level = levelOfResolution - 1; level >= 0; level-- )
	{
		// LOG info
		std::cout << "GvxVoxelizerEngine::mipmap : level : " << level << std::endl;

		// The coarser data handler is allocated dynamically due to memory consumption considerations.
		//dataStructureIOHandlerDOWN = new GvxDataStructureIOHandler( _fileName, level, _brickWidth, _dataTypes, true );
		dataStructureIOHandlerDOWN = new GvxDataStructureIOHandler( filename, level, brickWidth, dataTypes, true );

		// Iterate through nodes of the structure
		unsigned int nodePos[ 3 ];
		for ( nodePos[2] = 0; nodePos[2] < dataStructureIOHandlerUP->_nodeGridSize; nodePos[2]++ )
		for ( nodePos[1] = 0; nodePos[1] < dataStructureIOHandlerUP->_nodeGridSize; nodePos[1]++ )
		{
			// LOG info
			std::cout << "mipmap - LEVEL [ " << level << " ] - Node [ " << "x" << " / " << nodePos[1] << " / " << nodePos[2] << " ] - " << dataStructureIOHandlerUP->_nodeGridSize << std::endl;
			
		for ( nodePos[ 0 ] = 0; nodePos[ 0 ] < dataStructureIOHandlerUP->_nodeGridSize; nodePos[ 0 ]++ )
		{
			// Retrieve the current node info
			unsigned int node = dataStructureIOHandlerUP->getNode( nodePos );
			
			// If node is empty, go to next node
			if ( GvxDataStructureIOHandler::isEmpty( node ) )
			{
				continue;
			}

			// Iterate through voxels of the current node
			unsigned int voxelPos[ 3 ];
			for ( voxelPos[ 2 ] = brickWidth * nodePos[ 2 ]; voxelPos[ 2 ] < dataStructureIOHandlerUP->_brickWidth * ( nodePos[ 2 ] + 1 ); voxelPos[ 2 ] +=2 )
			for ( voxelPos[ 1 ] = brickWidth * nodePos[ 1 ]; voxelPos[ 1 ] < dataStructureIOHandlerUP->_brickWidth * ( nodePos[ 1 ] + 1 ); voxelPos[ 1 ] +=2 )
			for ( voxelPos[ 0 ] = brickWidth * nodePos[ 0 ]; voxelPos[ 0 ] < dataStructureIOHandlerUP->_brickWidth * ( nodePos[ 0 ] + 1 ); voxelPos[ 0 ] +=2 )
			{
				float voxelDataDOWNf[ 4 ] = { 0.f, 0.f, 0.f, 0.f };
				float voxelDataDOWNf2[ 4 ] = { 0.f, 0.f, 0.f, 0.f };

				// As the underlying structure is an octree,
				// to compute data at coaser level,
				// we need to iterate through 8 voxels and take the mean value.
				for ( unsigned int z = 0; z < 2; z++ )
				for ( unsigned int y = 0; y < 2; y++ )
				for ( unsigned int x = 0; x < 2; x++ )
				{
					// Retrieve position of voxel in the UP resolution version
					unsigned int voxelPosUP[ 3 ];
					voxelPosUP[ 0 ] = voxelPos[ 0 ] + x;
					voxelPosUP[ 1 ] = voxelPos[ 1 ] + y;
					voxelPosUP[ 2 ] = voxelPos[ 2 ] + z;

					// Get associated data (in the UP resolution version)
					unsigned char voxelDataUP[ 4 ];
					dataStructureIOHandlerUP->getVoxel( voxelPosUP, voxelDataUP, 0 );
					voxelDataDOWNf[ 0 ] += voxelDataUP[ 0 ];
					voxelDataDOWNf[ 1 ] += voxelDataUP[ 1 ];
					voxelDataDOWNf[ 2 ] += voxelDataUP[ 2 ];
					voxelDataDOWNf[ 3 ] += voxelDataUP[ 3 ];
										
#ifdef NORMALS
					// Get associated normal (in the UP resolution version)
					dataStructureIOHandlerUP->getVoxel( voxelPosUP, voxelDataUP, 1 );
					voxelDataDOWNf2[ 0 ] += 2.f * voxelDataUP[ 0 ] - 1.f;
					voxelDataDOWNf2[ 1 ] += 2.f * voxelDataUP[ 1 ] - 1.f;
					voxelDataDOWNf2[ 2 ] += 2.f * voxelDataUP[ 2 ] - 1.f;
					voxelDataDOWNf2[ 3 ] += 0.f;
#endif
				}

				// Coarser voxel is scaled from current UP voxel (2 times smaller for octree)
				unsigned int voxelPosDOWN[3];
				voxelPosDOWN[ 0 ] = voxelPos[ 0 ] / 2;
				voxelPosDOWN[ 1 ] = voxelPos[ 1 ] / 2;
				voxelPosDOWN[ 2 ] = voxelPos[ 2 ] / 2;

				// Set data in coarser voxel
				unsigned char vd[4];		// "vd" stands for "voxel data"
				vd[ 0 ] = static_cast< unsigned char >( voxelDataDOWNf[ 0 ] / 8.f );
				vd[ 1 ] = static_cast< unsigned char >( voxelDataDOWNf[ 1 ] / 8.f );
				vd[ 2 ] = static_cast< unsigned char >( voxelDataDOWNf[ 2 ] / 8.f );
				vd[ 3 ] = static_cast< unsigned char >( voxelDataDOWNf[ 3 ] / 8.f );
				dataStructureIOHandlerDOWN->setVoxel( voxelPosDOWN, vd, 0 );

#ifdef NORMALS
				// Set normal in coarser voxel
				float norm = sqrtf( voxelDataDOWNf2[ 0 ] * voxelDataDOWNf2[ 0 ] + voxelDataDOWNf2[ 1 ] * voxelDataDOWNf2[ 1 ] + voxelDataDOWNf2[ 2 ] * voxelDataDOWNf2[ 2 ] );
				vd[ 0 ] = static_cast< unsigned char >( ( 0.5f * ( voxelDataDOWNf2[ 0 ] / norm ) + 0.5f ) * 255.f );
				vd[ 1 ] = static_cast< unsigned char >( ( 0.5f * ( voxelDataDOWNf2[ 1 ] / norm ) + 0.5f ) * 255.f );
				vd[ 2 ] = static_cast< unsigned char >( ( 0.5f * ( voxelDataDOWNf2[ 2 ] / norm ) + 0.5f ) * 255.f );
				vd[ 3 ] = 0;
				dataStructureIOHandlerDOWN->setVoxel( voxelPosDOWN, vd, 1 );
#endif
			}
		}
		}

		// Generate the border data of the coarser scene
		dataStructureIOHandlerDOWN->computeBorders();
		
		// Destroy the coarser data handler (due to memory consumption considerations)
		delete dataStructureIOHandlerUP;
		
		// The mip-map pyramid hierarchy is built recursively from adjacent levels.
		// Now that the coarser version has been generated, a coarser one need to be generated from it.
		// So, the coarser one is the UP version.
		dataStructureIOHandlerUP = dataStructureIOHandlerDOWN;
	}

	// Free memory
	delete dataStructureIOHandlerDOWN;
}

/******************************************************************************
 * Load/import the scene the scene
 ******************************************************************************/
bool GvxRAWReader::read()
{
	bool result = false;

	//string filename = string( _filePath + _fileName + _fileExtension );

	string filename = string( "D:\\Projects\\GigaVoxelsTrunk\\Data\\Voxels\\Raw\\skull.raw" );
	_dataResolution = 256;
	_mode = eBinary;

	FILE* file = NULL;

	std::cout << "- read file : " << filename << std::endl;

	if ( _mode == eBinary )
	{
		// TO DO
		// ...
		file = fopen( filename.c_str(), "rb" );
		if ( file != NULL )
		{
			// Create a file/streamer handler to read/write GigaVoxels data
			//unsigned int levelOfResolution = static_cast< unsigned int >( log( static_cast< float >( _dataResolution ) ) / log( static_cast< float >( 2 ) ) );
			unsigned int levelOfResolution = 5;
			unsigned int brickWidth = 8;
			GvxDataTypeHandler::VoxelDataType dataType = GvxDataTypeHandler::gvUCHAR4;
			_dataStructureIOHandler = new GvxDataStructureIOHandler( filename, levelOfResolution, brickWidth, dataType, true );

			// Read current node info and update brick number if not empty
			unsigned char ptr[ 4 ];
			size_t size = sizeof( unsigned char );	// RGB-triplet
			size_t count = 4;

			// Set voxel data
			unsigned int voxelPosition[ 3 ];
			unsigned char voxelData[ 4 ];

			for ( unsigned int z = 0; z < _dataResolution; z++ )
			{
				for ( unsigned int y = 0; y < _dataResolution; y++ )
				{
					for ( unsigned int x = 0; x < _dataResolution; x++ )
					{
						// Voxel position
						voxelPosition[ 0 ] = x;
						voxelPosition[ 1 ] = y;
						voxelPosition[ 2 ] = z;

						// Voxel data
						//fread( &ptr[ 0 ], size, count, file );
						unsigned char voxelData_0;
						fread( &voxelData_0, sizeof( unsigned char ), 1, file );
				
						//std::cout << "- [ " << x << " ; " << y << " ; " << z << " ] " << "- [ " << ptr[ 0 ] << " ; " << ptr[ 1 ] << " ; " << ptr[ 2 ] << " ; " << ptr[ 0 ] << " ] " << std::endl;
						//std::cout << "- [ " << x << " ; " << y << " ; " << z << " ] " << "- [ " << voxelData_0 << " ; " << voxelData_1 << " ; " << voxelData_2 << " ; " << voxelData_3 << " ] " << std::endl;
						//std::cout << "- [ " << x << " ; " << y << " ; " << z << " ] " << "- [ " << atoi( &voxelData_0 ) << " ; " << atoi( &voxelData_0 ) << " ; " << atoi( &voxelData_0 ) << " ; " << atoi( &voxelData_0 ) << " ] " << std::endl;

						voxelData[ 0 ] = voxelData_0;
						voxelData[ 1 ] = voxelData_0;
						voxelData[ 2 ] = voxelData_0;

						//voxelData[ 3 ] = 255;

						voxelData[ 3 ] = voxelData_0;

						//voxelData[ 3 ] = ptr[ 3 ];

						//voxelData[ 0 ] = atoi( &voxelData_0 );
						//voxelData[ 1 ] = atoi( &voxelData_0 );
						//voxelData[ 2 ] = atoi( &voxelData_0 );

						//voxelData[ 3 ] = 255;
						//voxelData[ 3 ] = atoi( &voxelData_0 );

						// Write voxel data
						_dataStructureIOHandler->setVoxel( voxelPosition, voxelData, 0 );	// channel 0
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
		file = fopen( filename.c_str(), "r" );
		if ( file != NULL )
		{
			// Create a file/streamer handler to read/write GigaVoxels data
			//unsigned int levelOfResolution = static_cast< unsigned int >( log( static_cast< float >( _dataResolution ) ) / log( static_cast< float >( 2 ) ) );
			unsigned int levelOfResolution = 5;
			unsigned int brickWidth = 8;
			GvxDataTypeHandler::VoxelDataType dataType = GvxDataTypeHandler::gvUCHAR4;
			_dataStructureIOHandler = new GvxDataStructureIOHandler( filename, levelOfResolution, brickWidth, dataType, true );

			// Read current node info and update brick number if not empty
			unsigned char* ptr = NULL;
			size_t size = sizeof( unsigned char );	// RGB-triplet
			size_t count = 3;

			// Set voxel data
			unsigned int voxelPosition[ 3 ];
			unsigned char voxelData[ 4 ];

			for ( unsigned int z = 0; z < _dataResolution; z++ )
			{
				for ( unsigned int y = 0; y < _dataResolution; y++ )
				{
					for ( unsigned int x = 0; x < _dataResolution; x++ )
					{
						// Voxel position
						voxelPosition[ 0 ] = x;
						voxelPosition[ 1 ] = y;
						voxelPosition[ 2 ] = z;

						// Voxel data
						fread( &ptr, size, count, file );
						voxelData[ 0 ] = ptr[ 0 ];
						voxelData[ 1 ] = ptr[ 1 ];
						voxelData[ 2 ] = ptr[ 2 ];
						voxelData[ 3 ] = 255;

						// Write voxel data
						_dataStructureIOHandler->setVoxel( voxelPosition, voxelData, 0 );	// channel 0
					}
				}
			}
		}
		else
		{
			assert( false );
		}
	}

	std::cout << "- update borders" << _dataStructureIOHandler->_level << std::endl;
	_dataStructureIOHandler->computeBorders();

	std::cout << "- mipmap generation" << _dataStructureIOHandler->_level << std::endl;
	mipmap();

	return result;
}
