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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvUtils/GvBrickLoaderChannelInitializer.h"
#include "GvCore/GvError.h"

// TinyXML
#include <tinyxml.h>
//#include <tinystr.h>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvUtils
{

/******************************************************************************
 * Constructor
 *
 * @param pName filename
 * @param pDataSize volume resolution
 * @param pBlocksize brick resolution
 * @param pBordersize brick broder size
 * @param pUseCache  flag to tell wheter or not a cache mechanismn is required when reading files (nodes and bricks)
 ******************************************************************************/
template< typename TDataTypeList >
GvDataLoader< TDataTypeList >
::GvDataLoader( const std::string& pName, const uint3& pBlocksize, int pBordersize, bool pUseCache )
{
	uint resolution = 0;
	int parse = this->parseXMLFile( pName.c_str(), resolution );
	assert( parse == 0 );
		
	this->_bricksRes.x = pBlocksize.x;
	this->_bricksRes.y = pBlocksize.y;
	this->_bricksRes.z = pBlocksize.z;
		
	// Fill member variables
	this->_numChannels = Loki::TL::Length< TDataTypeList >::value;
	this->_volumeRes = make_uint3( resolution );//TO CHANGE
	printf( "%d\n", _volumeRes.x );
	this->_borderSize = pBordersize;
	this->_mipMapOrder = 2;
	this->_useCache = pUseCache;

	// Compute number of mipmaps levels
	int dataResMin = mincc( this->_volumeRes.x, mincc( this->_volumeRes.y, this->_volumeRes.z ) );
	int blocksNumLevels		= static_cast< int >( log( static_cast< float >( this->_bricksRes.x ) ) / log( static_cast< float >( this->_mipMapOrder ) ) );
	this->_numMipMapLevels	= static_cast< int >( log( static_cast< float >( dataResMin ) ) / log( static_cast< float >( this->_mipMapOrder ) ) ) + 1 ;
	this->_numMipMapLevels	= this->_numMipMapLevels - blocksNumLevels;
	if ( this->_numMipMapLevels < 1 )
	{
		this->_numMipMapLevels = 1;
	}

	// Compute full brick resolution (with borders)
	uint3 true_bricksRes = this->_bricksRes + make_uint3( 2 * this->_borderSize );

	// Build the list of all filenames that producer will have to load (nodes and bricks).
	//this->makeFilesNames( pName.c_str() );

	// If cache mechanismn is required, read all files (nodes and bricks),
	// and store data in associated buffers.
	if ( this->_useCache )
	{
		// Iterate through mipmap levels
		
		for ( int level = 0; level < _numMipMapLevels; level++ )
		{
			
			// Retrieve node filename at current mipmap level
			//
			// Files are stored by mipmap level in the list :
			// - first : node file
			// - then : brick file for each channel
			std::string fileNameIndex = _filesNames[ ( _numChannels + 1 ) * level ];
			
			// Open node file
			FILE* fileIndex = fopen( fileNameIndex.c_str(), "rb" );
			if ( fileIndex )
			{
				
				// Retrieve node file size
#ifdef WIN32
				_fseeki64( fileIndex, 0, SEEK_END );

				__int64 size = _ftelli64( fileIndex );
				__int64 expectedSize = (__int64)powf( 8.0f, static_cast< float >( level ) ) * sizeof( unsigned int );
#else
				fseeko( fileIndex, 0, SEEK_END );

				off_t size = ftello( fileIndex );
				off_t expectedSize = (off_t)powf( 8.0f, static_cast< float >( level ) ) * sizeof( unsigned int );
#endif
				// Handle error
				if ( size != expectedSize )
				{					
					std::cerr << "GvDataLoader::GvDataLoader: file size expected = " << expectedSize
								<< ", size returned = " << size << " for " << fileNameIndex << std::endl;
				}

				// Allocate a buffer in which read node data will be stored
				unsigned int* tmpCache = new unsigned int[ size / 4 ];

				// Position file pointer at beginning of file
#ifdef WIN32
				_fseeki64( fileIndex, 0, SEEK_SET );
#else
				fseeko( fileIndex, 0, SEEK_SET );
#endif
				// Read node file and store data in the tmpCache buffer
				if ( fread( tmpCache, 1, static_cast< size_t >( size ), fileIndex ) != size )
				{
					// Handle error if reading node file has failed
					std::cout << "GvDataLoader::GvDataLoader: Unable to open file " << this->_filesNames[ level ] << std::endl;
					this->_useCache = false;
				}
				
				// Close node file
				fclose( fileIndex );

				// Store node data in associated cache
				_blockIndexCache.push_back( tmpCache );
			}
			else
			{
				// Handle error if opening node file has failed
				std::cout << "GvDataLoader::GvDataLoader : Unable to open file index " << fileNameIndex << std::endl;
			}

			// For current mipmap level, iterate through channels
			for ( size_t channel = 0; channel < _numChannels; channel++ )
			{
				// Retrieve brick filename at current channel (at current mipmap level)
				//
				// Files are stored by mipmap level in the list :
				// - first : node file
				// - then : brick file for each channel
				
				// Open brick file
				unsigned int fileIndex = ( _numChannels + 1 ) * level + channel + 1;
				if ( fileIndex >= this->_filesNames.size() )
				{
					assert( false );
					std::cout << "GvDataLoader::GvDataLoader() => File index error." << std::endl;
				}
				FILE* brickFile = fopen( this->_filesNames[ fileIndex ].c_str(), "rb" );
				if ( brickFile )
				{
					// Retrieve brick file size
#ifdef WIN32
					_fseeki64( brickFile, 0, SEEK_END );

					__int64 size = _ftelli64( brickFile );
#else
					fseeko( brickFile, 0, SEEK_END );

					off_t size = ftello( brickFile );
#endif
					// Allocate a buffer in which read brick data will be stored
					unsigned char* tmpCache;
#if USE_GPUFETCHDATA
					cudaHostAlloc( (void**)&tmpCache, size, cudaHostAllocMapped | cudaHostAllocWriteCombined );
					
					void* deviceptr;
					cudaHostGetDevicePointer( &deviceptr, tmpCache, 0 );
					std::cout << "Device pointer host mem: " << static_cast< uint >( deviceptr ) << "\n";
#else
					// cudaMallocHost( (void**)&tmpCache, size ); // pinned memory
					// cudaHostAlloc( (void **)&tmpCache, size );
					tmpCache = new unsigned char[ static_cast< size_t >( size ) ];
#endif
					GV_CHECK_CUDA_ERROR( "GvDataLoader::GvDataLoader: cache alloc" );

					// Position file pointer at beginning of file
#ifdef WIN32
					_fseeki64( brickFile, 0, SEEK_SET );
#else
					fseeko( brickFile, 0, SEEK_SET );
#endif
					
					// Read brick file and store data in the tmpCache buffer
					if ( fread( tmpCache, 1, static_cast< size_t >( size ), brickFile ) != size )
					{
						// Handle error if reading brick file has failed
						std::cout << "GvDataLoader::GvDataLoader: Can't read file" << std::endl;
						this->_useCache = false;
					}
					
					// Close brick file
					fclose( brickFile );

					// Store brick data in associated cache
					_blockCache.push_back( tmpCache );
				}
				else
				{
					// Handle error if opening brick file has failed
					std::cout << "GvDataLoader::GvDataLoader: Unable to open file " << this->_filesNames[(_numChannels + 1) * level + channel + 1] << std::endl;
					this->_useCache = false;

				}
				
			}
		}
	}

}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TDataTypeList >
GvDataLoader< TDataTypeList >
::~GvDataLoader()
{
	// Free memory of bricks data
	for	( size_t i = 0; i < _blockCache.size(); i++ )
	{
		if ( _blockCache[ i ] )
		{
#if USE_GPUFETCHDATA
			//cudaFree( _blockCache[ i ] );
			cudaFreeHost( _blockCache[ i ] );
#else
			delete[] _blockCache[ i ];
#endif
		}
	}

	// Free memory of nodes data
	for	( size_t i = 0; i < _blockIndexCache.size(); i++ )
	{
		if ( _blockIndexCache[ i ] )
		{
			delete [] _blockIndexCache[ i ];
		}
	}
}

/******************************************************************************
 * Build the list of all filenames that producer will have to load (nodes and bricks).
 * Given a root name, this function create filenames by adding brick resolution,
 * border size, file extension, etc...
 *
 * @param pFilename the root filename from which all filenames will be built
 ******************************************************************************/
template< typename TDataTypeList >
void GvDataLoader< TDataTypeList >
::makeFilesNames( const char* pFilename )
{
	//// Set common filenames parameters
	//std::string sourceFileName = std::string( pFilename );
	//std::string nodesFileNameExt = ".nodes";
	//std::string bricksFileNameExt = ".bricks";

	//// Iterate through mipmap levels
	//for	( int i = 0; i < _numMipMapLevels; i++ )
	//{
	//	std::stringstream ssNodes;

	//	// Build nodes file
	//	ssNodes << sourceFileName << "_BR" << _bricksRes.x << "_B" << _borderSize << "_L" << i << nodesFileNameExt;
	//	_filesNames.push_back( ssNodes.str() );

	//	// Build bricks file
	//	GvFileNameBuilder< TDataTypeList > fnb( _bricksRes.x, _borderSize, i, sourceFileName, bricksFileNameExt, _filesNames );
	//	GvCore::StaticLoop< GvFileNameBuilder< TDataTypeList >, Loki::TL::Length< TDataTypeList >::value - 1 >::go( fnb );
	//}

	//int parse = parseXMLFile(pFilename);
	//assert(parse ==0);
}


/******************************************************************************
 * Parses the XML configuration file
 *
 * @param pFilename the filename of the XML file
 ******************************************************************************/
template< typename TDataTypeList >
int GvDataLoader< TDataTypeList >
::parseXMLFile( const char* pFilename , uint& resolution)
{
	resolution = 1;
	std::string directory = std::string( pFilename );
	directory = directory.substr(0, directory.find_last_of("\\/")+1);
	int nbLevels =0;
	int nbChannels =0;

	std::vector<std::string> nodes;
	std::vector<std::string> bricks;
	
	TiXmlDocument doc(pFilename);
	bool loadOkay = doc.LoadFile();
	if (loadOkay)
	{
		TiXmlNode* model = doc.FirstChild();
		if (strcmp(model->Value(),root)==0)
		{
			TiXmlAttribute* attrib=model->ToElement()->FirstAttribute();
			bool directorySet = false;
			bool nbLevelsSet = false;
			while (attrib)
			{
				if (strcmp(attrib->Name(),rootName)==0)
				{	
					printf("Loading model %s\n",attrib->Value());	
				} 
				else if (strcmp(attrib->Name(),rootDir)==0)
				{	
					//printf("In directory %s\n",attrib->Value());
					directory = directory + std::string(attrib->Value()) + "/";
					directorySet = true;
				} 
				else if (strcmp(attrib->Name(),nodeTreeNbLevels)==0)
				{
					//printf("Nb levels %s\n",attrib->Value());
					nbLevels = atoi(attrib->Value());
					resolution *= powf(2,nbLevels-1);
					//printf("%d\n",(int)(resolution));
					nbLevelsSet = true;
				} else 
				{
					printf("XML WARNING Unknown attribute: %s\n",attrib->Value());
				}
				attrib = attrib->Next();
			}
			if (directorySet && nbLevelsSet) 
			{
				
				model = model->FirstChild();
				TiXmlNode* level ;
				TiXmlNode* channel ;

				
				while (model)
				{
					if (strcmp(model->Value() , nodeTree)==0)
					{
						level = model->FirstChild();
						while (level)
						{
							if (strcmp(level->Value(),node)==0)
							{
								//printf("Node\n");
								attrib=level->ToElement()->FirstAttribute();
								while (attrib)
								{
									if (strcmp(attrib->Name(),nodeId)==0)
									{	
										//printf("Id : %s\n",attrib->Value());	
									} else if ( strcmp(attrib->Name(),nodeFilename)==0)
									{
										//printf("Filename : %s\n",(directory+std::string(attrib->Value())).c_str());
										nodes.push_back( directory+std::string(attrib->Value()) );
									} else 
									{
										printf("XML WARNING Unknown attribute: %s\n",attrib->Value());
									}

									attrib = attrib->Next();
								}
							} else {
								printf("XML WARNING Unexpected token : %s expected Level\n",level->Value());
							}

							level = level->NextSibling();
						}
				
					}
					else if ( strcmp(model->Value() , brickData)==0)
					{
						attrib=model->ToElement()->FirstAttribute();
						while (attrib)
						{
							if (strcmp(attrib->Name(),brickDataResolution)==0)
							{	
								//printf("Resolution : %s\n",attrib->Value());	
								resolution *= atoi(attrib->Value());
							} else if ( strcmp(attrib->Name(),brickDataBorderSize)==0)
							{
								//printf("Border size : %s\n",attrib->Value());
							} else 
							{
								//printf("XML WARNING Unknown attribute: %s\n",attrib->Value());
							}

							attrib = attrib->Next();
						}


						channel = model->FirstChild();
						while (channel)
						{
							nbChannels++;
							if (strcmp(channel->Value(),brickChannel)==0)
							{
								//printf("Channel\n");
								attrib=channel->ToElement()->FirstAttribute();
								while (attrib)
								{
									if (strcmp(attrib->Name(),channelId)==0)
									{	
										//printf("Id : %s\n",attrib->Value());	
									} else if ( strcmp(attrib->Name(),channelName)==0)
									{
										//printf("Name : %s\n",attrib->Value());
									} else if ( strcmp(attrib->Name(),channelType)==0)
									{
										//printf("Type : %s\n",attrib->Value());
									} else 
									{
										printf("XML WARNING Unknown attribute: %s\n",attrib->Value());
									}
									attrib = attrib->Next();
								}

								level = channel->FirstChild();
								while (level)
								{
									if (strcmp(level->Value(),brickLevel)==0)
									{
										//printf("Brick\n");
										attrib=level->ToElement()->FirstAttribute();
										while (attrib)
										{
											if (strcmp(attrib->Name(),levelId)==0)
											{	
												//printf("Id : %s\n",attrib->Value());	
											} else if ( strcmp(attrib->Name(),levelFilename)==0)
											{
												//printf("Filename : %s\n",(directory+std::string(attrib->Value())).c_str());
												bricks.push_back(directory+std::string( attrib->Value()) );
											} else 
											{
												printf("XML WARNING Unknown attribute: %s\n",attrib->Value());
											}

											attrib = attrib->Next();
										}
									} else {
										printf("XML WARNING Unexpected token : %s expected Level\n",level->Value());
									}

									level = level->NextSibling();
								}
							}
							else 
							{
								printf("XML WARNING Unexpected token : %s expected Channel\n",channel->Value());
							}
							channel = channel->NextSibling();
						}
					}
					else 
					{
						printf("XML WARNING Unknown token : %s\n",model->Value());
					}
					model = model->NextSibling();
				}				
			} else {
				printf("XML ERROR Wrong Syntax : Missing model informations (directory or nbLevels)\n",root,model->Value() );
				return -1;
			}
		}
		else {
			printf("XML ERROR Wrong Syntax : expected \"%s\", read \"%s\"\n",root,model->Value() );
			return -1;
		}
	}
	else
	{
		printf("XML ERROR Failed to load file %s \n",pFilename);
		return -1;
	}

	//printf( "%d %d\n", nbLevels ,nbChannels );
	for ( int k = 0; k < nbLevels ; k++ )
	{
		_filesNames.push_back( nodes[ k ] );
		//printf( "%s\n", nodes[ k ].c_str() );
		for ( int p = 0; p < nbChannels ; p++ )
		{
			_filesNames.push_back( bricks[ k + p * nbLevels ] );
			//printf( "%s\n", bricks[ k + p * nbLevels ].c_str() );
		}
	}
	
	return 0;
}

/******************************************************************************
 * Retrieve the node encoded address given a mipmap level and a 3D node indexed position
 *
 * @param pLevel mipmap level
 * @param pBlockPos the 3D node indexed position
 *
 * @return the node encoded address
 ******************************************************************************/
template< typename TDataTypeList >
unsigned int GvDataLoader< TDataTypeList >
::getBlockIndex( int pLevel, const uint3& pBlockPos ) const
{
	// Retrieve the resolution at given level (i.e. the number of voxels in each dimension)
	uint3 levelSize = getLevelRes( pLevel );

	// Number of nodes at given level
	uint3 blocksInLevel = levelSize / this->_bricksRes;
	
	unsigned int indexValue = 0;

	// Check wheter or not, cache mechanism is used
	if ( _useCache )
	{
		// _blockIndexCache is the buffer containing all read nodes data.
		// This a 2D array containing, for each mipmap level, the list of nodes addresses.

		// Compute the index of the node in the buffer of nodes, given its position
		//
		// Nodes are stored in increasing order from X axis first, then Y axis, then Z axis.
		uint indexPos = pBlockPos.x + pBlockPos.y * blocksInLevel.x + pBlockPos.z * blocksInLevel.x * blocksInLevel.y;

		// Get the node address
		indexValue = _blockIndexCache[ pLevel ][ indexPos ];
	}
	else
	{
		// Compute the index of the node in the buffer of nodes, given its position
		//
		// Nodes are stored in increasing order from X axis first, then Y axis, then Z axis.
#ifdef WIN32
		__int64 indexPos = ( (__int64)pBlockPos.x + (__int64)( pBlockPos.y * blocksInLevel.x ) + (__int64)( pBlockPos.z * blocksInLevel.x * blocksInLevel.y ) ) * sizeof( unsigned int );
#else
		off_t indexPos = ( (off_t)pBlockPos.x + (off_t)( pBlockPos.y * blocksInLevel.x ) + (off_t)( pBlockPos.z * blocksInLevel.x * blocksInLevel.y ) ) * sizeof( unsigned int );
#endif
		// Retrieve node filename at given level
		std::string fileNameIndex = this->_filesNames[ ( _numChannels + 1 ) * pLevel ];

		// Open node file
		FILE* fileIndex = fopen( fileNameIndex.c_str(), "rb" );
		if ( fileIndex )
		{
			// Position file pointer at position corresponding to the requested node information
#ifdef WIN32
			_fseeki64( fileIndex, indexPos, 0 );
#else
			fseeko( fileIndex, indexPos, 0 );
#endif
			// Read node file and store requested node address in indexValue
			if ( fread( &indexValue, sizeof( unsigned int ), 1, fileIndex ) != 1 )
			{
				// Handle error if reading node file has failed
				std::cerr << "GvDataLoader<T>::getBlockIndex(): fread failed." << std::endl;
			}

			// Close node file
			fclose( fileIndex );
		}
		else
		{
			// Handle error if opening node file has failed
			std::cerr << "GvDataLoader<T>::getBlockIndex() : Unable to open file index "<<fileNameIndex << std::endl;
		}
	}

	return indexValue;
}

/******************************************************************************
 * Load a brick given a mipmap level, a 3D node indexed position,
 * the data pool and an offset in the data pool.
 *
 * @param pLevel mipmap level
 * @param pBlockPos the 3D node indexed position
 * @param pDataPool the data pool
 * @param pOffsetInPool offset in the data pool
 *
 * @return a flag telling wheter or not the brick has been loaded (some brick can contain no data).
 ******************************************************************************/
template< typename TDataTypeList >
bool GvDataLoader< TDataTypeList >
::loadBrick( int pLevel, const uint3& pBlockPos, GvCore::GPUPoolHost< GvCore::Array3D, TDataTypeList >* pDataPool, size_t pOffsetInPool )
{
	// Retrieve the resolution at given level (i.e. the number of voxels in each dimension)
	uint3 levelSize = getLevelRes( pLevel );

	//uint3 blocksInLevel = levelSize / this->_bricksRes;	// seem to be not used anymore

	// Compute full brick resolution (with borders)
	uint3 trueBlocksRes = this->_bricksRes + make_uint3( 2 * this->_borderSize );

	// Compute the brick size alignment in memory (with borders)
	size_t blockMemSize = static_cast< size_t >( trueBlocksRes.x * trueBlocksRes.y * trueBlocksRes.z );

	// Retrieve the node encoded address given a mipmap level and a 3D node indexed position
	unsigned int indexVal = getBlockIndex( pLevel, pBlockPos );

	// Test if node contains a brick
	if ( indexVal & GV_VTBA_BRICK_FLAG )
	{
		// Use a channel initializer to read the brick
		GvBrickLoaderChannelInitializer< TDataTypeList > channelInitializer( this, indexVal, blockMemSize, pLevel, pDataPool, pOffsetInPool );
		GvCore::StaticLoop< GvBrickLoaderChannelInitializer< TDataTypeList >, Loki::TL::Length< TDataTypeList >::value - 1 >::go( channelInitializer );

		return true;
	}
	
	return false;
} 

/******************************************************************************
 * Helper function used to determine the type of regions in the data structure.
 * The data structure is made of regions containing data, empty or constant regions.
 *
 * Retrieve the node and associated brick located in this region of space,
 * and depending of its type, if it contains data, load it.
 *
 * @param pPosition position of a region of space
 * @param pSize size of a region of space
 * @param pBrickPool data cache pool. This is where all data reside for each channel (color, normal, etc...)
 * @param pOffsetInPool offset in the brick pool
 *
 * @return the type of the region (.i.e returns constantness information for that region)
 ******************************************************************************/
template< typename TDataTypeList >
GvDataLoader< TDataTypeList >::VPRegionInfo GvDataLoader< TDataTypeList >
::getRegion( const float3& pPosition, const float3& pSize, GvCore::GPUPoolHost< GvCore::Array3D, TDataTypeList >* pBrickPool, size_t pOffsetInPool )
{
	// Retrieve the level of resolution associated to a given size of a region of space.
	int level =	getDataLevel( pSize, _bricksRes );

	// Retrieve the indexed coordinates associated to position of region of space at level of resolution.
	uint3 coordsInLevel = getCoordsInLevel( level, pPosition );
	// Retrieve the indexed coordinates of a block (i.e. a node) in the blocks grid of a region of space at a level of resolution.
	uint3 blockCoords = getBlockCoords( level, pPosition );

	// Check mipmap level bounds
	if ( level >= 0 && level < _numMipMapLevels )
	{
		// Try to load brick given localization parameters
		if ( loadBrick( level, blockCoords, pBrickPool, pOffsetInPool ) )
		{
			return GvDataLoader< TDataTypeList >::VP_UNKNOWN_REGION;
		}
		else
		{
			return GvDataLoader< TDataTypeList >::VP_CONST_REGION; // Correct ?
		}
	}
	else
	{
		// Handle error
		std::cout << "VolProducerBlocksOptim::getZone() : Invalid requested block dimentions" << std::endl;
		return GvDataLoader< TDataTypeList >::VP_CONST_REGION;
	}
}

/******************************************************************************
 * Provides constantness information about a region.
 *
 * @param pPosition position of a region of space
 * @param pSize size of a region of space
 *
 * @return the type of the region (.i.e returns constantness information for that region)
 ******************************************************************************/
template< typename TDataTypeList >
GvDataLoader< TDataTypeList >::VPRegionInfo GvDataLoader< TDataTypeList >
::getRegionInfo( const float3& pPosition, const float3& pSize/*, T* constValueOut*/ )
{
	// Retrieve the level of resolution associated to a given size of a region of space.
	int level =	getDataLevel( pSize, _bricksRes );
	// Retrieve the indexed coordinates of a block (i.e. a node) in the blocks grid of a region of space at a level of resolution.
	uint3 blockPosition = getBlockCoords( level, pPosition );

	// Retrieve the resolution at given level (i.e. the number of voxels in each dimension)
	uint3 levelsize = getLevelRes( level );
	//uint3 blocksinlevel = levelsize / this->_bricksRes;	// seem to be not used anymore...

	uint3 trueBlocksRes = this->_bricksRes+ make_uint3( 2 * this->_borderSize );
	//size_t blockmemsize = (size_t)(trueBlocksRes.x*trueBlocksRes.y*trueBlocksRes.z );

	// Check mipmap level bounds
	if ( level >= 0 && level < _numMipMapLevels )
	{
		// Retrieve the node encoded address given a mipmap level and a 3D node indexed position
		unsigned int indexValue = getBlockIndex( level, blockPosition );

		// If there is a brick
		if ( indexValue & 0x40000000U )
		{
			// If we are on a terminal node
			if ( indexValue & 0x80000000U )
			{
				return GvDataLoader< TDataTypeList >::VP_UNKNOWN_REGION;
			}
			else
			{
				return GvDataLoader< TDataTypeList >::VP_NON_CONST_REGION;
			}
		}
		else
		{
			return GvDataLoader< TDataTypeList >::VP_CONST_REGION;
		}
	}
	else
	{
		return GvDataLoader< TDataTypeList >::VP_CONST_REGION;
	}
}

/******************************************************************************
 * Retrieve the node located in a region of space,
 * and get its information (i.e. address containing its data type region).
 *
 * @param pPosition position of a region of space
 * @param pSize size of a region of space
 *
 * @return the node encoded information
 ******************************************************************************/
template< typename TDataTypeList >
uint GvDataLoader< TDataTypeList >
::getRegionInfoNew( const float3& pPosition, const float3& pSize )
{
	// Retrieve the level of resolution associated to a given size of a region of space.
	int level =	getDataLevel( pSize, _bricksRes );
	// Retrieve the indexed coordinates of a block (i.e. a node) in the blocks grid of a region of space at a level of resolution.
	uint3 blockPosition = getBlockCoords( level, pPosition );

	// Retrieve the resolution at given level (i.e. the number of voxels in each dimension)
	uint3 levelsize = getLevelRes( level );
	//uint3 blocksinlevel = levelsize / this->_bricksRes;	// seem to be not used anymore...

	// Compute full brick resolution (with borders)
	uint3 trueBlocksRes = this->_bricksRes + make_uint3( 2 * this->_borderSize );

	// Check mipmap level bounds
	if ( level >= 0 && level < _numMipMapLevels )
	{
		// Get the node encoded address in the pool.
		//
		// Retrieve the node encoded address given a mipmap level and a 3D node indexed position
		// Apply a mask on the two first bits to retrieve node information
		// (the other 30 bits are for x,y,z address).
		return ( getBlockIndex( level, blockPosition ) & 0xC0000000 );
	}
	return 0;
}

/******************************************************************************
 * Retrieve the resolution at a given level (i.e. the number of voxels in each dimension)
 *
 * @param level the level
 *
 * @return the resolution at given level
 ******************************************************************************/
template< typename TDataTypeList >
inline uint3 GvDataLoader< TDataTypeList >
::getLevelRes( uint level ) const
{
	//return _volumeRes / (1<<level); // WARNING: suppose mipMapOrder==2 !
	return _bricksRes * ( 1 << level ); // WARNING: suppose mipMapOrder==2 !
}

/******************************************************************************
 * Provides the size of the smallest features the producer can generate.
 *
 * @return the size of the smallest features the producer can generate
 ******************************************************************************/
template< typename TDataTypeList >
inline float3 GvDataLoader< TDataTypeList >
::getFeaturesSize() const
{
	return make_float3( 1.0f ) / make_float3( _volumeRes );
}

/******************************************************************************
 * Read a brick given parameters to retrieve data localization in brick files or in cache of bricks.
 *
 * @param pChannel channel index (i.e. color, normal, density, etc...)
 * @param pIndexVal associated node encoded address of the node in which the brick resides
 * @param pBlockMemSize brick size alignment in memory (with borders)
 * @param pLevel mipmap level
 * @param pData data array corresponding to given channel index in the data pool (i.e. bricks of voxels)
 * @param pOffsetInPool offset in the data pool
 ******************************************************************************/
template< typename TDataTypeList >
template< typename TChannelType >
inline void GvDataLoader< TDataTypeList >
::readBrick( int pChannel, unsigned int pIndexVal, unsigned int pBlockMemSize, unsigned int pLevel, GvCore::Array3D< TChannelType >* pData, size_t pOffsetInPool )
{
	// pIndexVal & 0x3FFFFFFFU : this expresssion correponds to the last 30 bits of the 32 bits node encoded address
	// These 30 bits corresponds to the address of the node on x,y,z axes.
	//
	// Compute the offset
	unsigned int filePos = ( pIndexVal & 0x3FFFFFFFU ) * pBlockMemSize * sizeof( TChannelType );

	// Check wheter or not, cache mechanism is used
	if ( _useCache )
	{
		if ( _blockCache[ pLevel * _numChannels + pChannel ] )
		{
			// Copy data from cache to the channel array of the data pool
			memcpy( pData->getPointer( pOffsetInPool )/* destination */,
					_blockCache[ pLevel * _numChannels + pChannel ] + filePos/* source */,
					pBlockMemSize * sizeof( TChannelType ) /* number of bytes*/ );
		}
	}
	else
	{
		// Open brick file
		FILE* file = fopen( _filesNames[ pLevel * ( _numChannels + 1 ) + pChannel + 1 ].c_str(), "rb" );
		if ( file )
		{
			// Position file pointer at position corresponding to the requested brick data
#ifdef WIN32
			_fseeki64( file, filePos, 0 );
#else
			fseeko( file, filePos, 0 );
#endif
			// Read brick file and store data in the channel array of the data pool
			fread( pData->getPointer( pOffsetInPool ), sizeof( TChannelType ), pBlockMemSize, file );

			// Close brick file
			fclose( file );
		}
	}
}

/******************************************************************************
 * Retrieve the indexed coordinates of a block (i.e. a node) in the blocks grid
 * associated to a given position of a region of space at a given level of resolution.
 *
 * @param pLevel level of resolution
 * @param pPosition position of a region of space
 *
 * @return the associated indexed coordinates
 ******************************************************************************/
template< typename TDataTypeList >
inline uint3 GvDataLoader< TDataTypeList >
::getBlockCoords( int pLevel, const float3& pPosition ) const
{
	return getCoordsInLevel( pLevel, pPosition ) / this->_bricksRes;
}

/******************************************************************************
 * Retrieve the level of resolution associated to a given size of a region of space.
 *
 * @param pSize size of a region of space
 * @param pResolution resolution of data (i.e. brick)
 *
 * @return the corresponding level
 ******************************************************************************/
template< typename TDataTypeList >
inline int GvDataLoader< TDataTypeList >
::getDataLevel( const float3& pSize, const uint3& pResolution ) const
{
	// Compute the node resolution (i.e. number of nodes in each dimension)
	uint3 numNodes = make_uint3( 1.0f / pSize );
	int level = static_cast< int >( log( static_cast< float >( numNodes.x ) ) / log( static_cast< float >( _mipMapOrder ) ) );

	// uint3 resinfulldata = make_uint3( make_float3( this->_volumeRes ) * pSize );
	// int level = (int)( log( resinfulldata.x / (float)pRes.x ) / log( (float)( _mipMapOrder ) ) );

	return level;
}

/******************************************************************************
 * Retrieve the indexed coordinates associated to a given position of a region of space
 * at a given level of resolution.
 *
 * @param pLevel level of resolution
 * @param pPosition position of a region of space
 *
 * @return the associated indexed coordinates
 ******************************************************************************/
template< typename TDataTypeList >
inline uint3 GvDataLoader< TDataTypeList >
::getCoordsInLevel( int pLevel, const float3& pPosition ) const
{
	// Retrieve the resolution at given level (i.e. the number of voxels in each dimension)
	uint3 levelResolution = getLevelRes( pLevel );
	uint3 coordsInLevel = make_uint3( make_float3( levelResolution ) * pPosition );

	return coordsInLevel;
}

} // namespace GvUtils
