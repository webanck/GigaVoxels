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

#ifndef _GV_DATA_LOADER_H_
#define _GV_DATA_LOADER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// STL
#include <vector>

// System
#include <string>
#include <sstream>

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/Array3D.h"
#include "GvCore/TypeHelpers.h"
#include "GvCore/vector_types_ext.h"
#include "GvUtils/GvIDataLoader.h"
#include "GvUtils/GvFileNameBuilder.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

const char * root ="Model";
const char * rootName="name";
const char * rootDir ="directory";
const char * nodeTree ="NodeTree";
const char * nodeTreeNbLevels = "nbLevels";
const char * node= "Level";
const char * nodeId = "id";
const char * nodeFilename = "filename";
const char * brickData = "BrickData";
const char * brickDataResolution = "brickResolution";
const char * brickDataBorderSize = "borderSize";
const char * brickChannel = "Channel";
const char * channelId = "id";
const char * channelName = "name";
const char * channelType = "type";
const char * brickLevel = "Level";
const char * levelId = "id";
const char * levelFilename = "filename";




/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvUtils
{

/** 
 * @class GvDataLoader
 *
 * @brief The GvDataLoader class provides...
 *
 * ...
 */
template< typename TDataTypeList >
class GvDataLoader : public GvIDataLoader< TDataTypeList >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Typedef of inherited enum
	 */
	typedef typename GvIDataLoader< TDataTypeList >::VPRegionInfo VPRegionInfo;

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param pName filename
	 * @param pDataSize volume resolution
	 * @param pBlocksize brick resolution
	 * @param pBordersize brick broder size
	 * @param pUseCache  flag to tell wheter or not a cache mechanismn is required when reading files (nodes and bricks)
	 */
	GvDataLoader( const std::string& pName, const uint3& pBlocksize, int pBordersize, bool pUseCache = false );

	/**
	 * Destructor
	 */
	 virtual ~GvDataLoader();

	/**
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
	 */
	virtual VPRegionInfo getRegion( const float3& pPosition, const float3& pSize, GvCore::GPUPoolHost< GvCore::Array3D, TDataTypeList >* pBrickPool, size_t pOffsetInPool );
	/**
	 * Provides constantness information about a region.
	 *
	 * @param pPosition position of a region of space
	 * @param pSize size of a region of space
	 *
	 * @return the type of the region (.i.e returns constantness information for that region)
	 */
	virtual VPRegionInfo getRegionInfo( const float3& pPosition, const float3& pSize/*, TDataTypeList *constValueOut = NULL*/ );

	/**
	 * Retrieve the node located in a region of space,
	 * and get its information (i.e. address containing its data type region).
	 *
	 * @param pPosition position of a region of space
	 * @param pSize size of a region of space
	 *
	 * @return the node encoded information
	 */
	virtual uint getRegionInfoNew( const float3& pPosition, const float3& pSize );

	/**
	 * Retrieve the resolution at a given level (i.e. the number of voxels in each dimension)
	 *
	 * @param pLevel the level
	 *
	 * @return the resolution at given level
	 */
	inline uint3 getLevelRes( uint pLevel ) const;

	/**
	 * Provides the size of the smallest features the producer can generate.
	 *
	 * @return the size of the smallest features the producer can generate.
	 */
	inline virtual float3 getFeaturesSize() const;

	/**
	 * Read a brick given parameters to retrieve data localization in brick files or in cache of bricks.
	 *
	 * @param pChannel channel index (i.e. color, normal, density, etc...)
	 * @param pIndexVal associated node encoded address of the node in which the brick resides
	 * @param pBlockMemSize brick size alignment in memory (with borders)
	 * @param pLevel mipmap level
	 * @param pData data array corresponding to given channel index in the data pool (i.e. bricks of voxels)
	 * @param pOffsetInPool offset in the data pool
	 */
	template< typename TChannelType >
	inline void readBrick( int pChannel, unsigned int pIndexVal, unsigned int pBlockMemSize, unsigned int pLevel, GvCore::Array3D< TChannelType >* pData, size_t pOffsetInPool );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Volume resolution
	 */
	uint3 _volumeRes;

	/**
	 * Brick resolution
	 */
	uint3 _bricksRes;

	/**
	 * Brick size
	 */
	int _borderSize;

	/**
	 * Number of mipmap levels
	 */
	int _numMipMapLevels;

	/**
	 * Mip map order
	 */
	int _mipMapOrder;

	/**
	 * Flag to tell wheter or not a cache mechanismn is required
	 * when reading files (nodes and bricks).
	 * If set, read data is stored buffers.
	 */
	bool _useCache;

	/**
	 * Buffer containing all read nodes data.
	 *
	 * This a 2D array containing, for each mipmap level,
	 * the list of nodes addresses.
	 *
	 * Nodes are stored in increasing order from X axis first, then Y axis, then Z axis.
	 */
	std::vector< unsigned int* > _blockIndexCache;

	/**
	  * Buffer containing all read bricks data
	  *
	  * Bricks are stored in increasing order from X axis first, then Y axis, then Z axis.
	 */
	std::vector< unsigned char* > _blockCache;

	/**
	 * List of all filenames that producer will have to load (nodes and bricks).
	 */
	std::vector< std::string > _filesNames;

	/**
	 * Number of channel in the data structure (color, normal, density, etc...)
	 */
	size_t _numChannels;

	/******************************** METHODS *********************************/

	/**
	 * Retrieve the indexed coordinates of a block (i.e. a node) in the blocks grid
	 * associated to a given position of a region of space at a given level of resolution.
	 *
	 * @param pLevel level of resolution
	 * @param pPosition position of a region of space
	 *
	 * @return the associated indexed coordinates
	 */
	inline uint3 getBlockCoords( int pLevel, const float3& pPosition ) const;

	/**
	 * Retrieve the level of resolution associated to a given size of a region of space.
	 *
	 * @param pSize size of a region of space
	 * @param pResolution resolution of data (i.e. brick)
	 *
	 * @return the corresponding level
	 */
	inline int getDataLevel( const float3& pSize, const uint3& pResolution ) const;

	/**
	 * Retrieve the indexed coordinates associated to a given position of a region of space
	 * at a given level of resolution.
	 *
	 * @param pLevel level of resolution
	 * @param pPosition position of a region of space
	 *
	 * @return the associated indexed coordinates
	 */
	inline uint3 getCoordsInLevel( int pLevel, const float3& pPosition ) const;

	/**
	 * Build the list of all filenames that producer will have to load.
	 * Given a root name, this function create filenames by adding brick resolution,
	 * border size, file extension, etc...
	 *
	 * @param pFilename the root filename from which all filenames will be built
	 */
	void makeFilesNames( const char* pFilename );


	/**
	 * Parses the XML configuration file
	 *
	 * @param pFilename the filename of the XML file
	 */
	int parseXMLFile( const char* pFilename , uint & resolution);


	/**
	 * Retrieve the node encoded address given a mipmap level and a 3D node indexed position
	 *
	 * @param pLevel mipmap level
	 * @param pBlockPos the 3D node indexed position
	 *
	 * @return the node encoded address
	 */
	unsigned int getBlockIndex( int level, const uint3& bpos ) const;

	/**
	 * Load a brick given a mipmap level, a 3D node indexed position,
	 * the data pool and an offset in the data pool.
	 *
	 * @param pLevel mipmap level
	 * @param pBlockPos the 3D node indexed position
	 * @param pDataPool the data pool
	 * @param pOffsetInPool offset in the data pool
	 *
	 * @return a flag telling wheter or not the brick has been loaded (some brick can contain no data).
	 */
	bool loadBrick( int pLevel, const uint3& pBlockPos, GvCore::GPUPoolHost< GvCore::Array3D, TDataTypeList >* pDataPool, size_t pOffsetInPool );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

};

} // namespace GvUtils

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvDataLoader.inl"

#endif
