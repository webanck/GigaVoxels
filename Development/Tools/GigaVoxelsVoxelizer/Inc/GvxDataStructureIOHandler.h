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

#ifndef _GVX_DATA_STRUCTURE_IO_HANDLER_H_
#define _GVX_DATA_STRUCTURE_IO_HANDLER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

#include <string>

// STL
#include <vector>

// Project
#include "GvxDataTypeHandler.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace Gvx
{

/** 
 * GvxDataStructureIOHandler ...
 */
class GvxDataStructureIOHandler
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/
	
	/******************************* ATTRIBUTES *******************************/

	/**
	 * Maximum level of resolution
	 */
	const unsigned int _level;

	/**
	 * Node grid size of the underlying data structure (i.e. octree, N-Tree, etc...).
	 * This is the number of nodes in each dilension.
	 */
	const unsigned int _nodeGridSize;

	/**
	 * Voxel grid size of the underlying data structure (i.e. octree, N-Tree, etc...).
	 * This is the number of voxels in each dimension.
	 */
	const unsigned int _voxelGridSize;

	/**
	 * Brick width.
	 * This is the number of voxels in each dimension of a brick
	 * of the underlying data structure (i.e. octree, N-Tree, etc...).
	 */
	const unsigned int _brickWidth;

	/**
	 * Brick size.
	 * This is the total number of voxels in a brick by taking to account borders.
	 * Currently, there is only a border of one voxel on each side of bricks.
	 */
	const unsigned int _brickSize;	
	
	/******************************** METHODS *********************************/

	/**
     * Constructor
	 *
	 * @param pName Name of the data (.i.e. sponza, dragon, sibenik, etc...)
	 * @param pLevel level of resolution of the data structure
	 * @param pBrickWidth width of bricks in the data structure
	 * @param pDataType type of voxel data (i.e. uchar4, float, float4, etc...)
	 * @param pNewFiles a flag telling wheter or not "new files" are used
	 */
	GvxDataStructureIOHandler( const std::string& pName, 
								unsigned int pLevel,
								unsigned int pBrickWidth,
								GvxDataTypeHandler::VoxelDataType pDataType,
								bool pNewFiles );

	/**
     * Constructor
	 *
	 * @param pName Name of the data (.i.e. sponza, dragon, sibenik, etc...)
	 * @param pLevel level of resolution of the data structure
	 * @param pBrickWidth width of bricks in the data structure
	 * @param pDataTypes types of voxel data (i.e. uchar4, float, float4, etc...)
	 * @param pNewFiles a flag telling wheter or not "new files" are used
	 */
	GvxDataStructureIOHandler( const std::string& pName, 
								unsigned int pLevel,
								unsigned int pBrickWidth,
								const std::vector< GvxDataTypeHandler::VoxelDataType >& pDataTypes,
								bool pNewFiles );

	/**
     * Destructor
	 */
	~GvxDataStructureIOHandler();
	
	/**
	 * Get the node info associated to an indexed node position
	 *
	 * @param pNodePos an indexed node position
	 *
	 * @return node info (address + brick index)
	 */
	unsigned int getNode( unsigned int nodePos[ 3 ] );

	/**
	 * Set data in a voxel at given data channel
	 *
	 * @param pVoxelPos voxel position
	 * @param pVoxelData voxel data
	 * @param pDataChannel data channel index
	 */
	void setVoxel( unsigned int pVoxelPos[ 3 ], void* pVoxelData, unsigned int pDataChannel );

	/**
	 * Set data in a voxel at given data channel
	 *
	 * @param pNormalizedVoxelPos float normalized voxel position
	 * @param pVoxelData voxel data
	 * @param pDataChannel data channel index
	 */
	void setVoxel( float pNormalizedVoxelPos[ 3 ], void* pVoxelData, unsigned int pDataChannel );

	/**
	 * Get data in a voxel at given data channel
	 *
	 * @param pVoxelPos voxel position
	 * @param pVoxelData voxel data
	 * @param pDataChannel data channel index
	 */
	void getVoxel( unsigned int pVoxelPos[ 3 ], void* voxelData, unsigned int pDataChannel );

	/**
	 * Get data in a voxel at given data channel
	 *
	 * @param pNormalizedVoxelPos float normalized voxel position
	 * @param pVoxelData voxel data
	 * @param pDataChannel data channel index
	 */
	void getVoxel( float pNormalizedVoxelPos[ 3 ], void* pVoxelData, unsigned int pDataChannel );

	/**
	 * Get the current brick number
	 *
	 * @return the current brick number
	 */
	unsigned int getBrickNumber() const;

	/**
	 * Get brick data in a node at given data channel
	 *
	 * @param pNodePos node position
	 * @param pBrickData brick data
	 * @param pDataChannel data channel index
	 */
	void getBrick( unsigned int pNodePos[ 3 ], void* pBrickData, unsigned int pDataChannel );

	/**
	 * Set brick data in a node at given data channel
	 *
	 * @param pNodePos node position
	 * @param pBrickData brick data
	 * @param pDataChannel data channel index
	 */
	void setBrick( unsigned int pNodePos[ 3 ], void* pBrickData, unsigned int pDataChannel );

	/**
	 * Get the voxel size at current level of resolution
	 *
	 * @return the voxel size
	 */
	float getVoxelSize() const;

	/** @name Position functions
	 *  Position functions
	 */
	/**@{*/

	/**
	 * Convert a normalized node position to its indexed node position
	 *
	 * @param pNormalizedNodePos normalized node position
	 * @param pNodePos indexed node position
	 */
	void getNodePosition(  float pNormalizedNodePos[ 3 ], unsigned int pNodePos[ 3 ] );

	/**
	 * Convert a normalized voxel position to its indexed voxel position
	 *
	 * @param pNormalizedVoxelPos normalized voxel position
	 * @param pVoxelPos indexed voxel position
	 */
	void getVoxelPosition( float pNormalizedVoxelPos[ 3 ], unsigned int pVoxelPos[ 3 ] );

	/**
	 * Convert a normalized voxel position to its indexed voxel position in its associated brick
	 *
	 * @param pNormalizedVoxelPos normalized voxel position
	 * @param pVoxelPosInBrick indexed voxel position in its associated brick
	 */
	void getVoxelPositionInBrick( float pNormalizedVoxelPos[ 3 ], unsigned int pVoxelPosInBrick[ 3 ] );
	
	/**@}*/

	/**
	 * Fill all brick borders of the data strucuture with data.
	 */
	void computeBorders();

	/**
	 * Tell wheter or not a node is empty given its node info.
	 *
	 * @param pNode a node info
	 *
	 * @return a flag telling wheter or not a node is empty
	 */
	static bool isEmpty( unsigned int pNode );
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

	
	/**
	 * Retrieve node info and brick data associated to a node position.
	 * Data is retrieved from disk if not already in cache, otherwise exit.
	 *
	 * Data is stored in buffers if not yet in cache.
	 *
	 * Note : Data is written on disk a previous node have been processed
	 * and a new one is requested.
	 *
	 * @param pNodePos node position
	 */
	void loadNodeandBrick( unsigned int pNodePos[ 3 ] );


protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
	
	/** @name Files
	 *  Files
	 */
	/**@{*/

	/**
	 * Node file
	 */
	FILE* _nodeFile;

	/**
	 * List of brick files
	 */
	std::vector< FILE* > _brickFiles;

	/**
	 * Node filename
	 */
	std::string _fileNameNode;

	/**
	 * List of brick filenames
	 */
	std::vector< std::string > _fileNamesBrick;

	/**
	 * List of data types associated to the data structure channels (unsigned char, unsigned char4, float, float4, etc...)
	 */
	std::vector< GvxDataTypeHandler::VoxelDataType > _dataTypes;

	/**@}*/

	/**
	 * Real bricks number (empty regions contain no brick)
	 */
	unsigned int _brickNumber;

	// current brick and node

	/**
	 * Flag to tell wheter or not the current node and brick have been loaded in memory (and stored in buffers)
	 */
	bool _isBufferLoaded;

	/**
	 * Buffer of node position.
	 * It corresponds to the current indexed node position.
	 */
	unsigned int _nodeBufferPos[ 3 ];

	/**
	 * Buffer of node info associated to current node position.
	 * It corresponds to the childAddress of an GvStructure::OctreeNode.
	 * If node is not empty, the asssociated brick index is also stored inside.
	 */
	unsigned int _nodeBuffer;

	/**
	 * Brick data buffer associated to current nodeBuffer.
	 * This is where all data reside for each channel (color, normal, etc...)
	 */
	std::vector< void* > _brickBuffers;

	/**
	 * Empty node flag
	 */
	static const unsigned int _cEmptyNodeFlag;

	/******************************** METHODS *********************************/	

	/**
	 * Save node info and brick data associated to current node position on disk.
	 */
	void saveNodeandBrick();

	/**
	 * Initialize all the files that will be generated.
	 *
	 * Note : Associated brick buffer(s) will be created/initialized.
	 *
	 * @param pName name of the data (i.e. sponza, sibenik, dragon, etc...)
	 * @param pNewFiles a flag telling wheter or not new files are used
	 */
	void openFiles( const std::string& name, bool newFiles );

	/**
	 * Retrieve the node file name.
	 * An example of GigaVoxels node file could be : "fux_BR8_B1_L0.nodes"
	 * where "fux" is the name, "8" is the brick width BR, "1" is the brick border size B,
	 * "0" is the level of resolution L and "nodes" is the file extension.
	 *
	 * @param pName name of the data file
	 * @param pLevel data structure level of resolution
	 * @param pBrickWidth width of bricks
	 *
	 * @return the node file name in GigaVoxels format.
	 */
	static std::string getFileNameNode( const std::string& pName, unsigned int pLevel, unsigned int pBrickWidth );

	/**
	 * Retrieve the brick file name.
	 * An example of GigaVoxels brick file could be : "fux_BR8_B1_L0_C0_uchar4.bricks"
	 * where "fux" is the name, "8" is the brick width BR, "1" is the brick border size B,
	 * "0" is the level of resolution L, "0" is the data channel index C,
	 * "uchar4" the data type name and "bricks" is the file extension.
	 *
	 * @param pName name of the data file
	 * @param pLevel data structure level of resolution
	 * @param pBrickWidth width of bricks
	 * @param pDataChannelIndex data channel index
	 * @param pDataTypeName data type name
	 *
	 * @return the brick file name in GigaVoxels format.
	 */
	static std::string getFileNameBrick( const std::string& pName, unsigned int pLevel, unsigned int pBrickWidth, unsigned int pDataChannelIndex, const std::string& pDataTypeName );

	/**
	 * Create a brick node info (address + brick index)
	 *
	 * @param pBrickNumber a brick index
	 *
	 * @return a brick node info
	 */
	static unsigned int createBrickNode( unsigned int pBrickNumber );

	/**
	 * Retrieve the brick offset of a brick given a node info.
	 *
	 * @param pNode a node info
	 *
	 * @return the brick offset
	 */
	static unsigned int getBrickOffset( unsigned int pNode );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvxDataStructureIOHandler( const GvxDataStructureIOHandler& );

	/**
	 * Copy operator forbidden.
	 */
	GvxDataStructureIOHandler& operator=( const GvxDataStructureIOHandler& );

};

}

#endif
