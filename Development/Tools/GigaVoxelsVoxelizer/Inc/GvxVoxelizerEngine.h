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

#ifndef _GVX_VOXELIZER_ENGINE_H_
#define _GVX_VOXELIZER_ENGINE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "GvxDataStructureIOHandler.h"
#include "GvxDataTypeHandler.h"

// STL
#include <vector>

// CImg
#define cimg_use_magick	// Beware, this definition must be placed before including CImg.h
#include <CImg.h>

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
 * @class GvxVoxelizerEngine
 *
 * @brief The GvxVoxelizerEngine class provides a client interface to voxelize a mesh.
 *
 * It is the core class that, given data at triangle (vertices, normals, textures, etc...),
 * generate voxel data in the GigaVoxels data structure (i.e. octree).
 */
class GvxVoxelizerEngine
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Flag to tell wheter or not to handle textures
	 */
	bool _useTexture;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvxVoxelizerEngine();

	/**
	 * Destructor
	 */
	~GvxVoxelizerEngine();

	/**
	 * Initialize the voxelizer
	 *
	 * Call before voxelization
	 *
	 * @param pLevel Max level of resolution
	 * @param pBrickWidth Width a brick
	 * @param pName Filename to be processed
	 * @param pDataType Data type that will be processed
	 */
	void init( int pLevel, int pBrickWidth, const std::string& pName, GvxDataTypeHandler::VoxelDataType pDataType );
	
	/**
	 * Finalize the voxelizer
	 *
	 * Call after voxelization
	 */
	void end();

	/**
	  * Voxelize a triangle.
	 *
	 * Given vertex attributes previously set for a triangle (positions, normals,
	 * colors and texture coordinates), it voxelizes triangle (by writing data). 
	 */
	void voxelizeTriangle();

	/**
	 * Store a 3D position in the vertex buffer.
	 * During voxelization, each triangle attribute is stored.
	 * Due to kind of circular buffer technique, calling setVertex() method on each vertex
	 * of a triangle, register each position internally.
	 *
	 * @param pX x coordinate
	 * @param pY y coordinate
	 * @param pZ z coordinate
	 */
	void setVertex( float pX, float pY, float pZ );

	/**
	 * Store a nomal in the buffer of normals.
	 * During voxelization, each triangle attribute is stored.
	 * Due to kind of circular buffer technique, calling setNormal() method on each vertex
	 * of a triangle, register each normal internally.
	 *
	 * @param pX x normal component
	 * @param pY y normal component
	 * @param pZ z normal component
	 */
	void setNormal( float pX, float pY, float pZ );

	/**
	 * Store a color in the color buffer.
	 * During voxelization, each triangle attribute is stored.
	 * Due to kind of circular buffer technique, calling setColor() method on each vertex
	 * of a triangle, register each color internally.
	 *
	 * @param pR red color component
	 * @param pG green color component
	 * @param pB blue color component
	 */
	void setColor( float pR, float pG, float pB );

	/**
	 * Store a texture coordinates in the texture coordinates buffer.
	 * During voxelization, each triangle attribute is stored.
	 * Due to kind of circular buffer technique, calling setTexCoord() method on each vertex
	 * of a triangle, register each texture coordinate internally.
	 *
	 * @param pR r texture coordinate
	 * @param pS s texture coordinate
	 */
	void setTexCoord( float pR, float pS );

	/**
	 * Construct image from reading an image file.
	 *
	 * @param pFilename the image filename
	 */
	void setTexture( const std::string& pFilename );

	/**
	 * Set The number of times we apply the filter
	 */
	void setNbFilterApplications(int pValue);

	/**
	 * Set the type of the filter 
	 * 0 = mean
	 * 1 = gaussian
	 * 2 = laplacian
	 */
	void setFilterType (int pValue);

	/**
	 * Set the _normals value
	 */
	void setNormals( bool value);
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/
	
	/**
	 * List of types that will be used during voxelization (i.e. ucahr4, float, float4, etc...)
	 */
	std::vector< GvxDataTypeHandler::VoxelDataType > _dataTypes;

	/**
	 * File/stream handler.
	 * It ios used to read and/ or write to GigaVoxels files (internal format).
	 */
	GvxDataStructureIOHandler* _dataStructureIOHandler;
	
	/**
	 * Filename to be processed.
	 * Currently, this is just a name like "sponza", not a real path+filename.
	 */
	std::string _fileName;
	
	/**
	 * Brick width
	 */
	int _brickWidth;
	
	/**
	 * Max level of resolution
	 */
	int _level;

	/**
	 * The number of times we apply the filter
	 */
	int _nbFilterApplications;

	/**
	 * the type of the filter 
	 * 0 = mean
	 * 1 = gaussian
	 * 2 = laplacian
	 */
	int _filterType;

	/**
	 * bool that says whether or not we produce normals
	 */
	bool _normals;


	// primitives

	/**
	 * Vertices of a triangle
	 */
	float _v1[ 3 ];
	float _v2[ 3 ];
	float _v3[ 3 ];

	/**
	 * Normals of a triangle
	 */
	float _n1[ 3 ];
	float _n2[ 3 ];
	float _n3[ 3 ];

	/**
	 * Colors of a triangle
	 */
	float _c1[ 3 ];
	float _c2[ 3 ];
	float _c3[ 3 ];

	/**
	 * Texture coordinates of a triangle
	 */
	float _t1[ 2 ];
	float _t2[ 2 ];
	float _t3[ 2 ];

	/**
	 * Class representing an image (up to 4 dimensions wide), each pixel being of type T, i.e. "float".
	 * This is the main class of the CImg Library.
	 * It declares and constructs an image, allows access to its pixel values, and is able to perform various image operations.
	 */
	cimg_library::CImg< float > _texture;

	/******************************** METHODS *********************************/

	/**
	 * Apply the update borders algorithmn.
	 * Fill borders with data.
	 */
	void updateBorders();

	/**
	 * Apply the normalize algorithmn
	 */
	void normalize();

	/**
	 * Apply the filtering algorithm
	 */
	void applyFilter();

	/**
	 * Apply the mip-mapping algorithmn.
	 * Given a pre-filtered voxel scene at a given level of resolution,
	 * it generates a mip-map pyramid hierarchy of coarser levels (until 0).
	 */
	void mipmap();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvxVoxelizerEngine( const GvxVoxelizerEngine& );

	/**
	 * Copy operator forbidden.
	 */
	GvxVoxelizerEngine& operator=( const GvxVoxelizerEngine& );

};

}

#endif
