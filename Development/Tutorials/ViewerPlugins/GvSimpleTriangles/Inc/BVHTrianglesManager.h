/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2013 INRIA <http://www.inria.fr/>
 *
 * Authors : GigaVoxels Team
 *
 * GigaVoxels is distributed under a dual-license scheme.
 * You can obtain a specific license from Inria at gigavoxels-licensing@inria.fr.
 * Otherwise the default license is the GPL version 3.
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

#ifndef _BVH_TRIANGLES_MANAGER_H_
#define _BVH_TRIANGLES_MANAGER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/Array3D.h>
#include <GvCore/GPUPool.h>
#include <GvCore/vector_types_ext.h>

// Project
#include "GPUTreeBVHCommon.h"
#include "BVHTriangles.hcu"

// STL
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>

// Assimp
#include <assimp/scene.h>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>

#include <sys/types.h>
#include <sys/stat.h>
///#include <unistd.h>

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

using namespace GvCore; // FIXME

/******************************************************************************
 * ...
 ******************************************************************************/
template< class T >
void writeStdVector( const std::string& fileName, std::vector< T >& vec )
{
	std::ofstream dataOut( fileName.c_str(), std::ios::out | std::ios::binary );

	dataOut.write( (const char *)&( vec[ 0 ] ), vec.size() * sizeof( T ) );

	dataOut.close();
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< class T >
void readStdVector( const std::string& fileName, std::vector< T >& vec )
{
	struct stat results;
	if ( stat( fileName.c_str(), &results ) == 0 )
	{
		// results.st_size

		uint numElem = results.st_size / sizeof( T );

		vec.resize( numElem );

		std::ifstream dataIn( fileName.c_str(), std::ios::in | std::ios::binary );
		
		dataIn.read( (char *)&( vec[ 0 ] ), vec.size() * sizeof( T ) );
		
		dataIn.close();
	}
	else
	{
		std::cout << "Unable to open file: " << fileName << "\n";
	}
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class BVHTrianglesManager
 *
 * @brief The BVHTrianglesManager class provides ...
 *
 * The BVHTrianglesManager class ...
 *
 * @param TDataTypeList data type list choosen by user (ex : (float4, uchar4) for position and color )
 * @param DataPageSize data page size in vertices (size is given by BVH_DATA_PAGE_SIZE)
 */
template< class TDataTypeList, uint DataPageSize >
class BVHTrianglesManager 
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	//typedef unsigned long long PosHasType;
	typedef uint PosHasType;

	/**
	 * Type definition of a custom data pool (array mapped on GPU)
	 */
	typedef GvCore::GPUPoolHost< GvCore::Array3D, TDataTypeList > DataBufferType;

	// Advanced

	/** 
	 * @struct BBoxInfo
	 *
	 * @brief The BBoxInfo struct provides ...
	 *
	 * The BBoxInfo struct ...
	 */
	struct BBoxInfo
	{

		/******************************* ATTRIBUTES *******************************/

		/**
		 * Object ID
		 */
		uint objectID;

		/**
		 * ...
		 */
		PosHasType posHash;

		/**
		 * ...
		 */
		//uchar surfaceHash;
		
		/******************************** METHODS *********************************/

		/**
		 * Constructor
		 */
		BBoxInfo( uint oid, AABB& bbox/*, float2 minMaxSurf*/ );
		
		/**
		 * Constructor
		 */
		BBoxInfo( uint oid, float3 v0, float3 v1, float3 v2 );
		
		/**
		 * ...
		 */
		bool operator<( const BBoxInfo& b ) const;

	};	// end of struct BBoxInfo

	/******************************* ATTRIBUTES *******************************/

	///// MESH /////

	/** 
	 * @struct Triangle
	 *
	 * @brief The Triangle struct provides ...
	 *
	 * The Triangle struct ...
	 */
	struct Triangle
	{
		uint vertexID[ 3 ];
	};

	/**
	 * Buffer of vertex positions
	 */
	std::vector< float3 > meshVertexPositionList;
	
	/**
	 * Buffer of vertex colors
	 */
	std::vector< float4 > meshVertexColorList;

	/**
	 * Buffer of triangles
	 */
	std::vector< Triangle > meshTriangleList;

	// If field added, think of updating splitting and buffer filling.
	
	/**
	 * Node buffer
	 */
	GvCore::Array3D< VolTreeBVHNode >* _nodesBuffer;
	
	/**
	 * Data pool
	 */
	DataBufferType* _dataBuffer;
	
	/**
	 * ...
	 */
	uint numDataNodesCounter;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	BVHTrianglesManager();

	/**
	 * Destructor
	 */
	~BVHTrianglesManager();
	
	/**
	 * Helper function to stringify a value
	 */
	inline std::string stringify( int x );

	/**
	 * Iterate through triangles and split them if required (depending on a size criteria)
	 *
	 * @param criticalEdgeLength max "triangle edge length" criteria beyond which a split must occur
	 *
	 * @return flag to tell wheter or not a split has happend
	 */
	bool splitTrianges( float criticalEdgeLength );

	/**
	 * ...
	 */
	std::string getBaseFileName( const std::string& fileName );

	/**
	 * ...
	 */
	void loadPowerPlant( const std::string& baseFileName );

	/**
	 * ...
	 */
	void loadMesh( const std::string& meshFileName );

	/**
	 * ...
	 */
	void saveRawMesh( const std::string& fileName );

	/**
	 * ...
	 */
	void loadRawMesh( const std::string& fileName );

	/**
	 * Generate buffers
	 */
	void generateBuffers( uint arrayFlag );

	// TODO: case where data is linked
	/**
	 * Fill nodes buffer
	 */
	VolTreeBVHNode fillNodesBuffer( std::vector< VolTreeBVHNode >& bvhNodesList, std::vector< BBoxInfo >& bboxInfoList,
									int level, uint2 curInterval, uint& nextNodeBufferOffset,
									std::vector< VolTreeBVHNode >& bvhNodesBufferBuildingList );

	/**
	 * ...
	 */
	void recursiveAddEscapeIdx( uint nodeAddress, uint escapeTo );

	/////////////

	/**
	 * ...
	 */
	void loadPowerPlantDirectoryStructure( const std::string& baseFileName );

	/**
	 * ...
	 */
	void addMeshFile( const std::string& meshFileName );

	/**
	 * ...
	 */
	void renderGL();

	/**
	 * ...
	 */
	void renderDebugGL();

	/**
	 * ...
	 */
	void recursiveRenderDebugGL( uint curNodeIdx );

	/**
	 * ...
	 */
	void displayTriangles( uint index );

	/**
	 * ...
	 */
	void renderFullGL();

	/**
	 * ...
	 */
	void recursiveRenderFullGL( uint curNodeIdx );

	/**
	 * Get the node pool
	 */
	GvCore::Array3D< VolTreeBVHNode >* getNodesBuffer();

	/**
	 * Get the data pool
	 */
	DataBufferType* getDataBuffer();

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	BVHTrianglesManager( const BVHTrianglesManager& );

	/**
	 * Copy operator forbidden.
	 */
	BVHTrianglesManager& operator=( const BVHTrianglesManager& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "BVHTrianglesManager.inl"

#endif
