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

#ifndef _SCENE_H_
#define _SCENE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GL
#include <GL/glew.h>

// GigaVoxels
#include <GvCore/vector_types_ext.h>

// Assimp
#include <assimp/scene.h>

// STL
#include <vector>
#include <list>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

struct node {
	unsigned int first;
	int count;
};

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class Scene
 *
 * @brief The Scene class provides helper class allowing to extract a scene from a file 
 * and drawing it with openGL
 *
 * 
 */
class Scene
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	Scene( unsigned int pMaxDepth = 5 );

	/**
	 * Destructor
	 */
	~Scene();

	/**
	 * ...
	 */
	bool init( const char* pSceneFile );

	/**
	 * ...
	 */
	void draw() const;

	/**
	 * ...
	 */
	void draw( unsigned int depth, uint3 locCode ) const;
	
	/**
	 * ...
	 */
	void draw( unsigned int depth ) const;

	/**
	 * ...
	 */
	uint intersectMesh( unsigned int depth, uint3 locCode ) const;

	/**
	 * ...
	 */
	void setOctreeNode( unsigned int depth, uint3 locCode, unsigned int pFirst, int pCount );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Initialize graphics resources
	 *
	 * @return a flag telling wheter or not process has succeeded
	 */
	bool initializeGraphicsResources();

	/**
	 * Finalize graphics resources
	 *
	 * @return a flag telling wheter or not process has succeeded
	 */
	bool finalizeGraphicsResources();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * 3D scene
	 */
	const aiScene* _scene;

	/**
	 * Vertex array object
	 */
	GLuint _vao; 

	/**
	 * Vertex buffer objects
	 * - vertices, normals and indexes
	 */
	GLuint _buffers[ 3 ];

	unsigned int mNbTriangle;

	node* _octree;
	
	// Octree max depth
	unsigned int mDepthMax;

	unsigned int _depthMaxPrecomputed;

	/******************************** METHODS *********************************/
	
	/**
	 * ...
	 */
	void organizeIBO( std::vector<unsigned int> & IBO, const float *vertices );
	
	/**
	 * ...
	 */
	bool organizeNode( unsigned int d, uint3 locCode, std::vector<unsigned int> & IBO, const float *vertices );

	/**
	 * ...
	 */
	inline uint3 getFather( uint3 locCode ) const;

	/**
	 * Get the index of the brick in the octree
	 */
	inline unsigned int getIndex( unsigned int d, uint3 locCode ) const;
	
	/**
	 * ...
	 */
	inline bool triangleIntersectBick( const float3 & brickPos, 
								const float3 & brickSize, 
								unsigned int triangleIndex, 
								const std::vector<unsigned int> & IBO, 
								const float *vertices );
	/**
	* ...
	*/
	inline bool triangleAabbIntersection2D( const float2 & a, 
										    const float2 & b, 
											const float2 & c,
											const float4 & aabb );

};


/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "Scene.inl"

#endif // !_SCENE_H_
