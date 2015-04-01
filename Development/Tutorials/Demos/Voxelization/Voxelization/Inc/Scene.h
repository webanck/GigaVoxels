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

/**
 * ...
 */
struct node
{
	/**
	 * ...
	 */
	unsigned int first;

	/**
	 * ...
	 */
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
	void init( char* pSceneFile );

	/**
	 * Draw the scene
	 */
	void draw() const;

	/**
	 * Draw the scene contained in a specific region of space
	 *
	 * @param depth depth (i.e. level of resolution)
	 */
	void draw( unsigned int depth ) const;

	/**
	 * Draw the scene contained in a specific region of space
	 *
	 * @param depth depth (i.e. level of resolution)
	 * @param locCode localization code
	 */
	void draw( unsigned int depth, const uint3& locCode ) const;

	/**
	 * ...
	 */
	uint intersectMesh( unsigned int depth, const uint3& locCode ) const;

	/**
	 * Initialize a node in the node buffer
	 *
	 * @param pDepth node's depth localization info
	 * @param pCode node's code localization info
	 * @param pFirst start index of node in node buffer
	 * @param pCount number of primitives in node (i.e triangles)
	 */
	void setOctreeNode( unsigned int pDepth, const uint3& pCode, unsigned int pFirst, int pCount );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	const aiScene* _scene;

	/**
	 * ...
	 */
	GLuint mVAO[ 1 ]; 

	/**
	 * ...
	 */
	GLuint mBuffers[ 3 ];

	/**
	 * ...
	 */
	unsigned int mNbTriangle;

	/**
	 * ...
	 */
	node* mOctree;

	/**
	 * Octree max depth
	 */
	unsigned int mDepthMax;

	/**
	 * ...
	 */
	unsigned int mDepthMaxPrecomputed;

	/**
	 * ...
	 */
	unsigned int mIBOLengthMax;

	/******************************** METHODS *********************************/
	
	/**
	 * ...
	 */
	void organizeIBO( std::vector< unsigned int >& IBO, const float* vertices );
	
	/**
	 * ...
	 */
	bool organizeNode( unsigned int d, const uint3& locCode, std::vector< unsigned int >& IBO, const float* vertices );

	/**
	 * ...
	 */
	inline uint3 getFather( const uint3& locCode ) const;

	/**
	 * Compute global index of a node in the node buffer given its depth and code localization info
	 *
	 * TODO : use generic code => only valid for octree...
	 *
	 * @param pDepth node's depth localization info
	 * @param pCode node's code localization info
	 *
	 * return node's global index in the node buffer
	 */
	inline unsigned int getIndex( unsigned int pDepth, const uint3& pCode ) const;
	
	/**
	 * ...
	 */
	inline bool triangleIntersectBick( const float3& brickPos, 
								const float3& brickSize, 
								unsigned int triangleIndex, 
								const std::vector< unsigned int >& IBO, 
								const float* vertices );
	/**
	 * ...
	 */
	inline bool vertexIsInBrick( const float3& brickPos, 
								const float3& brickSize, 
								unsigned int vertexIndex,
								const float* vertices );

	/**
	 * ...
	 */
	void organizeIBOGlsl();

	/**
	 * ...
	 */
	bool organizeNodeGlsl( unsigned int d, const uint3& locCode, GLuint TBO, GLuint triangleCounter );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "Scene.inl"

#endif // !_SCENE_H_
