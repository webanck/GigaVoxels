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

#include "Scene.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

#include <iostream>

// System
#include <cfloat>
#include <limits>
#include <cassert>

// Assimp
#include <assimp/cimport.h>
#include <assimp/postprocess.h>

//Debugue
#include <iostream>

//Shader
#include <GvUtils/GvShaderManager.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * ...
 */
#define BUFFER_OFFSET(i) ((void*)(i))

/**
 * Assimp library object to load 3D model (with a log mechanism)
 */
static aiLogStream stream;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
Scene::Scene( unsigned int pMaxDepth )
:   _scene( NULL )
,	mNbTriangle( 0 )
,	mDepthMaxPrecomputed( 4 )
,	mDepthMax( pMaxDepth )
{
	// Compute max length of node buffer (and VBO)
	// - elements are stored LOD by LOD
	// - LOD 0 ( 1 node ) - LOD 1 ( 8 nodes ) - LOD 2 ( 64 nodes ) - LOD 3 ( 512 nodes ) - LOD 4 ( 4096 nodes ) - LOD 5 ...
	// lenght_octree = Somme( 8^i, i = 0:mDepthMax ) + 8 
	const unsigned int length = ( powf( 8, mDepthMax + 1 ) - 1 ) / (float)7 ;
	mOctree = new node[ length ];

	// Initialize node buffer
	for ( unsigned int i = 0; i < length; i++ )
	{
		mOctree[ i ].first = 0;
		mOctree[ i ].count = -1;
	}
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
Scene::~Scene()
{	
		glDeleteBuffers( 3, &mBuffers[ 0 ] );
		glDeleteVertexArrays( 1, &mVAO[ 0 ] );
		delete[] mOctree;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Scene::init( char* pSceneFile )
{
	assert( pSceneFile != NULL );

	// Import the geometry
	stream = aiGetPredefinedLogStream( aiDefaultLogStream_STDOUT, NULL );
	aiAttachLogStream( &stream );
	_scene = aiImportFile( pSceneFile, 0 );	// aiProcessPreset_TargetRealtime_Fast );

	// Scale the geometry
	float minx = +std::numeric_limits< float >::max();//FLT_MAX;
	float miny = +std::numeric_limits< float >::max();//FLT_MAX;
	float minz = +std::numeric_limits< float >::max();//FLT_MAX;
	float maxx = -std::numeric_limits< float >::max();//-FLT_MAX;
	float maxy = -std::numeric_limits< float >::max();//-FLT_MAX;
	float maxz = -std::numeric_limits< float >::max();//-FLT_MAX;

	// Iterate through meshes to compute global bounding box
	for ( unsigned int meshIndex = 0; meshIndex < _scene->mNumMeshes; ++meshIndex )
	{
		// Get current mesh
		const aiMesh* pMesh = _scene->mMeshes[ meshIndex ];

		// Iterate through mesh vertices
		for ( unsigned int vertexIndex = 0; vertexIndex < pMesh->mNumVertices; ++vertexIndex )
		{
			minx = std::min( minx, pMesh->mVertices[ vertexIndex ].x );
			miny = std::min( miny, pMesh->mVertices[ vertexIndex ].y );
			minz = std::min( minz, pMesh->mVertices[ vertexIndex ].z );
			maxx = std::max( maxx, pMesh->mVertices[ vertexIndex ].x );
			maxy = std::max( maxy, pMesh->mVertices[ vertexIndex ].y );
			maxz = std::max( maxz, pMesh->mVertices[ vertexIndex ].z );
		}
	}

	// Set the scale to apply on the mesh to be between [ 0.0 ; 1.0 ]
	float scale = 0.95f / std::max( std::max( maxx - minx, maxy - miny ), maxz - minz );

	// Iterate through meshes to scale them
	for ( unsigned int meshIndex = 0; meshIndex < _scene->mNumMeshes; ++meshIndex )
	{
		// Get current mesh
		const aiMesh* pMesh = _scene->mMeshes[ meshIndex ];

		// Iterate through mesh vertices
		for ( unsigned int vertexIndex = 0; vertexIndex < pMesh->mNumVertices; ++vertexIndex )
		{
			pMesh->mVertices[ vertexIndex ].x = ( pMesh->mVertices[ vertexIndex ].x - ( maxx + minx ) * 0.5f ) * scale + 0.5f;
			pMesh->mVertices[ vertexIndex ].y = ( pMesh->mVertices[ vertexIndex ].y - ( maxy + miny ) * 0.5f ) * scale + 0.5f;
			pMesh->mVertices[ vertexIndex ].z = ( pMesh->mVertices[ vertexIndex ].z - ( maxz + minz ) * 0.5f ) * scale + 0.5f;
		}
	}

	// WARNING : we assume here that faces of the mesh are triangles. Plus we don't take care of scene tree structure...

	// Computing number of vertices and triangles:
	unsigned int nbVertices = 0;
	mNbTriangle = 0;

	// Iterate through meshes to scale them
	for ( unsigned int meshIndex = 0; meshIndex < _scene->mNumMeshes; ++meshIndex )
	{
		nbVertices += _scene->mMeshes[ meshIndex ]->mNumVertices;
		mNbTriangle += _scene->mMeshes[ meshIndex ]->mNumFaces;
	}

	float* vertices = new float[ 3 * nbVertices ];
	float* normals = new float[ 3 * nbVertices ]();
	unsigned int* count = new unsigned int[ nbVertices ](); // To count the normals, to average

	std::vector< unsigned int > IBO = std::vector< unsigned int >();
	// Reserve place to store the IBO
	// We assume that a triangle can intersect 3 brick in the worst case
	IBO.reserve( 3 * ( 1 + mDepthMaxPrecomputed * 3 ) * mNbTriangle );		// TODO, if too big use several index buffers ?
	// Resize vector to store the depth lvl 0 
	IBO.resize( 3 * mNbTriangle );

	unsigned int offsetIBO = 0;
	unsigned int offsetVBO = 0;

	// First pass to fill IBO's depth lvl 0, vertices, and normal.
	for ( unsigned int meshIndex = 0; meshIndex < _scene->mNumMeshes; ++meshIndex )
	{
		const aiMesh* pMesh = _scene->mMeshes[ meshIndex ];

		for ( unsigned int faceIndex = 0; faceIndex < pMesh->mNumFaces; ++faceIndex )
		{
			const aiFace* pFace = &pMesh->mFaces[ faceIndex ];

			// Remark : we can compute different normal for same vertex, but new one overwrites the old one
			for ( unsigned int vertIndex = 0; vertIndex < pFace->mNumIndices; ++vertIndex )
			{
				unsigned int index = pFace->mIndices[ vertIndex ];

				float normal[ 3 ];

				if ( ! pMesh->HasNormals() )
				{
					// We compute normal with cross product :

					// retrieve vertex index of the face
					int a = pFace->mIndices[0];
					int b = pFace->mIndices[1];
					int c = pFace->mIndices[2];

					float e1[3] = { pMesh->mVertices[b].x - pMesh->mVertices[a].x,
						pMesh->mVertices[b].y - pMesh->mVertices[a].y,
						pMesh->mVertices[b].z - pMesh->mVertices[a].z };

					float e2[3] = { pMesh->mVertices[c].x - pMesh->mVertices[a].x,
						pMesh->mVertices[c].y - pMesh->mVertices[a].y,
						pMesh->mVertices[c].z - pMesh->mVertices[a].z };

					normals[offsetVBO + 3*index + 0] += e1[1]*e2[2] - e1[2]*e2[1];
					normals[offsetVBO + 3*index + 1] += e1[2]*e2[0] - e1[0]*e2[2];
					normals[offsetVBO + 3*index + 2] += e1[0]*e2[1] - e1[1]*e2[0];

				}
				else
				{
					normals[ offsetVBO + 3*index + 0 ] += pMesh->mNormals[ index ].x;
					normals[ offsetVBO + 3*index + 1 ] += pMesh->mNormals[ index ].y;
					normals[ offsetVBO + 3*index + 2 ] += pMesh->mNormals[ index ].z;
				}
				// To average normals
				count[ index ]++;

				// Fill position buffer
				vertices[ offsetVBO + 3*index + 0 ] = pMesh->mVertices[ index ].x;
				vertices[ offsetVBO + 3*index + 1 ] = pMesh->mVertices[ index ].y;
				vertices[ offsetVBO + 3*index + 2 ] = pMesh->mVertices[ index ].z;

				// Fill index buffer
				IBO[ offsetIBO + vertIndex ] = index;
			}
			offsetIBO += 3;
		}
		offsetVBO += _scene->mMeshes[ meshIndex ]->mNumVertices ;
	}

	// We average and normalize the sum of normals
	for ( unsigned int i = 0; i < nbVertices; i++ )
	{
		// Average 
		normals[ 3 *i  + 0 ] /= count[ i ];
		normals[ 3 * i + 1 ] /= count[ i ];
		normals[ 3 * i + 2 ] /= count[ i ];

		// Normalizing the normal 
		float normal = sqrt ( normals[ 3*i + 0 ]*normals[ 3*i + 0 ] + 
			normals[ 3*i + 1 ]*normals[ 3*i + 1 ] +
			normals[ 3*i + 2 ]*normals[ 3*i + 2 ] );
		normals[ 3 * i + 0 ] /= normal;
		normals[ 3 * i + 1 ] /= normal;
		normals[ 3 * i + 2 ] /= normal;
	}

	// Organize the octree 
	organizeIBO( IBO, vertices );

	// Create VAO (vertex array object)
	glGenVertexArrays( 1, mVAO );

	// Bind VAO (vertex array object)
	glBindVertexArray( mVAO[ 0 ] );

	// Create buffers for vertex data
	glGenBuffers( 4, mBuffers );

	// Vertex position buffer
	glBindBuffer( GL_ARRAY_BUFFER, mBuffers[ 0 ] );
	glBufferData( GL_ARRAY_BUFFER, sizeof( float ) * 3 * nbVertices, vertices, GL_STATIC_DRAW );
	glEnableVertexAttribArray( (GLuint)0 );
	glVertexAttribPointer( (GLuint)0/*attribute index*/, 3/*nb components per vertex*/, GL_FLOAT/*type*/, GL_FALSE/*un-normalized*/, 0/*memory stride*/, static_cast< GLubyte* >( NULL )/*byte offset from buffer*/ );

	// Vertex normal buffer
	glBindBuffer( GL_ARRAY_BUFFER, mBuffers[ 1 ] );
	glBufferData( GL_ARRAY_BUFFER, sizeof( float ) * 3 * nbVertices, normals, GL_STATIC_DRAW );
	glEnableVertexAttribArray( (GLuint)1 );
	glVertexAttribPointer( (GLuint)1/*attribute index*/, 3/*nb components per vertex*/, GL_FLOAT/*type*/, GL_FALSE/*un-normalized*/, 0/*memory stride*/, static_cast< GLubyte* >( NULL )/*byte offset from buffer*/ );

	// Index atribute buffer : to enable organizeIBOGlsl
	/*unsigned int *index = new unsigned int[ mNbTriangle ];
	for ( int i= 0; i< mNbTriangle; i ++ ) {
	index[i] = i;
	}
	glBindBuffer(GL_ARRAY_BUFFER, mBuffers[3]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(unsigned int) * mNbTriangle, index, GL_STATIC_DRAW);
	glEnableVertexAttribArray( (GLuint)2 );
	glVertexAttribPointer( (GLuint)2 , 1, GL_UNSIGNED_INT, 0, 0, 0);*/

	//index buffer
	// We resize the IBO at its maximal size
	//IBO.resize( max( IBO.size(), 3 * ( 1 + mDepthMaxPrecomputed * 3 ) * mNbTriangle ) );
	mIBOLengthMax = IBO.size();
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, mBuffers[ 2 ] );
	glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( unsigned int ) * IBO.size(), IBO.data(), GL_DYNAMIC_DRAW );

	// Unbind VAO (vertex array object)
	glBindVertexArray( 0 );

	// Orginize octree
	//organizeIBOGlsl();

	// deleting tab used 
	delete[] vertices;
	delete[] normals;
	//delete[] index;
}

/******************************************************************************
 * Draw the scene
 ******************************************************************************/
void Scene::draw() const
{
	// Render VAO
	glBindVertexArray( mVAO[ 0 ] );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, mBuffers[ 2 ] );
	
	glDrawElements( GL_TRIANGLES, mNbTriangle * 3, GL_UNSIGNED_INT, 0 );
	
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
	glBindVertexArray( 0 );
}

/******************************************************************************
 * Draw the scene contained in a specific region of space
 *
 * @param depth depth (i.e. level of resolution)
 ******************************************************************************/
void Scene::draw( unsigned int depth ) const
{
	int minNodeIndex = ( powf( 8, depth ) - 1 ) / (float)7;
	int maxNodeIndex = ( powf( 8, depth + 1 ) - 1 ) / (float)7;
	for ( int i = minNodeIndex; i < maxNodeIndex; i++ )
	{
		if ( mOctree[ i ].count != 0 )
		{
			// Render VAO
			glBindVertexArray( mVAO[ 0 ] );
			glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, mBuffers[ 2 ] );
			
			glDrawElements( GL_TRIANGLES,  
				mOctree[ i ].count, 
				GL_UNSIGNED_INT,
				(void*)( sizeof( unsigned int ) * mOctree[ i ].first ) );

			glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
			glBindVertexArray( 0 );
		}
	}
}

/******************************************************************************
 * Draw the scene contained in a specific region of space
 *
 * @param pDepth depth (i.e. level of resolution)
 * @param pCode localization code
 ******************************************************************************/
void Scene::draw( unsigned int pDepth, const uint3& pCode ) const
{
	// Compute index of the node in the octree
	unsigned int i = getIndex( pDepth, pCode );
	if ( pDepth > mDepthMaxPrecomputed )
	{
		return draw( pDepth - 1, getFather( pCode ) );
	}
	else
	{
		if ( mOctree[i].count != 0 )
		{
			// Render VAO
			glBindVertexArray( mVAO[ 0 ] );
			glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, mBuffers[ 2 ] );
				
			glDrawElements( GL_TRIANGLES,  
				mOctree[ i ].count, 
				GL_UNSIGNED_INT,
				(void*)( sizeof( unsigned int ) * mOctree[ i ].first ) );
			
			glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
			glBindVertexArray( 0 );
		}
	}
}

/******************************************************************************
 * Test intersection of a node and the mesh, given its localization info (depth and code)
 *
 * @param depth depth
 * @param locCode localization code
 ******************************************************************************/
uint Scene::intersectMesh( unsigned int depth, const uint3& locCode ) const
{
	// Compute index of the node in the octree
	unsigned int i = getIndex( depth, locCode );

	// TO DO
	// - crash at level 7, mOctree memory overflow...

	if ( mOctree[ i ].count == -1 )
	{
		return intersectMesh( depth - 1, getFather( locCode ) );
	}
	else
	{
		if ( mOctree[ i ].count != 0 )
		{
			if ( depth == mDepthMax )
			{
				// Max level of resolution
				return  2;
			}
			else
			{
				// Data inside
				return 1;
			}
		}
	}

	// Empty node
	return  0;
}

/******************************************************************************
 * Fill the index buffer from vertices
 *
 * @param pIndices index buffer
 * @param pVertices position buffer
 ******************************************************************************/
void Scene::organizeIBO( std::vector< unsigned int >& pIndices, const float* pVertices )
{
	// Temporary buffer
	std::list< uint3 >* nodes = new std::list< uint3 >();
	std::list< uint3 >* nodesNextDepth = new std::list< uint3 >();

	unsigned int depth = 0;

	// Init algorithm with first node at depth 0
	mOctree[ 0 ].first = 0;
	mOctree[ 0 ].count = mNbTriangle * 3 - 1;	// TODO : only work for triangles primitives...

	// Move on the next depth level
	depth++;

	// Initialize node addresses (8 children)
	// - [ x,y,z ] triplet
	//
	// TODO : only work for octrees...
	nodes->push_back( make_uint3( 0, 0, 0 ) );
	nodes->push_back( make_uint3( 1, 0, 0 ) );
	nodes->push_back( make_uint3( 0, 1, 0 ) );
	nodes->push_back( make_uint3( 1, 1, 0 ) );
	nodes->push_back( make_uint3( 0, 0, 1 ) );
	nodes->push_back( make_uint3( 1, 0, 1 ) );
	nodes->push_back( make_uint3( 0, 1, 1 ) );
	nodes->push_back( make_uint3( 1, 1, 1 ) );

	// Iterate through depth
	while ( depth <= mDepthMaxPrecomputed )
	{
		// For each nodes of this depth we organize the index buffer 
		while ( ! nodes->empty() )
		{
			// Retrieve next node to treat it 
			uint3 currentBrick = nodes->front();

			// Remove first node (reduce size)
			nodes->pop_front();

			// Check if at least one primitive (i.e. triangle) intersect current node
			if ( organizeNode( depth, currentBrick, pIndices, pVertices ) )
			{
				// If there are triangles in this node, add its children to node list (8 children)
				// - [ x,y,z ] triplet
				//
				// TODO : only work for octrees...
				nodesNextDepth->push_back( currentBrick * 2 );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 1, 0, 0 ) );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 0, 1, 0 ) );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 1, 1, 0 ) );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 0, 0, 1 ) );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 1, 0, 1 ) );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 0, 1, 1 ) );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 1, 1, 1 ) );
			}
		}

		// We go on to the next depth level
		depth++;

		// Swap list of nodes
		std::list< uint3 >* nodesAux = nodes;
		nodes = nodesNextDepth; // new list of nodes to process
		nodesNextDepth = nodesAux; // empty list of nodes
	}

	// Free memory
	delete nodes;
	delete nodesNextDepth;
}

/******************************************************************************
 * ...
 *
 * @param pDepth node's depth localization info
 * @param pCode node's code localization info
 * @param pIndices index buffer
 * @param pVertices position buffer
 *
 * @return a flag telling wheter or not at least one primitive (i.e. triangle) intersect the node
 ******************************************************************************/
bool Scene::organizeNode( unsigned int pDepth, const uint3& pCode, std::vector< unsigned int >& pIndices, const float* pVertices )
{
	// Retrieve brick size and real coordinate and father index
	//
	// Note : we assume here that the border size is 1.0 / 8.0, but the value is TDataStructureType::BrickBorderSize
	// We have to use template...
	float3 brickPos = make_float3( pCode ) / make_float3( 1 << pDepth ) - make_float3( 1.0 / 8.0 ) / make_float3( 1 << pDepth );
	float3 brickSize = make_float3( 1.0 ) / make_float3( 1 << pDepth ) + make_float3( 2.0 / 8.0 ) / make_float3( 1 << pDepth );

	unsigned int index = getIndex( pDepth, pCode );
	unsigned int indexFather = getIndex( pDepth - 1, getFather( pCode ) ); 
	unsigned int count = 0; // Number of triangle that intersect the brick
	unsigned int begin = pIndices.size();

	// We test for each father's triangle if they intersect this birck
	for ( unsigned int i = mOctree[ indexFather ].first; i < mOctree[ indexFather ].first + mOctree[ indexFather ].count; i+=3 )
	{
		// Test if triangle intersect node
		if ( triangleIntersectBick( brickPos, brickSize, i, pIndices, pVertices ) )
		{
			count += 3;

			pIndices.push_back( pIndices[ i ] );
			pIndices.push_back( pIndices[ i + 1 ] );
			pIndices.push_back( pIndices[ i + 2 ] );
		}
	}

	mOctree[ index ].first = begin;
	mOctree[ index ].count = count;

	return ( count > 0 );
}

/******************************************************************************
 * Initialize a node in the node buffer
 *
 * @param pDepth node's depth localization info
 * @param pCode node's code localization info
 * @param pFirst start index of node in node buffer
 * @param pCount number of primitives in node (i.e triangles)
 ******************************************************************************/
void Scene::setOctreeNode( unsigned int pDepth, const uint3& pCode, unsigned int pFirst, int pCount )
{
	// We assume that depth < mDepthMax
	if ( pDepth > mDepthMaxPrecomputed )
	{
		// Retrieve node's gobal index in node buffer
		unsigned int i = getIndex( pDepth, pCode );
		mOctree[ i ].count = pCount;
		mOctree[ i ].first = pFirst;
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Scene::organizeIBOGlsl() 
{
	// Init Glsl program 
	GLuint prog = GvUtils::GvShaderManager::createShaderProgram( "Data/Shaders/Voxelization/organizeTriangle_VS.glsl", "Data/Shaders/Voxelization/organizeTriangle_GS.glsl", "Data/Shaders/Voxelization/organizeTriangle_FS.glsl" );
	GvUtils::GvShaderManager::linkShaderProgram( prog );

	// Init counter
	GLuint triangleCounter;
	glGenBuffers(1, &triangleCounter);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, triangleCounter);
	glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint) , NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

	// Set atomicCounter to number of triangle

	// declare a pointer to hold the values in the buffer 
	GLuint *userCounters;
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, triangleCounter);
	// map the buffer, userCounters will point to the buffers data
	userCounters = (GLuint*)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER, 
		0 , 
		sizeof(GLuint) , 
		GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT
		);
	// set the memory to number of triangle
	memset(userCounters, mNbTriangle-1, sizeof(GLuint) );
	// unmap the buffer
	glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
	glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

	// Attach IBO to a texture buffer 
	GLuint TBO;
	glGenTextures( 1, &TBO );
	glBindTexture( GL_TEXTURE_BUFFER, TBO );
	// Attach the storage of index buffer object to buffer texture
	glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, mBuffers[2] );
	glBindTexture( GL_TEXTURE_BUFFER, 0 );

	// setup openGl parameter 
	glPushAttrib(GL_VIEWPORT_BIT | GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT );
	glViewport( 0, 0, 1, 1 ); // DO IT HERE ???
	glDisable( GL_DEPTH_TEST );
	glDisable( GL_CULL_FACE );
	glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE );
	glUseProgram( prog );

	// Setup glsl uniform variable
	glBindImageTexture( 0, TBO, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI );
	glUniform1i( glGetUniformLocation( prog, "IBO" ), 0 );
	
	std::list< uint3 > *nodes = new std::list< uint3 >();
	std::list< uint3 > *nodesNextDepth = new std::list< uint3 >();
	unsigned int d = 0;

	// Init algorithm with first node at depth 0
	mOctree[0].first = 0;
	mOctree[0].count = mNbTriangle * 3 - 1;
	// Move on the next depth level
	d++;
	nodes->push_back( make_uint3( 0, 0, 0 ) );
	nodes->push_back( make_uint3( 1, 0, 0 ) );
	nodes->push_back( make_uint3( 0, 1, 0 ) );
	nodes->push_back( make_uint3( 1, 1, 0 ) );
	nodes->push_back( make_uint3( 0, 0, 1 ) );
	nodes->push_back( make_uint3( 1, 0, 1 ) );
	nodes->push_back( make_uint3( 0, 1, 1 ) );
	nodes->push_back( make_uint3( 1, 1, 1 ) );

	while ( d <= mDepthMaxPrecomputed ) {

		// For each nodes of this depth we orginize the IBO 
		while ( ! nodes->empty() ) {

			// Retrieve next node to treat it 
			uint3 currentBrick = nodes->front();
			nodes->pop_front();

			if ( organizeNodeGlsl( d, currentBrick , TBO, triangleCounter ) )
			{
				// If there are triangle in this brick we add its child to node list

				nodesNextDepth->push_back( currentBrick * 2 );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 1, 0, 0 ) );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 0, 1, 0 ) );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 1, 1, 0 ) );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 0, 0, 1 ) );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 1, 0, 1 ) );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 0, 1, 1 ) );
				nodesNextDepth->push_back( currentBrick * 2 + make_uint3( 1, 1, 1 ) );

			}
		}

		// We go on the next depth level
		d++;
		std::list< uint3 > *nodesAux = nodes;
		nodes = nodesNextDepth;
		nodesNextDepth = nodesAux ;
	}


	// Restore old variable
	glPopMatrix();
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();
	glPopAttrib();

	// Disable program
	glUseProgram( 0 );

	// Detach the storage of index buffer object to buffer texture
	glBindTexture( GL_TEXTURE_BUFFER, TBO );
	glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, 0 );
	glBindTexture( GL_TEXTURE_BUFFER, 0 );
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool Scene::organizeNodeGlsl( unsigned int d, const uint3& pCode, GLuint TBO, GLuint triangleCounter )
{
	return true;
}
