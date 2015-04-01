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

// Assimp
#include <assimp/cimport.h>
#include <assimp/postprocess.h>
#include <assimp/config.h>

// STL
#include <iostream>

// System
#include <cassert>
#include <cfloat>
#include <limits>

// GigaVoxels
#include <GvUtils/GvShaderManager.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

#define BUFFER_OFFSET(i) ((void*)(i))

/**
 * Assimp library object to load 3D model (with a log mechanism)
 */
static aiLogStream logStream;

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
:  _scene( NULL )
,	_vao( 0 )
,	mNbTriangle( 0 )
,	_octree( NULL )
, 	_depthMaxPrecomputed( 5 ) // with 5 can take around 15 seconds
,	mDepthMax( pMaxDepth )
{
	_buffers[ 0 ] = 0;
	_buffers[ 1 ] = 0;
	_buffers[ 2 ] = 0;

	// Attach stdout to the logging system.
	// Get one of the predefine log streams. This is the quick'n'easy solution to 
	// access Assimp's log system. Attaching a log stream can slightly reduce Assimp's
	// overall import performance.
	logStream = aiGetPredefinedLogStream( aiDefaultLogStream_STDOUT, NULL );

	// Attach a custom log stream to the libraries' logging system.
	aiAttachLogStream( &logStream );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
Scene::~Scene()
{
	// Releaqe graphics memory
	finalizeGraphicsResources();

	// Detach a custom log stream from the libraries' logging system.
	aiDetachLogStream( &logStream );
}

/******************************************************************************
 * Initialize the scene
 *
 * @param pSceneFile 3D scene filename (i.e. 3D model, etc...)
 ******************************************************************************/
bool Scene::init( const char* pSceneFile )
{
	// Finalize graphics resources
	finalizeGraphicsResources();

	// Initialize the graphics resources
	// - vertex array and associated vertex buffer objects
	initializeGraphicsResources();

	// Import 3D scene
	//aiSetImportPropertyInteger( AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_POINT | aiPrimitiveType_LINE );
	//const unsigned int flags = 0;
	const unsigned int flags = aiProcessPreset_TargetRealtime_Fast;
	//const unsigned int flags = aiProcessPreset_TargetRealtime_Quality;
	//const unsigned int flags = aiProcessPreset_TargetRealtime_MaxQuality;
	_scene = aiImportFile( pSceneFile, flags );

	// Scale the geometry
	float minx = +std::numeric_limits<float>::max();//FLT_MAX;
	float miny = +std::numeric_limits<float>::max();//FLT_MAX;
	float minz = +std::numeric_limits<float>::max();//FLT_MAX;
	float maxx = -std::numeric_limits<float>::max();//-FLT_MAX;
	float maxy = -std::numeric_limits<float>::max();//-FLT_MAX;
	float maxz = -std::numeric_limits<float>::max();//-FLT_MAX;

	// Compute the 3D scene bounds
	//
	// Iterate through meshes
	for ( unsigned int meshIndex = 0; meshIndex < _scene->mNumMeshes; ++meshIndex )
	{
		// Retrieve current mesh
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

	// Scale and bias data to fit [0;1]x[0;1]x[0;1] 3D domain
	//
	// Iterate through meshes
	float scale = 0.95f / std::max( std::max( maxx - minx, maxy - miny ), maxz - minz );
	for ( unsigned int meshIndex = 0; meshIndex < _scene->mNumMeshes; ++meshIndex )
	{
		// Retrieve current mesh
		const aiMesh* pMesh = _scene->mMeshes[ meshIndex ];

		// Iterate through mesh vertices
		for ( unsigned int vertexIndex = 0; vertexIndex < pMesh->mNumVertices; ++vertexIndex )
		{
			pMesh->mVertices[ vertexIndex ].x = ( pMesh->mVertices[ vertexIndex ].x - ( maxx + minx ) * 0.5f ) * scale + 0.5f;
			pMesh->mVertices[ vertexIndex ].y = ( pMesh->mVertices[ vertexIndex ].y - ( maxy + miny ) * 0.5f ) * scale + 0.5f;
			pMesh->mVertices[ vertexIndex ].z = ( pMesh->mVertices[ vertexIndex ].z - ( maxz + minz ) * 0.5f ) * scale + 0.5f;
		}
	}
	
	// WARNING : we assume here that faces of the mesh are triangle. Plus we don't take care of scene tree structure...
		
	// Computing number of vertices and triangles:
	unsigned int nbVertices = 0;
	mNbTriangle = 0;
	// Iterate through meshes
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
	// Estimate the length of the IBO 
	IBO.reserve( 3 * 3 * ( ( powf( 2, mDepthMax + 1 ) - 1 )  ) * mNbTriangle );
	// Resize vector to store the depth lvl 0 
	IBO.resize( 3*mNbTriangle );

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
					int a = pFace->mIndices[ 0 ];
					int b = pFace->mIndices[ 1 ];
					int c = pFace->mIndices[ 2 ];

					float e1[ 3 ] = { pMesh->mVertices[b].x - pMesh->mVertices[a].x,
						pMesh->mVertices[b].y - pMesh->mVertices[a].y,
						pMesh->mVertices[b].z - pMesh->mVertices[a].z };

					float e2[ 3 ] = { pMesh->mVertices[c].x - pMesh->mVertices[a].x,
						pMesh->mVertices[c].y - pMesh->mVertices[a].y,
						pMesh->mVertices[c].z - pMesh->mVertices[a].z };

					normals[ offsetVBO + 3 * index + 0 ] += e1[1]*e2[2] - e1[2]*e2[1];
					normals[ offsetVBO + 3 * index + 1 ] += e1[2]*e2[0] - e1[0]*e2[2];
					normals[ offsetVBO + 3 * index + 2 ] += e1[0]*e2[1] - e1[1]*e2[0];

				}
				else
				{
					normals[ offsetVBO + 3 * index + 0 ] += pMesh->mNormals[ index ].x;
					normals[ offsetVBO + 3 * index + 1 ] += pMesh->mNormals[ index ].y;
					normals[ offsetVBO + 3 * index + 2 ] += pMesh->mNormals[ index ].z;
				}
				// To average normals
				count[ index ]++;

				vertices[ offsetVBO + 3 * index + 0 ] = pMesh->mVertices[ index ].x;
				vertices[ offsetVBO + 3 * index + 1 ] = pMesh->mVertices[ index ].y;
				vertices[ offsetVBO + 3 * index + 2 ] = pMesh->mVertices[ index ].z;

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
		normals[ 3 * i + 0 ] /= count[ i ];
		normals[ 3 * i + 1 ] /= count[ i ];
		normals[ 3 * i + 2 ] /= count[ i ];

		// Normalizing the normal 
		float normal = sqrt ( normals[ 3 * i + 0 ] * normals[ 3 * i + 0 ] + 
							normals[ 3 * i + 1 ] * normals[ 3 * i + 1 ] +
							normals[ 3 * i + 2 ] * normals[ 3 * i + 2 ] );
		normals[ 3 * i + 0 ] /= normal;
		normals[ 3 * i + 1 ] /= normal;
		normals[ 3 * i + 2 ] /= normal;
	}

	// Rmk : Assimp seems to create one vertex per triangle so the above average step isn't usefull...

	// Organize the octree 
	organizeIBO( IBO, vertices );

	// Bind VAO
	glBindVertexArray( _vao );

	// vertex coordinates buffer
	glBindBuffer( GL_ARRAY_BUFFER, _buffers[ 0 ] );
	glBufferData( GL_ARRAY_BUFFER, sizeof( float ) * 3 * nbVertices, vertices, GL_STATIC_DRAW );
	glEnableVertexAttribArray( (GLuint)0 );
	glVertexAttribPointer( (GLuint)0, 3, GL_FLOAT, 0, 0, 0 );

	// normals buffer
	glBindBuffer( GL_ARRAY_BUFFER, _buffers[ 1 ] );
	glBufferData( GL_ARRAY_BUFFER, sizeof( float ) * 3 * nbVertices, normals, GL_STATIC_DRAW );
	glEnableVertexAttribArray( (GLuint)1 );
	glVertexAttribPointer( (GLuint)1 , 3, GL_FLOAT, 0, 0, 0 );

	// index buffer
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _buffers[ 2 ] );
	glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( unsigned int ) * IBO.size(), IBO.data(), GL_STATIC_DRAW );

	// Unbind the VAO
	glBindVertexArray( 0 );

	// Deleting temporary arrays
	delete[] vertices;
	delete[] normals;
	delete[] count;

	return true;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Scene::draw() const
{
	// render VAO
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _buffers[ 2 ] );
	glBindVertexArray( _vao );
	
	glDrawElements( GL_TRIANGLES, mNbTriangle * 3, GL_UNSIGNED_INT, 0 );
	
	glBindVertexArray( 0 );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Scene::draw( unsigned int pDepth ) const
{
	for ( int i = ( powf( 8, pDepth ) - 1 ) / (float)7; i < ( powf( 8, pDepth + 1 ) - 1 ) / (float)7; i++ )
	{
		if ( _octree[ i ].count != 0 )
		{
			// Render VAO	
			glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _buffers[ 2 ] );
			glBindVertexArray( _vao );
			glDrawElements( GL_TRIANGLES,  
							_octree[ i ].count, 
							GL_UNSIGNED_INT,
							(void*)( sizeof( unsigned int ) * _octree[ i ].first ) );
			glBindVertexArray( 0 );
			glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
		}
	}
}

/******************************************************************************
* ...
******************************************************************************/
void Scene::draw( unsigned int pDepth, uint3 pLocNode ) const
{
	if ( pDepth > _depthMaxPrecomputed )
	{
		return draw( pDepth - 1, getFather( pLocNode ) );
	}
	else
	{
		// Compute index of the node in the octree
		unsigned int i = getIndex( pDepth, pLocNode );
		if ( _octree[ i ].count != 0 )
		{
			// Render VAO	
			glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _buffers[ 2 ] );
			glBindVertexArray( _vao );
			glDrawElements( GL_TRIANGLES,  
							_octree[ i ].count, 
							GL_UNSIGNED_INT,
							(void*)( sizeof( unsigned int ) * _octree[ i ].first ) );
			glBindVertexArray( 0 );
			glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
		}
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
uint Scene::intersectMesh( unsigned int depth, uint3 locCode ) const
{
	// Compute index of the node in the octree
	unsigned int i = getIndex( depth, locCode );
	if ( _octree[i].count == -1 )
	{
		return intersectMesh( depth - 1, getFather( locCode ) );
	}
	else
	{
		if ( _octree[ i ].count != 0 )
		{
			if ( depth == mDepthMax )
			{
				return  2;
			}
			return 1;
		}
		return  0;
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Scene::organizeIBO( std::vector< unsigned int >& IBO, const float* vertices ) 
{
	std::list< uint3 >* nodes = new std::list< uint3 >();
	std::list< uint3 >* nodesNextDepth = new std::list< uint3 >();
	unsigned int d = 0;

	// Init algorithm with first node at depth 0
	_octree[ 0 ].first = 0;
	_octree[ 0 ].count = mNbTriangle * 3 ;
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

	while ( d <= _depthMaxPrecomputed )
	{
		// For each nodes of this depth we orginize the IBO 
		while ( ! nodes->empty() )
		{
			// Retrieve next node to treat it 
			uint3 currentBrick = nodes->front();
			nodes->pop_front();

			if ( organizeNode( d, currentBrick, IBO, vertices ) )
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
		std::list< uint3 >* nodesAux = nodes;
		nodes = nodesNextDepth;
		nodesNextDepth = nodesAux ;
	}
}
	
/******************************************************************************
 * ...
 ******************************************************************************/
bool Scene::organizeNode( unsigned int d, uint3 locCode, std::vector< unsigned int >& IBO, const float* vertices )
{
	// Retrieve brick size and real coordinate and father index
	// Rmk : we assume here that the border size is 1.0 / 8.0, but the value is TDataStructureType::BrickBorderSize.
	// We have to use template...
	float3 brickPos = make_float3( locCode ) / make_float3( 1 << d ) - make_float3( 1.0 / 8.0 ) /  make_float3( 1 << d );
	float3 brickSize = make_float3( 1.0 ) / make_float3( 1 << d ) + make_float3( 2.0 / 8.0 ) /  make_float3( 1 << d );

	unsigned int index = getIndex( d, locCode );
	unsigned int indexFather = getIndex( d - 1, getFather( locCode ) );
	unsigned int count = 0; // Number of triangle that intersect the brick
	unsigned int begin = IBO.size();

	// We test for each fater's triangle if they intertect this birck
	for ( unsigned int i = _octree[ indexFather ].first; i < _octree[ indexFather ].first + _octree[ indexFather ].count ; i+=3 )
	{
		if ( triangleIntersectBick( brickPos, brickSize, i, IBO, vertices ) )
		{
			count += 3;
			IBO.push_back( IBO[ i ] );
			IBO.push_back( IBO[ i + 1 ] );
			IBO.push_back( IBO[ i + 2 ] );
		}
	}
	
	_octree[ index ].first = begin;
	_octree[ index ].count = count; 

	return ( count > 0 );
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Scene::setOctreeNode( unsigned int depth, uint3 locCode, unsigned int pFirst, int pCount )
{
	// We assume that depth < mDepthMax
	if ( depth > _depthMaxPrecomputed )
	{
		unsigned int i = getIndex( depth, locCode );
		_octree[ i ].count = pCount;
		_octree[ i ].first = pFirst;
	}
}

/******************************************************************************
 * Initialize graphics resources
 *
 * @return a flag telling wheter or not process has succeeded
 ******************************************************************************/
bool Scene::initializeGraphicsResources()
{
	// Vertex array object
	glGenVertexArrays( 1, &_vao );
	
	// Vertex buffer objects
	// - create buffers for our vertex data (vertices, normals and indexes)
	glGenBuffers( 3, _buffers );

	// length_octree = Somme( 8^i, i = 0:mDepthMax ) + 8 
	unsigned int length = ( powf( 8, mDepthMax + 1 ) - 1 ) / (float)7 ;
	_octree = new node[ length ];
	for ( unsigned int i = 0; i < length; i++ )
	{
		_octree[ i ].first = 0;
		_octree[ i ].count = -1;
	}
		
	return true;
}

/******************************************************************************
 * Finalize graphics resources
 *
 * @return a flag telling wheter or not process has succeeded
 ******************************************************************************/
bool Scene::finalizeGraphicsResources()
{
	// Clean Assimp library ressources
	if ( _scene != NULL )
	{
		//	If the call to aiImportFile() succeeds, the imported data is returned in an aiScene structure. 
		// The data is intended to be read-only, it stays property of the ASSIMP 
		// library and will be stable until aiReleaseImport() is called. After you're 
		// done with it, call aiReleaseImport() to free the resources associated with 
		// this file.
		aiReleaseImport( _scene );
		
		_scene = NULL;
	}

	// Release graphics recources
	glDeleteBuffers( 3, _buffers );
	glDeleteVertexArrays( 1, &_vao );

	delete[] _octree;

	return true;
}
