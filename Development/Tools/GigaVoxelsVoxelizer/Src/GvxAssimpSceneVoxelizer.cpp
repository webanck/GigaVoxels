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

#include "GvxAssimpSceneVoxelizer.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// System
#include <cassert>
#include <iostream>
#include <cfloat>

// Assimp
#include <assimp/postprocess.h>

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
GvxAssimpSceneVoxelizer::GvxAssimpSceneVoxelizer()
:	GvxSceneVoxelizer()
,	_scene( NULL )
{
	// Attach stdout to the logging system.
	// Get one of the predefine log streams. This is the quick'n'easy solution to 
	// access Assimp's log system. Attaching a log stream can slightly reduce Assimp's
	// overall import performance.
	_logStream = aiGetPredefinedLogStream( aiDefaultLogStream_STDOUT, NULL );

	// Attach a custom log stream to the libraries' logging system.
	aiAttachLogStream( &_logStream );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvxAssimpSceneVoxelizer::~GvxAssimpSceneVoxelizer()
{
	//	If the call to aiImportFile() succeeds, the imported data is returned in an aiScene structure. 
	// The data is intended to be read-only, it stays property of the ASSIMP 
	// library and will be stable until aiReleaseImport() is called. After you're 
	// done with it, call aiReleaseImport() to free the resources associated with 
	// this file.
	aiReleaseImport( _scene );
	
	// Detach a custom log stream from the libraries' logging system.
	aiDetachLogStream( &_logStream );
}

/**
 * ...
 */
#define aiProcessPreset_TargetRealtime_Smooth ( \
	aiProcess_CalcTangentSpace		|  \
	aiProcess_GenSmoothNormals 	    |  \
	aiProcess_JoinIdenticalVertices |  \
	aiProcess_FixInfacingNormals    |  \
	aiProcess_Triangulate			|  \
	aiProcess_GenUVCoords           |  \
	aiProcess_SortByPType           |  \
	0 )

/******************************************************************************
 * Load/import the scene the scene
 ******************************************************************************/
bool GvxAssimpSceneVoxelizer::loadScene()
{
	bool result = false;

	// Load the scene from the 3D model file.
	// Read the given file and returns its content.
	// Return a pointer to the imported data or NULL if the import failed.
	string filename = string( getFilePath() + getFileName() + getFileExtension() );
	_scene = const_cast< aiScene* >( aiImportFile( filename.data(), aiProcessPreset_TargetRealtime_Smooth ) );
	// NOTE
	// The aiProcessPreset_TargetRealtime_Fast macro is an OR combination of the following flags :
	// - aiProcess_CalcTangentSpace
	// - aiProcess_GenNormals
	// - aiProcess_JoinIdenticalVertices
	// - aiProcess_Triangulate
	// - aiProcess_GenUVCoords
	// - aiProcess_SortByPType
				
	// Check import status
	assert( _scene != NULL );
	if ( _scene != NULL )
	{
		result = true;
	}

	return result;
}

/******************************************************************************
 * Normalize the scene.
 * It determines the whole scene bounding box and then modifies vertices
 * to scale the scene.
 ******************************************************************************/
bool GvxAssimpSceneVoxelizer::normalizeScene()
{
	assert( _scene != NULL );
	if ( _scene == NULL )
	{
		// TO DO
		// Handle error
		// ...

		return false;
	}

	// Initialize scene bounds
	float xmin = +FLT_MAX;
	float ymin = +FLT_MAX;
	float zmin = +FLT_MAX;
	float xmax = -FLT_MAX;
	float ymax = -FLT_MAX;
	float zmax = -FLT_MAX;

	// Iterate through meshes to determine the whole scene bounding box
	for ( unsigned int i = 0; i < _scene->mNumMeshes; i++ )
	{
		// Get the current mesh
		const aiMesh* mesh = _scene->mMeshes[ i ];
		assert( mesh != NULL );
		if ( mesh != NULL )
		{
			// Iterate through vertices
			for ( unsigned int j = 0 ; j < mesh->mNumVertices ; j++ )
			{
				// Get the current vertex
				const aiVector3D& vertex = mesh->mVertices[ j ];

				// Compute/update scene bounding box
				xmin = std::min< float >( vertex.x, xmin );
				ymin = std::min< float >( vertex.y, ymin );
				zmin = std::min< float >( vertex.z, zmin );
				xmax = std::max< float >( vertex.x, xmax );
				ymax = std::max< float >( vertex.y, ymax );
				zmax = std::max< float >( vertex.z, zmax );
			}
		}
		else
		{
			// TO DO
			// Handle error
			// ...
			// return false;
		}
	}

	// Iterate through meshes to normalize the scene
	float scale = 0.95f / std::max< float >( std::max< float >( xmax - xmin, ymax - ymin ), zmax - zmin );
	for ( unsigned int i = 0; i < _scene->mNumMeshes; i++ )
	{
		// Get the current mesh
		const aiMesh* mesh = _scene->mMeshes[ i ];

		// Iterate through vertices
		for ( unsigned int j = 0 ; j < mesh->mNumVertices ; j++ )
		{
			// Get the current vertex
			aiVector3D& vertex = mesh->mVertices[ j ];

			// Scale the scene
			vertex.x = ( vertex.x  - 0.5f * ( xmax + xmin ) ) * scale + 0.5f;
			vertex.y = ( vertex.y  - 0.5f * ( ymax + ymin ) ) * scale + 0.5f;
			vertex.z = ( vertex.z  - 0.5f * ( zmax + zmin ) ) * scale + 0.5f;
		}
	}

	return true;
}

/******************************************************************************
 * Voxelize the scene
 ******************************************************************************/
bool GvxAssimpSceneVoxelizer::voxelizeScene()
{
	assert( _scene != NULL );
	if ( _scene == NULL )
	{
		// TO DO
		// Handle error
		// ...

		return false;
	}

	// Count number of faces
	unsigned int nbFaces = 0;
	for ( unsigned int i = 0; i < _scene->mNumMeshes; i++ )
	{
		nbFaces += _scene->mMeshes[ i ]->mNumFaces;
	}

	// Render
	std::cout << "GvxVoxelizer at level : " << getMaxResolution() << std::endl;

	// Create a voxelizer
	//GvxVoxelizer vcpu;
	//assert( _voxelizerEngine != NULL );		// TO DO : a modfier !!!!!!!!!!!!!!!!!
	//if ( _voxelizerEngine == NULL )
	//{
	//	// TO DO
	//	// Handle error
	//	// ...
	//
	//	return false;
	//}
	
	// Initialize the voxelization phase
	_voxelizerEngine.init( getMaxResolution(), getBrickWidth(), getFileName(), getDataType() );
	_voxelizerEngine.setColor( 1, 1, 1 );
	_voxelizerEngine.setColor( 1, 1, 1 );
	_voxelizerEngine.setColor( 1, 1, 1 );

	// Iterate through meshes to voxelize them
	unsigned int nbProcessedFaces = 0;
	for ( unsigned int i = 0; i < _scene->mNumMeshes; i++ )
	{
		// Get the current mesh
		const aiMesh* mesh = _scene->mMeshes[ i ];
		assert( mesh != NULL );
		if ( mesh == NULL )
		{
			// TO DO
			// Handle error
			// ...
			//return;
		}

		// Check for material
		if ( _scene->mNumMaterials == 0 )
		{
			// QUESTION
			// Is that an error to handle ?
			// ...

			continue;
		}

		// Retrieve mesh material
		aiMaterial* material = _scene->mMaterials[ mesh->mMaterialIndex ];
		if ( material == NULL )
		{
			// QUESTION
			// Is that an error to handle ?
			// ...

			continue;
		}

		// Get the number of diffuse and ambient textures type
		int nbDiffuseTextures = material->GetTextureCount( aiTextureType_DIFFUSE );
		int nbAmbientTextures = material->GetTextureCount( aiTextureType_AMBIENT );

		if ( nbDiffuseTextures > 0 )
		{
			// Handle diffuse texture type
			aiString textureName;
			material->GetTexture( aiTextureType_DIFFUSE, 0, &textureName );

			string texName = textureName.data;

			// On Linux, adding path as "data\image.png"
			// generates run time-crash with CImag library.
			// Backslahes need to be converted.
#ifndef WIN32
		for ( int i = 0; i < texName.size(); i++ )
		{
			if ( texName[ i ] == '\\' )
			{
				texName[ i ] = '/';
			}
		}
#endif

			_voxelizerEngine.setTexture( getFilePath() + texName );
			_voxelizerEngine._useTexture = true;
		}
		else if ( nbAmbientTextures > 0 )
		{
			// Handle ambient texture type
			aiString textureName;
			material->GetTexture( aiTextureType_AMBIENT, 0, &textureName );

			string texName = textureName.data;

			// On Linux, adding path as "data\image.png"
			// generates run time-crash with CImag library.
			// Backslahes need to be converted.
#ifndef WIN32
		for ( int i = 0; i < texName.size(); i++ )
		{
			if ( texName[ i ] == '\\' )
			{
				texName[ i ] = '/';
			}
		}
#endif

			_voxelizerEngine.setTexture( getFilePath() + texName );
			_voxelizerEngine._useTexture = true;
		}
		else
		{
			// Handle mesh without texture
			aiColor4D ambient;
			aiColor4D diffuse;
			aiColor4D specular;
			aiGetMaterialColor( material, AI_MATKEY_COLOR_AMBIENT, &ambient );
			aiGetMaterialColor( material, AI_MATKEY_COLOR_DIFFUSE, &diffuse );
			aiGetMaterialColor( material, AI_MATKEY_COLOR_DIFFUSE, &specular );

			_voxelizerEngine._useTexture = false;

			_voxelizerEngine.setColor( ambient.r + diffuse.r, ambient.g + diffuse.g, ambient.b + diffuse.b );
			_voxelizerEngine.setColor( ambient.r + diffuse.r, ambient.g + diffuse.g, ambient.b + diffuse.b );
			_voxelizerEngine.setColor( ambient.r + diffuse.r, ambient.g + diffuse.g, ambient.b + diffuse.b );
		}

		// Iterate through mesh faces to voxelize them
		for ( unsigned int face = 0; face < mesh->mNumFaces; face++ )
		{
			// WARNING : We only handle triangles
			if ( mesh->mFaces[ face ].mNumIndices != 3 )
			{
				std::cout << "Error voxelizeScene : this polygon is not a triangle." << std::endl;
				continue;
			}

			// Iterate through face vertices
			for ( int vertex = 0; vertex < 3; vertex++ )
			{
				unsigned int ind = mesh->mFaces[ face ].mIndices[ vertex ];

				// TO DO : question => check if there are always normals ?
				// Retrieve normal
				_voxelizerEngine.setNormal( mesh->mNormals[ ind ].x, mesh->mNormals[ ind ].y, mesh->mNormals[ ind ].z);

				// Retrieve texture coordinates
				if ( _voxelizerEngine._useTexture )
				{
					_voxelizerEngine.setTexCoord( mesh->mTextureCoords[ 0 ][ ind ].x, mesh->mTextureCoords[ 0 ][ ind ].y );
				}

				// Retrieve vertex coordinates
				_voxelizerEngine.setVertex( mesh->mVertices[ ind ].x, mesh->mVertices[ ind ].y, mesh->mVertices[ ind ].z );
			}

			// Generate a voxel from face data
			_voxelizerEngine.voxelizeTriangle();

			// Update the processed faces counter
			nbProcessedFaces++;
			if ( nbProcessedFaces % 100 == 0 )
			{
				std::cout << "At level : " << getMaxResolution() << " " << nbProcessedFaces << "/" << nbFaces << " faces done." << std::endl;
			}
		}
	}

	// End of the voxelization phase
	_voxelizerEngine.end();

	return true;
}
