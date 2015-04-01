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

#include "Mesh.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Assimp
#include <assimp.h>
#include <aiScene.h>
#include <aiPostProcess.h>
#include <aiConfig.h>

// Cuda
#include <vector_types.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>

// STL
#include <iostream>

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvUtils;

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
Mesh::Mesh()
:	IMesh()
,	_scene( NULL )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
Mesh::~Mesh()
{
	finalize();
}

/******************************************************************************
 * Finalize
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
bool Mesh::finalize()
{
	// Clean Assimp library ressources
	if ( _scene != NULL )
	{
		aiReleaseImport( _scene );
		_scene = NULL;
	}
	
	return true;
}

/******************************************************************************
 * Render
 ******************************************************************************/
void Mesh::render( const float4x4& pModelViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	// Configure rendering state
	_shaderProgram->use();
	glBindVertexArray( _vertexArray );

	// Set uniforms
	GLuint location = glGetUniformLocation( _shaderProgram->_program, "uModelViewMatrix" );
	if ( location >= 0 )
	{
		glUniformMatrix4fv( location, 1, GL_FALSE, pModelViewMatrix._array );
	}
	location = glGetUniformLocation( _shaderProgram->_program, "uProjectionMatrix" );
	if ( location >= 0 )
	{
		glUniformMatrix4fv( location, 1, GL_FALSE, pProjectionMatrix._array );
	}

	location = glGetUniformLocation( _shaderProgram->_program, "uProjectionMatrix" );
	if ( location >= 0 )
	{
		glUniformMatrix4fv( location, 1, GL_FALSE, pProjectionMatrix._array );
	}
	
	location = glGetUniformLocation( _shaderProgram->_program, "uViewportMatrix" );
	if ( location >= 0 )
	{
		float viewportMatrix[ 16 ];	// no non-uniform scale

		viewportMatrix[ 0 ] = static_cast< float >( pViewport.z ) * 0.5f;
		viewportMatrix[ 1 ] = 0.0f;
		viewportMatrix[ 2 ] = 0.0f;
		viewportMatrix[ 3 ] = 0.0f;

		viewportMatrix[ 4 ] = 0.0f;
		viewportMatrix[ 5 ] = static_cast< float >( pViewport.w ) * 0.5f;
		viewportMatrix[ 6 ] = 0.0f;
		viewportMatrix[ 7 ] = 0.0f;

		viewportMatrix[ 8 ] = 0.0f;
		viewportMatrix[ 9 ] = 0.0f;
		viewportMatrix[ 10 ] = 0.5f;
		viewportMatrix[ 11 ] = 0.0f;

		viewportMatrix[ 12 ] = static_cast< float >( pViewport.x ) + static_cast< float >( pViewport.z ) * 0.5f;
		viewportMatrix[ 13 ] = static_cast< float >( pViewport.y ) + static_cast< float >( pViewport.w ) * 0.5f;
		viewportMatrix[ 14 ] = 0.5f;
		viewportMatrix[ 15 ] = 1.0f;

		glUniformMatrix4fv( location, 1, GL_FALSE, viewportMatrix );
	}

	location = glGetUniformLocation( _shaderProgram->_program, "uMeshColor" );
	if ( location >= 0 )
	{
		float4 color = make_float4( 1.f, 0.f, 0.f, 1.f );
		glUniform4f( location, color.x, color.y, color.z, color.w );
	}

	location = glGetUniformLocation( _shaderProgram->_program, "uColor" );
	if ( location >= 0 )
	{
		float4 color = make_float4( 1.f, 0.f, 0.f, 1.f );
		glUniform3f( location, _color.x, _color.y, _color.z );
	}

	location = glGetUniformLocation( _shaderProgram->_program, "uLineWidth" );
	if ( location >= 0 )
	{
		//float _wireframeLineWidth = 1.0f;
		float _wireframeLineWidth = 0.5f;
		glUniform1f( location, _wireframeLineWidth );
	}

	location = glGetUniformLocation( _shaderProgram->_program, "uLineColor" );
	if ( location >= 0 )
	{
		float3 _wireframeColor = make_float3( 0.f, 0.f, 0.f );
		glUniform4f( location, _wireframeColor.x, _wireframeColor.y, _wireframeColor.z, 1.0f );
	}

	if ( _hasNormals )
	{
		location = glGetUniformLocation( _shaderProgram->_program, "uNormalMatrix" );
		if ( location >= 0 )
		{
			// TO DO
			// - retrieve or compute Normal matrix
			//
			float normalMatrix[ 9 ];
			glUniformMatrix3fv( location, 1, GL_FALSE, normalMatrix );
		}
	}

	// Draw mesh
	if ( _useIndexedRendering )
	{
		glDrawElements( GL_TRIANGLES, _nbFaces * 3, GL_UNSIGNED_INT, NULL );
	}
	else
	{
		glDrawArrays( GL_TRIANGLES, 0, 0 );
	}
	
	// Reset rendering state
	glBindVertexArray( 0 );
	glUseProgram( 0 );
}

/******************************************************************************
 * Read mesh data
 ******************************************************************************/
bool Mesh::read( const char* pFilename, std::vector< float3 >& pVertices, std::vector< float3 >& pNormals, std::vector< float2 >& pTexCoords, std::vector< unsigned int >& pIndices )
{
	// Delete the 3D scene if needed
	if ( _scene != NULL )
	{
		aiReleaseImport( _scene );
		_scene = NULL;
	}

	// ---- Load the 3D scene ----
	aiSetImportPropertyInteger( AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_POINT | aiPrimitiveType_LINE );
	//const unsigned int flags = 0;
	//const unsigned int flags = aiProcessPreset_TargetRealtime_Fast;
	//const unsigned int flags = aiProcessPreset_TargetRealtime_Quality;
	const unsigned int flags = aiProcessPreset_TargetRealtime_MaxQuality;
	_scene = aiImportFile( pFilename, flags );
	
	assert( _scene != NULL );

	// Compute mesh bounds
	float minX = +std::numeric_limits< float >::max();
	float minY = +std::numeric_limits< float >::max();
	float minZ = +std::numeric_limits< float >::max();
	float maxX = -std::numeric_limits< float >::max();
	float maxY = -std::numeric_limits< float >::max();
	float maxZ = -std::numeric_limits< float >::max();

	// Iterate through meshes
	for ( unsigned int i = 0; i < _scene->mNumMeshes; ++i )
	{
		// Retrieve current mesh
		const aiMesh* mesh = _scene->mMeshes[ i ];

		// Iterate through vertices
		for ( unsigned int j = 0; j < mesh->mNumVertices; ++j )
		{
			minX = std::min( minX, mesh->mVertices[ j ].x );
			minY = std::min( minY, mesh->mVertices[ j ].y );
			minZ = std::min( minZ, mesh->mVertices[ j ].z );
			maxX = std::max( maxX, mesh->mVertices[ j ].x );
			maxY = std::max( maxY, mesh->mVertices[ j ].y );
			maxZ = std::max( maxZ, mesh->mVertices[ j ].z );
		}
	}

	// WARNING : we assume here that faces of the mesh are triangle. Plus we don't take care of scene tree structure...

	// Computing number of vertices and triangles:
	//
	// Iterate through meshes
	for ( unsigned int i = 0; i < _scene->mNumMeshes; ++i )
	{
		_nbVertices += _scene->mMeshes[ i ]->mNumVertices;
		_nbFaces += _scene->mMeshes[ i ]->mNumFaces;
	}
	pVertices.reserve( _nbVertices );
	pNormals.reserve( _nbVertices );
	pTexCoords.reserve( _nbVertices );
	pIndices.reserve( _nbFaces * 3 );

	// Iterate through meshes
	for ( unsigned int i = 0; i < _scene->mNumMeshes; ++i )
	{
		// Retrieve current mesh
		const aiMesh* mesh = _scene->mMeshes[ i ];

		// Iterate through vertices
		//
		// TO DO : extract IF for normal and texture coordinates to speed code
		for ( unsigned int j = 0; j < mesh->mNumVertices; ++j )
		{
			// Retrieve vertex position
			pVertices.push_back( make_float3( mesh->mVertices[ j ].x, mesh->mVertices[ j ].y, mesh->mVertices[ j ].z ) );
			
			// Retrieve vertex normal
			if ( _hasNormals )
			{
				pNormals.push_back( make_float3( mesh->mNormals[ j ].x, mesh->mNormals[ j ].y, mesh->mNormals[ j ].z ) );
			}

			// Retrieve texture coordinates
			if ( _hasTextureCoordinates )
			{
				// ...
			}
		}
	}

	// Retrieve face information for indexed rendering
	if ( _useIndexedRendering )
	{
		// Iterate through meshes
		for ( unsigned int i = 0; i < _scene->mNumMeshes; ++i )
		{
			// Retrieve current mesh
			const aiMesh* mesh = _scene->mMeshes[ i ];

			// Iterate through faces
			for ( unsigned int j = 0; j < mesh->mNumFaces; ++j )
			{
				// Retrieve current face
				const struct aiFace* face = &mesh->mFaces[ j ];

				// Remark : we can compute different normal for same vertex, but new one overwrites the old one
				for ( unsigned int k = 0; k < face->mNumIndices; ++k )
				{
					pIndices.push_back( face->mIndices[ k ] );
				}
			}
		}
	}
	
	return true;
}
