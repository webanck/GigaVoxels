/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2014 INRIA <http://www.inria.fr/>
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

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// STL
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * ...
 */
#define BUFFER_OFFSET( a ) ((char*)NULL + ( a ) )

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

// Assimp importer
Assimp::Importer importer;

// Assimp scene object
const aiScene* scene = NULL;

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 *
 * - bounding box set to infinity
 *
 * @param p: program identifier
 ******************************************************************************/
Mesh::Mesh( GLuint p )
:	_modelFilename()
{
	// Transformations
	_translation[ 0 ] = 0.0f;
	_translation[ 1 ] = 0.0f;
	_translation[ 2 ] = 0.0f;
	_rotation[ 0 ] = 0.0f;
	_rotation[ 1 ] = 0.0f;
	_rotation[ 2 ] = 0.0f;
	_rotation[ 3 ] = 0.0f;
	_scale = 1.0f;

	lightPos[ 0 ] = 1;
	lightPos[ 1 ] = 1;
	lightPos[ 2 ] = 1;

	_shaderProgram = p;

	for ( int k = 0; k < 3; k++ )
	{
		boxMin[ k ] = numeric_limits< float >::max();
		boxMax[ k ] = numeric_limits< float >::min();
	}
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
Mesh::~Mesh()
{
	// Clean Assimp library ressources
	if ( scene != NULL )
	{
		// If the call to aiImportFile() succeeds, the imported data is returned in an aiScene structure. 
		// The data is intended to be read-only, it stays property of the ASSIMP 
		// library and will be stable until aiReleaseImport() is called. After you're 
		// done with it, call aiReleaseImport() to free the resources associated with 
		// this file.
		importer.FreeScene();
		
		// Reset pointer
		scene = NULL;
	}
		
	// Iterate through meshes
	for ( int i = 0; i < _meshes.size(); i++ )
	{
		if ( _meshes[ i ]._vertexBuffer  )
		{
			glDeleteBuffers( 1, &( _meshes[ i ]._vertexBuffer ) );
		}
		if ( _meshes[ i ]._indexBuffer  )
		{
			glDeleteBuffers( 1, &( _meshes[ i ]._indexBuffer ) );
		}
	}
}

/******************************************************************************
 * Loading and binding a texture.
 *
 * @param filename: texture file
 * @param id: texture ID
 ******************************************************************************/
void Mesh::loadTexture( const char* filename, GLuint id )
{
	string f;

	// Get the right file name
	QDir d( filename );
	QDirIterator it( QDir( QString( _repository.c_str() ) ), QDirIterator::Subdirectories );
	while ( it.hasNext() )
	{
		it.next();
		QString file = it.fileName();
		if ( file == QString( Filename( filename ).c_str() ) )
		{
			f = it.filePath().toStdString();
		}
	}

	ifstream fin( f.c_str() );
	if ( ! fin.fail() )
	{
		fin.close();
	}
	else
	{
		cout << "Couldn't open texture file: " << f << "." << endl;

		return;
	}

	// Load image
	QImage img = QGLWidget::convertToGLFormat( QImage( f.c_str() ) );

	// Initialize texture
	glBindTexture( GL_TEXTURE_2D, id );
	// - allocate texture memory
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, img.width(), img.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, img.bits() );
	// - configure texture
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	// TO DO : add man filter
	
	// TO DO : check if that's useful ?
	glEnable( GL_TEXTURE_2D );
}

/******************************************************************************
 * Retrieves the directory of a filename.
 *
 * @param filename: file path.
 ******************************************************************************/
string Directory( const string& filename )
{
	size_t pos = filename.find_last_of( "\\/" );
	return ( string::npos == pos ) ? "" : filename.substr( 0, pos + 1 );
}

/******************************************************************************
 * Retrieves the base filename of a file path.
 *
 * @param path: file path.
 ******************************************************************************/
string Filename( const string& path )
{
	return path.substr( path.find_last_of( "/\\" ) + 1 );
}

/******************************************************************************
 * Collects all the information about the meshes 
 * (vertices, normals, textures, materials, ...)
 *
 * @param scene: assimp loaded meshes.
 ******************************************************************************/
void Mesh::InitFromScene( const aiScene* scene )
{
	aiColor4D coltemp;
	int materialIndex;
	aiReturn texFound;
	int nbT;
	aiString file;
	float shininess;

	// Vertices, normals and textures

	// Iterate through meshes
	for ( int i = 0; i < scene->mNumMeshes; i++ )
	{
		// Retrieve current mesh
		const aiMesh* mesh = scene->mMeshes[ i ];

		oneMesh M;
		M._hasAmbientTextures = false;
		M._hasDiffuseTextures = false;
		M._hasSpecularTextures = false;
		M._shininess = 20;
		for ( int a = 0; a < 3; a++ )
		{
			M._ambient[ a ] = 0.5;
			M._diffuse[ a ] = 0;
			M._specular[ a ] = 0.75;
		}	
		M._ambient[ 3 ] = 1.0;
		M._diffuse[ 3 ] = 1.0;
		M._specular[ 3 ] = 1.0;

		// Retrieve positions if any
		if ( mesh->HasPositions() )
		{
			// Iterate through vertices
			for ( int j = 0; j < mesh->mNumVertices; j++ )
			{
				// Retrieve current position
				const aiVector3D* pos = &( mesh->mVertices[ j ] );

				M._vertices.push_back( pos->x );
				if ( boxMax[ 0 ] < pos->x )
				{
					boxMax[ 0 ] = pos->x;
				}
				if ( boxMin[ 0 ] > pos->x )
				{
					boxMin[ 0 ] = pos->x;
				}
				M._vertices.push_back( pos->y );
				if ( boxMax[ 1 ] < pos->y )
				{
					boxMax[ 1 ] = pos->y;
				}
				if ( boxMin[ 1 ] > pos->y )
				{
					boxMin[ 1 ] = pos->y;
				}
				M._vertices.push_back( pos->z );
				if ( boxMax[ 2 ] < pos->z )
				{
					boxMax[ 2 ] = pos->z;
				}
				if ( boxMin[ 2 ] > pos->z )
				{
					boxMin[ 2 ] = pos->z;
				}
			}
		}

		// Retrieve normals if any
		if ( mesh->HasNormals() )
		{
			// Iterate through vertices
			for ( int j = 0; j < mesh->mNumVertices; j++ )
			{
				// Retrieve current normal
				const aiVector3D* normal = &( mesh->mNormals[ j ] );
				
				M._normals.push_back( normal->x );
				M._normals.push_back( normal->y );
				M._normals.push_back( normal->z );
			}
		}

		// Retrieve texture coordinates if any
		// TO DO : check what's this index and test number before to avoid crash
		if ( mesh->HasTextureCoords( 0 ) )
		{
			// Iterate through vertices
			for ( int j = 0; j < mesh->mNumVertices; j++ )
			{
				M._texCoords.push_back( mesh->mTextureCoords[ 0 ][ j ].x );
				M._texCoords.push_back( mesh->mTextureCoords[ 0 ][ j ].y );
			}
		}
		
		// Indices
		//
		// Iterate through faces
		for ( int k = 0 ; k < mesh->mNumFaces ; k++ )
		{
			// Retrieve current face
			const aiFace& Face = mesh->mFaces[ k ];

			//if ( Face.mNumIndices == 3 )
			//{
			M.mode = GL_TRIANGLES;
			M._indices.push_back( Face.mIndices[ 0 ] );
			M._indices.push_back( Face.mIndices[ 1 ] );
			M._indices.push_back( Face.mIndices[ 2 ] );
			//} 
		}

		// Materials
		if ( scene->HasMaterials() )
		{
			materialIndex = mesh->mMaterialIndex;
			aiMaterial* material = scene->mMaterials[ materialIndex ];

			// Ambient
			nbT = material->GetTextureCount( aiTextureType_AMBIENT );
			if ( nbT > 0 )
			{
				M._hasAmbientTextures = true;
			} 
			for ( int j = 0; j < nbT; j++ )
			{
				material->GetTexture( aiTextureType_AMBIENT, j, &file );
				M.texFiles[ 0 ].push_back( file.data ); 
				GLuint id;
				glGenTextures( 1, &id );
				M.texIDs[ 0 ].push_back( id );
				loadTexture( file.data, id );
			}
			material->Get( AI_MATKEY_COLOR_AMBIENT, coltemp );
			if ( ! ( coltemp.r == 0 && coltemp.g == 0 && coltemp.b ==0 ) )
			{
				M._ambient[ 0 ] = coltemp.r;
				M._ambient[ 1 ] = coltemp.g;
				M._ambient[ 2 ] = coltemp.b;
				M._ambient[ 3 ] = coltemp.a;
			}

			// Diffuse
			nbT = material->GetTextureCount( aiTextureType_DIFFUSE );
			if ( nbT > 0 )
			{
				M._hasDiffuseTextures = true;
			} 
			for ( int j = 0; j < nbT; j++ )
			{
				material->GetTexture( aiTextureType_DIFFUSE, j, &file );
				M.texFiles[ 1 ].push_back( file.data ); 
				GLuint id;
				glGenTextures( 1, &id );
				M.texIDs[ 1 ].push_back( id );
				loadTexture( file.data, id );
			}
			material->Get( AI_MATKEY_COLOR_DIFFUSE, coltemp );
			if ( ! ( coltemp.r == 0 && coltemp.g == 0 && coltemp.b == 0 ) )
			{
				M._diffuse[ 0 ] = coltemp.r;
				M._diffuse[ 1 ] = coltemp.g;
				M._diffuse[ 2 ] = coltemp.b;
				M._diffuse[ 3 ] = coltemp.a;
			}

			// Specular
			nbT = material->GetTextureCount( aiTextureType_SPECULAR );
			if ( nbT > 0 )
			{
				M._hasSpecularTextures = true;
			} 
			for ( int j = 0; j < nbT; j++ )
			{
				material->GetTexture( aiTextureType_SPECULAR, j, &file );
				M.texFiles[ 2 ].push_back( file.data ); 
				GLuint id;
				glGenTextures( 1, &id );
				M.texIDs[ 2 ].push_back( id );
				loadTexture( file.data, id );
			}
			material->Get( AI_MATKEY_COLOR_SPECULAR, coltemp );
			if ( ! ( coltemp.r == 0 && coltemp.g == 0 && coltemp.b == 0 ) )
			{
				M._specular[ 0 ] = coltemp.r;
				M._specular[ 1 ] = coltemp.g;
				M._specular[ 2 ] = coltemp.b;
				M._specular[ 3 ] = coltemp.a;
			}
			material->Get( AI_MATKEY_SHININESS, shininess );
			if ( shininess != 0.f )
			{
				M._shininess = shininess;
			}
		}

		// Store current mesh data
		_meshes.push_back( M );
	}

	// Creating the object's bounding box
	boundingBoxSide = max( boxMax[ 0 ] - boxMin[ 0 ], max( boxMax[ 1 ] - boxMin[ 1 ], boxMax[ 2 ] - boxMin[ 2 ] ) );
	for ( int k = 0; k < 3; k++ )
	{
		center[ k ] = 0.5 * ( boxMax[ k ] + boxMin[ k ] );
		boxMax[ k ] = center[ k ] + boundingBoxSide * 0.5;
		boxMin[ k ] = center[ k ] - boundingBoxSide * 0.5;
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool Mesh::chargerMesh( const string& filename )
{
	// Check if file exists
	ifstream fin( filename.c_str() );
	if ( ! fin.fail() )
	{
		fin.close();
	}
	else
	{
		cout << "Couldn't open file." << endl;

		return false;
	}

	_repository = Directory( filename );
	QString s( _repository.c_str() );
	QDir d( s );
	_repository = d.absolutePath().toStdString();

	// Load scene (use importer)
	scene = importer.ReadFile( filename, aiProcessPreset_TargetRealtime_MaxQuality );
	if ( ! scene )
	{
		cout << "Import failed." << endl;

		return false;
	}

	// Collects all the information about the meshes (vertices, normals, textures, materials, ...)
	InitFromScene( scene );
	cout << "Import scene succeeded.\n" << endl;

	return true;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Mesh::creerVBO()
{
	// Iterate through meshes
	for ( int i = 0; i < _meshes.size(); i++ )
	{
		// Create vertex and index buffers
		glGenBuffers( 1, &( _meshes[ i ]._vertexBuffer ) );
		glGenBuffers( 1, &( _meshes[ i ]._indexBuffer ) );
		
		// Initialize vertex buffer
		glBindBuffer( GL_ARRAY_BUFFER, _meshes[ i ]._vertexBuffer );
		// - allocate vertex buffer memory
		glBufferData( GL_ARRAY_BUFFER, sizeof( GLfloat ) * _meshes[ i ]._vertices.size() 
			+ sizeof( GLfloat ) * _meshes[ i ]._normals.size()
			+ sizeof( GLfloat ) * _meshes[ i ]._texCoords.size(), NULL, GL_STATIC_DRAW );
		// - send positions
		glBufferSubData( GL_ARRAY_BUFFER, 0, sizeof( GLfloat ) * _meshes[ i ]._vertices.size(), _meshes[ i ]._vertices.data() );
		// - send normals
		glBufferSubData( GL_ARRAY_BUFFER, sizeof( GLfloat ) * _meshes[ i ]._vertices.size(), sizeof( GLfloat ) * _meshes[ i ]._normals.size(), _meshes[ i ]._normals.data() );
		// - send texture coordinates
		glBufferSubData( GL_ARRAY_BUFFER, sizeof( GLfloat ) * _meshes[ i ]._vertices.size() 
			+ sizeof( GLfloat ) * _meshes[ i ]._normals.size(), sizeof( GLfloat ) * _meshes[ i ]._texCoords.size(), _meshes[ i ]._texCoords.data() );
		
		// Initialize index buffer
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _meshes[ i ]._indexBuffer );
		// - allocate index buffer memory
		glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( GLuint ) * _meshes[ i ]._indices.size(), _meshes[ i ]._indices.data(), GL_STATIC_DRAW );
	}
}

/******************************************************************************
 * Render part of mesh
 *
 * @pram pIndex part of mesh
 ******************************************************************************/
void Mesh::renderMesh( int pIndex )
{
	glEnable( GL_TEXTURE_2D );
	/*if (_meshes[ pIndex ]._hasAmbientTextures)
	{
		//cout << "_hasAmbientTextures mesh num" << pIndex << endl;
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D, _meshes[ pIndex ].texIDs[ 0 ][ 0 ] );
	}*/
	if ( _meshes[ pIndex ]._hasDiffuseTextures )
	{
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D, _meshes[ pIndex ].texIDs[ 1 ][ 0 ] );
	} 
	/*if ( _meshes[ pIndex ]._hasSpecularTextures )
	{
		//cout << "_hasSpecularTextures mesh num" << pIndex << endl;
		glActiveTexture( GL_TEXTURE2 );
		glBindTexture( GL_TEXTURE_2D, _meshes[ pIndex ].texIDs[ 2 ][ 0 ] );
	}*/
	
	glBindBuffer( GL_ARRAY_BUFFER, _meshes[ pIndex ]._vertexBuffer );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _meshes[ pIndex ]._indexBuffer );

	glEnableClientState( GL_VERTEX_ARRAY );
	glEnableClientState( GL_NORMAL_ARRAY );
	glEnableClientState( GL_TEXTURE_COORD_ARRAY );

	glVertexPointer( 3, GL_FLOAT, 0, 0 );
	glNormalPointer( GL_FLOAT, 0, BUFFER_OFFSET( sizeof( GLfloat ) * _meshes[ pIndex ]._vertices.size() ) ) ;
	glTexCoordPointer( 2, GL_FLOAT, 0, BUFFER_OFFSET( sizeof( GLfloat ) * _meshes[ pIndex ]._vertices.size() + sizeof( GLfloat ) * _meshes[ pIndex ]._normals.size() ) ) ;
	
	// Render mesh
	glDrawElements( _meshes[ pIndex ].mode, _meshes[ pIndex ]._indices.size(), GL_UNSIGNED_INT, 0 );
	
	glDisableClientState( GL_VERTEX_ARRAY );
	glDisableClientState( GL_NORMAL_ARRAY );
	glDisableClientState( GL_TEXTURE_COORD_ARRAY );
	
	glBindTexture( GL_TEXTURE_2D, 0 ); 
	
	glDisable( GL_TEXTURE_2D );
}

/******************************************************************************
 * Render scene (i.e. all meshes)
 ******************************************************************************/
void Mesh::render()
{
	// Save current matrix stack
	glPushMatrix();

	// Apply local transformations
	glRotatef( _rotation[ 0 ], _rotation[ 1 ], _rotation[ 2 ], _rotation[ 3 ] );
	glScalef( _scale, _scale, _scale );
	glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );

	// Configure OpenGL state
	glEnable( GL_CULL_FACE );

	// Use shader program and send uniforms
	glUseProgram( _shaderProgram );
	glUniform1i( glGetUniformLocation( _shaderProgram, "samplerd" ), 0 );
	// Iterate through meshes
	for ( int i = 0; i < _meshes.size(); i++ )
	{	
		if ( hasTexture( i ) )
		{
			glUniform1i( glGetUniformLocation( _shaderProgram, "hasTex" ), 1 );
		}
		else
		{
			glUniform1i( glGetUniformLocation( _shaderProgram, "hasTex" ), 0 );
		}
		glUniform3f( glGetUniformLocation( _shaderProgram, "lightPos" ), lightPos[ 0 ], lightPos[ 1 ], lightPos[ 2 ] );
		glUniform4f( glGetUniformLocation( _shaderProgram, "ambientLight" ), _meshes[ i ]._ambient[ 0 ], _meshes[ i ]._ambient[ 1 ], _meshes[ i ]._ambient[ 2 ], _meshes[i ] ._ambient[ 3 ] );
		glUniform4f( glGetUniformLocation( _shaderProgram, "specularColor" ), _meshes[ i ]._specular[ 0 ], _meshes[ i ]._specular[ 1 ], _meshes[ i ]._specular[ 2 ], _meshes[ i ]._specular[ 3 ] );
		glUniform1f( glGetUniformLocation( _shaderProgram, "shininess" ), _meshes[ i ]._shininess );
		
		// Render mesh
		renderMesh( i );
	}
	glUseProgram( 0 );

	// Configure OpenGL state
	glDisable( GL_CULL_FACE );
	
	// Restore current matrix stack
	glPopMatrix();
}

/******************************************************************************
 * Get the list of meshes
 *
 * @return the list of meshes
 ******************************************************************************/
const vector< oneMesh >& Mesh::getMeshes() const
{
	return _meshes;
}

/******************************************************************************
 * Get the number of meshes
 *
 * @return the number of meshes
 ******************************************************************************/
int Mesh::getNbMeshes()
{
	return _meshes.size();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Mesh::getAmbient( float tab[ 4 ], int i )
{
	tab[ 0 ] = _meshes[ i ]._ambient[ 0 ];
	tab[ 1 ] = _meshes[ i ]._ambient[ 1 ];
	tab[ 2 ] = _meshes[ i ]._ambient[ 2 ];
	tab[ 3 ] = _meshes[ i ]._ambient[ 3 ];
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Mesh::getDiffuse( float tab[ 4 ], int i )
{
	tab[ 0 ] = _meshes[ i ]._diffuse[ 0 ];
	tab[ 1 ] = _meshes[ i ]._diffuse[ 1 ];
	tab[ 2 ] = _meshes[ i ]._diffuse[ 2 ];
	tab[ 3 ] = _meshes[ i ]._diffuse[ 3 ];	
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Mesh::getSpecular( float tab[ 4 ], int i )
{
	tab[ 0 ] = _meshes[ i ]._specular[ 0 ];
	tab[ 1 ] = _meshes[ i ]._specular[ 1 ];
	tab[ 2 ] = _meshes[ i ]._specular[ 2 ];
	tab[ 3 ] = _meshes[ i ]._specular[ 3 ];
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Mesh::getShininess( float &s, int i )
{
	s = _meshes[ i ]._shininess;
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool Mesh::hasTexture( int i )
{
	return ( _meshes[ i ]._hasAmbientTextures || _meshes[ i ]._hasDiffuseTextures || _meshes[ i ]._hasSpecularTextures );
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Mesh::setLightPosition( float x, float y, float z )
{
	lightPos[ 0 ] = x;
	lightPos[ 1 ] = y;
	lightPos[ 2 ] = z;
}

/******************************************************************************
 * ...
 ******************************************************************************/
float Mesh::getScaleFactor()
{
	return 1.05 * boundingBoxSide;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Mesh::getTranslationFactors( float translation[ 3 ])
{
	for ( int k = 0; k < 3; k++ )
	{
		translation[ k ] = center[ k ];
	}
}

/******************************************************************************
 * Get the 3D model filename to load
 *
 * @return the 3D model filename to load
 ******************************************************************************/
const char* Mesh::get3DModelFilename() const
{
	return _modelFilename.c_str();
}

/******************************************************************************
 * Set the 3D model filename to load
 *
 * @param pFilename the 3D model filename to load
 ******************************************************************************/
void Mesh::set3DModelFilename( const char* pFilename )
{
	_modelFilename = pFilename;
}

/******************************************************************************
 * Get the translation
 *
 * @param pX the translation on x axis
 * @param pY the translation on y axis
 * @param pZ the translation on z axis
 ******************************************************************************/
void Mesh::getTranslation( float& pX, float& pY, float& pZ ) const
{
	pX = _translation[ 0 ];
	pY = _translation[ 1 ];
	pZ = _translation[ 2 ];
}

/******************************************************************************
 * Set the translation
 *
 * @param pX the translation on x axis
 * @param pY the translation on y axis
 * @param pZ the translation on z axis
 ******************************************************************************/
void Mesh::setTranslation( float pX, float pY, float pZ )
{
	_translation[ 0 ] = pX;
	_translation[ 1 ] = pY;
	_translation[ 2 ] = pZ;
}

/******************************************************************************
 * Get the rotation
 *
 * @param pAngle the rotation angle (in degree)
 * @param pX the x component of the rotation vector
 * @param pY the y component of the rotation vector
 * @param pZ the z component of the rotation vector
 ******************************************************************************/
void Mesh::getRotation( float& pAngle, float& pX, float& pY, float& pZ ) const
{
	pAngle = _rotation[ 0 ];
	pX = _rotation[ 1 ];
	pY = _rotation[ 2 ];
	pZ = _rotation[ 3 ];
}

/******************************************************************************
 * Set the rotation
 *
 * @param pAngle the rotation angle (in degree)
 * @param pX the x component of the rotation vector
 * @param pY the y component of the rotation vector
 * @param pZ the z component of the rotation vector
 ******************************************************************************/
void Mesh::setRotation( float pAngle, float pX, float pY, float pZ )
{
	_rotation[ 0 ] = pAngle;
	_rotation[ 1 ] = pX;;
	_rotation[ 2 ] = pY;;
	_rotation[ 3 ] = pZ;;
}

/******************************************************************************
 * Get the uniform scale
 *
 * @param pValue the uniform scale
 ******************************************************************************/
void Mesh::getScale( float& pValue ) const
{
	pValue = _scale;
}

/******************************************************************************
 * Set the uniform scale
 *
 * @param pValue the uniform scale
 ******************************************************************************/
void Mesh::setScale( float pValue )
{
	_scale = pValue;
}
