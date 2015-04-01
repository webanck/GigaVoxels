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
{
	lightPos[ 0 ] = 1;
	lightPos[ 1 ] = 1;
	lightPos[ 2 ] = 1;

	program = p;

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
	QDirIterator it( QDir( QString( Dir.c_str() ) ), QDirIterator::Subdirectories );
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

	QImage img = QGLWidget::convertToGLFormat( QImage( f.c_str() ) );

	glBindTexture( GL_TEXTURE_2D, id );
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, img.width(), img.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, img.bits() );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
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

	/**Vertices, Normals and Textures**/
	for ( int i = 0; i < scene->mNumMeshes; i++ )
	{
		const aiMesh* mesh = scene->mMeshes[ i ];
		oneMesh M;
		M.hasATextures = false;
		M.hasDTextures = false;
		M.hasSTextures = false;
		M.shininess = 20;
		for ( int a = 0; a < 3; a++ )
		{
			M.ambient[ a ] = 0.5;
			M.diffuse[ a ] = 0;
			M.specular[ a ] = 0.75;
		}	
		M.ambient[ 3 ] = 1.0;
		M.diffuse[ 3 ] = 1.0;
		M.specular[ 3 ] = 1.0;

		for ( int j = 0; j < mesh->mNumVertices; j++ )
		{
			if ( mesh->HasPositions() )
			{
				const aiVector3D* pos = &( mesh->mVertices[ j ] );
				M.Vertices.push_back( pos->x );
				if ( boxMax[ 0 ] < pos->x )
				{
					boxMax[ 0 ] = pos->x;
				}
				if ( boxMin[ 0 ] > pos->x )
				{
					boxMin[ 0 ] = pos->x;
				}
				M.Vertices.push_back( pos->y );
				if ( boxMax[ 1 ] < pos->y )
				{
					boxMax[ 1 ] = pos->y;
				}
				if ( boxMin[ 1 ] > pos->y )
				{
					boxMin[ 1 ] = pos->y;
				}
				M.Vertices.push_back( pos->z );
				if ( boxMax[ 2 ] < pos->z )
				{
					boxMax[ 2 ] = pos->z;
				}
				if ( boxMin[ 2 ] > pos->z )
				{
					boxMin[ 2 ] = pos->z;
				}
			}
			if ( mesh->HasNormals() )
			{
				const aiVector3D* normal = &( mesh->mNormals[ j ] );
				M.Normals.push_back( normal->x );
				M.Normals.push_back( normal->y );
				M.Normals.push_back( normal->z );
			}
			if ( mesh->HasTextureCoords( 0 ) )
			{
				M.Textures.push_back( mesh->mTextureCoords[ 0 ][ j ].x );
				M.Textures.push_back( mesh->mTextureCoords[ 0 ][ j ].y );
			}
		}

		/**Indices**/
		for ( int k = 0 ; k < mesh->mNumFaces ; k++ )
		{
			const aiFace& Face = mesh->mFaces[ k ];
			//if ( Face.mNumIndices == 3 )
			//{
			M.mode = GL_TRIANGLES;
			M.Indices.push_back( Face.mIndices[ 0 ] );
			M.Indices.push_back( Face.mIndices[ 1 ] );
			M.Indices.push_back( Face.mIndices[ 2 ] );
			//} 
		}

		/**Materials**/
		if ( scene->HasMaterials() )
		{
			materialIndex = mesh->mMaterialIndex;
			aiMaterial* material = scene->mMaterials[ materialIndex ];

			nbT = material->GetTextureCount( aiTextureType_AMBIENT );
			if ( nbT > 0 )
			{
				M.hasATextures = true;
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
				M.ambient[ 0 ] = coltemp.r;
				M.ambient[ 1 ] = coltemp.g;
				M.ambient[ 2 ] = coltemp.b;
				M.ambient[ 3 ] = coltemp.a;
			}

			nbT = material->GetTextureCount( aiTextureType_DIFFUSE );
			if ( nbT > 0 )
			{
				M.hasDTextures = true;
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
				M.diffuse[ 0 ] = coltemp.r;
				M.diffuse[ 1 ] = coltemp.g;
				M.diffuse[ 2 ] = coltemp.b;
				M.diffuse[ 3 ] = coltemp.a;
			}

			nbT = material->GetTextureCount( aiTextureType_SPECULAR );
			if ( nbT > 0 )
			{
				M.hasSTextures = true;
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
				M. specular[ 0 ] = coltemp.r;
				M.specular[ 1 ] = coltemp.g;
				M.specular[ 2 ] = coltemp.b;
				M.specular[ 3 ] = coltemp.a;
			}
			material->Get( AI_MATKEY_SHININESS, shininess );
			if ( shininess != 0.f )
			{
				M.shininess = shininess;
			}
		}
		meshes.push_back( M );
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
	Dir = Directory( filename );
	scene = importer.ReadFile( filename, aiProcessPreset_TargetRealtime_MaxQuality );
	QString s( Dir.c_str() );
	QDir d( s );
	Dir = d.absolutePath().toStdString();
	if ( ! scene )
	{
		cout << "Import failed." << endl;

		return false;
	}

	InitFromScene( scene );
	cout << "Import scene succeeded.\n" << endl;

	return true;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Mesh::creerVBO()
{
	for ( int i = 0; i < meshes.size(); i++ )
	{
		glGenBuffers( 1, &( meshes[ i ].VB ) );
		glGenBuffers( 1, &( meshes[ i ].IB ) );
		glBindBuffer( GL_ARRAY_BUFFER, meshes[ i ].VB ); 
		glBufferData( GL_ARRAY_BUFFER, sizeof( GLfloat ) * meshes[ i ].Vertices.size() 
			+ sizeof( GLfloat ) * meshes[ i ].Normals.size()
			+ sizeof( GLfloat ) * meshes[ i ].Textures.size(), NULL, GL_STATIC_DRAW );
		glBufferSubData( GL_ARRAY_BUFFER, 0, sizeof( GLfloat ) * meshes[ i ].Vertices.size(), meshes[ i ].Vertices.data() );
		glBufferSubData( GL_ARRAY_BUFFER, sizeof( GLfloat ) * meshes[ i ].Vertices.size(), sizeof( GLfloat ) * meshes[ i ].Normals.size(), meshes[ i ].Normals.data() );
		glBufferSubData( GL_ARRAY_BUFFER, sizeof( GLfloat ) * meshes[ i ].Vertices.size() 
			+ sizeof( GLfloat ) * meshes[ i ].Normals.size(), sizeof( GLfloat ) * meshes[ i ].Textures.size(), meshes[ i ].Textures.data() );
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, meshes[ i ].IB );
		glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( GLuint ) * meshes[ i ].Indices.size(), meshes[ i ].Indices.data(), GL_STATIC_DRAW );
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Mesh::renderMesh(int i)
{
	glEnable( GL_TEXTURE_2D );
	/*if (meshes[ i ].hasATextures)
	{
		//cout << "hasATextures mesh num" << i << endl;
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D, meshes[ i ].texIDs[ 0 ][ 0 ] );
	}*/
	if ( meshes[ i ].hasDTextures )
	{
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D, meshes[ i ].texIDs[ 1 ][ 0 ] );
	} 
	/*if ( meshes[ i ].hasSTextures )
	{
		//cout << "hasSTextures mesh num" << i << endl;
		glActiveTexture( GL_TEXTURE2 );
		glBindTexture( GL_TEXTURE_2D, meshes[ i ].texIDs[ 2 ][ 0 ] );
	}*/
	glBindBuffer( GL_ARRAY_BUFFER, meshes[ i ].VB );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, meshes[ i ].IB );

	glEnableClientState( GL_VERTEX_ARRAY );
	glEnableClientState( GL_NORMAL_ARRAY );
	glEnableClientState( GL_TEXTURE_COORD_ARRAY );
	glVertexPointer( 3, GL_FLOAT, 0, 0 );
	glNormalPointer( GL_FLOAT, 0, BUFFER_OFFSET( sizeof( GLfloat ) * meshes[ i ].Vertices.size() ) ) ;
	glTexCoordPointer( 2, GL_FLOAT, 0, BUFFER_OFFSET( sizeof( GLfloat ) * meshes[ i ].Vertices.size() + sizeof( GLfloat ) * meshes[ i ].Normals.size() ) ) ;
	glDrawElements( meshes[ i ].mode, meshes[ i ].Indices.size(), GL_UNSIGNED_INT, 0 );
	glDisableClientState( GL_VERTEX_ARRAY );
	glDisableClientState( GL_NORMAL_ARRAY );
	glDisableClientState( GL_TEXTURE_COORD_ARRAY );
	glBindTexture( GL_TEXTURE_2D, 0 ); 
	glDisable( GL_TEXTURE_2D );
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Mesh::render()
{
	glEnable( GL_CULL_FACE );
	glUseProgram( program );
	glUniform1i( glGetUniformLocation( program, "samplerd" ), 0 );
	for ( int i = 0; i < meshes.size(); i++ )
	{	
		if ( hasTexture( i ) )
		{
			glUniform1i( glGetUniformLocation( program, "hasTex" ), 1 );
		}
		else
		{
			glUniform1i( glGetUniformLocation( program, "hasTex" ), 0 );
		}
		glUniform3f( glGetUniformLocation( program, "lightPos" ), lightPos[ 0 ], lightPos[ 1 ], lightPos[ 2 ] );
		glUniform4f( glGetUniformLocation( program, "ambientLight" ), meshes[ i ].ambient[ 0 ], meshes[ i ].ambient[ 1 ], meshes[ i ].ambient[ 2 ], meshes[i ] .ambient[ 3 ] );
		glUniform4f( glGetUniformLocation( program, "specularColor" ), meshes[ i ].specular[ 0 ], meshes[ i ].specular[ 1 ], meshes[ i ].specular[ 2 ], meshes[ i ].specular[ 3 ] );
		glUniform1f( glGetUniformLocation( program, "shininess" ), meshes[ i ].shininess );
		renderMesh( i );				
	}
	glDisable( GL_CULL_FACE );
	glUseProgram( 0 );
}

/******************************************************************************
 * ...
 ******************************************************************************/
vector<oneMesh> Mesh::getMeshes()
{
	return meshes;
}

/******************************************************************************
 * ...
 ******************************************************************************/
int Mesh::getNumberOfMeshes()
{
	return meshes.size();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Mesh::getAmbient( float tab[ 4 ], int i )
{
	tab[ 0 ] = meshes[ i ].ambient[ 0 ];
	tab[ 1 ] = meshes[ i ].ambient[ 1 ];
	tab[ 2 ] = meshes[ i ].ambient[ 2 ];
	tab[ 3 ] = meshes[ i ].ambient[ 3 ];
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Mesh::getDiffuse(float tab[4], int i)
{
	tab[ 0 ] = meshes[ i ].diffuse[ 0 ];
	tab[ 1 ] = meshes[ i ].diffuse[ 1 ];
	tab[ 2 ] = meshes[ i ].diffuse[ 2 ];
	tab[ 3 ] = meshes[ i ].diffuse[ 3 ];	
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Mesh::getSpecular( float tab[ 4 ], int i )
{
	tab[ 0 ] = meshes[ i ].specular[ 0 ];
	tab[ 1 ] = meshes[ i ].specular[ 1 ];
	tab[ 2 ] = meshes[ i ].specular[ 2 ];
	tab[ 3 ] = meshes[ i ].specular[ 3 ];
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Mesh::getShininess( float &s, int i )
{
	s = meshes[ i ].shininess;
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool Mesh::hasTexture( int i )
{
	return ( meshes[ i ].hasATextures || meshes[ i ].hasDTextures || meshes[ i ].hasSTextures );
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
