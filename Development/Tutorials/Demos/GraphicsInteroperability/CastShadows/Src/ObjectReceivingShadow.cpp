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

#include "ObjectReceivingShadow.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaSpace
#include <GvCore/Array3D.h>
#include <GvCore/Array3DGPULinear.h>
#include <GvCore/GvError.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * ...
 */
#define BUFFER_OFFSET( a ) ( (char*)NULL + ( a ) )

/**
 * ...
 */
float vertices[ 12 ] =
{
	15.0, 15.0, -3.0,
	-15.0, 15.0, -3.0, 
	-15.0, -15.0, -3.0,
	15.0, -15.0, -3.0
};

/**
 * ...
 */
float normals[ 12 ] =
{
	0.0, 0.0, 1.0,
	0.0, 0.0, 1.0,
	0.0, 0.0, 1.0,
	0.0, 0.0, 1.0
};

/**
 * ...
 */
GLuint indices[ 4 ] = { 0, 1, 2, 3 };

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
ObjectReceivingShadow::ObjectReceivingShadow()
{
	lightPos[0] = 1;
	lightPos[1] = 1;
	lightPos[2] = 1;

	object = NULL;
	loadedObject = false;//************set to true if the model should be loaded from an OBJ file*************//
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
ObjectReceivingShadow::~ObjectReceivingShadow()
{
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ObjectReceivingShadow::init()
{
	QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	QString shaderRepository = dataRepository + QDir::separator() + QString( "Shaders" );
	QString vertexShaderFilename = shaderRepository + QDir::separator() + QString( "CastShadows" ) + QDir::separator() + QString( "objectReceivingShadowVert.glsl" );
	QString fragmentShaderFilename = shaderRepository + QDir::separator() + QString( "CastShadows" ) + QDir::separator() + QString( "objectReceivingShadowFrag.glsl" );
	QString meshRepository = dataRepository + QDir::separator() + QString( "3DModels" );
	QString meshFilename = meshRepository + QDir::separator() + QString( "Butterfly" ) + QDir::separator() + QString( "Butterfly.obj" );

	// Shader initialization
	vshader = useShader( GL_VERTEX_SHADER, vertexShaderFilename.toLatin1().constData() );
	fshader = useShader( GL_FRAGMENT_SHADER, fragmentShaderFilename.toLatin1().constData() );
	program = glCreateProgram();
	glAttachShader( program, vshader );
	glAttachShader( program, fshader );
	glLinkProgram( program );
	linkStatus( program );
	GV_CHECK_GL_ERROR();		

	// VBO initialization
	if ( loadedObject )
	{
		object = new Mesh( program );
		object->chargerMesh( meshFilename.toStdString() );//as an example
		object->creerVBO();
	}
	else
	{
		glGenBuffers( 1, &idVBO );
		glGenBuffers( 1, &idIndices );
		glBindBuffer( GL_ARRAY_BUFFER, idVBO );
		glBufferData( GL_ARRAY_BUFFER, sizeof( vertices ) + sizeof( normals ), &vertices[ 0 ], GL_STATIC_DRAW );
		glBufferSubData( GL_ARRAY_BUFFER, 0, sizeof( vertices ), vertices );
		glBufferSubData( GL_ARRAY_BUFFER, sizeof( vertices ), sizeof( normals ),normals );
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, idIndices );
		glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( indices ), indices, GL_STATIC_DRAW );
		GV_CHECK_GL_ERROR();

		// Texture buffer arrays linked to GigaVoxels
		glGenTextures( 1, &_childArrayTBO );
		glBindTexture( GL_TEXTURE_BUFFER, _childArrayTBO );
		
		// Attach the storage of buffer object to buffer texture
		glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, childBufferName );
		GV_CHECK_GL_ERROR();
		glGenTextures( 1, &_dataArrayTBO );
		glBindTexture( GL_TEXTURE_BUFFER, _dataArrayTBO );
		
		// Attach the storage of buffer object to buffer texture
		glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, dataBufferName );
	}
	GV_CHECK_GL_ERROR();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ObjectReceivingShadow::render()
{
	//retrieveing the object model matrix
	float objectModelMatrix[ 16 ];
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();
	glScalef( 40, 40, 40 );
	glTranslatef( 2.0, -1.0, -10.5 );
	glGetFloatv( GL_MODELVIEW_MATRIX, objectModelMatrix );
	glPopMatrix();

	// Start of rendering process
	glPushMatrix();
	glScalef( 40, 40, 40 );
	glTranslatef( 2.0, -1.0, -10.5 );
	//uniform info
	glProgramUniform3fEXT( program, glGetUniformLocation( program, "lightPos" ), lightPos[ 0 ], lightPos[ 1 ], lightPos[ 2 ] );
	glProgramUniform3fEXT( program, glGetUniformLocation( program, "worldLight" ), worldLight[ 0 ], worldLight[ 1 ], worldLight[ 2 ] );
	glProgramUniformMatrix4fvEXT( program, glGetUniformLocation( program, "gvModelMatrix" ), 1, GL_FALSE, modelMatrix );
	glProgramUniformMatrix4fvEXT( program, glGetUniformLocation( program, "objectModelMatrix" ), 1, GL_FALSE, objectModelMatrix );
	glProgramUniform3uiEXT( program, glGetUniformLocation( program, "uBrickCacheSize" ), brickCacheSize[ 0 ], brickCacheSize[ 1 ], brickCacheSize[ 2 ] );
	glProgramUniform3fEXT( program, glGetUniformLocation( program, "uBrickPoolResInv" ), brickPoolResInv[ 0 ], brickPoolResInv[ 1 ], brickPoolResInv[ 2 ] );
	glProgramUniform1uiEXT( program, glGetUniformLocation( program, "uMaxDepth" ), maxDepth);
	GV_CHECK_GL_ERROR();
	glProgramUniform1iEXT( program, glGetUniformLocation( program, "uNodePoolChildArray" ), 3 );
	glProgramUniform1iEXT( program, glGetUniformLocation( program, "uNodePoolDataArray" ), 4 );
	GV_CHECK_GL_ERROR();
	// Using program
	glUseProgram( program );
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_3D, texBufferName );
	glUniform1i( glGetUniformLocation( program, "uDataPool" ), 0);
	glBindImageTextureEXT(3, _childArrayTBO, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI );
	glBindImageTextureEXT(4, _dataArrayTBO, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI );
	// Rendering
	if ( loadedObject )
	{
		object->render();
	}
	else
	{
		glUniform3f( glGetUniformLocation( program, "ambientLight" ), 0.75, 0.75, 0.75 );
		glUniform3f( glGetUniformLocation( program, "specularColor" ), 0.7, 0.7, 0.7 );
		glUniform1f( glGetUniformLocation( program, "shininess" ), 20 );
		glUniform1i( glGetUniformLocation( program, "hasTex" ), 0 );
		glBindBuffer( GL_ARRAY_BUFFER, idVBO );
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, idIndices );
		glEnableClientState( GL_VERTEX_ARRAY );
		glEnableClientState( GL_NORMAL_ARRAY );
		glVertexPointer( 3, GL_FLOAT, 0, 0 );
		glNormalPointer( GL_FLOAT, 0, BUFFER_OFFSET( sizeof( vertices ) ) );
		glDrawElements( GL_QUADS, 4, GL_UNSIGNED_INT, 0 );	// 4 is the number of indices
		glDisableClientState( GL_VERTEX_ARRAY );
		glDisableClientState( GL_NORMAL_ARRAY );
		GV_CHECK_GL_ERROR();
		glUseProgram( 0 );
	}
	glPopMatrix();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ObjectReceivingShadow::setLightPosition( float x, float y, float z )
{
	lightPos[ 0 ] = x;
	lightPos[ 1 ] = y;
	lightPos[ 2 ] = z;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ObjectReceivingShadow::setBrickCacheSize( unsigned int x, unsigned int y, unsigned int z )
{
	brickCacheSize[ 0 ] = x;
	brickCacheSize[ 1 ] = y;
	brickCacheSize[ 2 ] = z;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ObjectReceivingShadow::setBrickPoolResInv( float x, float y, float z )
{
	brickPoolResInv[ 0 ] = x;
	brickPoolResInv[ 1 ] = y;
	brickPoolResInv[ 2 ] = z;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ObjectReceivingShadow::setMaxDepth( unsigned int v )
{
	maxDepth = v;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ObjectReceivingShadow::setVolTreeChildArray( GvCore::Array3DGPULinear< uint >* v, GLint id )
{
	volTreeChildArray = v;
	childBufferName = id;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ObjectReceivingShadow::setVolTreeDataArray( GvCore::Array3DGPULinear< uint >* v, GLint id )
{
	volTreeDataArray = v;
	dataBufferName = id;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ObjectReceivingShadow::setModelMatrix( float m00, float m01, float m02, float m03,
	float m10, float m11, float m12, float m13,
	float m20, float m21, float m22, float m23,
	float m30, float m31, float m32, float m33 )
{
	modelMatrix[ 0 ] = m00;
	modelMatrix[ 1 ] = m01;
	modelMatrix[ 2 ] = m02;
	modelMatrix[ 3 ] = m03;

	modelMatrix[ 4 ] = m10;
	modelMatrix[ 5 ] = m11;
	modelMatrix[ 6 ] = m12;
	modelMatrix[ 7 ] = m13;

	modelMatrix[ 8 ] = m20;
	modelMatrix[ 9 ] = m21;
	modelMatrix[ 10 ] = m22;
	modelMatrix[ 11 ] = m23;

	modelMatrix[ 12 ] = m30;
	modelMatrix[ 13 ] = m31;
	modelMatrix[ 14 ] = m32;
	modelMatrix[ 15 ] = m33;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ObjectReceivingShadow::setWorldLight( float x, float y, float z )
{
	worldLight[ 0 ] = x;
	worldLight[ 1 ] = y;
	worldLight[ 2 ] = z;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ObjectReceivingShadow::setTexBufferName( GLint v )
{
	texBufferName = v;
}
