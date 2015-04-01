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
 * Default plan vertices
 */
float vertices[ 12 ] =
{
	15.0, 15.0, -3.0,
	-15.0, 15.0, -3.0, 
	-15.0, -15.0, -3.0,
	15.0, -15.0, -3.0
};

/**
 * Default plan normals
 */
float normals[ 12 ] =
{
	0.0, 0.0, 1.0,
	0.0, 0.0, 1.0,
	0.0, 0.0, 1.0,
	0.0, 0.0, 1.0
};

/**
 * Default plan indices
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

	_lightPos[ 0 ] = 1;
	_lightPos[ 1 ] = 1;
	_lightPos[ 2 ] = 1;

	_object = NULL;
	_loadedObject = false;//************set to true if the model should be loaded from an OBJ file*************//
	//_loadedObject = true;//************set to true if the model should be loaded from an OBJ file*************//
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
ObjectReceivingShadow::~ObjectReceivingShadow()
{
	delete _object;
	_object = NULL;

	if ( _vertexBuffer )
	{
		glDeleteBuffers( 1, &_vertexBuffer );
	}
	if ( _indexBuffer )
	{
		glDeleteBuffers( 1, &_indexBuffer );
	}
	if ( _childArrayTexture )
	{
		glDeleteTextures( 1, &_childArrayTexture );
	}
	if ( _dataArrayTexture )
	{
		glDeleteTextures( 1, &_dataArrayTexture );
	}
}

/******************************************************************************
 * Initialize
 ******************************************************************************/
void ObjectReceivingShadow::init()
{
	// Data repository
	QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	
	// Shader repository
	QString shaderRepository = dataRepository + QDir::separator() + QString( "Shaders" );
	QString vertexShaderFilename = shaderRepository + QDir::separator() + QString( "GvCastShadows" ) + QDir::separator() + QString( "objectReceivingShadowVert.glsl" );
	QString fragmentShaderFilename = shaderRepository + QDir::separator() + QString( "GvCastShadows" ) + QDir::separator() + QString( "objectReceivingShadowFrag.glsl" );

	/*
	// 3D model repository
	QString meshRepository = dataRepository + QDir::separator() + QString( "3DModels" );
	QString meshFilename = meshRepository + QDir::separator() + QString( "Butterfly" ) + QDir::separator() + QString( "Butterfly.obj" );
	*/
	if ( _loadedObject )
	{
		assert( ! _modelFilename.empty() );
	}

	// Shader initialization
	_vertexShader = useShader( GL_VERTEX_SHADER, vertexShaderFilename.toLatin1().constData() );
	_fragmentShader = useShader( GL_FRAGMENT_SHADER, fragmentShaderFilename.toLatin1().constData() );
	_shaderProgram = glCreateProgram();
	glAttachShader( _shaderProgram, _vertexShader );
	glAttachShader( _shaderProgram, _fragmentShader );
	glLinkProgram( _shaderProgram );
	linkStatus( _shaderProgram );
	GV_CHECK_GL_ERROR();		

	// VBO initialization
	if ( _loadedObject )
	{
		//_modelFilename = meshFilename.toStdString();

		_object = new Mesh( _shaderProgram );
		_object->chargerMesh( _modelFilename );//as an example
		_object->creerVBO();
	}
	else
	{
		// Create vertex and index buffers
		glGenBuffers( 1, &_vertexBuffer );
		glGenBuffers( 1, &_indexBuffer );
		
		// Initialize vertex buffer
		glBindBuffer( GL_ARRAY_BUFFER, _vertexBuffer );
		// - allocate vertex buffer memory
		glBufferData( GL_ARRAY_BUFFER, sizeof( vertices ) + sizeof( normals ), &vertices[ 0 ], GL_STATIC_DRAW );
		// - send positions
		glBufferSubData( GL_ARRAY_BUFFER, 0, sizeof( vertices ), vertices );
		// - send normals
		glBufferSubData( GL_ARRAY_BUFFER, sizeof( vertices ), sizeof( normals ), normals );

		// Initialize index buffer
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _indexBuffer );
		// - allocate index buffer memory
		glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( indices ), indices, GL_STATIC_DRAW );
		GV_CHECK_GL_ERROR();

		// Texture buffer array linked to GigaSpace
		glGenTextures( 1, &_childArrayTexture );
		glBindTexture( GL_TEXTURE_BUFFER, _childArrayTexture );
		// - attach the storage of buffer object to buffer texture
		glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, _nodePoolChildArrayTextureBuffer );
		GV_CHECK_GL_ERROR();
		
		// Texture buffer array linked to GigaSpace
		glGenTextures( 1, &_dataArrayTexture );
		glBindTexture( GL_TEXTURE_BUFFER, _dataArrayTexture );
		// - attach the storage of buffer object to buffer texture
		glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, _nodePoolDataArrayTextureBuffer );
		GV_CHECK_GL_ERROR();
	}
}

/******************************************************************************
 * Render
 ******************************************************************************/
void ObjectReceivingShadow::render()
{
	//retrieveing the object model matrix
	float objectModelMatrix[ 16 ];
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();
	//glScalef( 4, 4, 4 );
	//glTranslatef( 2.0, -1.0, 0.5 );
	glRotatef( _rotation[ 0 ], _rotation[ 1 ], _rotation[ 2 ], _rotation[ 3 ] );
	glScalef( _scale, _scale, _scale );
	glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );
	glGetFloatv( GL_MODELVIEW_MATRIX, objectModelMatrix );
	glPopMatrix();

	// Start of rendering process
	glPushMatrix();
	//glScalef( 4, 4, 4 );
	//glTranslatef( 2.0, -1.0, 0.5 );
	glRotatef( _rotation[ 0 ], _rotation[ 1 ], _rotation[ 2 ], _rotation[ 3 ] );
	glScalef( _scale, _scale, _scale );
	glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );
	
	//uniform info
	glProgramUniform3fEXT( _shaderProgram, glGetUniformLocation( _shaderProgram, "lightPos" ), _lightPos[ 0 ], _lightPos[ 1 ], _lightPos[ 2 ] );
	glProgramUniform3fEXT( _shaderProgram, glGetUniformLocation( _shaderProgram, "worldLight" ), _worldLight[ 0 ], _worldLight[ 1 ], _worldLight[ 2 ] );
	glProgramUniformMatrix4fvEXT( _shaderProgram, glGetUniformLocation( _shaderProgram, "gvModelMatrix" ), 1, GL_FALSE, _modelMatrix );
	glProgramUniformMatrix4fvEXT( _shaderProgram, glGetUniformLocation( _shaderProgram, "objectModelMatrix" ), 1, GL_FALSE, objectModelMatrix );
	glProgramUniform3uiEXT( _shaderProgram, glGetUniformLocation( _shaderProgram, "uBrickCacheSize" ), _brickCacheSize[ 0 ], _brickCacheSize[ 1 ], _brickCacheSize[ 2 ] );
	glProgramUniform3fEXT( _shaderProgram, glGetUniformLocation( _shaderProgram, "uBrickPoolResInv" ), _brickPoolResInv[ 0 ], _brickPoolResInv[ 1 ], _brickPoolResInv[ 2 ] );
	glProgramUniform1uiEXT( _shaderProgram, glGetUniformLocation( _shaderProgram, "uMaxDepth" ), _maxDepth);
	GV_CHECK_GL_ERROR();
	glProgramUniform1iEXT( _shaderProgram, glGetUniformLocation( _shaderProgram, "uNodePoolChildArray" ), 3 );
	glProgramUniform1iEXT( _shaderProgram, glGetUniformLocation( _shaderProgram, "uNodePoolDataArray" ), 4 );
	GV_CHECK_GL_ERROR();
	
	// Using program
	glUseProgram( _shaderProgram );
	
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_3D, _dataPoolTexture );
	glUniform1i( glGetUniformLocation( _shaderProgram, "uDataPool" ), 0);
	glBindImageTextureEXT( 3, _childArrayTexture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI );
	glBindImageTextureEXT( 4, _dataArrayTexture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI );

	// Rendering
	if ( _loadedObject )
	{
		_object->render();
	}
	else
	{
		glUniform3f( glGetUniformLocation( _shaderProgram, "ambientLight" ), 0.75, 0.75, 0.75 );
		glUniform3f( glGetUniformLocation( _shaderProgram, "specularColor" ), 0.7, 0.7, 0.7 );
		glUniform1f( glGetUniformLocation( _shaderProgram, "shininess" ), 20 );
		glUniform1i( glGetUniformLocation( _shaderProgram, "hasTex" ), 0 );
		
		glBindBuffer( GL_ARRAY_BUFFER, _vertexBuffer );
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _indexBuffer );
		
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
 * Set light position
 *
 * @param x ...
 * @param y ...
 * @param z ...
 ******************************************************************************/
void ObjectReceivingShadow::setLightPosition( float x, float y, float z )
{
	_lightPos[ 0 ] = x;
	_lightPos[ 1 ] = y;
	_lightPos[ 2 ] = z;
}

/******************************************************************************
 * Set brick cache size
 *
 * @param x ...
 * @param y ...
 * @param z ...
 ******************************************************************************/
void ObjectReceivingShadow::setBrickCacheSize( unsigned int x, unsigned int y, unsigned int z )
{
	_brickCacheSize[ 0 ] = x;
	_brickCacheSize[ 1 ] = y;
	_brickCacheSize[ 2 ] = z;
}

/******************************************************************************
 * Set brick pool resolution inverse
 *
 * @param x ...
 * @param y ...
 * @param z ...
 ******************************************************************************/
void ObjectReceivingShadow::setBrickPoolResInv( float x, float y, float z )
{
	_brickPoolResInv[ 0 ] = x;
	_brickPoolResInv[ 1 ] = y;
	_brickPoolResInv[ 2 ] = z;
}

/******************************************************************************
 * Set max depth
 *
 * @param x ...
 ******************************************************************************/
void ObjectReceivingShadow::setMaxDepth( unsigned int v )
{
	_maxDepth = v;
}

/******************************************************************************
 * Set the data structure's node pool's child array (i.e. the octree)
 *
 * @param v ...
 * @param id ...
 ******************************************************************************/
void ObjectReceivingShadow::setVolTreeChildArray( GvCore::Array3DGPULinear< uint >* v, GLint id )
{
	_nodePoolChildArray = v;
	_nodePoolChildArrayTextureBuffer = id;
}

/******************************************************************************
 * Set the data structure's node pool's data array (i.e. addresses of brick in cache)
 *
 * @param v ...
 * @param id ...
 ******************************************************************************/
void ObjectReceivingShadow::setVolTreeDataArray( GvCore::Array3DGPULinear< uint >* v, GLint id )
{
	_nodePoolDataArray = v;
	_nodePoolDataArrayTextureBuffer = id;
}

/******************************************************************************
 * Set the model matrix
 *
 * @param m ...
 * @param m ...
 * @param m ...
 * @param m ...
 * @param m ...
 * @param m ...
 * @param m ...
 * @param m ...
 * @param m ...
 * @param m ...
 * @param m ...
 * @param m ...
 * @param m ...
 * @param m ...
 * @param m ...
 * @param m ...
 ******************************************************************************/
void ObjectReceivingShadow::setModelMatrix( float m00, float m01, float m02, float m03,
	float m10, float m11, float m12, float m13,
	float m20, float m21, float m22, float m23,
	float m30, float m31, float m32, float m33 )
{
	_modelMatrix[ 0 ] = m00;
	_modelMatrix[ 1 ] = m01;
	_modelMatrix[ 2 ] = m02;
	_modelMatrix[ 3 ] = m03;

	_modelMatrix[ 4 ] = m10;
	_modelMatrix[ 5 ] = m11;
	_modelMatrix[ 6 ] = m12;
	_modelMatrix[ 7 ] = m13;

	_modelMatrix[ 8 ] = m20;
	_modelMatrix[ 9 ] = m21;
	_modelMatrix[ 10 ] = m22;
	_modelMatrix[ 11 ] = m23;

	_modelMatrix[ 12 ] = m30;
	_modelMatrix[ 13 ] = m31;
	_modelMatrix[ 14 ] = m32;
	_modelMatrix[ 15 ] = m33;
}

/******************************************************************************
 * Set light position in world coordinate system
 *
 * @param x ...
 * @param y ...
 * @param z ...
 ******************************************************************************/
void ObjectReceivingShadow::setWorldLight( float x, float y, float z )
{
	_worldLight[ 0 ] = x;
	_worldLight[ 1 ] = y;
	_worldLight[ 2 ] = z;
}

/******************************************************************************
 * ...
 *
 * @param v ...
 ******************************************************************************/
void ObjectReceivingShadow::setTexBufferName( GLint v )
{
	_dataPoolTexture = v;
}

/******************************************************************************
 * Get the 3D model filename to load
 *
 * @return the 3D model filename to load
 ******************************************************************************/
const char* ObjectReceivingShadow::get3DModelFilename() const
{
	return _modelFilename.c_str();
}

/******************************************************************************
 * Set the 3D model filename to load
 *
 * @param pFilename the 3D model filename to load
 ******************************************************************************/
void ObjectReceivingShadow::set3DModelFilename( const char* pFilename )
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
void ObjectReceivingShadow::getTranslation( float& pX, float& pY, float& pZ ) const
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
void ObjectReceivingShadow::setTranslation( float pX, float pY, float pZ )
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
void ObjectReceivingShadow::getRotation( float& pAngle, float& pX, float& pY, float& pZ ) const
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
void ObjectReceivingShadow::setRotation( float pAngle, float pX, float pY, float pZ )
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
void ObjectReceivingShadow::getScale( float& pValue ) const
{
	pValue = _scale;
}

/******************************************************************************
 * Set the uniform scale
 *
 * @param pValue the uniform scale
 ******************************************************************************/
void ObjectReceivingShadow::setScale( float pValue )
{
	_scale = pValue;
}
