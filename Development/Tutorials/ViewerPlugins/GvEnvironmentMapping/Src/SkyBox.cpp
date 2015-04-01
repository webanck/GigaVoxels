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

#include "SkyBox.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaSpace
#include <GsGraphics/GsShaderProgram.h>
#include <GvCore/GvError.h>

// System
#include <cassert>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>
#include <QImage>
#include <QGLWidget>

// glm
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform2.hpp>
#include <glm/gtx/projection.hpp>
#include <glm/gtc/type_ptr.hpp>

// STL
#include <vector>
#include <string>

// System
#include <cassert>

// CImg
#define cimg_use_magick	// Beware, this definition must be placed before including CImg.h
#include <CImg.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GsGraphics;

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
SkyBox::SkyBox()
:	_shaderProgram( NULL )
,	_cubeMap( 0 )
,	_vertexArray( 0 )
,	_vertexBuffer( 0 )
,	_indexBuffer( 0 )
,	_faceFilenames()
{
	// Initialize graphics resources
	//initialize();
}

/******************************************************************************
 * Desconstructor
 ******************************************************************************/
SkyBox::~SkyBox()
{
	// Release graphics resources
	finalize();
}

/******************************************************************************
 * Initialize
 ******************************************************************************/
bool SkyBox::initialize()
{
	bool statusOK = false;

	// Initialize shader program
	QString dataRepository = QCoreApplication::applicationDirPath();
	dataRepository += QDir::separator();
	dataRepository += QString( "Data" );
	dataRepository += QDir::separator();
	dataRepository += QString( "Shaders" );
	dataRepository += QDir::separator();
	dataRepository += QString( "GvEnvironmentMapping" );
	dataRepository += QDir::separator();
	const QString vertexShaderFilename = dataRepository + QString( "skyBox_vert.glsl" );
	const QString fragmentShaderFilename = dataRepository + QString( "skyBox_frag.glsl" );
	_shaderProgram = new GsShaderProgram();
	assert( _shaderProgram != NULL );
	statusOK = _shaderProgram->addShader( GsShaderProgram::eVertexShader, vertexShaderFilename.toStdString() );
	assert( statusOK );
	statusOK = _shaderProgram->addShader( GsShaderProgram::eFragmentShader, fragmentShaderFilename.toStdString() );
	assert( statusOK );
	statusOK = _shaderProgram->link();
	assert( statusOK );

	// Allocate texture storage
	dataRepository = QCoreApplication::applicationDirPath();
	dataRepository += QDir::separator();
	dataRepository += QString( "Data" );
	dataRepository += QDir::separator();
	dataRepository += QString( "SkyBox" );
	dataRepository += QDir::separator();
		
	// Initialize cube map
	QString filename;
	vector< string > faceFilenames( eNbSkyBoxFaces );
	//filename = dataRepository + QString( "right.png" );
	filename = dataRepository + QString( "posx.png" );
	faceFilenames[ ePositiveX ] = filename.toStdString();
	filename = dataRepository + QString( "negx.png" );
	faceFilenames[ eNegativeX ] = filename.toStdString();
	filename = dataRepository + QString( "posy.png" );
	faceFilenames[ ePositiveY ] = filename.toStdString();
	filename = dataRepository + QString( "negy.png" );
	faceFilenames[ eNegativeY ] = filename.toStdString();
	filename = dataRepository + QString( "posz.png" );
	faceFilenames[ ePositiveZ ] = filename.toStdString();
	filename = dataRepository + QString( "negz.png" );
	faceFilenames[ eNegativeZ ] = filename.toStdString();
	statusOK = load( faceFilenames );
	assert( statusOK );
	
	// Vertex buffer initialization
	glGenBuffers( 1, &_vertexBuffer );
	glBindBuffer( GL_ARRAY_BUFFER, _vertexBuffer );
	const unsigned int nbVertices = 8;
	GLsizeiptr vertexBufferSize = sizeof( GLfloat ) * nbVertices * 3;
	glBufferData( GL_ARRAY_BUFFER, vertexBufferSize, NULL, GL_STATIC_DRAW );
	// Fill vertex buffer (map it for writing)
	GLfloat* vertexBufferData = static_cast< GLfloat* >( glMapBuffer( GL_ARRAY_BUFFER, GL_WRITE_ONLY ) );
	unsigned int index = 0;
	static glm::vec3 vertices[] =
	{
		glm::vec3( -0.5f, -0.5f, 0.5f ),
		glm::vec3( 0.5f, -0.5f, 0.5f ),
		glm::vec3( 0.5f, 0.5f, 0.5f ),
		glm::vec3( -0.5f, 0.5f, 0.5f ),
		glm::vec3( -0.5f, -0.5f, -0.5f ),
		glm::vec3( 0.5f, -0.5f, -0.5f ),
		glm::vec3( 0.5f, 0.5f, -0.5f ),
		glm::vec3( -0.5f, 0.5f, -0.5f )
	};
	for ( unsigned int i = 0; i < 8; i++ )
	{
		vertexBufferData[ index++ ] = vertices[ i ].x;
		vertexBufferData[ index++ ] = vertices[ i ].y;
		vertexBufferData[ index++ ] = vertices[ i ].z;
	}
	glUnmapBuffer( GL_ARRAY_BUFFER );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	GV_CHECK_GL_ERROR();

	// Index buffer initialization
	glGenBuffers( 1, &_indexBuffer );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _indexBuffer );
	const unsigned int nbIndices = 6/*nb faces*/ * 2/*2 triangles per face*/ * 3/*nb indices per triangle*/;
	GLsizeiptr indexBufferSize = sizeof( GLuint ) * nbIndices;
	glBufferData( GL_ELEMENT_ARRAY_BUFFER, indexBufferSize, NULL, GL_STATIC_DRAW );
	// Fill index buffer (map it for writing)
	GLuint* indexBufferData = static_cast< GLuint* >( glMapBuffer( GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY ) );
	index = 0;
	// Bottom face
	/*indexBufferData[ index++ ] = 1;
	indexBufferData[ index++ ] = 0;
	indexBufferData[ index++ ] = 4;
	indexBufferData[ index++ ] = 5;*/
	// -
	indexBufferData[ index++ ] = 1;
	indexBufferData[ index++ ] = 0;
	indexBufferData[ index++ ] = 5;
	// -
	indexBufferData[ index++ ] = 5;
	indexBufferData[ index++ ] = 4;
	indexBufferData[ index++ ] = 0;
	// Top face
	/*indexBufferData[ index++ ] = 3;
	indexBufferData[ index++ ] = 2;
	indexBufferData[ index++ ] = 6;
	indexBufferData[ index++ ] = 7;*/
	// -
	indexBufferData[ index++ ] = 3;
	indexBufferData[ index++ ] = 2;
	indexBufferData[ index++ ] = 7;
	// -
	indexBufferData[ index++ ] = 7;
	indexBufferData[ index++ ] = 6;
	indexBufferData[ index++ ] = 2;
	// Left face
	/*indexBufferData[ index++ ] = 4;
	indexBufferData[ index++ ] = 0;
	indexBufferData[ index++ ] = 3;
	indexBufferData[ index++ ] = 7;*/
	// -
	indexBufferData[ index++ ] = 4;
	indexBufferData[ index++ ] = 0;
	indexBufferData[ index++ ] = 7;
	// -
	indexBufferData[ index++ ] = 7;
	indexBufferData[ index++ ] = 3;
	indexBufferData[ index++ ] = 0;
	// Right face
	/*indexBufferData[ index++ ] = 1;
	indexBufferData[ index++ ] = 5;
	indexBufferData[ index++ ] = 6;
	indexBufferData[ index++ ] = 2;*/
	// -
	indexBufferData[ index++ ] = 1;
	indexBufferData[ index++ ] = 5;
	indexBufferData[ index++ ] = 2;
	// -
	indexBufferData[ index++ ] = 2;
	indexBufferData[ index++ ] = 6;
	indexBufferData[ index++ ] = 5;
	// Front face
	/*indexBufferData[ index++ ] = 0;
	indexBufferData[ index++ ] = 1;
	indexBufferData[ index++ ] = 2;
	indexBufferData[ index++ ] = 3;*/
	// -
	indexBufferData[ index++ ] = 0;
	indexBufferData[ index++ ] = 1;
	indexBufferData[ index++ ] = 3;
	// -
	indexBufferData[ index++ ] = 3;
	indexBufferData[ index++ ] = 2;
	indexBufferData[ index++ ] = 1;
	// Rear face
	/*indexBufferData[ index++ ] = 5;
	indexBufferData[ index++ ] = 4;
	indexBufferData[ index++ ] = 7;
	indexBufferData[ index++ ] = 6;*/
	// -
	indexBufferData[ index++ ] = 5;
	indexBufferData[ index++ ] = 4;
	indexBufferData[ index++ ] = 6;
	// -
	indexBufferData[ index++ ] = 6;
	indexBufferData[ index++ ] = 7;
	indexBufferData[ index++ ] = 4;

	glUnmapBuffer( GL_ELEMENT_ARRAY_BUFFER );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );

	GV_CHECK_GL_ERROR();

	// Vertex array object initialization
	glGenVertexArrays( 1, &_vertexArray );
	glBindVertexArray( _vertexArray );

	glEnableVertexAttribArray( 0/*index*/ );	// vertex position
	glBindBuffer( GL_ARRAY_BUFFER, _vertexBuffer );
	glVertexAttribPointer( 0/*attribute index*/, 3/*nb components per vertex*/, GL_FLOAT/*type*/, GL_FALSE/*un-normalized*/, 0/*memory stride*/, static_cast< GLubyte* >( NULL )/*byte offset from buffer*/ );
	//glBindBuffer( GL_ARRAY_BUFFER, 0 );
	//glDisableVertexAttribArray( 0/*index*/ );

	// Required for indexed rendering
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _indexBuffer );
		
	glBindVertexArray( 0 );

	GV_CHECK_GL_ERROR();

	return statusOK;
}

/******************************************************************************
* Finalize
******************************************************************************/
bool SkyBox::finalize()
{
	_faceFilenames.clear();

	delete _shaderProgram;
	_shaderProgram = NULL;

	glDeleteTextures( 1, &_cubeMap );
	
	glDeleteBuffers( 1, &_vertexBuffer );
	glDeleteBuffers( 1, &_indexBuffer );
	glDeleteVertexArrays( 1, &_vertexArray );

	return true;
}

/******************************************************************************
* Load cubemap
******************************************************************************/
bool SkyBox::load( const vector< string >& pFaceFilenames )
{
	assert( pFaceFilenames.size() == eNbSkyBoxFaces );

	// Reset
	_faceFilenames.clear();
	_faceFilenames = pFaceFilenames;
	
	// Initialize cubemap
	glGenTextures( 1, &_cubeMap );

	glActiveTexture( GL_TEXTURE1 );
	glBindTexture( GL_TEXTURE_CUBE_MAP, _cubeMap );
	
	glTexParameteri( GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );

	// Allocate texture storage
	for ( unsigned int i = 0; i < eNbSkyBoxFaces; i++ )
	{
		// Check if there is a texture for a particular face
		if ( _faceFilenames[ i ].empty() )
		{
			continue;
		}

		// Load image with CImg (with the help of ImageMagick if required)
		cimg_library::CImg< unsigned char > image( _faceFilenames[ i ].c_str() );

		const GLenum target = GL_TEXTURE_CUBE_MAP_POSITIVE_X + i;
		const GLint level = 0;
		const GLint internalFormat = image.spectrum() == 4 ? GL_RGBA : GL_RGB;
		const GLsizei width = image.width();
		const GLsizei height = image.height();
		const GLint border = 0;
		const GLenum format = internalFormat;
		const GLenum type = GL_UNSIGNED_BYTE;

		// Interleave data for OpenGL
		image.permute_axes( "cxyz" );
		const GLvoid* pixels = image.data();
		
		glTexImage2D( target, level, internalFormat, width, height, border, format, type, pixels );
		GV_CHECK_GL_ERROR();
	}

	glBindTexture( GL_TEXTURE_CUBE_MAP, 0 );
	
	return true;
}

/******************************************************************************
 * Render
 ******************************************************************************/
void SkyBox::render( const float4x4& pModelViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport ) const
{
	// Activation des textures
	//glEnable( GL_TEXTURE_2D );
	glActiveTexture( GL_TEXTURE1 );
	glBindTexture( GL_TEXTURE_CUBE_MAP, _cubeMap );

	_shaderProgram->use();

	// Set custom uniforms
	//GLint location = glGetUniformLocation( _shaderProgram->_program, "uModelViewProjectionMatrix" );
	//if ( location >= 0 )
	//{
	//	//const GLfloat* value = NULL;
	//	glm::mat4 P = glm::mat4( 1.f );
	//	glm::mat4 MV = glm::mat4( 1.f );
	//	glm::mat4 MVP = P * MV;
	//	glUniformMatrix4fv( location, 1/*count*/, GL_FALSE/*transpose*/, glm::value_ptr( MVP ) );
	//}
	GLint location = glGetUniformLocation( _shaderProgram->_program, "uModelViewMatrix" );
	if ( location >= 0 )
	{
		//const GLfloat* value = NULL;
		glm::mat4 P = glm::mat4( 1.f );
		glm::mat4 MV = glm::mat4( 1.f );
		glm::mat4 MVP = P * MV;
		glUniformMatrix4fv( location, 1/*count*/, GL_FALSE/*transpose*/, pModelViewMatrix._array );
	}
	location = glGetUniformLocation( _shaderProgram->_program, "uProjectionMatrix" );
	if ( location >= 0 )
	{
		//const GLfloat* value = NULL;
		glm::mat4 P = glm::mat4( 1.f );
		glm::mat4 MV = glm::mat4( 1.f );
		glm::mat4 MVP = P * MV;
		glUniformMatrix4fv( location, 1/*count*/, GL_FALSE/*transpose*/, pProjectionMatrix._array );
	}
	location = glGetUniformLocation( _shaderProgram->_program, "uCubeMapSampler" );
	if ( location >= 0 )
	{
		glUniform1i( location, 1 );
	}

	const GLsizei _nbIndices = 6/*nb faces*/ * 2/*2 triangles per face*/ * 3/*nb indices per triangle*/;
	glBindVertexArray( _vertexArray );
	glDrawElements( GL_TRIANGLES/*mode*/, _nbIndices/*count*/, GL_UNSIGNED_INT/*type*/, 0/*indices*/ );
	glBindVertexArray( 0 );

	glUseProgram( 0 );
}
