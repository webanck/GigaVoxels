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

#include "Terrain.h"

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
Terrain::Terrain()
:	_shaderProgram( NULL )
,	_vertexArray( 0 )
,	_vertexBuffer( 0 )
,	_indexBuffer( 0 )
,	_heightmap( 0 )
{
	// Initialize graphics resources
	//initialize();
}

/******************************************************************************
 * Desconstructor
 ******************************************************************************/
Terrain::~Terrain()
{
	// Release graphics resources
	finalize();
}

/******************************************************************************
 * Initialize
 ******************************************************************************/
bool Terrain::initialize()
{
	bool statusOK = false;

	// Initialize shader program
	QString dataRepository = QCoreApplication::applicationDirPath();
	dataRepository += QDir::separator();
	dataRepository += QString( "Data" );
	dataRepository += QDir::separator();
	dataRepository += QString( "Shaders" );
	dataRepository += QDir::separator();
	dataRepository += QString( "GvInstancing" );
	dataRepository += QDir::separator();
	const QString vertexShaderFilename = dataRepository + QString( "heightmap_vert.glsl" );
	const QString fragmentShaderFilename = dataRepository + QString( "heightmap_frag.glsl" );
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
	dataRepository += QString( "Terrain" );
	dataRepository += QDir::separator();
		
	//// Initialize cube map
	QString filename;
	filename = dataRepository + QString( "heightmap512x512.png" );
	statusOK = load( filename.toStdString() );
	assert( statusOK );
	
	// Vertex buffer initialization
	glGenBuffers( 1, &_vertexBuffer );
	glBindBuffer( GL_ARRAY_BUFFER, _vertexBuffer );
	const unsigned int NUM_X = 100;
	const unsigned int NUM_Z = 100;
	const float SIZE_X = 1.f;
	const float SIZE_Z = 1.f;
	const float HALF_SIZE_X = SIZE_X * 0.5f;
	const float HALF_SIZE_Z = SIZE_Z * 0.5f;
	const unsigned int cNbVertices = NUM_X * NUM_Z;
	GLsizeiptr vertexBufferSize = sizeof( GLfloat ) * cNbVertices * 3;
	glBufferData( GL_ARRAY_BUFFER, vertexBufferSize, NULL, GL_STATIC_DRAW );
	// Fill vertex buffer (map it for writing)
	GLfloat* vertexBufferData = static_cast< GLfloat* >( glMapBuffer( GL_ARRAY_BUFFER, GL_WRITE_ONLY ) );
	for ( unsigned int i = 0; i < NUM_Z; i++ )
	{
		for ( unsigned int j = 0; j < NUM_X; j++ )
		{
			*vertexBufferData++ = ( ( static_cast< float >( j ) / static_cast< float >( NUM_X ) ) * 2.f - 1.f ) * HALF_SIZE_X;
			*vertexBufferData++ = 0.f;
			*vertexBufferData++ = ( ( static_cast< float >( i ) / static_cast< float >( NUM_Z ) ) * 2.f - 1.f ) * HALF_SIZE_Z;
		}
	}
	glUnmapBuffer( GL_ARRAY_BUFFER );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
	GV_CHECK_GL_ERROR();

	// Index buffer initialization
	glGenBuffers( 1, &_indexBuffer );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _indexBuffer );
	const unsigned int nbIndices = ( ( NUM_X - 1 ) *( NUM_Z - 1 ) )/*nb faces*/ * 2/*2 triangles per face*/ * 3/*nb indices per triangle*/;
	GLsizeiptr indexBufferSize = sizeof( GLuint ) * nbIndices;
	glBufferData( GL_ELEMENT_ARRAY_BUFFER, indexBufferSize, NULL, GL_STATIC_DRAW );
	// Fill index buffer (map it for writing)
	GLuint* indexBufferData = static_cast< GLuint* >( glMapBuffer( GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY ) );
	unsigned int i0;
	unsigned int i1;
	unsigned int i2;
	unsigned int i3;
	for ( unsigned int i = 0; i < ( NUM_Z - 1 ); i++ )
	{
		for ( unsigned int j = 0; j < ( NUM_X - 1 ); j++ )
		{
			i0 = j + NUM_X * i;
			i1 = i0 + 1;
			i2 = i0 + NUM_X;
			i3 = i2 + 1;

			*indexBufferData++ = i0;
			*indexBufferData++ = i1;
			*indexBufferData++ = i2;

			*indexBufferData++ = i1;
			*indexBufferData++ = i3;
			*indexBufferData++ = i2;
		}
	}
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
bool Terrain::finalize()
{
	delete _shaderProgram;
	_shaderProgram = NULL;

	glDeleteTextures( 1, &_heightmap );
	
	glDeleteBuffers( 1, &_vertexBuffer );
	glDeleteBuffers( 1, &_indexBuffer );
	glDeleteVertexArrays( 1, &_vertexArray );

	return true;
}

/******************************************************************************
* Load cubemap
******************************************************************************/
bool Terrain::load( const string& pFilename )
{
	assert( ! pFilename.empty() );

	// Initialize _heightmap
	glGenTextures( 1, &_heightmap );

	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D, _heightmap );
	
	//glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	//glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
	
	// Load image with CImg (with the help of ImageMagick if required)
	cimg_library::CImg< unsigned char > image( pFilename.c_str() );

	std::cout << "Terrain spectrum : " << image.spectrum() << std::endl;

	assert( image.spectrum() == 1 );
	
	const GLenum target = GL_TEXTURE_2D;
	const GLint level = 0;
	const GLint internalFormat = GL_RED;
	const GLsizei width = image.width();
	const GLsizei height = image.height();
	const GLint border = 0;
	const GLenum format = GL_RED;
	const GLenum type = GL_UNSIGNED_BYTE;

	// Interleave data for OpenGL
	image.permute_axes( "cxyz" );
	const GLvoid* pixels = image.data();
	
	glTexImage2D( target, level, internalFormat, width, height, border, format, type, pixels );
	GV_CHECK_GL_ERROR();
	
	glBindTexture( GL_TEXTURE_2D, 0 );
	
	return true;
}

/******************************************************************************
 * Render
 ******************************************************************************/
void Terrain::render( const float4x4& pModelViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport ) const
{
	// Activation des textures
	//glEnable( GL_TEXTURE_2D );
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D, _heightmap );

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
	location = glGetUniformLocation( _shaderProgram->_program, "heightMapTexture" );
	if ( location >= 0 )
	{
		glUniform1i( location, 0 );
	}

	// Terrain parameters
	const float SIZE_X = 1.f;
	const float SIZE_Z = 1.f;
	location = glGetUniformLocation( _shaderProgram->_program, "HALF_TERRAIN_SIZE" );
	if ( location >= 0 )
	{
		glUniform2i( location, SIZE_X, SIZE_Z );
	}
	location = glGetUniformLocation( _shaderProgram->_program, "scale" );
	if ( location >= 0 )
	{
		glUniform1f( location, 1.f );
	}
	location = glGetUniformLocation( _shaderProgram->_program, "half_scale" );
	if ( location >= 0 )
	{
		glUniform1f( location, 1.f/*scale*/ * 0.5f );
	}

	const unsigned int NUM_X = 100;
	const unsigned int NUM_Z = 100;
	const GLsizei _nbIndices = ( ( NUM_X - 1 ) *( NUM_Z - 1 ) )/*nb faces*/ * 2/*2 triangles per face*/ * 3/*nb indices per triangle*/;
	glBindVertexArray( _vertexArray );
	glDrawElements( GL_TRIANGLES/*mode*/, _nbIndices/*count*/, GL_UNSIGNED_INT/*type*/, 0/*indices*/ );
	glBindVertexArray( 0 );

	glUseProgram( 0 );
}
