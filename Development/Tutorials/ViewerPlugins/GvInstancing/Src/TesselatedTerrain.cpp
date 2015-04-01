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

#include "TesselatedTerrain.h"

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
TesselatedTerrain::TesselatedTerrain()
:	_shaderProgram( NULL )
,	_vertexArray( 0 )
,	_vertexBuffer( 0 )
,	_heightmap( 0 )
{
	// Initialize graphics resources
	//initialize();
}

/******************************************************************************
 * Desconstructor
 ******************************************************************************/
TesselatedTerrain::~TesselatedTerrain()
{
	// Release graphics resources
	finalize();
}

/******************************************************************************
 * Initialize
 ******************************************************************************/
bool TesselatedTerrain::initialize()
{
	std::cout << "Terrain - initialization" << std::endl;

	bool statusOK = false;

	std::cout << "Terrain - shader program initialization" << std::endl;

	// Initialize shader program
	QString dataRepository = QCoreApplication::applicationDirPath();
	dataRepository += QDir::separator();
	dataRepository += QString( "Data" );
	dataRepository += QDir::separator();
	dataRepository += QString( "Shaders" );
	dataRepository += QDir::separator();
	dataRepository += QString( "GvInstancing" );
	dataRepository += QDir::separator();
	const QString vertexShaderFilename = dataRepository + QString( "tesselatedTerrain_vert.glsl" );
	const QString tesselationControlShaderFilename = dataRepository + QString( "tesselatedTerrain_tesc.glsl" );
	const QString tesselationEvaluationShaderFilename = dataRepository + QString( "tesselatedTerrain_tese.glsl" );
	const QString fragmentShaderFilename = dataRepository + QString( "tesselatedTerrain_frag.glsl" );
	_shaderProgram = new GsShaderProgram();
	assert( _shaderProgram != NULL );
	statusOK = _shaderProgram->addShader( GsShaderProgram::eVertexShader, vertexShaderFilename.toStdString() );
	assert( statusOK );
	statusOK = _shaderProgram->addShader( GsShaderProgram::eTesselationControlShader, tesselationControlShaderFilename.toStdString() );
	assert( statusOK );
	statusOK = _shaderProgram->addShader( GsShaderProgram::eTesselationEvaluationShader, tesselationEvaluationShaderFilename.toStdString() );
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
		
	std::cout << "Terrain - heightmap texture initialization" << std::endl;

	// Initialize cube map
	QString filename;
	filename = dataRepository + QString( "heightmap512x512.png" );
	QString landFilename;
	landFilename = dataRepository + QString( "green.bmp" );
	statusOK = load( filename.toStdString(), landFilename.toStdString() );
	assert( statusOK );
	
	std::cout << "Terrain - vertex buffer initialization" << std::endl;

	// Vertex buffer initialization
	glGenBuffers( 1, &_vertexBuffer );
	glBindBuffer( GL_ARRAY_BUFFER, _vertexBuffer );
	const unsigned int NUM_X = 64;
	const unsigned int NUM_Z = 64;
	const float SIZE_X = 1.f;
	const float SIZE_Z = 1.f;
	const float HALF_SIZE_X = SIZE_X * 0.5f;
	const float HALF_SIZE_Z = SIZE_Z * 0.5f;
	const unsigned int cNbVertices = NUM_X * NUM_Z;
	GLsizeiptr vertexBufferSize = sizeof( GLfloat ) * cNbVertices * 2;
	//glBufferData( GL_ARRAY_BUFFER, vertexBufferSize, NULL, GL_DYNAMIC_DRAW );
	glBufferData( GL_ARRAY_BUFFER, vertexBufferSize, NULL, GL_STATIC_DRAW );
	// Fill vertex buffer (map it for writing)
	GLfloat* vertexBufferData = static_cast< GLfloat* >( glMapBuffer( GL_ARRAY_BUFFER, GL_WRITE_ONLY ) );
	for ( unsigned int i = 0; i < NUM_Z; i++ )
	{
		for ( unsigned int j = 0; j < NUM_X; j++ )
		{
			*vertexBufferData++ = static_cast< float >( j ) / static_cast< float >( NUM_X );
			*vertexBufferData++ = static_cast< float >( i ) / static_cast< float >( NUM_Z );
		}
	}
	glUnmapBuffer( GL_ARRAY_BUFFER );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
	GV_CHECK_GL_ERROR();

	std::cout << "Terrain - vertex array object initialization" << std::endl;

	// Vertex array object initialization
	glGenVertexArrays( 1, &_vertexArray );
	glBindVertexArray( _vertexArray );
	glEnableVertexAttribArray( 0/*index*/ );	// vertex position
	glBindBuffer( GL_ARRAY_BUFFER, _vertexBuffer );
	glVertexAttribPointer( 0/*attribute index*/, 2/*nb components per vertex*/, GL_FLOAT/*type*/, GL_FALSE/*un-normalized*/, 0/*memory stride*/, static_cast< GLubyte* >( NULL )/*byte offset from buffer*/ );
	//glBindBuffer( GL_ARRAY_BUFFER, 0 );
	//glDisableVertexAttribArray( 0/*index*/ );
	glBindVertexArray( 0 );
	GV_CHECK_GL_ERROR();

	return statusOK;
}

/******************************************************************************
* Finalize
******************************************************************************/
bool TesselatedTerrain::finalize()
{
	delete _shaderProgram;
	_shaderProgram = NULL;

	glDeleteTextures( 1, &_heightmap );
	
	glDeleteBuffers( 1, &_vertexBuffer );
	glDeleteVertexArrays( 1, &_vertexArray );

	return true;
}

/******************************************************************************
* Load cubemap
******************************************************************************/
bool TesselatedTerrain::load( const string& pFilename, const string& pLandFilename )
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
	//glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	//glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	//glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
	//glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
	
	// Load image with CImg (with the help of ImageMagick if required)
	cimg_library::CImg< unsigned char > image( pFilename.c_str() );

	std::cout << "Terrain spectrum : " << image.spectrum() << std::endl;

	assert( image.spectrum() == 1 );

	const GLenum target = GL_TEXTURE_2D;
	const GLint level = 0;
	const GLint internalFormat = GL_RGBA/*GL_RED*/;
	const GLsizei width = image.width();
	const GLsizei height = image.height();
	const GLint border = 0;
	const GLenum format = GL_RGBA/*GL_RED*/;
	const GLenum type = GL_UNSIGNED_BYTE;

	// Interleave data for OpenGL
	image.permute_axes( "cxyz" );
	const GLvoid* pixels = image.data();
	
	glTexImage2D( target, level, internalFormat, width, height, border, format, type, pixels );
	GV_CHECK_GL_ERROR();
	
	glBindTexture( GL_TEXTURE_2D, 0 );

	//------------------------------------------------------------------------
	// load the land texture data
	unsigned char* landTexture = NULL;
	// Load image with CImg (with the help of ImageMagick if required)
	cimg_library::CImg< unsigned char > landImage( pLandFilename.c_str() );

	std::cout << "Land spectrum : " << landImage.spectrum() << std::endl;

	assert( landImage.spectrum() == 3 );

	const GLenum landtarget = GL_TEXTURE_2D;
	const GLint landlevel = 0;
	const GLint landinternalFormat = GL_RGB;
	const GLsizei landwidth = landImage.width();
	const GLsizei landheight = landImage.height();
	const GLint landborder = 0;
	const GLenum landformat = GL_RGB;
	const GLenum landtype = GL_UNSIGNED_BYTE;

	// Interleave data for OpenGL
	landImage.permute_axes( "cxyz" );
	//const GLvoid* pixels = image.data();
	landTexture = landImage.data();

	if ( ! landTexture )
	{
		// TO DO
		// - clean
		// ...

		std::cout << "ERROR : landTexture is NULL " << std::endl;

		return false;
	}

	// generate the land texture as a mipmap
	glGenTextures( 1, &_land );            
	//glBindTexture( GL_TEXTURE_2D, _land );

	glActiveTexture( GL_TEXTURE1 );
	glBindTexture( GL_TEXTURE_2D, _land );

	//glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	//glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	/*glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );*/
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );

	//	gluBuild2DMipmaps( GL_TEXTURE_2D, GL_RGB, landwidth, landheight, GL_RGB, GL_UNSIGNED_BYTE, landTexture );

	glTexImage2D( landtarget, landlevel, landinternalFormat, landwidth, landheight, landborder, landformat, landtype, landTexture );
	GV_CHECK_GL_ERROR();
	//------------------------------------------------------------------------

	//glBindTexture( GL_TEXTURE_2D, 0 );
	
	return true;
}

/******************************************************************************
 * Render
 ******************************************************************************/
void TesselatedTerrain::render( const float4x4& pModelViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport ) const
{
	//std::cout << "Terrain - render" << std::endl;

	glEnable( GL_CULL_FACE );
	//glCullFace( GL_BACK );

	_shaderProgram->use();

	// Activation des textures
	glEnable( GL_TEXTURE_2D );
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D, _heightmap );

	// land
	glActiveTexture( GL_TEXTURE1 );
	glBindTexture( GL_TEXTURE_2D, _land );

	// Set custom uniforms
	GLint location;
	location = glGetUniformLocation( _shaderProgram->_program, "uModelViewMatrix" );
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
	location = glGetUniformLocation( _shaderProgram->_program, "uHeightMap" );
	if ( location >= 0 )
	{
		glUniform1i( location, 0 );
	}
	location = glGetUniformLocation( _shaderProgram->_program, "uLandTexture" );
	if ( location >= 0 )
	{
		glUniform1i( location, 1 );
	}
	else
	{
		std::cout << "ERROR : glGetUniformLocation( _shaderProgram->_program, ""uLandTexture"" )" << std::endl;
	}

	const unsigned int NUM_X = 64;
	const unsigned int NUM_Z = 64;
	const unsigned int cNbVertices = NUM_X * NUM_Z;
	const GLsizei count = cNbVertices;
	
	glPatchParameteri( GL_PATCH_VERTICES, 1 );

	glBindVertexArray( _vertexArray );
	glDrawArrays( GL_PATCHES/*mode*/, 0/*first*/, count/*count*/ );
	glBindVertexArray( 0 );

	glUseProgram( 0 );

	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D, 0 );

	// land
	glActiveTexture( GL_TEXTURE1 );
	glBindTexture( GL_TEXTURE_2D, 0 );

	glDisable( GL_CULL_FACE );
}
