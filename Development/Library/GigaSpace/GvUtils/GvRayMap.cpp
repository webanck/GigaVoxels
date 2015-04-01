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

#include "GvUtils/GvRayMap.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvRendering/GvGraphicsResource.h"
#include "GvCore/GvError.h"
#include "GvUtils/GvShaderManager.h"
#include "GvCore/vector_types_ext.h"
#include "GsGraphics/GsShaderProgram.h"

// CUDA
#include <driver_types.h>

// System
#include <cassert>
#include <cstddef>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvUtils;
using namespace GsGraphics;
using namespace GvRendering;

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
GvRayMap::GvRayMap()
:	_rayMap( 0 )
,	_rayMapType( eClassical )
,	_graphicsResource( NULL )
,	_isInitialized( false )
,	_shaderProgram( NULL )
,	_width( 0 )
,	_height( 0 )
,	_frameBuffer( 0 )
{
}

/******************************************************************************
 * Desconstructor
 ******************************************************************************/
GvRayMap::~GvRayMap()
{
	// Check if graphics resources have been initialized
	finalize();
}

/******************************************************************************
 * Initialize
 *
 * @return a flag to tell wheter or not it succeeds.
 ******************************************************************************/
bool GvRayMap::initialize()
{
	// First, check if graphics resources have already been initialized
	if ( _isInitialized )
	{
		finalize();
	}

	// Create associated CUDA graphics resource
	_graphicsResource = new GvGraphicsResource();
	if ( _graphicsResource == NULL )
	{
		// TO DO
		// Handle error
		// ...

		return false;
	}
	
	// Create associated OpenGL buffer
	glGenTextures( 1, &_rayMap );
	GV_CHECK_GL_ERROR();

	// Create associated OpenGL frame buffer object
	glGenFramebuffers(1, &_frameBuffer );
	GV_CHECK_GL_ERROR();

	// Update internal state
	_isInitialized = true;

	return true;
}

/******************************************************************************
 * Finalize
 *
 * @return a flag to tell wheter or not it succeeds.
 ******************************************************************************/
bool GvRayMap::finalize()
{
	if ( _graphicsResource != NULL )
	{
		delete _graphicsResource;
		_graphicsResource = NULL;
	}

	if ( _rayMap != 0 )
	{
		glDeleteBuffers( 1, &_rayMap );

		GV_CHECK_GL_ERROR();
	}

	if ( _frameBuffer != 0 )
	{
		glDeleteFramebuffers( 1, &_frameBuffer );
	}

	// Update internal state
	_isInitialized = false;

	return true;
}

/******************************************************************************
 * Set the ray map dimensions
 *
 * @param pWidth width
 * @param pHeight height
 ******************************************************************************/
void GvRayMap::setResolution( unsigned int pWidth, unsigned int pHeight )
{
	assert( ! ( pWidth == 0 || pHeight == 0 ) );

	if ( ! _isInitialized )
	{
		initialize();
	}

	// Update internal state
	_width = pWidth;
	_height= pHeight;

	cudaError_t error;

	// Unregister graphics resource
	if ( _graphicsResource->isRegistered() )
	{
		error = _graphicsResource->unregister();
		if ( error != cudaSuccess )
		{
			// TO DO
			// Handle error
			// ...

			assert( false );
		}
	}

	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _rayMap );

	// Texture parameters
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );

	// Allocate texture storage
	GLenum target = GL_TEXTURE_RECTANGLE_EXT;
	GLint level = 0;
	GLint internalFormat = GL_RGBA32F;
	GLsizei width = pWidth;
	GLsizei height = pHeight;
	GLint border = 0;
	GLenum format = GL_RGBA;
	GLenum type = GL_FLOAT;
	const GLvoid* pixels = NULL;
	glTexImage2D( target, level, internalFormat, width, height, border, format, type, pixels );
	
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
	
	GV_CHECK_GL_ERROR();

	// Register graphics resource
	//
	// Attach an OpenGL texture or renderbuffer object to an internal graphics resource 
	// that will be mapped in CUDA memory space during rendering.
	error = _graphicsResource->registerImage( _rayMap, GL_TEXTURE_RECTANGLE_EXT, cudaGraphicsRegisterFlagsReadOnly );
	if ( error != cudaSuccess )
	{
		// TO DO
		// Handle error
		// ...

		assert( false );
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool GvRayMap::createShaderProgram( const char* pFileNameVS, const char* pFileNameFS )
{
	// Create and link a GLSL shader program
	// Initialize shader program
	_shaderProgram = new GsShaderProgram();
	std::string vertexShaderFilename( pFileNameVS );
	std::string fragmentShaderFilename( pFileNameFS );
	_shaderProgram->addShader( GsShaderProgram::eVertexShader, vertexShaderFilename );
	_shaderProgram->addShader( GsShaderProgram::eFragmentShader, fragmentShaderFilename );
	_shaderProgram->link();

	return true;
}

/******************************************************************************
 * Render
 ******************************************************************************/
void GvRayMap::render()
{
	assert( _rayMap != 0 );
	assert( _graphicsResource != NULL );
	assert( _isInitialized );

	glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );

	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_EXT, _rayMap, 0 );
	GV_CHECK_GL_ERROR();

	_shaderProgram->use();
	GV_CHECK_GL_ERROR();

	GLuint vploc = glGetAttribLocation( _shaderProgram->_program, "iPosition" );
	GV_CHECK_GL_ERROR();

	//// Specify values of uniform variables for shader program object
	//glProgramUniform2fEXT( _rayMapProgram,
	//						glGetUniformLocation( _rayMapProgram, "screenResolutionInverse" )
	//											, static_cast< GLfloat >( 1.0f / _width )
	//											, static_cast< GLfloat >( 1.0f / _height ) );
	//GV_CHECK_GL_ERROR();

	// Specify values of uniform variables for shader program object
	glProgramUniform2fEXT( _shaderProgram->_program,
							glGetUniformLocation( _shaderProgram->_program, "uImageResolution" )
												, static_cast< GLfloat >( _width )
												, static_cast< GLfloat >( _height ) );
	GV_CHECK_GL_ERROR();

		// Extract view transformations
	float4x4 modelViewMatrix;
	glGetFloatv( GL_MODELVIEW_MATRIX, modelViewMatrix._array );
	float4x4 invModelViewMatrix = inverse( modelViewMatrix );
	float4x4 projectionMatrix;
	glGetFloatv( GL_PROJECTION_MATRIX, projectionMatrix._array );
	//float4x4 invMVP = inverse( mul( projectionMatrix, modelViewMatrix ) );
	//------ TEST
	//invModelViewMatrix = invMVP;

	// Specify values of uniform variables for shader program object
	glProgramUniform3fEXT( _shaderProgram->_program,
							glGetUniformLocation( _shaderProgram->_program, "uCameraXAxis" )
												, invModelViewMatrix._array[ 0 ], invModelViewMatrix._array[ 1 ], invModelViewMatrix._array[ 2 ] );
	GV_CHECK_GL_ERROR();

	// Specify values of uniform variables for shader program object
	glProgramUniform3fEXT( _shaderProgram->_program,
							glGetUniformLocation( _shaderProgram->_program, "uCameraYAxis" )
												, invModelViewMatrix._array[ 4 ], invModelViewMatrix._array[ 5 ], invModelViewMatrix._array[ 6 ] );
	GV_CHECK_GL_ERROR();

	// Specify values of uniform variables for shader program object
	glProgramUniform3fEXT( _shaderProgram->_program,
							glGetUniformLocation( _shaderProgram->_program, "uCameraZAxis" )
												, invModelViewMatrix._array[ 8 ], invModelViewMatrix._array[ 9 ], invModelViewMatrix._array[ 10 ] );
	GV_CHECK_GL_ERROR();

	// Specify values of uniform variables for shader program object
	float fnear = projectionMatrix._array[ 14 ] / ( projectionMatrix._array[ 10 ] - 1.0f );
	//std::cout << "fnear = " << fnear << std::endl;
	glProgramUniform1fEXT( _shaderProgram->_program,
							glGetUniformLocation( _shaderProgram->_program, "uCameraViewPlaneDistance" )
												, fnear );
	GV_CHECK_GL_ERROR();

	// Specify values of uniform variables for shader program object
	glProgramUniform1fEXT( _shaderProgram->_program,
							glGetUniformLocation( _shaderProgram->_program, "uFishEyePsiMaxAngle" )
												, 90.0 );
	GV_CHECK_GL_ERROR();

	// TO DO : check for OpenGL extensions "GL_ARB_shader_subroutine"
	GLuint rayDirectionSubroutineIndex = glGetSubroutineIndex( _shaderProgram->_program, GL_FRAGMENT_SHADER, "getClassicalRayDirection" );
	GV_CHECK_GL_ERROR();
	GLuint fishEyeRayDirectionSubroutineIndex = glGetSubroutineIndex(  _shaderProgram->_program, GL_FRAGMENT_SHADER, "getFishEyeRayDirection" );
	GV_CHECK_GL_ERROR();
	GLuint reflectionMapRayDirectionSubroutineIndex = glGetSubroutineIndex(  _shaderProgram->_program, GL_FRAGMENT_SHADER, "getReflectionMapRayDirection" );
	GV_CHECK_GL_ERROR();
	GLuint refractionMapRayDirectionSubroutineIndex = glGetSubroutineIndex(  _shaderProgram->_program, GL_FRAGMENT_SHADER, "getRefractionMapRayDirection" );
	GV_CHECK_GL_ERROR();

	switch ( _rayMapType )
	{
		case eClassical:
			glUniformSubroutinesuiv( GL_FRAGMENT_SHADER, 1, &rayDirectionSubroutineIndex );
			break;

		case eFishEye:
			glUniformSubroutinesuiv( GL_FRAGMENT_SHADER, 1, &fishEyeRayDirectionSubroutineIndex );
			break;

		case eReflectionMap:
			glUniformSubroutinesuiv( GL_FRAGMENT_SHADER, 1, &reflectionMapRayDirectionSubroutineIndex );
			break;

		case eRefractionMap:
			glUniformSubroutinesuiv( GL_FRAGMENT_SHADER, 1, &refractionMapRayDirectionSubroutineIndex );
			break;

		default:
			assert( false );
			std::cout << "GvRayMap::render() wrong enum passed to switch statement for rayMap type." << std::endl;
			break;
	}
	GV_CHECK_GL_ERROR();

	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();

	glEnable( GL_TEXTURE_RECTANGLE_EXT );
	glDisable( GL_DEPTH_TEST );

	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _rayMap );
	
	// Draw a quad on full screen
	glBegin( GL_QUADS );
	glVertexAttrib2f( vploc, -1.0f, -1.0f );
	glVertexAttrib2f( vploc,  1.0f, -1.0f );
	glVertexAttrib2f( vploc,  1.0f,  1.0f );
	glVertexAttrib2f( vploc, -1.0f,  1.0f );
	glEnd();

	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );

	glDisable( GL_TEXTURE_RECTANGLE_EXT );
	//glEnable( GL_DEPTH_TEST );

	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();

	glUseProgram( 0 );

	glBindFramebuffer( GL_FRAMEBUFFER, 0 );
}

/******************************************************************************
 * Get the associated graphics resource
 *
 * @return the associated graphics resource
 ******************************************************************************/
GvRendering::GvGraphicsResource* GvRayMap::getGraphicsResource()
{
	return _graphicsResource;
}

/******************************************************************************
 * Get the shader program
 *
 * @return the shader program
 ******************************************************************************/
const GsGraphics::GsShaderProgram* GvRayMap::getShaderProgram() const
{
	return _shaderProgram;
}

/******************************************************************************
 * Edit the shader program
 *
 * @return the shader program
 ******************************************************************************/
GsGraphics::GsShaderProgram* GvRayMap::editShaderProgram()
{
	return _shaderProgram;
}

/******************************************************************************
 * Get the ray map type
 *
 * @return the ray map type
 ******************************************************************************/
GvRayMap::RayMapType GvRayMap::getRayMapType() const
{
	return _rayMapType;
}

/******************************************************************************
 * Set the ray map type
 *
 * @param pValue the ray map type
 ******************************************************************************/
void GvRayMap::setRayMapType( GvRayMap::RayMapType pValue )
{
	_rayMapType = pValue;
}
