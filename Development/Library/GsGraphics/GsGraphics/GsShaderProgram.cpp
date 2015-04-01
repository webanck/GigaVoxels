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

#include "GsGraphics/GsShaderProgram.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// STL
#include <string>
#include <iostream>
#include <vector>

#include <fstream>
#include <sstream>

#include <fstream>
#include <cerrno>

//#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GsGraphics;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * GLSL Compute shader features
 */
#ifndef GL_COMPUTE_SHADER
#define GL_COMPUTE_SHADER 0x91B9
#endif

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
GsShaderProgram::GsShaderProgram()
:	_vertexShaderFilename()
,	_tesselationControlShaderFilename()
,	_tesselationEvaluationShaderFilename()
,	_geometryShaderFilename()
,	_fragmentShaderFilename()
,	_computeShaderFilename()
,	_vertexShaderSourceCode()
,	_tesselationControlShaderSourceCode()
,	_tesselationEvaluationShaderSourceCode()
,	_geometryShaderSourceCode()
,	_fragmentShaderSourceCode()
,	_computeShaderSourceCode()
,	_program( 0 )
,	_vertexShader( 0 )
,	_tesselationControlShader( 0 )
,	_tesselationEvaluationShader( 0 )
,	_geometryShader( 0 )
,	_fragmentShader( 0 )
,	_computeShader( 0 )
,	_linked( false )
{
	// Initialize graphics resources
	initialize();
}

/******************************************************************************
 * Desconstructor
 ******************************************************************************/
GsShaderProgram::~GsShaderProgram()
{
	// Release graphics resources
	finalize();
}

/******************************************************************************
 * Initialize
 ******************************************************************************/
bool GsShaderProgram::initialize()
{
	// First, check if a program has already been created
	// ...
	assert( _program == 0 );

	// Create program object
	_program = glCreateProgram();
	if ( _program == 0 )
	{
		// LOG
		// ...

		return false;
	}

	return true;
}

/******************************************************************************
 * Finalize
 ******************************************************************************/
bool GsShaderProgram::finalize()
{
	// Check all data to release
	// ...

	glDetachShader( _program, _vertexShader );
	glDetachShader( _program, _tesselationControlShader );
	glDetachShader( _program, _tesselationEvaluationShader );
	glDetachShader( _program, _geometryShader );
	glDetachShader( _program, _fragmentShader );
	glDetachShader( _program, _computeShader );

	glDeleteShader( _vertexShader );
	glDeleteShader( _tesselationControlShader );
	glDeleteShader( _tesselationEvaluationShader );
	glDeleteShader( _geometryShader );
	glDeleteShader( _fragmentShader );
	glDeleteShader( _computeShader );

	// Delete program object
	glDeleteProgram( _program );

	_linked = false;

	return true;
}

/******************************************************************************
 * Compile shader
 ******************************************************************************/
bool GsShaderProgram::addShader( GsShaderProgram::ShaderType pShaderType, const std::string& pShaderFileName )
{
	assert( _program != 0 );

	// Retrieve file content
	std::string shaderSourceCode;
	bool isReadFileOK = getFileContent( pShaderFileName, shaderSourceCode );
	if ( ! isReadFileOK )
	{
		// LOG
		// ...

		return false;
	}

	// Create shader object
	GLuint shader = 0;
	switch ( pShaderType )
	{
		case GsShaderProgram::eVertexShader:
			shader = glCreateShader( GL_VERTEX_SHADER );
			break;

		case GsShaderProgram::eTesselationControlShader:
			shader = glCreateShader( GL_TESS_CONTROL_SHADER );
			break;

		case GsShaderProgram::eTesselationEvaluationShader:
			shader = glCreateShader( GL_TESS_EVALUATION_SHADER );
			break;

		case GsShaderProgram::eGeometryShader:
			shader = glCreateShader( GL_GEOMETRY_SHADER );
			break;

		case GsShaderProgram::eFragmentShader:
			shader = glCreateShader( GL_FRAGMENT_SHADER );
			break;

//TODO
			//- protect code if not defined
		case GsShaderProgram::eComputeShader:
			shader = glCreateShader( GL_COMPUTE_SHADER );

			// LOG
			// ...
			// GL_COMPUTE_SHADER is available only if the GL version is 4.3 or higher

			break;

		default:

			// LOG
			// ...

			return false;
	}

	// Check shader creation error
	if ( shader == 0 )
	{
		// LOG
		// ...

		return false;
	}

	switch ( pShaderType )
	{
		case GsShaderProgram::eVertexShader:
			_vertexShader = shader;
			_vertexShaderFilename = pShaderFileName;
			_vertexShaderSourceCode = shaderSourceCode;
			break;

		case GsShaderProgram::eTesselationControlShader:
			_tesselationControlShader = shader;
			_tesselationControlShaderFilename = pShaderFileName;
			_tesselationControlShaderSourceCode = shaderSourceCode;
			break;

		case GsShaderProgram::eTesselationEvaluationShader:
			_tesselationEvaluationShader = shader;
			_tesselationEvaluationShaderFilename = pShaderFileName;
			_tesselationEvaluationShaderSourceCode = shaderSourceCode;
			break;

		case GsShaderProgram::eGeometryShader:
			_geometryShader = shader;
			_geometryShaderFilename = pShaderFileName;
			_geometryShaderSourceCode = shaderSourceCode;
			break;

		case GsShaderProgram::eFragmentShader:
			_fragmentShader = shader;
			_fragmentShaderFilename = pShaderFileName;
			_fragmentShaderSourceCode = shaderSourceCode;
			break;

		case GsShaderProgram::eComputeShader:
			_computeShader = shader;
			_computeShaderFilename = pShaderFileName;
			_computeShaderSourceCode = shaderSourceCode;
			break;

		default:
			break;
	}

	// Replace source code in shader object
	const char* source = shaderSourceCode.c_str();
	glShaderSource( shader, 1, &source, NULL );

	// Compile shader object
	glCompileShader( shader );

	// Check compilation status
	GLint compileStatus;
	glGetShaderiv( shader, GL_COMPILE_STATUS, &compileStatus );
	if ( compileStatus == GL_FALSE )
	{
		// LOG
		// ...

		GLint logInfoLength = 0;
		glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &logInfoLength );
		if ( logInfoLength > 0 )
		{
			// Return information log for shader object
			GLchar* infoLog = new GLchar[ logInfoLength ];
			GLsizei length = 0;
			glGetShaderInfoLog( shader, logInfoLength, &length, infoLog );

			// LOG
			std::cout << "\nGsShaderProgram::addShader() - compilation ERROR" << std::endl;
			std::cout << "File : " << pShaderFileName << std::endl;
			std::cout << infoLog << std::endl;			

			delete[] infoLog;
		}

		return false;
	}
	else
	{
		// Attach shader object to program object
		glAttachShader( _program, shader );
	}

	return true;
}

/******************************************************************************
 * Link program
 ******************************************************************************/
bool GsShaderProgram::link()
{
	assert( _program != 0 );

	if ( _linked )
	{
		return true;
	}

	if ( _program == 0 )
	{
		return false;
	}

	// Link program object
	glLinkProgram( _program );

	// Check linking status
	GLint linkStatus = 0;
	glGetProgramiv( _program, GL_LINK_STATUS, &linkStatus );
	if ( linkStatus == GL_FALSE )
	{
		// LOG
		// ...

		GLint logInfoLength = 0;
		glGetProgramiv( _program, GL_INFO_LOG_LENGTH, &logInfoLength );
		if ( logInfoLength > 0 )
		{
			// Return information log for program object
			GLchar* infoLog = new GLchar[ logInfoLength ];
			GLsizei length = 0;
			glGetProgramInfoLog( _program, logInfoLength, &length, infoLog );

			// LOG
			std::cout << "\nGsShaderProgram::link() - compilation ERROR" << std::endl;
			std::cout << infoLog << std::endl;

			delete[] infoLog;
		}

		return false;
	}
	
	// Update internal state
	_linked = true;
	
	return true;
}

/******************************************************************************
 * ...
 *
 * @param pFilename ...
 *
 * @return ...
 ******************************************************************************/
bool GsShaderProgram::getFileContent( const std::string& pFilename, std::string& pFileContent )
{
	std::ifstream file( pFilename.c_str(), std::ios::in );
	if ( file )
	{
		// Initialize a string to store file content
		file.seekg( 0, std::ios::end );
		pFileContent.resize( file.tellg() );
		file.seekg( 0, std::ios::beg );

		// Read file content
		file.read( &pFileContent[ 0 ], pFileContent.size() );

		// Close file
		file.close();

		return true;
	}
	else
	{
		// LOG
		// ...
	}

	return false;
}

/******************************************************************************
 * Tell wheter or not pipeline has a given type of shader
 *
 * @param pShaderType the type of shader to test
 *
 * @return a flag telling wheter or not pipeline has a given type of shader
 ******************************************************************************/
bool GsShaderProgram::hasShaderType( ShaderType pShaderType ) const
{
	bool result = false;

	GLuint shader = 0;
	switch ( pShaderType )
	{
		case GsShaderProgram::eVertexShader:
			shader = _vertexShader;
			break;

		case GsShaderProgram::eTesselationControlShader:
			shader = _tesselationControlShader;
			break;

		case GsShaderProgram::eTesselationEvaluationShader:
			shader = _tesselationEvaluationShader;
			break;

		case GsShaderProgram::eGeometryShader:
			shader = _geometryShader;
			break;

		case GsShaderProgram::eFragmentShader:
			shader = _fragmentShader;
			break;

		case GsShaderProgram::eComputeShader:
			shader = _computeShader;
			break;

		default:

			assert( false );

			break;
	}

	return ( shader != 0 );
}

/******************************************************************************
 * Get the source code associated to a given type of shader
 *
 * @param pShaderType the type of shader
 *
 * @return the associated shader source code
 ******************************************************************************/
std::string GsShaderProgram::getShaderSourceCode( ShaderType pShaderType ) const
{
	std::string shaderSourceCode( "" );
	switch ( pShaderType )
	{
		case GsShaderProgram::eVertexShader:
			shaderSourceCode = _vertexShaderSourceCode;
			break;
			
		case GsShaderProgram::eTesselationControlShader:
			shaderSourceCode = _tesselationControlShaderSourceCode;
			break;

		case GsShaderProgram::eTesselationEvaluationShader:
			shaderSourceCode = _tesselationEvaluationShaderSourceCode;
			break;
			
		case GsShaderProgram::eGeometryShader:
			shaderSourceCode = _geometryShaderSourceCode;
			break;

		case GsShaderProgram::eFragmentShader:
			shaderSourceCode = _fragmentShaderSourceCode;
			break;

		case GsShaderProgram::eComputeShader:
			shaderSourceCode = _computeShaderSourceCode;
			break;

		default:

			assert( false );

			break;
	}

	return shaderSourceCode;
}

/******************************************************************************
 * Get the filename associated to a given type of shader
 *
 * @param pShaderType the type of shader
 *
 * @return the associated shader filename
 ******************************************************************************/
std::string GsShaderProgram::getShaderFilename( ShaderType pShaderType ) const
{
	std::string shaderFilename( "" );
	switch ( pShaderType )
	{
		case GsShaderProgram::eVertexShader:
			shaderFilename = _vertexShaderFilename;
			break;
			
		case GsShaderProgram::eTesselationControlShader:
			shaderFilename = _tesselationControlShaderFilename;
			break;

		case GsShaderProgram::eTesselationEvaluationShader:
			shaderFilename = _tesselationEvaluationShaderFilename;
			break;
			
		case GsShaderProgram::eGeometryShader:
			shaderFilename = _geometryShaderFilename;
			break;

		case GsShaderProgram::eFragmentShader:
			shaderFilename = _fragmentShaderFilename;
			break;

		case GsShaderProgram::eComputeShader:
			shaderFilename = _computeShaderFilename;
			break;

		default:

			assert( false );

			break;
	}

	return shaderFilename;
}

/******************************************************************************
 * ...
 *
 * @param pShaderType the type of shader
 *
 * @return ...
 ******************************************************************************/
bool GsShaderProgram::reloadShader( ShaderType pShaderType )
{
	if ( ! hasShaderType( pShaderType ) )
	{
		// LOG
		// ...

		return false;
	}

	// Retrieve file content
	std::string shaderSourceCode;
	std::string shaderFilename = getShaderFilename( pShaderType );
	bool isReadFileOK = getFileContent( shaderFilename, shaderSourceCode );
	if ( ! isReadFileOK )
	{
		// LOG
		// ...

		return false;
	}
	
	GLuint shader = 0;
	switch ( pShaderType )
	{
		case GsShaderProgram::eVertexShader:
			shader = _vertexShader;
			break;

		case GsShaderProgram::eTesselationControlShader:
			shader = _tesselationControlShader;
			break;

		case GsShaderProgram::eTesselationEvaluationShader:
			shader = _tesselationEvaluationShader;
			break;

		case GsShaderProgram::eGeometryShader:
			shader = _geometryShader;
			break;

		case GsShaderProgram::eFragmentShader:
			shader = _fragmentShader;
			break;

		case GsShaderProgram::eComputeShader:
			shader = _computeShader;
			break;

		default:
			break;
	}

	// Check shader creation error
	if ( shader == 0 )
	{
		// LOG
		// ...

		return false;
	}

	// Replace source code in shader object
	const char* source = shaderSourceCode.c_str();
	glShaderSource( shader, 1, &source, NULL );

	// Compile shader object
	glCompileShader( shader );

	// Check compilation status
	GLint compileStatus;
	glGetShaderiv( shader, GL_COMPILE_STATUS, &compileStatus );
	if ( compileStatus == GL_FALSE )
	{
		// LOG
		// ...

		GLint logInfoLength = 0;
		glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &logInfoLength );
		if ( logInfoLength > 0 )
		{
			// Return information log for shader object
			GLchar* infoLog = new GLchar[ logInfoLength ];
			GLsizei length = 0;
			glGetShaderInfoLog( shader, logInfoLength, &length, infoLog );

			// LOG
			std::cout << "\nGsShaderProgram::reloadShader() - compilation ERROR" << std::endl;
			std::cout << infoLog << std::endl;

			delete[] infoLog;
		}

		return false;
	}

	// Link program
	//
	// - first, unliked the program
	_linked = false;
	if ( ! link() )
	{
		return false;
	}
	
	return true;
}
