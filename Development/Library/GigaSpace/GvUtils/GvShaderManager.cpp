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

#include "GvUtils/GvShaderManager.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// STL
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvUtils;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Size of the string, the shorter is better
 */
#define STRING_BUFFER_SIZE 2048

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/**
 * Size of the string, the shorter is better
 */
char stringBuffer[ STRING_BUFFER_SIZE ];

/**
 * ...
 */
bool linkNeeded = false;

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * GLSL shader program creation
 *
 * @param pFileNameVS ...
 * @param pFileNameGS ...
 * @param pFileNameFS ...
 * @param pProgramID ...
 * @param pLazyRecompile ...
 *
 * @return ...
 ******************************************************************************/
GLuint GvShaderManager::createShaderProgram( const char* pFileNameVS, const char* pFileNameGS, const char* pFileNameFS, GLuint pProgramID, bool pLazyRecompile )
{
	bool reload = pProgramID != 0;

	if ( reload && pLazyRecompile )
	{
		return pProgramID;
	}

	linkNeeded = true;

	GLuint vertexShaderID = 0;
	GLuint geometryShaderID = 0;
	GLuint fragmentShaderID = 0;

	if ( ! reload )
	{
		// Create GLSL program
		pProgramID = glCreateProgram();
	}
	else
	{
		GLsizei count;
		GLuint shaders[ 3 ];
		glGetAttachedShaders( pProgramID, 3, &count, shaders );

		for ( int i = 0; i < count; i++ )
		{
			GLint shadertype;
			glGetShaderiv( shaders[ i ], GL_SHADER_TYPE, &shadertype );

			if ( shadertype == GL_VERTEX_SHADER )
			{
				vertexShaderID = shaders[ i ];
			}
			else if ( shadertype == GL_GEOMETRY_SHADER )
			{
				geometryShaderID = shaders[ i ];
			}
			else if ( shadertype == GL_FRAGMENT_SHADER )
			{
				fragmentShaderID = shaders[ i ];
			}
		}
	}
	
	if ( pFileNameVS )
	{
		// Create vertex shader
		vertexShaderID = createShader( pFileNameVS, GL_VERTEX_SHADER, vertexShaderID );
		if ( ! reload )
		{
			// Attach vertex shader to program object
			glAttachShader( pProgramID, vertexShaderID );
		}
	}

	if ( pFileNameGS )
	{
		// Create geometry shader
		geometryShaderID = createShader( pFileNameGS, GL_GEOMETRY_SHADER, geometryShaderID );
		if ( ! reload )
		{
			// Attach vertex shader to program object
			glAttachShader( pProgramID, geometryShaderID );
		}
	}
	
	if ( pFileNameFS )
	{
		// Create fragment shader
		fragmentShaderID = createShader( pFileNameFS, GL_FRAGMENT_SHADER, fragmentShaderID );
		if ( ! reload )
		{
			// Attach fragment shader to program object
			glAttachShader( pProgramID, fragmentShaderID );
		}
	}
	
	return pProgramID;
}


/******************************************************************************
 * GLSL shader creation (of a certain type, vertex shader, fragment shader or geometry shader)
 *
 * @param pFileName ...
 * @param pShaderType ...
 * @param pShaderID ...
 *
 * @return ...
 ******************************************************************************/
GLuint GvShaderManager::createShader( const char* pFileName, GLuint pShaderType, GLuint pShaderID )
{
	if ( pShaderID == 0 )
	{
		pShaderID = glCreateShader( pShaderType );
	}
	
	std::string shaderSource = loadTextFile( pFileName );

	// Manage #includes
    shaderSource = manageIncludes( shaderSource, std::string( pFileName ) );

    // Passing shader source code to GL
	// Source used for "pShaderID" shader, there is only "1" source code and the string is NULL terminated (no sizes passed)
	const char* src = shaderSource.c_str();
	glShaderSource( pShaderID, 1, &src, NULL );

	// Compile shader object
	glCompileShader( pShaderID );

	// Check compilation status
	GLint ok;
	glGetShaderiv( pShaderID, GL_COMPILE_STATUS, &ok );
	if ( ! ok )
	{
		int ilength;
		glGetShaderInfoLog( pShaderID, STRING_BUFFER_SIZE, &ilength, stringBuffer );
		
		std::cout << "Compilation error (" << pFileName << ") : " << stringBuffer; 
	}

	return pShaderID;
}

/******************************************************************************
 * ...
 *
 * @param pProgramID ...
 * @param pStat ...
 ******************************************************************************/
void GvShaderManager::linkShaderProgram( GLuint pProgramID )
{
	int linkStatus;
	glGetProgramiv( pProgramID, GL_LINK_STATUS, &linkStatus );
	if ( linkNeeded )
	{
		// Link all shaders togethers into the GLSL program
		glLinkProgram( pProgramID );
		checkProgramInfos( pProgramID, GL_LINK_STATUS );

		// Validate program executability giving current OpenGL states
		glValidateProgram( pProgramID );
		checkProgramInfos( pProgramID, GL_VALIDATE_STATUS );
		//std::cout << "Program " << pProgramID << " linked\n";

		linkNeeded = false;
	}
}

/******************************************************************************
 * Text file loading for shaders sources
 *
 * @param pMacro ...
 *
 * @return ...
 ******************************************************************************/
std::string GvShaderManager::loadTextFile( const char* pName )
{
	//Source file reading
	std::string buff("");
	
	std::ifstream file;
	file.open( pName );
	if ( file.fail() )
	{
		std::cout<< "loadFile: unable to open file: " << pName;
	}
	
	buff.reserve( 1024 * 1024 );

	std::string line;
	while ( std::getline( file, line ) )
	{
		buff += line + "\n";
	}

	const char* txt = buff.c_str();

	return std::string( txt );
}

/******************************************************************************
 * ...
 *
 * @param pSrc ...
 * @param pSourceFileName ...
 *
 * @return ...
 ******************************************************************************/
std::string GvShaderManager::manageIncludes( std::string pSrc, std::string pSourceFileName )
{
	std::string res;
	res.reserve( 100000 );

	char buff[ 512 ];
	sprintf( buff, "#include" );
	
	size_t includepos = pSrc.find( buff, 0 );

	while ( includepos != std::string::npos )
	{
		bool comment = pSrc.substr( includepos - 2, 2 ) == std::string( "//" );

		if ( ! comment )
		{
			size_t fnamestartLoc = pSrc.find( "\"", includepos );
			size_t fnameendLoc = pSrc.find( "\"", fnamestartLoc + 1 );

			size_t fnamestartLib = pSrc.find( "<", includepos );
			size_t fnameendLib = pSrc.find( ">", fnamestartLib + 1 );

			size_t fnameEndOfLine = pSrc.find( "\n", includepos );

			size_t fnamestart;
			size_t fnameend;

			bool uselibpath = false;
			if ( ( fnamestartLoc == std::string::npos || fnamestartLib < fnamestartLoc ) && fnamestartLib < fnameEndOfLine )
			{
				fnamestart = fnamestartLib;
				fnameend = fnameendLib;
				uselibpath = true;
			}
			else if ( fnamestartLoc != std::string::npos && fnamestartLoc < fnameEndOfLine )
			{
				fnamestart = fnamestartLoc;
				fnameend = fnameendLoc;
				uselibpath = false;
			}
			else
			{
                std::cerr << "manageIncludes : invalid #include directive into \"" << pSourceFileName.c_str() << "\"\n";
				return pSrc;
			}

			std::string incfilename = pSrc.substr( fnamestart + 1, fnameend - fnamestart - 1 );
			std::string incsource;

			if ( uselibpath )
			{
				std::string usedPath;

				// TODO: Add paths types into the manager -> search only onto shaders paths.
				std::vector< std::string > pathsList;
				// ResourcesManager::getManager()->getPaths( pathsList );
                pathsList.push_back( "./" );

				for ( std::vector< std::string >::iterator it = pathsList.begin(); it != pathsList.end(); it++ )
				{
					std::string fullpathtmp = (*it) + incfilename;
					
					FILE* file = 0;
					file = fopen( fullpathtmp.c_str(), "r" );
					if ( file )
					{
						usedPath = (*it);
						fclose( file );
						break;
					}
					else
					{
						usedPath = "";
					}
				}
				
				if ( usedPath != "" )
				{
					incsource = loadTextFile( ( usedPath + incfilename ).c_str() );
				}
				else
				{
                    std::cerr << "manageIncludes : Unable to find included file \"" << incfilename.c_str() << "\" in system paths.\n";
					return pSrc;
				}
			} else
			{
				incsource = loadTextFile(
					( pSourceFileName.substr( 0, pSourceFileName.find_last_of( "/", pSourceFileName.size() ) + 1 )
						+ incfilename ).c_str()
				);
			}

			incsource = manageIncludes( incsource, pSourceFileName );
			incsource = incsource.substr( 0, incsource.size() - 1 );
			
			std::string preIncludePart = pSrc.substr( 0, includepos );
			std::string postIncludePart = pSrc.substr( fnameend + 1, pSrc.size() - fnameend );

			int numline = 0;
			size_t newlinepos = 0;
			do
			{
				newlinepos = preIncludePart.find( "\n", newlinepos + 1 );
				numline++;
			}
			while ( newlinepos != std::string::npos );
			numline--;
			
			char buff2[ 512 ];
			sprintf( buff2, "\n#line 0\n" );
			std::string linePragmaPre( buff2 );
			sprintf( buff2, "\n#line %d\n", numline );
			std::string linePragmaPost( buff2 );
			
			res = preIncludePart + linePragmaPre + incsource + linePragmaPost + postIncludePart;

			pSrc = res;
		}
		includepos = pSrc.find( buff, includepos + 1 );
	}

	return pSrc;
}

/******************************************************************************
 * ...
 *
 * @param pProgramID ...
 * @param pStat ...
 ******************************************************************************/
void GvShaderManager::checkProgramInfos( GLuint pProgramID, GLuint pStat )
{
	GLint ok = 0;
	glGetProgramiv( pProgramID, pStat, &ok );
	if ( ! ok )
	{
		int ilength;
		glGetProgramInfoLog( pProgramID, STRING_BUFFER_SIZE, &ilength, stringBuffer );
		
		std::cout << "Program error :\n" << stringBuffer << "\n"; 
		
		int cc;
		std::cin >> cc;
	}
}
