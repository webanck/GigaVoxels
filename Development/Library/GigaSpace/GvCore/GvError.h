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

#ifndef _GV_ERROR_H_
#define _GV_ERROR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

// System 
#include <cstdlib>
#include <cstdio>

// CUDA
#include <cuda_runtime.h>

// OpenGL
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
 
// Windows
#ifdef WIN32
# include <windows.h>
#endif

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{

/**
 * Check for OpenGL errors
 *
 * @param pFile File in which errors are checked
 * @param pLine Line of file on which errors are checked
 *
 * @return Flag to say wheter or not there has been an error
 */
inline bool checkGLError( const char* pFile, const int pLine )
{
	// Check for error
	GLenum error = glGetError();
	if ( error != GL_NO_ERROR )
	{

// Windows specific stuff
#ifdef _WIN32
		char buf[ 512 ];
		sprintf( buf, "\n%s(%i) : GL Error : %s\n\n", pFile, pLine, gluErrorString( error ) );
		OutputDebugStringA( buf );
#endif

		fprintf( stderr, "GL Error in file '%s' in line %d :\n", pFile, pLine );
		fprintf( stderr, "%s\n", gluErrorString( error ) );

		return false;
	}

	return true;
}

} // namespace GvCore

/******************************************************************************
 ****************************** MACRO DEFINITION ******************************
 ******************************************************************************/

/**
 * MACRO
 * 
 * Call a Cuda method in a safe way (by checking error)
 */
#define GV_CUDA_SAFE_CALL( call )													\
{																					\
    cudaError_t error = call;														\
    if ( cudaSuccess != error )														\
	{																				\
		/* Write error info */														\
		fprintf( stderr, "\nCuda error :\n\t- file : '%s' \n\t- line %i : %s",		\
				__FILE__, __LINE__, cudaGetErrorString( error ) );					\
																					\
		/* Exit program */															\
        exit( EXIT_FAILURE );														\
    }																				\
}

// TO DO : add a flag to Release mode to optimize code => no check
/**
 * MACRO
 * 
 * Check for CUDA error
 */
#ifdef _DEBUG

// Debug mode version
#define GV_CHECK_CUDA_ERROR( pText )												\
{																					\
	/* Check for error */															\
	cudaError_t error = cudaGetLastError();											\
	if ( cudaSuccess != error )														\
	{																				\
		/* Write error info */														\
		fprintf( stderr, "\nCuda error : %s \n\t- file : '%s' \n\t- line %i : %s",	\
				pText, __FILE__, __LINE__, cudaGetErrorString( error ) );			\
																					\
		/* Exit program */															\
		exit( EXIT_FAILURE );														\
	}																				\
																					\
	/* Blocks until the device has completed all preceding requested tasks */		\
	error = cudaDeviceSynchronize();												\
	if ( cudaSuccess != error )														\
	{																				\
		fprintf( stderr, "Cuda error : %s in file '%s' in line %i : %s.\n",			\
				pText, __FILE__, __LINE__, cudaGetErrorString( error ) );			\
																					\
		/* Exit program */															\
		exit( EXIT_FAILURE );														\
	}																				\
}

#else

// TO DO : optimize code for Release/Distribution => don't call cudaGetLastError()

// Release mode version
#define GV_CHECK_CUDA_ERROR( pText )												\
{																					\
	/* Check for error */															\
	cudaError_t error = cudaGetLastError();											\
	if ( cudaSuccess != error )														\
	{																				\
		/* Write error info */														\
		fprintf( stderr, "\nCuda error : %s \n\t- file : '%s' \n\t- line %i : %s",	\
				pText, __FILE__, __LINE__, cudaGetErrorString( error ) );			\
																					\
		/* Exit program */															\
		exit( EXIT_FAILURE );														\
	}																				\
}

#endif

/******************************************************************************
 ****************************** MACRO DEFINITION ******************************
 ******************************************************************************/

/**
 * MACRO
 *
 * Check for OpenGL errors
 */
#ifdef NDEBUG

	// Release mode version
	#define GV_CHECK_GL_ERROR()

#else

	// Debug mode version
	#define GV_CHECK_GL_ERROR()														\
	if ( ! GvCore::checkGLError( __FILE__, __LINE__ ) )								\
	{																				\
		/* Exit program */															\
		exit( EXIT_FAILURE );														\
	}

#endif

#endif // !_GV_ERROR_H_
