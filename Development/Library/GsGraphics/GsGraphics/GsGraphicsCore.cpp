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

#include "GsGraphics/GsGraphicsCore.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// System
#include <cassert>
#include <cstdio>

// STL
#include <iostream>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GsGraphics;

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
GsGraphicsCore::GsGraphicsCore()
{
	/*glDispatchComputeEXT = (PFNGLDISPATCHCOMPUTEEXTPROC)wglGetProcAddress( "glDispatchCompute" );

	glTexStorage2D = (PFNGLTEXSTORAGE2DPROC)wglGetProcAddress( "glTexStorage2D" );*/
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GsGraphicsCore::~GsGraphicsCore()
{
}

/******************************************************************************
 * Print information about the device
 ******************************************************************************/
void GsGraphicsCore::printInfo()
{
	// Determine the OpenGL and GLSL versions
	const GLubyte* vendor = glGetString( GL_VENDOR );
	const GLubyte* renderer = glGetString( GL_RENDERER );
	const GLubyte* version = glGetString( GL_VERSION );
	const GLubyte* glslVersion = glGetString( GL_SHADING_LANGUAGE_VERSION );
	GLint major;
	GLint minor;
	glGetIntegerv( GL_MAJOR_VERSION, &major );
	glGetIntegerv( GL_MINOR_VERSION, &minor );
	printf( "\n" );
	printf( "GL Vendor : %s\n", vendor );
	printf( "GL Renderer : %s\n", renderer );
	printf( "GL Version (string) : %s\n", version );
	printf( "GL Version (integer) : %d.%d\n", major, minor );
	printf( "GLSL Version : %s\n", glslVersion );
	
	// TO DO
	// - check for NVX_gpu_memory_info experimental OpenGL extension
	//
	// TO DO
	// - track dedicated real-time parameters
	GLint glGpuMemoryInfoDedicatedVidmemNvx;
	GLint glGpuMemoryInfoTotalAvailableMemoryNvx;
	GLint glGpuMemoryInfoCurrentAvailableVidmemNvx;
	GLint glGpuMemoryInfoEvictionCountNvx;
	GLint glGpuMemoryInfoEvictedMemoryNvx;
	glGetIntegerv( 0x9047, &glGpuMemoryInfoDedicatedVidmemNvx );
	glGetIntegerv( 0x9048, &glGpuMemoryInfoTotalAvailableMemoryNvx );
	glGetIntegerv( 0x9049, &glGpuMemoryInfoCurrentAvailableVidmemNvx );
	glGetIntegerv( 0x904A, &glGpuMemoryInfoEvictionCountNvx );
	glGetIntegerv( 0x904B, &glGpuMemoryInfoEvictedMemoryNvx );
	std::cout << "\nNVIDIA Memory Status" << std::endl;
	std::cout << "- GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX : " << glGpuMemoryInfoDedicatedVidmemNvx << " kB" <<  std::endl;
	std::cout << "- GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX : " << glGpuMemoryInfoTotalAvailableMemoryNvx << " kB" <<  std::endl;
	std::cout << "- GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX : " << glGpuMemoryInfoCurrentAvailableVidmemNvx << " kB" <<  std::endl;
	std::cout << "- GL_GPU_MEMORY_INFO_EVICTION_COUNT_NVX : " << glGpuMemoryInfoEvictionCountNvx << std::endl;
	std::cout << "- GL_GPU_MEMORY_INFO_EVICTED_MEMORY_NVX : " << glGpuMemoryInfoEvictedMemoryNvx << " kB" << std::endl;

	// Compute Shaders
	std::cout << "\nOpenGL Compute Shader features" << std::endl;
	GLint glMAXCOMPUTEWORKGROUPINVOCATIONS;
	glGetIntegerv( GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &glMAXCOMPUTEWORKGROUPINVOCATIONS );
	std::cout << "- GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS : " << glMAXCOMPUTEWORKGROUPINVOCATIONS << std::endl;
	/*GLint glMAXCOMPUTEWORKGROUPCOUNT[3];
	glGetIntegerv( GL_MAX_COMPUTE_WORK_GROUP_COUNT, glMAXCOMPUTEWORKGROUPCOUNT );
	std::cout << "GL_MAX_COMPUTE_WORK_GROUP_COUNT : " << glMAXCOMPUTEWORKGROUPCOUNT[ 0 ] << " - " << glMAXCOMPUTEWORKGROUPCOUNT[ 1 ] << " - " << glMAXCOMPUTEWORKGROUPCOUNT[ 2 ] << std::endl;
	GLint glMAXCOMPUTEWORKGROUPSIZE[3];
	glGetIntegerv( GL_MAX_COMPUTE_WORK_GROUP_SIZE, glMAXCOMPUTEWORKGROUPSIZE );
	std::cout << "GL_MAX_COMPUTE_WORK_GROUP_SIZE : " << glMAXCOMPUTEWORKGROUPSIZE[ 0 ] << " - " << glMAXCOMPUTEWORKGROUPSIZE[ 1 ] << " - " << glMAXCOMPUTEWORKGROUPSIZE[ 2 ] << std::endl;
*/
	GLint cx, cy, cz;
	glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &cx );
	glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &cy );
	glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &cz );
	//fprintf( stderr, "Max Compute Work Group Count = %5d, %5d, %5d\n", cx, cy, cz );
	std::cout << "- GL_MAX_COMPUTE_WORK_GROUP_COUNT : " << cx << " - " << cy << " - " << cz << std::endl;

	glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &cx );
	glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &cy );
	glGetIntegeri_v( GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &cz );
	std::cout << "- GL_MAX_COMPUTE_WORK_GROUP_SIZE : " << cx << " - " << cy << " - " << cz << std::endl;
}
