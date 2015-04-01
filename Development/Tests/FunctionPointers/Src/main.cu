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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "Kernel.h"

// Cuda
#include <cuda_runtime.h>

// System
#include <cstring>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <sstream>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

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
 * Main entry program
 *
 * @param pArgc number of arguments
 * @param pArgv list of arguments
 *
 * @return exit code
 ******************************************************************************/
int main( int pArgc, char** pArgv )
{
	// CUDA variables used for benchmark (events, timers)
	cudaError_t cudaResult;
	cudaEvent_t startEvent;
	cudaEvent_t stopEvent;
	cudaEventCreate( &startEvent );
	cudaEventCreate( &stopEvent );

	// Declare variables
	unsigned char* h_inputData = NULL;
	unsigned char* d_inputData = NULL;
	unsigned char* d_outputData = NULL;
	const unsigned int cNbElements = 1000000;
	const unsigned int cNbIterations = 25;
	float elapsedTime = 0.0f;
	float totalElapsedTime = 0.0f;

	// Allocate data
	h_inputData = new unsigned char[ cNbElements ];
	cudaMalloc( (void**)&d_inputData, cNbElements * sizeof( unsigned char ) );
	cudaMalloc( (void**)&d_outputData, cNbElements * sizeof( unsigned char ) );

	// Initialize input data
	for ( unsigned int i = 0; i < cNbElements; i++ )
	{
		h_inputData[ i ] = i / 255;
	}
	// Copy data on device
	cudaMemcpy( d_inputData, h_inputData, cNbElements * sizeof( unsigned char ), cudaMemcpyHostToDevice );

	// Setup kernel execution parameters
	dim3 gridSize( cNbElements / 256 + 1, 1, 1 );
	dim3 blockSize( 256, 1, 1 );

	// Benchmark
	//
	// - standard kernel
	for ( unsigned int i = 0; i < cNbIterations; i++ )
	{
		// Start event
		cudaEventRecord( startEvent, 0 );

		// Launch standard kernel
		Kernel_StandardAlgorithm<<< gridSize, blockSize >>>( cNbElements, d_inputData, d_outputData );
	
		// Stop event and record elapsed time
		cudaEventRecord( stopEvent, 0 );
		cudaEventSynchronize( stopEvent );
		cudaEventElapsedTime( &elapsedTime, startEvent, stopEvent );
		totalElapsedTime += elapsedTime;

		// Checking errors
		cudaResult = cudaGetLastError();
	}
	std::cout << "Kernel_StandardAlgorithm : " << ( totalElapsedTime / cNbIterations ) << " ms" << std::endl;

	// Benchmark
	//
	// - kernel with call to device function pointer benchmark
	totalElapsedTime = 0.0f;
	for ( unsigned int i = 0; i < cNbIterations; i++ )
	{
		// Start event
		cudaEventRecord( startEvent, 0 );

		// Launch kernel with device function pointer call
		Kernel_AlgorithmWithFunctionPointer<<< gridSize, blockSize >>>( cNbElements, d_inputData, d_outputData );
	
		// Stop event and record elapsed time
		cudaEventRecord( stopEvent, 0 );
		cudaEventSynchronize( stopEvent );
		cudaEventElapsedTime( &elapsedTime, startEvent, stopEvent );
		totalElapsedTime += elapsedTime;

		// Checking errors
		cudaResult = cudaGetLastError();
	}
	std::cout << "Kernel_AlgorithmWithFunctionPointer : " << ( totalElapsedTime / cNbIterations ) << " ms" << std::endl;

	// Benchmark
	//
	// - standard kernel
	totalElapsedTime = 0.0f;
	for ( unsigned int i = 0; i < cNbIterations; i++ )
	{
		// Start event
		cudaEventRecord( startEvent, 0 );

		// Launch standard kernel
		Kernel_StandardAlgorithm_2<<< gridSize, blockSize >>>( cNbElements, d_inputData, d_outputData );
	
		// Stop event and record elapsed time
		cudaEventRecord( stopEvent, 0 );
		cudaEventSynchronize( stopEvent );
		cudaEventElapsedTime( &elapsedTime, startEvent, stopEvent );
		totalElapsedTime += elapsedTime;

		// Checking errors
		cudaResult = cudaGetLastError();
	}
	std::cout << "Kernel_StandardAlgorithm with 2 functions : " << ( totalElapsedTime / cNbIterations ) << " ms" << std::endl;

	// Benchmark
	//
	// - standard kernel
	totalElapsedTime = 0.0f;
	for ( unsigned int i = 0; i < cNbIterations; i++ )
	{
		// Start event
		cudaEventRecord( startEvent, 0 );

		// Launch standard kernel
		Kernel_AlgorithmWithFunctionPointer_2Functions<<< gridSize, blockSize >>>( cNbElements, d_inputData, d_outputData );
	
		// Stop event and record elapsed time
		cudaEventRecord( stopEvent, 0 );
		cudaEventSynchronize( stopEvent );
		cudaEventElapsedTime( &elapsedTime, startEvent, stopEvent );
		totalElapsedTime += elapsedTime;

		// Checking errors
		cudaResult = cudaGetLastError();
	}
	std::cout << "Kernel_AlgorithmWithFunctionPointer with 2 functions : " << ( totalElapsedTime / cNbIterations ) << " ms" << std::endl;

	// Clean up to ensure correct profiling
	cudaResult = cudaDeviceReset();

	return 0;
}
