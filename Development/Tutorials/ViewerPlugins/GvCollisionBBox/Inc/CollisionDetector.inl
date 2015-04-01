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

//#include <cudaProfiler.h>
#include <cuda_profiler_api.h>

namespace GvCollision {

/******************************************************************************
 * TODO
 ******************************************************************************/
template< class TVolTreeKernelType >
bool collision_Point_VolTree( TVolTreeKernelType pVolumeTree,
		    float3 pPoint,
		    float pPrecision ) {

	collision_Point_VolTree_Kernel< TVolTreeKernelType > <<< 1,1 >>>( 
				pVolumeTree,
				pPoint,
				pPrecision
			);

	bool ret;
	GV_CUDA_SAFE_CALL( cudaMemcpyFromSymbol( &ret, GvCollision::collision, sizeof( ret ), 0, cudaMemcpyDeviceToHost ) );

	return ret;
}

/******************************************************************************
 * TODO
 ******************************************************************************/
template< class TVolTreeKernelType >
void collision_BBOX_VolTree(
			const TVolTreeKernelType &volumeTree,
		    const std::vector< unsigned int > &precisions,
	   		const std::vector< float3 > &positions,
			const std::vector< float3 > &extents,
			const std::vector< float4x4 > &basis,
	   		std::vector< float > &results )
{
	//CUresult error;
	cudaError_t error;
	
	//error = cuProfilerStart();
	//error = cudaProfilerStart();
	//if ( error == cudaErrorProfilerNotInitialized )
	//if ( error != cudaSuccess )
	//{
	//	//std::cout << "ERROR : cuProfilerStart() is called without initializing profiler" << std::endl;
	//	std::cout << "ERROR : cudaProfilerStart() is called without initializing profiler" << std::endl;
	//}

	// Events to time the collision time
	cudaEvent_t startCollision, stopCollision;
	cudaEventCreate( &startCollision );
	cudaEventCreate( &stopCollision );

	uint arraysSize = precisions.size();

	// Enforce arrays size
	assert( positions.size() == arraysSize );
	assert( extents.size() == arraysSize );
	assert( basis.size() == arraysSize );

	// Allocate memory on the device side and copy arrays to said memory
	// TODO gestion des erreurs
	unsigned int *devicePrecisions;
	float3 *devicePositions;
	float3 *deviceExtents;
	float4x4 *deviceBasis;
	float *deviceResults;

	cudaMalloc(( void ** )&devicePrecisions, arraysSize * sizeof( unsigned int ));
	cudaMalloc(( void ** )&devicePositions, arraysSize * sizeof( float3 ));
	cudaMalloc(( void ** )&deviceExtents, arraysSize * sizeof( float3 ));
	cudaMalloc(( void ** )&deviceBasis, arraysSize * sizeof( float4x4 ));
	cudaMalloc(( void ** )&deviceResults, arraysSize * sizeof( float ));

	GV_CHECK_CUDA_ERROR( "collision_BBOX_VolTree : allocation" );

	// Copy arrays to device
	// TODO gestion des erreurs
	cudaMemcpy( devicePrecisions, &precisions[0], arraysSize * sizeof( unsigned int ), cudaMemcpyHostToDevice );
	cudaMemcpy( devicePositions, &positions[0], arraysSize * sizeof( float3 ), cudaMemcpyHostToDevice );
	cudaMemcpy( deviceExtents, &extents[0], arraysSize * sizeof( float3 ), cudaMemcpyHostToDevice );
	cudaMemcpy( deviceBasis, &basis[0], arraysSize * sizeof( float4x4 ), cudaMemcpyHostToDevice );

	GV_CHECK_CUDA_ERROR( "collision_BBOX_VolTree : copy parameters" );

	// Call the kernel
	// TODO : nb threads/blocs
	cudaEventRecord( startCollision, 0 );
	collision_BBOX_VolTree_Kernel< TVolTreeKernelType > <<< arraysSize, 1 >>>( 
				volumeTree,
				devicePrecisions,
				devicePositions,
				deviceExtents,
				deviceBasis,
				deviceResults,
				arraysSize
			);
	cudaEventRecord( stopCollision, 0 );
	cudaEventSynchronize( stopCollision );

	float elapsedTime;
	cudaEventElapsedTime( &elapsedTime, startCollision, stopCollision );

	std::cout << "Time : " << elapsedTime << "ms" << std::endl;


	results.reserve( arraysSize );

	GV_CHECK_CUDA_ERROR( "collision_BBOX_VolTree : kernel call" );

	// Copy the results back
	cudaMemcpy( &results[0], deviceResults, arraysSize * sizeof( float ), cudaMemcpyDeviceToHost );
	GV_CHECK_CUDA_ERROR( "collision_BBOX_VolTree : copy results" );

	// Free the memory.
	cudaFree( devicePrecisions );
	cudaFree( devicePositions );
	cudaFree( deviceExtents );
	cudaFree( deviceBasis );
	cudaFree( deviceResults );

	GV_CHECK_CUDA_ERROR( "collision_BBOX_VolTree : free memory" );

	cudaEventDestroy( startCollision );
	cudaEventDestroy( stopCollision );
	GV_CHECK_CUDA_ERROR( "collision_BBOX_VolTree : events" );

	//error = cuProfilerStop();
	//error = cudaProfilerStop();
	//if ( error == cudaErrorProfilerNotInitialized )
	//if ( error != cudaSuccess )
	//{
	//	//std::cout << "ERROR : cuProfilerStop() is called without initializing profiler" << std::endl;
	//	std::cout << "ERROR : cudaProfilerStop() is called without initializing profiler" << std::endl;
	//}
}

}; // GvCollision
