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

#ifndef _PARTICLE_H_
#define _PARTICLE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <cuda_runtime.h>
#include <vector_types.h>
#include <curand_kernel.h>

// GigaVoxels
#include <GvCore/vector_types_ext.h>

// OpenGL
#include <GL/glew.h>

// Project
#include "CollisionDetectorKernel.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/
class Particle;
namespace Particles {

	/**
	 * Whether to run or not the application.
	 */
	__constant__ bool cRunAnimation = false;

	/**
	 * Gravity
	 */
	__constant__ float cGravity = 0.001f;

	/**
	 * Rebound
	 */
	__constant__ float cRebound = 0.8f;

	/**
	 * Fill the array of particles.
	 */
	__global__ 
	void createParticlesKernel( unsigned long long seed );

	void createParticles( unsigned long long seed, int nParticles );

	/**
	 * Array of particles.
	 */
	__device__
	Particle *particles;

	/**
	 * Number of particles in the array.
	 */
	__device__
	unsigned int nParticles;

	/**
	 * Index of the VBO containing the position of the particle (for the display).
	 */
	GLuint positionBuffer;

	/**
	 * VBO containing the position of the particle (for the display).
	 */
	struct cudaGraphicsResource *cuda_vbo_resource;

	/**
	 * Animate the particles.
	 */
	template< class TVolTreeKernelType, class GPUCacheType >
	void animation( const TVolTreeKernelType pVolumeTree,
			GPUCacheType pGPUCache,
			float3 *vboCollision,
			unsigned int nParticles );

	/**
	 * Animate the particles (device part).
	 */
	template< class TVolTreeKernelType, class GPUCacheType >
	__global__
	void animationKernel( const TVolTreeKernelType pVolumeTree,
			GPUCacheType pGPUCache,
			float3 *vboCollision );
}

/**
 * @class Particle
 *
 * @brief TODO
 *
 * TODO
 *
 */
struct Particle
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
	/**
	 * Position (center of the particle)
	 */
	float3 _position;

	/**
	 * Size of the x, y and z side.
	 */
	float3 _extents;

	/**
	 * Speed vector of the particle.
	 */
	float3 _speed;

	/**
	 * Basis.
	 */
	float4x4 _rotation;

	/**
	 * Angular speeds of the particle.
	 */
	float3 _angularSpeed;

	/******************************** METHODS *********************************/
	/**
	 * Initialize a particle with random values.
	 */
	__device__
	void init( curandState &state );

	/**
	 *
	 */
	void collisionDetection();

	/**
	 *
	 */
	__device__
	void collisionReaction( float3 normal );

	/**
	 * TODO
	 */
	template< class TVolTreeKernelType, class GPUCacheType >
	__device__
	float3 collision_BBOX_VolTree_Kernel (
				const TVolTreeKernelType pVolumeTree,
				GPUCacheType pGPUCache ) const;

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/


	/******************************** METHODS *********************************/


	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

#include "Particle.inl"

#endif // !_PARTICLE_H_
