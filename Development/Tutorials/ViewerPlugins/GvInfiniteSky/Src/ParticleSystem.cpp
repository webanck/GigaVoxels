/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2012 INRIA <http://www.inria.fr/>
 *
 * Authors : GigaVoxels Team
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

#include "ParticleSystem.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// STL
#include <iostream>

// System
#include <cstdlib>

// Cuda
#include <cuda_runtime.h>

// GigaVoxels
#include <GvCore/vector_types_ext.h>
#include <GvCore/GvError.h>

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
 * Constructor
 ******************************************************************************/
ParticleSystem::ParticleSystem( const float3& pPoint1, const float3& pPoint2 )
:	_p1( pPoint1 )
,	_p2( pPoint2 )
,	_nbParticles( 998 )
,	_d_particleBuffer( NULL )
,	_sphereRadiusFader( 1.f )
,	_fixedSizeSphereRadius( 0.f )
{
    _particleBuffer = new float4[_nbParticles];
    initBuf();
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
ParticleSystem::~ParticleSystem()
{
	// TO DO
	// Handle destruction
	// ...
}

/******************************************************************************
 * Initialise le buffer GPU contenant les positions
 ******************************************************************************/
/*
void ParticleSystem::initGPUBuf()
{
    if ( _d_particleBuffer != NULL )
    {
        GV_CUDA_SAFE_CALL( cudaFree( _d_particleBuffer ) );
        _d_particleBuffer = NULL;
    }

    float4* part_buf = new float4[ _nbParticles ];

    for ( unsigned int i = 0; i < _nbParticles; ++i )
    {
        // Position (generee aleatoirement)
        float4 sphere = genPos( rand() );
        part_buf[ i ] = sphere;
    }


    size_t size = _nbParticles * sizeof( float4 );

    //_d_particleBuffer = new GvCore::Array3DGPULinear( make_int3( _nbParticles, 1, 1 ), 0 );
    if ( cudaSuccess != cudaMalloc( &_d_particleBuffer, size ) )
    {
        return;
    }

    GV_CUDA_SAFE_CALL( cudaMemcpy( _d_particleBuffer, part_buf, size, cudaMemcpyHostToDevice ) );

    // TO DO
    // Delete the temporary buffer : part_buf
    // ...
}
*/

void ParticleSystem::initBuf()
{

    for ( unsigned int i = 0; i < _nbParticles; ++i )
    {
        // Position (generee aleatoirement)
        float4 sphere = genPos( rand() );
        _particleBuffer[i] = sphere;
    }

    // TO DO
    // Delete the temporary buffer : part_buf
    // ...
}

void ParticleSystem::loadGPUBuf(){

    if ( _d_particleBuffer != NULL )
    {
        GV_CUDA_SAFE_CALL( cudaFree( _d_particleBuffer ) );
        _d_particleBuffer = NULL;
    }
    size_t size = _offset * sizeof( float4 );

    //_d_particleBuffer = new GvCore::Array3DGPULinear( make_int3( _nbParticles, 1, 1 ), 0 );
    if ( cudaSuccess != cudaMalloc( &_d_particleBuffer, size ) )
    {
        return;
    }

    GV_CUDA_SAFE_CALL( cudaMemcpy( _d_particleBuffer, _particleBuffer, size, cudaMemcpyHostToDevice ) );


}

/******************************************************************************
 * Get the buffer of data (sphere positions and radius)
 *
 * @return the buffer of data (sphere positions and radius)
 ******************************************************************************/
float4* ParticleSystem::getGPUBuf()
{
	return _d_particleBuffer;
}

/******************************************************************************
 * Genere une position aleatoire
 *
 * @param pSeed ...
 ******************************************************************************/
float4 ParticleSystem::genPos( int pSeed )
{
    float4 p;

	srand( pSeed );

	float min;	// min de l'interval des valeurs sur l'axe
	float max;	// max de l'interval des valeurs sur l'axe

    // Radius
    p.w = .005f + static_cast< float >( rand() ) / ( static_cast< float >( RAND_MAX ) / ( 0.02f - 0.005f ) );	// rayon de l'etoile dans [0.005 : 0.02]
    // Global size gain
    p.w *= _sphereRadiusFader;

    // genere la coordonnee en X
	if ( _p1.x < _p2.x )
	{
		min = _p1.x;
		max = _p2.x;
	}
	else
	{
		min = _p2.x;
		max = _p1.x;
	}
    p.x = (min+p.w+p.w) + (float)rand() / ((float)RAND_MAX / ((max-p.w-p.w)-(min+p.w+p.w)));

    // genere la coordonnee en Y
	if ( _p1.y < _p2.y )
	{
		min = _p1.y;
		max = _p2.y;
	}
	else
	{
		min = _p2.y;
		max = _p1.y;
	}
    p.y = (min+p.w+p.w) + (float)rand() / ((float)RAND_MAX / ((max-p.w-p.w)-(min+p.w+p.w)));

    // genere la coordonnee en Z
	if ( _p1.z < _p2.z )
	{
		min = _p1.z;
		max = _p2.z;
	}
	else
	{
		min = _p2.z;
		max = _p1.z;
	}
    p.z = (min+p.w+p.w) + (float)rand() / ((float)RAND_MAX / ((max-p.w-p.w)-(min+p.w+p.w)));

	return p;
}

/******************************************************************************
 * Get the number of particles
 *
 * @return the number of particles
 ******************************************************************************/
unsigned int ParticleSystem::getNbParticles() const
{
    //return _nbParticles;
    return _offset;
}


/******************************************************************************
 * Set the number of particles
 *
 * @param pValue the number of particles
 ******************************************************************************/
void ParticleSystem::setNbParticles( unsigned int pValue )
{
    //_nbParticles = pValue;
    _offset = pValue;
}

/******************************************************************************
 * ...
 ******************************************************************************/
float ParticleSystem::getSphereRadiusFader() const
{
	return _sphereRadiusFader;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ParticleSystem::setSphereRadiusFader( float pValue )
{
	_sphereRadiusFader = pValue;
}

/******************************************************************************
 * ...
 ******************************************************************************/
float ParticleSystem::getFixedSizeSphereRadius() const
{
	return _fixedSizeSphereRadius;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ParticleSystem::setFixedSizeSphereRadius( float pValue )
{
	_fixedSizeSphereRadius = pValue;
}
