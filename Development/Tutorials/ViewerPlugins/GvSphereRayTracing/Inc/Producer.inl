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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "ParticleSystem.h"

// System
#include <cassert>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
Producer< TDataStructureType, TDataProductionManager >
::Producer()
:	GvUtils::GvSimpleHostProducer< ProducerKernel< TDataStructureType >, TDataStructureType, TDataProductionManager >()
,	_particleSystem( NULL )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
Producer< TDataStructureType, TDataProductionManager >
::~Producer()
{
	// Finalize the producer and its particle system
	finalize();
}

/******************************************************************************
 * Initialize producer and generate particles
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::initialize( TDataStructureType* pDataStructure, TDataProductionManager* pDataProductionManager )
{
	// Call parent class
	GvUtils::GvSimpleHostProducer< ProducerKernel< TDataStructureType >, TDataStructureType, TDataProductionManager >::initialize( pDataStructure, pDataProductionManager );


	// Points definissant l'interval du cube
	float3 p1 = make_float3( 0.f, 0.f, 0.f );
	float3 p2 = make_float3( 1.f, 1.f, 1.f );

	// Nombre de particules a generer aleatoirement dans le cube
    _particleSystem = new ParticleSystem( p1, p2 );

	
}

/******************************************************************************
 * Finalize the producer and its particle system
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::finalize()
{
	// TO DO
	// Check if there are special things to do here... ?
	// ...
	delete _particleSystem;
	_particleSystem = NULL;	
}

/******************************************************************************
 * Spheres ray-tracing
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline unsigned int Producer< TDataStructureType, TDataProductionManager >
::getNbSpheres() const
{
	unsigned int nbSpheres = 0;

	assert( _particleSystem != NULL );
	if ( _particleSystem != NULL )
	{
		nbSpheres = _particleSystem->getNbParticles();
	}

	return nbSpheres;
}

/******************************************************************************
 * Spheres ray-tracing
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::setNbSpheres( unsigned int pValue )
{
	assert( _particleSystem != NULL );
	if ( _particleSystem != NULL )
	{
		_particleSystem->setNbParticles( pValue );
	}

	// Update the particle system
    updateParticleSystem();
}


/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::generateNewParticleBuffer(){

    assert( _particleSystem != NULL );
    if ( _particleSystem != NULL )
    {
        _particleSystem->initBuf();
    }
    // Update the particle system
    updateParticleSystem();

}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline float Producer< TDataStructureType, TDataProductionManager >
::getSphereRadiusFader() const
{
	float sphereRadiusFader = 0.f;

	assert( _particleSystem != NULL );
	if ( _particleSystem != NULL )
	{
		sphereRadiusFader = _particleSystem->getSphereRadiusFader();
	}

	return sphereRadiusFader;
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::setSphereRadiusFader( float pValue )
{
	assert( _particleSystem != NULL );
	if ( _particleSystem != NULL )
	{
		_particleSystem->setSphereRadiusFader( pValue );
	}

	// Update the particle system
	updateParticleSystem();
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline float Producer< TDataStructureType, TDataProductionManager >
::getFixedSizeSphereRadius() const
{
	float fixedSizeSphereRadius = 0.f;

	assert( _particleSystem != NULL );
	if ( _particleSystem != NULL )
	{
		fixedSizeSphereRadius = _particleSystem->getFixedSizeSphereRadius();
	}

	return fixedSizeSphereRadius;
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::setFixedSizeSphereRadius( float pValue )
{
    assert( _particleSystem != NULL );
    if ( _particleSystem != NULL )
    {
        _particleSystem->setFixedSizeSphereRadius( pValue );
    }

    // Update the particle system
    updateParticleSystem();
}

/******************************************************************************
 * Update the associated particle system
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::updateParticleSystem()
{
    assert( _particleSystem != NULL );
    if ( _particleSystem != NULL )
    {
        // Re-load spheres buffer
        _particleSystem->loadGPUBuf();

        // Update Kernel Producer info
        this->_kernelProducer.setPositionBuffer( _particleSystem->getGPUBuf() );
    }
}
