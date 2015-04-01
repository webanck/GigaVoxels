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

#ifndef _PARTICLE_SYSTEM_H_
#define _PARTICLE_SYSTEM_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <vector_types.h>

// STL library
#include <vector>

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

/** 
 * @class ParticleSystem
 *
 * @brief The ParticleSystem class provides the mecanisms to generate star positions.
 *
 * ...
 */
class ParticleSystem
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * Recupere deux points de la diagonale du cube pour connaitre l'intervalle de valeurs
	 */
	ParticleSystem( const float3& pPoint1, const float3& pPoint2 );

	/**
	 * Destructor
	 */
	~ParticleSystem();

    /**
     * Initialise le buffer CPU contenant les positions
     */
    void initBuf();

	/**
     * Charge le buffer GPU contenant les positions
	 */
    void loadGPUBuf();

	/**
	  * Get the buffer of data (sphere positions and radius)
	 *
	 * @return the buffer of data (sphere positions and radius)
	 */
	// Array3DGPULinear< float4 >* getGPUBuf();
	float4* getGPUBuf();

	/**
	 * Get the number of particles
	 *
	 * @return the number of particles
	 */
	unsigned int getNbParticles() const;

	/**
	 * Set the number of particles
	 *
	 * @param pValue the number of particles
	 */
	void setNbParticles( unsigned int pValue );

	/**
	 * Spheres ray-tracing methods
	 */
	float getSphereRadiusFader() const;
	void setSphereRadiusFader( float pValue );
	float getFixedSizeSphereRadius() const;
	void setFixedSizeSphereRadius( float pValue );
	//bool hasMeanSizeOfSpheres() const;
	//void setMeanSizeOfSpheres( bool pFlag );
	
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

private :

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Point devant en bas a gauche du cube
	 */
	float3 _p1;

	/**
	 * Point derriere en haut a droite du cube
	 */
	float3 _p2;

	/**
	 * Max number of particles
	 */
	unsigned int _nbParticles;

	/**
	 * The buffer of data (sphere positions and radius)
	 */
    float4* _d_particleBuffer;
    int _bufferSize;

    //std::vector<float4> _particleBuffer;
    float4* _particleBuffer;

    /**
     * offset to find the next free position in the buffer
     */
    int _offset;

	/**
	 * Spheres ray-tracing parameters
	 */
	float _sphereRadiusFader;
	float _fixedSizeSphereRadius;

	/******************************** METHODS *********************************/

	/**
	 * Genere une position aleatoire
	 *
	 * @param pSeed ...
	 */
    float4 genPos( int pSeed );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "ParticleSystem.inl"

#endif
