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

// Qt
#include <QString>

// STL
#include <vector>

// CUDA
#include <vector_types.h>

// GL
#include <GL/glew.h>

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
 * @brief The ParticleSystem class provides ...
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
	 */
	ParticleSystem();

	/**
	 * Destructor
	 */
	virtual ~ParticleSystem();
		
	/**
	 * Initialize
	 */
	bool initialize();

	/**
	 * Finalize
	 */
	bool finalize();

	/**
	 * Load data from file
	 *
	 * @param pFilename the filename
	 */
	bool load();

	///**
	// * Initialize GL
	// */
	//bool initializeGL();

	/**
	 * Render the particle system
	 */
	void render();

	/**
	 * ...
	 */
	unsigned int getBrickNbPoints() const;
	void setBrickNbPoints( unsigned int pValue );
	bool hasBrickDrawOneSlice() const;
	void setBrickDrawOneSlice( bool pFlag );
	void setBrickPresenceFlags( unsigned int pBrickPresenceFlags[][ 8 ][ 8 ] );
	float getBrickPointSize() const;
	void setBrickPointSize( float pValue );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * List of points
	 */
	std::vector< float3 > _brick;
	std::vector< float3 > _points;

	/**
	 * ...
	 */
	unsigned int _brickNbPoints;

	/**
	 * Vertex array
	 */
	GLuint _vertexBuffer;

	/**
	 * Presence flags of points inside a brick
	 */
	unsigned int _brickPresenceFlags[ 8 ][ 8 ][ 8 ];

	/**
	 * ...
	 */
	bool _hasBrickDrawOneSlice;

	/**
	 * ...
	 */
	float _pointSize;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

#endif // !_PARTICLE_SYSTEM_H_
