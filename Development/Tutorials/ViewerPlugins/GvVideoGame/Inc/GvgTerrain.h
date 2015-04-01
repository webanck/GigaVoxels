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

#ifndef _GVG_TERRAIN_H_
#define _GVG_TERRAIN_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include <GvgObject.h>

// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

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
 * @class GvgTerrain
 *
 * @brief The GvgTerrain class provides ...
 *
 * @ingroup ...
 *
 * ...
 */
class GvgTerrain : public GvgObject
{

	/**************************************************************************
	 ***************************** FRIEND SECTION *****************************
	 **************************************************************************/

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Create method
	 *
	 * @return an instance of terrain
	 */
	static GvgTerrain* create();

	/**
	 * Destroy method
	 */
	void destroy();

	/**
	 * Initialize
	 *
	 * @return a flag to tell wheter or not it succeeds
	 */
	bool initialize();

	/**
	 * Finalize
	 *
	 * @return a flag to tell wheter or not it succeeds
	 */
	bool finalize();

	/**
	 * Render the terrain
	 */
	void render();
				
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Heightmap height
	 */
	unsigned int _heightmapHeight;

	/**
	 * Heightmap width
	 */
	unsigned int _heightmapWidth;

	/**
	 * Heightmap heights
	 */
	float** _heightmapHeights;

	/**
	 * Vertex array
	 */
	GLuint _terrainBuffer;

	/**
	 * Index array
	 */
	GLuint _terrainIndexBuffer;
		
	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvgTerrain();

	/**
	 * Destructor
	 */
	virtual ~GvgTerrain();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	unsigned char* loadBitmapFile( char* filename, BITMAPINFOHEADER* bitmapInfoHeader );

	/**
	 * Copy constructor forbidden.
	 */
	GvgTerrain( const GvgTerrain& );

	/**
	 * Copy operator forbidden.
	 */
	GvgTerrain& operator=( const GvgTerrain& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#endif // !_GVG_TERRAIN_H_
