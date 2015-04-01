/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2014 INRIA <http://www.inria.fr/>
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

#ifndef _OBJECT_RECEIVING_SHADOW_H_
#define _OBJECT_RECEIVING_SHADOW_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

#include "ShaderManager.h"
#include "Mesh.h"
namespace GvCore
{
	template<typename T>
	class Array3DGPULinear;
}

/** 
 * @class ObjectReceivingShadow
 *
 * @brief The ObjectReceivingShadow class provides the mecanisms to manage mesh/scene objects
 * that should receive shadows.
 *
 * This class is the base class for all "receiving shadow" objects.
 */
class ObjectReceivingShadow
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * ...
	 */
	ObjectReceivingShadow();

	/**
	 * ...
	 */
	void init();

	/**
	 * ...
	 */
	void render();

	/**
	 * ...
	 */
	void setLightPosition( float x, float y, float z );

	/**
	 * ...
	 */
	~ObjectReceivingShadow();

	/**
	 * ...
	 */
	void setBrickCacheSize( unsigned int x, unsigned int y, unsigned int z );

	/**
	 * ...
	 */
	void setBrickPoolResInv( float x, float y, float z );

	/**
	 * ...
	 */
	void setMaxDepth( unsigned int v );

	/**
	 * ...
	 */
	void setVolTreeChildArray( GvCore::Array3DGPULinear< unsigned int >* v, GLint id );

	/**
	 * ...
	 */
	void setVolTreeDataArray( GvCore::Array3DGPULinear< unsigned int >* v, GLint id );

	/**
	 * ...
	 */
	void setModelMatrix( float m00, float m01, float m02, float m03,
						float m10, float m11, float m12, float m13,
						float m20, float m21, float m22, float m23,
						float m30, float m31, float m32, float m33 );

	/**
	 * ...
	 */
	void setWorldLight( float x, float y, float z );

	/**
	 * ...
	 */
	void setTexBufferName( GLint v );

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

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	bool loadedObject;//true if the user loads an object, false if he creates it manually

	/**
	 * ...
	 */
	//if the user wants to load a file 
	Mesh* object;

	/**
	 * ...
	 */
	//if he wants to define the object manually
	GLuint idVBO;

	/**
	 * ...
	 */
	GLuint idIndices;

	/**
	 * ...
	 */
	GLuint vshader;

	/**
	 * ...
	 */
	GLuint fshader;

	/**
	 * ...
	 */
	GLuint program;

	/**
	 * ...
	 */
	//Light positions
	float lightPos[ 3 ];

	/**
	 * ...
	 */
	float worldLight[ 3 ];

	/**
	 * ...
	 */
	//GigaVoxels object casting shadows stuff 
	unsigned int brickCacheSize[ 3 ];

	/**
	 * ...
	 */
	float brickPoolResInv[ 3 ];

	/**
	 * ...
	 */
	unsigned int maxDepth;

	/**
	 * ...
	 */
	GLuint _childArrayTBO;

	/**
	 * ...
	 */
	GLuint _dataArrayTBO;

	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< unsigned int >* volTreeChildArray;

	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< unsigned int >* volTreeDataArray;

	/**
	 * ...
	 */
	GLint childBufferName;

	/**
	 * ...
	 */
	GLint dataBufferName;

	/**
	 * ...
	 */
	GLint texBufferName;

	/**
	 * ...
	 */
	float modelMatrix[ 16 ];

	/******************************** METHODS *********************************/

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

//#include "ObjectReceivingShadow.inl"

#endif // !_OBJECT_RECEIVING_SHADOW_H_
