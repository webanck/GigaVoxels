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

#ifndef _MESH_H_
#define _MESH_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "ShaderManager.h"

// Assimp
//for assimp 2
//#include <assimp/Importer.hpp> // C++ importer interface
//#include <assimp/assimp.hpp>
//#include <assimp/aiConfig.h>
#include <assimp/Importer.hpp> // C++ importer interface
#include <assimp/scene.h> // Output data structure
#include <assimp/postprocess.h> // Post processing flags

// GL
#include <GL/glew.h>

// Qt
#include <QDir>
#include <QDirIterator>
#include <QGLWidget>

// STL
#include <vector>
#include <iostream>
#include <fstream>
#include <limits>

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
 * @class oneMesh
 *
 * @brief The oneMesh class provides the interface to manage meshes
 * 
 * This class is the base class for all mesh objects.
 */
struct oneMesh
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	GLuint VB;//vertex buffer id

	/**
	 * ...
	 */
	GLuint IB;//index buffer id

	/**
	 * ...
	 */
	std::vector< GLfloat > Vertices;

	/**
	 * ...
	 */
	std::vector< GLfloat > Normals;

	/**
	 * ...
	 */
	std::vector< GLfloat > Textures;

	/**
	 * ...
	 */
	std::vector<GLuint> Indices;

	/**
	 * ...
	 */
	GLenum mode;//GL_QUADS OR GL_TRIANGLES		

	/**
	 * ...
	 */
	float ambient[ 4 ];

	/**
	 * ...
	 */
	float diffuse[ 4 ];

	/**
	 * ...
	 */
	float specular[ 4 ];

	/**
	 * ...
	 */
	std::vector< std::string > texFiles[ 3 ];//one for ambient, diffuse, specular 

	/**
	 * ...
	 */
	std::vector< GLuint > texIDs[ 3 ];

	/**
	 * ...
	 */
	bool hasATextures;

	/**
	 * ...
	 */
	bool hasDTextures;

	/**
	 * ...
	 */
	bool hasSTextures;

	/**
	 * ...
	 */
	float shininess;

	/******************************** METHODS *********************************/

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

	/******************************** METHODS *********************************/

};

/** 
 * @class Mesh
 *
 * @brief The Mesh class provides the interface to manage a scene as a collection of meshes
 * 
 * This class is the base class for all scene objects.
 */
class Mesh
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
	Mesh( GLuint p = 0 );

	/**
	 * ...
	 */
	void loadTexture( const char* filename, GLuint id );

	/**
	 * ...
	 */
	void InitFromScene( const aiScene* scene );

	/**
	 * ...
	 */
	bool chargerMesh( const std::string& filename ); //loads file

	/**
	 * ...
	 */
	void creerVBO();

	/**
	 * ...
	 */
	void renderMesh( int i );

	/**
	 * ...
	 */
	void render(); //renders scene

	/**
	 * ...
	 */
	std::vector< oneMesh > getMeshes();

	/**
	 * ...
	 */
	int getNumberOfMeshes();

	/**
	 * ...
	 */
	void getAmbient( float tab[ 4 ], int i );

	/**
	 * ...
	 */
	void getDiffuse( float tab[ 4 ], int i );

	/**
	 * ...
	 */
	void getSpecular( float tab[ 4 ], int i );

	/**
	 * ...
	 */
	void getShininess( float &s, int i );

	/**
	 * ...
	 */
	void setLightPosition( float x, float y, float z );

	/**
	 * ...
	 */
	bool hasTexture( int i );

	/**
	 * ...
	 */
	float getScaleFactor();

	/**
	 * ...
	 */
	void getTranslationFactors( float translation[ 3 ] );

	/**
	 * ...
	 */
	~Mesh();

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

	/******************************** METHODS *********************************/

	/**
	 * ...
	 */
	float boundingBoxSide;

	/**
	 * ...
	 */
	float center[3];

	/**
	 * ...
	 */
	std::vector< oneMesh > meshes;//all the meshes in the scene

	/**
	 * ...
	 */
	std::string Dir;

	/**
	 * ...
	 */
	GLuint program;

	/**
	 * ...
	 */
	float lightPos[ 3 ];

	/**
	 * ...
	 */
	float boxMin[ 3 ];

	/**
	 * ...
	 */
	float boxMax[ 3 ];

};

/**
 * ...
 */
std::string Directory( const std::string& filename );

/**
 * ...
 */
std::string Filename( const std::string& path );

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

//#include "Mesh.inl"

#endif // !_MESH_H_
