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

#ifndef _DEPTH_PEELING_H_
#define _DEPTH_PEELING_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

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

// GigaVoxels
namespace GvUtils
{
	class GvShaderProgram;
}

// Assimp
struct aiScene;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class Renderer
 *
 * @brief The Renderer class provides an implementation of a renderer
 * specialized for CUDA.
 *
 * That is the commun renderer that users may use.
 * It implements the renderImpl() method from GvRendering::GvIRenderer base class
 * and has access to the data structure, cache and producer through the inherited
 * GvVolumeTreeRenderer base class.
 */
class DepthPeeling
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
	 * @param pVolumeTree data structure to render
	 * @param pVolumeTreeCache cache
	 * @param pProducer producer of data
	 */
	DepthPeeling();

	/**
	 * Destructor
	 */
	virtual ~DepthPeeling();

	/**
	 * Initialize
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool initialize();

	/**
	 * Finalize
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool finalize();

	/**
	 * This function is the specific implementation method called
	 * by the parent GvIRenderer::render() method during rendering.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 * @param pViewport the viewport configuration
	 */
	void render();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Shader program
	 */
	GvUtils::GvShaderProgram* _meshShaderProgram;

	/**
	 * Shader program
	 */
	GvUtils::GvShaderProgram* _frontToBackPeelingShaderProgram;

	/**
	 * Shader program
	 */
	GvUtils::GvShaderProgram* _blenderShaderProgram;

	/**
	 * Shader program
	 */
	GvUtils::GvShaderProgram* _finalShaderProgram;

	// OpenGL buffer object
	GLuint _frameBuffers[ 2 ];
	GLuint _textures[ 2 ];
	GLuint _depthTextures[ 2 ];
	GLuint _blenderTexture;
	GLuint _blenderFrameBuffer;

	/**
	 * Proxy geometry 3D model is loaded into an Assimp scene
	 */
	const aiScene* _scene;

	/**
	 * VAO for mesh rendering (vertex array object)
	 */
	GLuint _meshVertexArray;
	GLuint _meshVertexBuffer;
	GLuint _meshIndexBuffer;
	
	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "DepthPeeling.inl"

#endif // _DEPTH_PEELING_H_
