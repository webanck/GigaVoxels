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

#ifndef _SHADOW_MAP_H_
#define _SHADOW_MAP_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GL
#include <GL/glew.h>

// GigaVoxels
#include <GvCore/vector_types_ext.h>

// glm
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform2.hpp>
#include <glm/gtx/projection.hpp>

// STL
#include <string>
#include <vector>
#include <map>

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

// Project
class IMesh;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class ShadowMap
 *
 * @brief The ShadowMap class provides an interface for shadow management.
 *
 * ...
 */
class ShadowMap
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Scale and bias matrix used to transform NDC space coordinates from [ -1 ; 1 ]
	 * to Texture space coordinates in [ 0 ; 1 ]
	 */
	static const glm::mat4 sScaleBiasMatrix;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	ShadowMap();

	/**
	 * Destructor
	 */
	virtual ~ShadowMap();

	/**
	 * Initialize
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	virtual bool initialize();

	/**
	 * Finalize
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	virtual bool finalize();

	/**
	 * This function is the specific implementation method called
	 * by the parent GvIRenderer::render() method during rendering.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 * @param pViewport the viewport configuration
	 */
	virtual void render( const float4x4& pModelViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport );

	/**
	 * Light viewing system parameters
	 */
	void setLightEye( const glm::vec3& pValue );
	void setLightCenter( const glm::vec3& pValue );
	void setLightUp( const glm::vec3& pValue );
	void setLightFovY( float pValue );
	void setLightAspectRatio( float pValue );
	void setLightZNear( float pValue );
	void setLightZFar( float pValue );

	/**
	 * Camera viewing system parameters
	 */
	void setCameraEye( const glm::vec3& pValue );
	void setCameraCenter( const glm::vec3& pValue );
	void setCameraUp( const glm::vec3& pValue );
	void setCameraFovY( float pValue );
	void setCameraAspectRatio( float pValue );
	void setCameraZNear( float pValue );
	void setCameraZFar( float pValue );
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Shader program
	 */
	GvUtils::GvShaderProgram* _shaderProgramFirstPass;

	/**
	 * Shader program
	 */
	GvUtils::GvShaderProgram* _shaderProgramSecondPass;

	/**
	 * Shadow map's FBO (frame buffer object)
	 */
	GLuint _shadowMapFBO;
	GLsizei _shadowMapWidth;
	GLsizei _shadowMapHeight;
	GLuint _depthTexture;
	GLsizei _width;
	GLsizei _height;

	/**
	 * Mesh
	 */
	IMesh* _mesh;

	/**
	 * Light viewing system parameters
	 */
	glm::vec3 _lightEye;
	glm::vec3 _lightCenter;
	glm::vec3 _lightUp;
	float _lightFovY;
	float _lightAspectRatio;
	float _lightZNear;
	float _lightZFar;

	/**
	 * Camera viewing system parameters
	 */
	glm::vec3 _cameraEye;
	glm::vec3 _cameraCenter;
	glm::vec3 _cameratUp;
	float _cameraFovY;
	float _cameraAspectRatio;
	float _cameraZNear;
	float _cameraZFar;

	/******************************** METHODS *********************************/

	/**
	 * spitOutDepthBuffer
	 */
	void spitOutDepthBuffer();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	ShadowMap( const ShadowMap& );

	/**
	 * Copy operator forbidden.
	 */
	ShadowMap& operator=( const ShadowMap& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "ShadowMap.inl"

#endif // _SHADOW_MAP_H_
