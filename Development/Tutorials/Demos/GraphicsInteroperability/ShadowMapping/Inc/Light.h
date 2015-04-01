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

#ifndef _LIGHT_H_
#define _LIGHT_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GL
#include <GL/glew.h>

// glm
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform2.hpp>
#include <glm/gtx/projection.hpp>

// GigaVoxels
#include <GvCore/vector_types_ext.h>

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

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class Light
 *
 * @brief The Light class provides an interface for lights.
 *
 * ...
 */
class Light
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Light type
	 */
	enum ELightType
	{
		eDirectionalLight,
		eSpotLight,
		eNbLightTypes
	};

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	Light();

	/**
	 * Destructor
	 */
	virtual ~Light();

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
	virtual void render() const;

	/**
	 * Get the light intensity
	 *
	 * @return the light intensity
	 */
	const float3& getIntensity() const;

	/**
	 * Set the light intensity
	 *
	 * @param pValue the light intensity
	 */
	void setIntensity( const float3& pValue );

	/**
	 * Get the light direction
	 *
	 * @return the light direction
	 */
	const float3& getDirection() const;

	/**
	 * Set the light direction
	 *
	 * @param pValue the light direction
	 */
	void setDirection( const float3& pValue );

	/**
	 * Get the light position
	 *
	 * @return the light position
	 */
	const float4& getPosition() const;

	/**
	 * Set the light position
	 *
	 * @param pValue the light position
	 */
	void setPosition( const float4& pValue );

	/**
	 * Get the associated view matrix
	 *
	 * @return the associated view matrix
	 */
	glm::mat4 getViewMatrix() const;

	/**
	 * Get the associated projection matrix
	 *
	 * @return the associated projection matrix
	 */
	glm::mat4 getProjectionMatrix() const;

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Intensity (ambient, diffuse and specular)
	 */
	float3 _intensity;

	/**
	 * Direction
	 */
	float3 _direction;

	/**
	 * Position
	 */
	float4 _position;

	/**
	 * VAO for mesh rendering (vertex array object)
	 */
	GLuint _vertexArray;
	GLuint _vertexBuffer;
	GLuint _indexBuffer;

	/**
	 *
	 */
	unsigned int _nbVertices;
	unsigned int _nbFaces;

	/**
	 * Spiral arms color
	 */
	float3 _color;

	/**
	 * Eye point
	 */
	glm::vec3 _eye;

	/**
	 * Reference point indicating the center of the scene
	*/
	glm::vec3 _center;

	/**
	 * Up vector
	 */
	glm::vec3 _up;

	/**
	 * Field of view angle in the y direction (in degrees)
	 */
	float _fovY;

	/**
	 * Aspect ratio that determines the field of view in the x direction.
	 * The aspect ratio is the ratio of x (width) to y (height).
	 */
	float _aspect;
	
	/**
	 * Distance from the viewer to the near clipping plane (always positive)
	 */
	float _zNear;
	
	/**
	 * Distance from the viewer to the far clipping plane (always positive)
	 */
	float _zFar;

	/******************************** METHODS *********************************/

	/**
	 * Initialize graphics resource
	 */
	bool initializeGraphicsResources( const std::vector< float3 >& pVertices, const std::vector< unsigned int >& pIndices );
	
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
	Light( const Light& );

	/**
	 * Copy operator forbidden.
	 */
	Light& operator=( const Light& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "Light.inl"

#endif // _LIGHT_H_
