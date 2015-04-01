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

#ifndef _PROXY_GEOMETRY_H_
#define _PROXY_GEOMETRY_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GL
#include <GL/glew.h>

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

// GigaVoxels
namespace GsGraphics
{
	class GsShaderProgram;
}

// Project
class IMesh;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class ProxyGeometry
 *
 * @brief The ProxyGeometry class provides an interface for proxy geometry management.
 *
 * Proxy geometry are used to provide depth map of front faces and back faces
 */
class ProxyGeometry
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	GLuint _depthMinTex;
	GLuint _depthMaxTex;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	ProxyGeometry();

	/**
	 * Destructor
	 */
	virtual ~ProxyGeometry();

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
	 * Set buffer size
	 *
	 * @param pWidth buffer width
	 * @param pHeight buffer height
	 */
	void setBufferSize( int pWidth, int pHeight );

	/**
	 * Get the 3D model filename to load
	 *
	 * @return the 3D model filename to load
	 */
	const std::string& get3DModelFilename() const;

	/**
	 * Set the 3D model filename to load
	 *
	 * @param pFilename the 3D model filename to load
	 */
	void set3DModelFilename( const std::string& pFilename );

	/**
	 * Tell wheter or not the screen-based criteria is activated
	 *
	 * @return a flag telling wheter or not the screen-based criteria is activated
	 */
	bool getScreenBasedCriteria() const;

	/**
	 * Set the flag telling wheter or not the screen-based criteria is activated
	 *
	 * @param pFlag a flag telling wheter or not the screen-based criteria is activated
	 */
	void setScreenBasedCriteria( bool pFlag );

	/**
	 * Get the screen-based criteria coefficient
	 *
	 * @return the screen-based criteria coefficient
	 */
	float getScreenBasedCriteriaCoefficient() const;

	/**
	 * Set the screen-based criteria coefficient
	 *
	 * @param pValue the screen-based criteria coefficient
	 */
	void setScreenBasedCriteriaCoefficient( float pValue );

	/**
	 * Get the material alpha correction coefficient
	 *
	 * @return the material alpha correction coefficient
	 */
	float getMaterialAlphaCorrectionCoefficient() const;

	/**
	 * Set the material alpha correction coefficient
	 *
	 * @param pValue the material alpha correction coefficient
	 */
	void setMaterialAlphaCorrectionCoefficient( float pValue );

	/**
	 * Get the associated mesh
	 *
	 * @return the associated mesh
	 */
	const IMesh* getMesh() const;

	/**
	 * Get the associated mesh
	 *
	 * @return the associated mesh
	 */
	IMesh* editMesh();
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Mesh
	 */
	IMesh* _mesh;

	/**
	 * Shader program
	 */
	GsGraphics::GsShaderProgram* _shaderProgram;

	/**
	 * Shader program
	 */
	GsGraphics::GsShaderProgram* _meshShaderProgram;

	/**
	 * Shadow map's FBO (frame buffer object)
	 */
	GLuint _frameBuffer;
	/*GLuint _depthMinTex;
	GLuint _depthMaxTex;*/
	GLuint _depthTex;
	GLsizei _bufferWidth;
	GLsizei _bufferHeight;

	/**
	 * 3D model filename
	 */
	std::string _filename;

	/**
	 * Flag telling wheter or not the screen-based criteria is activated
	 */
	bool _screenBasedCriteria;

	/**
	 * the screen-based criteria coefficient
	 */
	float _screenBasedCriteriaCoefficient;

	/**
	 * Material alpha correction coefficient
	 *
	 * - its the traversed distance at which full opacity will be reached inside matter
	 */
	float _materialAlphaCorrectionCoefficient;
	
	/******************************** METHODS *********************************/

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
	ProxyGeometry( const ProxyGeometry& );

	/**
	 * Copy operator forbidden.
	 */
	ProxyGeometry& operator=( const ProxyGeometry& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "ProxyGeometry.inl"

#endif // _PROXY_GEOMETRY_H_
