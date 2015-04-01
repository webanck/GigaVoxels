/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2013 INRIA <http://www.inria.fr/>
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

#ifndef _RENDERER_GLSL_H_
#define _RENDERER_GLSL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvConfig.h>
#include <GvRendering/GvRenderer.h>

// Cuda
#include <driver_types.h>

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

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class RendererGLSL
 *
 * @brief The RendererGLSL class provides an implementation of a renderer
 * specialized for GLSL.
 *
 * It implements the renderImpl() method from GvRenderer::GvIRenderer base class
 * and has access to the data structure, cache and producer through the inherited
 * VolumeTreeRenderer base class.
 */
template< typename TDataStructureType, typename TVolumeTreeCacheType >
class RendererGLSL : public GvRendering::GvRenderer< TDataStructureType, TVolumeTreeCacheType >
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
	 * It initializes all OpenGL-related stuff
	 *
	 * @param pVolumeTree data structure to render
	 * @param pVolumeTreeCache cache
	 * @param pProducer producer of data
	 */
	RendererGLSL( TDataStructureType* pVolumeTree, TVolumeTreeCacheType* pVolumeTreeCache );

	/**
	 * Destructor
	 */
	virtual ~RendererGLSL();

	/**
	 * This function is the specific implementation method called
	 * by the parent GvIRenderer::render() method during rendering.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 * @param pViewport the viewport configuration
	 */
	virtual void render( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport );

	/**
	 * Get the associated shader program
	 *
	 * @return the associated shader program
	 */
	GsGraphics::GsShaderProgram* getShaderProgram()
	{
		return _shaderProgram;
	}

	/**
	 * Get the cone aperture scale
	 *
	 * @return the cone aperture scale
	 */
	float getConeApertureScale() const;

	/**
	 * Set the cone aperture scale
	 *
	 * @param pValue the cone aperture scale
	 */
	void setConeApertureScale( float pValue );

	/**
	 * Get the max number of loops during the main GigaSpace pipeline pass (GLSL shader)
	 *
	 * @return the max number of loops
	 */
	unsigned int getMaxNbLoops() const;

	/**
	 * Set the max depth
	 *
	 * @param pValue the max depth
	 */
	void setMaxDepth( unsigned int pValue );

	/**
	 * Set the max number of loops during the main GigaSpace pipeline pass (GLSL shader)
	 *
	 * @param pValue the max number of loops
	 */
	void setMaxNbLoops( unsigned int pValue );

#ifdef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_GVRENDERERCUDA
	virtual void preRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport );
	virtual void postRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport );
#endif

	/**
	 * Initialize shader program
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool initializeShaderProgram();

	/**
	 * Finalize shader program
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool finalizeShaderProgram();

	/**
	 * Initialize shader program
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool initializeShaderProgramUniforms();
		
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
	 * Shader program
	 */
	GsGraphics::GsShaderProgram* _shaderProgram;

	/**
	 * Buffer of requests
	 */
	GLuint _updateBufferTBO;

	/**
	 * Node time stamps buffer
	 */
	GLuint _nodeTimeStampTBO;

	/**
	 * Brick time stamps buffer
	 */
	GLuint _brickTimeStampTBO;

	/**
	 * Node pool's child array (i.e. encoded data structure [octree, N3-Tree, etc...])
	 */
	GLuint _childArrayTBO;

	/**
	 * Node pool's data array (i.e. addresses of bricks associated to each node in cache)
	 */
	GLuint _dataArrayTBO;

	// Unmap GigaVoxels graphics resources from CUDA environment in order to be used by OpenGL
	cudaGraphicsResource* _graphicsResources[ 7 ];

	/**
	 * Cone aperture scale
	 */
	float _coneApertureScale;

	/**
	 * Max number of loops during the main GigaSpace pipeline pass (GLSL shader)
	 */
	unsigned int _maxNbLoops;

	/**
	 * Viewing System uniform parameters
	 */
	GLint _viewPosLoc;
	GLint _viewPlaneLoc;
	GLint _viewAxisXLoc;
	GLint _viewAxisYLoc;
	GLint _pixelSizeLoc;
	GLint _frustumNearInvLoc;
	// Cone aperture management
	GLint _coneApertureScaleLoc;
	// GigaSpace pipeline uniform parameters
	GLint _maxNbLoopsLoc;

	/**
	 * Locations of uniform variables
	 */
	GLint _nodeBufferLoc;
	GLint _dataBufferLoc;
	GLint _requestBufferLoc;
	GLint _nodeTimestampBufferLoc;
	GLint _brickTimestampBufferLoc;
	GLint _currentTimeLoc;

	/**
	 * Locations of uniform variables
	 */
	GLint _nodePoolResInvLoc;
	GLint _brickPoolResInvLoc;
	GLint _nodeCacheSizeLoc;
	GLint _brickCacheSizeLoc;
	GLint _dataPool_Channel_0_Loc;
	GLint _dataPool_Channel_1_Loc;
	GLint _maxDepthLoc;

	/**
	 * Locations of uniform variables
	 */
	GLint _positionLoc;

	// GigaVoxels/GigaSpace arrays
	GvCore::Array3DGPULinear< uint >* _nodeBuffer;
	GvCore::Array3DGPULinear< uint >* _dataBuffer;
	GvCore::Array3DGPULinear< uint >* _requestBuffer;
	GvCore::Array3DGPULinear< uint >* _nodeTimestampBuffer;
	GvCore::Array3DGPULinear< uint >* _brickTimestampBuffer;

	/******************************** METHODS *********************************/

	/**
	 * Start the rendering process.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 * @param pViewport the viewport configuration
	 */
	virtual void doRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "RendererGLSL.inl"

#endif // !_RENDERER_GLSL_H_
