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

#ifndef _VOLUME_TREE_RENDERER_CUDA_H_
#define _VOLUME_TREE_RENDERER_CUDA_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <driver_types.h>

// GigaVoxels
#include <GvCore/GvCoreConfig.h>
#include <GvCore/GPUPool.h>
#include <GvCore/RendererTypes.h>
#include <GvCore/StaticRes3D.h>
#include <GvCore/vector_types_ext.h>
#include <GvStructure/GvVolumeTree.h>
#include <GvStructure/GvDataProductionManager.h>
#include <GvRendering/GvRenderer.h>
#include <GvRendering/GvGraphicsInteroperabiltyHandler.h>
#include <GvRendering/GvGraphicsInteroperabiltyHandlerKernel.h>

// Project
#include "VolumeTreeRendererCUDAKernel.h"

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
 * @class GvVolumeTreeRendererCUDA
 *
 * @brief The GvVolumeTreeRendererCUDA class provides an implementation of a renderer
 * specialized for CUDA.
 *
 * That is the commun renderer that users may use.
 * It implements the renderImpl() method from GvRenderer::GvIRenderer base class
 * and has access to the data structure, cache and producer through the inherited
 * GvVolumeTreeRenderer base class.
 */
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
class VolumeTreeRendererCUDA : public GvRendering::GvRenderer< TVolumeTreeType, TVolumeTreeCacheType >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * CUDA block dimension used during rendering (kernel launch).
	 * Screen is split in 2D blocks of 8 per 8 pixels.
	 */
	typedef GvCore::StaticRes3D< 8, 8, 1 > RenderBlockResolution;

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param pVolumeTree data structure to render
	 * @param pVolumeTreeCache cache
	 * @param pProducer producer of data
	 */
	VolumeTreeRendererCUDA( TVolumeTreeType* pVolumeTree, TVolumeTreeCacheType* pVolumeTreeCache );

	/**
	 * Destructor
	 */
	virtual ~VolumeTreeRendererCUDA();

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
	 * Attach an OpenGL buffer object (i.e. a PBO, a VBO, etc...) to an internal graphics resource 
	 * that will be mapped to a color or depth slot used during rendering.
	 *
	 * @param pGraphicsResourceSlot the associated internal graphics resource (color or depth) and its type of access (read, write or read/write)
	 * @param pBuffer the OpenGL buffer
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool connect( GvRendering::GvGraphicsInteroperabiltyHandler::GraphicsResourceSlot pGraphicsResourceSlot, GLuint pBuffer );

	/**
	 * Attach an OpenGL texture or renderbuffer object to an internal graphics resource 
	 * that will be mapped to a color or depth slot used during rendering.
	 *
	 * @param pGraphicsResourceSlot the associated internal graphics resource (color or depth) and its type of access (read, write or read/write)
	 * @param pImage the OpenGL texture or renderbuffer object
	 * @param pTarget the target of the OpenGL texture or renderbuffer object
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool connect( GvRendering::GvGraphicsInteroperabiltyHandler::GraphicsResourceSlot pGraphicsResourceSlot, GLuint pImage, GLenum pTarget );

	/**
	 * Dettach an OpenGL buffer object (i.e. a PBO, a VBO, etc...), texture or renderbuffer object
	 * to its associated internal graphics resource mapped to a color or depth slot used during rendering.
	 *
	 * @param pGraphicsResourceSlot the internal graphics resource slot (color or depth)
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool disconnect( GvRendering::GvGraphicsInteroperabiltyHandler::GraphicsResourceSlot pGraphicsResourceSlot );

	/**
	 * Disconnect all registered graphics resources
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool resetGraphicsResources();

	/**
	 * TEST - optimization
	 *
	 * Launch the post-render phase
	 */
	virtual void doPostRender();
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Frame size
	 */
	uint2 _frameSize;

	/**
	 * Fast build mode flag
	 */
	bool _fastBuildMode;

#ifndef USE_SIMPLE_RENDERER
	/**
	 * Cuda stream
	 */
	cudaStream_t _cudaStream[ 1 ];

	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< float >* d_rayBufferT;

	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< float >* d_rayBufferTmax;

	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< float4 >* d_rayBufferAccCol;

	/**
	 * Use a debug options
	 */
	int2 _currentDebugRay;
#endif

	/**
	 * ...
	 */
	uint _frameNumAfterUpdate;

	/**
	 * ...
	 */
	uint _numUpdateFrames;	// >= 1

	/**
	 * Graphics interoperability handler
	 */
	GvRendering::GvGraphicsInteroperabiltyHandler* _graphicsInteroperabiltyHandler;

	/******************************** METHODS *********************************/

	/**
	 * Get the graphics interoperability handler
	 *
	 * @return the graphics interoperability handler
	 */
	GvRendering::GvGraphicsInteroperabiltyHandler* getGraphicsInteroperabiltyHandler();

	/**
	 * Initialize Cuda objects
	 */
	void initializeCuda();

	/**
	 * Finalize Cuda objects
	 */
	void finalizeCuda();

	/**
	 * Initialize internal buffers used during rendering
	 * (i.e. input/ouput color and depth buffers, ray buffers, etc...).
	 * Buffers size are dependent of the frame size.
	 *
	 * @param pFrameSize the frame size
	 */
	void initFrameObjects( const uint2& pFrameSize );

	/**
	 * Destroy internal buffers used during rendering
	 * (i.e. input/ouput color and depth buffers, ray buffers, etc...)
	 * Buffers size are dependent of the frame size.
	 */
	void deleteFrameObjects();

	/**
	 * Start the rendering process.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 */
	void doRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix );

#ifndef USE_SIMPLE_RENDERER
	/**
	 * ...
	 *
	 * @param pFrameAfterUpdate ...
	 *
	 * @return ...
	 */
	float getVoxelSizeMultiplier( uint pFrameAfterUpdate );
#endif

	/**
	 * Bind all graphics resources used by the GL interop handler during rendering.
	 *
	 * Internally, it binds textures and surfaces to arrays associated to mapped graphics reources.
	 *
	 * NOTE : this method should be in the GvGraphicsInteroperabiltyHandler but it seems that
	 * there are conflicts with textures ans surfaces symbols. The binding succeeds but not the
	 * read/write operations.
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool bindGraphicsResources();

	/**
	 * Unbind all graphics resources used by the GL interop handler during rendering.
	 *
	 * Internally, it unbinds textures and surfaces to arrays associated to mapped graphics reources.
	 *
	 * NOTE : this method should be in the GvGraphicsInteroperabiltyHandler but it seems that
	 * there are conflicts with textures ans surfaces symbols. The binding succeeds but not the
	 * read/write operations.
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool unbindGraphicsResources();

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
	VolumeTreeRendererCUDA( const VolumeTreeRendererCUDA& );

	/**
	 * Copy operator forbidden.
	 */
	VolumeTreeRendererCUDA& operator=( const VolumeTreeRendererCUDA& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "VolumeTreeRendererCUDA.inl"

#endif // _VOLUME_TREE_RENDERER_CUDA_H_
