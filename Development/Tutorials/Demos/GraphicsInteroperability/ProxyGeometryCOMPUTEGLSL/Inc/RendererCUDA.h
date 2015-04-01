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

#ifndef _RENDERER_CUDA_H_
#define _RENDERER_CUDA_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvRendering/GvRendererCUDA.h>

// Project
#include "RendererCUDAKernel.h"
#include "ProxyGeometry.h"

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
 * @class RendererCUDA
 *
 * @brief The RendererCUDA class provides...
 *
 * ...
 */
template< typename TDataStructureType, typename TDataProductionManagerType, typename TShaderType >
class RendererCUDA : public GvRendering::GvRendererCUDA< TDataStructureType, TDataProductionManagerType, TShaderType >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * CUDA block dimension used during rendering (kernel launch).
	 * Screen is split in 2D blocks of blockDim.x per blockDim.y pixels.
	 */
	typedef typename GvRendering::GvRendererCUDA< TDataStructureType, TDataProductionManagerType, TShaderType >::RenderBlockResolution RenderBlockResolution;

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param pDataStructure data structure to render
	 * @param pDataProductionManager data production manager that will handle requests emitted during rendering
	 */
	RendererCUDA( TDataStructureType* pDataStructure, TDataProductionManagerType* pDataProductionManager );

	/**
	 * Destructor
	 */
	virtual ~RendererCUDA();

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
	 * Get the associated proxy geometry
	 *
	 * @return the proxy geometry
	 */
	const ProxyGeometry* getProxyGeometry() const;

	/**
	 * Get the associated proxy geometry
	 *
	 * @return the proxy geometry
	 */
	ProxyGeometry* editProxyGeometry();

	/**
	 * Set the associated proxy geometry
	 *
	 * @param pProxyGeometry the proxy geometry
	 */
	void setProxyGeometry( ProxyGeometry* pProxyGeometry );

	/**
	 * Register the graphics resources associated to proxy geometry
	 *
	 * @return a flag to tell wheter or not it succeeds
	 */
	bool registerProxyGeometryGraphicsResources();
	
	/**
	 * Unregister the graphics resources associated to proxy geometry
	 *
	 * @return a flag to tell wheter or not it succeeds
	 */
	bool unregisterProxyGeometryGraphicsResources();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Graphics resource associated to the proxy geomtry minimum depth GL buffer
	 */
	struct cudaGraphicsResource *_rayMinResource;

	/**
	 * Graphics resource associated to the proxy geomtry minimum depth GL buffer
	 */
	struct cudaGraphicsResource *_rayMaxResource;

	/**
	 * Proxy geometry
	 */
	ProxyGeometry* _proxyGeometry;

	/******************************** METHODS *********************************/

	/**
	 * Start the rendering process.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 */
	virtual void doRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport );

	/**
	 * Copy constructor forbidden.
	 */
	RendererCUDA( const RendererCUDA& );

	/**
	 * Copy operator forbidden.
	 */
	RendererCUDA& operator=( const RendererCUDA& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "RendererCUDA.inl"

#endif // _RENDERER_CUDA_H_
