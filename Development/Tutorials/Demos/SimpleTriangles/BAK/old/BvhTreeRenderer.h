/*
 * Copyright (C) 2011 Fabrice Neyret <Fabrice.Neyret@imag.fr>
 * Copyright (C) 2011 Cyril Crassin <Cyril.Crassin@icare3d.org>
 * Copyright (C) 2011 Morgan Armand <morgan.armand@gmail.com>
 *
 * This file is part of Gigavoxels.
 *
 * Gigavoxels is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Gigavoxels is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Gigavoxels.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _BVHTREERENDERER_H_
#define _BVHTREERENDERER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Gigavoxels
#include <GvCore/StaticRes3D.h>
#include <GvCore/GPUPool.h>
#include <GvCore/RendererTypes.h>

#include "RendererBVHTrianglesCommon.h"

#include "BvhTree.h"
#include "BvhTreeCache.h"

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

template< typename BvhTreeType, typename BvhTreeCacheType, typename ProducerType >
class BvhTreeRenderer
{
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * ...
	 */
	typedef GvCore::StaticRes3D< NUM_RAYS_PER_BLOCK_X, NUM_RAYS_PER_BLOCK_Y, 1 > RenderBlockResolution;

	// typedef base class types, since there is no unqualified name lookups for templated classes.

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param volumeTree ...
	 * @param gpuProd ...
	 * @param nodePoolRes ...
	 * @param brickPoolRes ...
	 */
	BvhTreeRenderer( BvhTreeType* bvhTree, BvhTreeCacheType* bvhTreeCache, ProducerType* gpuProd );

	/**
	 * Destructor
	 */
	~BvhTreeRenderer();

	/**
	 * ...
	 *
	 * @param modelMatrix ...
	 * @param viewMatrix ...
	 * @param projectionMatrix ...
	 * @param viewport ...
	 */
	void renderImpl(const float4x4 &modelMatrix, const float4x4 &viewMatrix, const float4x4 &projectionMatrix, const int4 &viewport);

	/**
	 * ...
	 *
	 * @param colorResource ...
	 */
	void setColorResource(struct cudaGraphicsResource *colorResource);

	/**
	 * ...
	 *
	 * @param depthResource ...
	 */
	void setDepthResource(struct cudaGraphicsResource *depthResource);

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	BvhTreeType			*_bvhTree;
	/**
	 * ...
	 */
	BvhTreeCacheType	*_bvhTreeCache;
	/**
	 * ...
	 */
	struct cudaGraphicsResource *_colorResource;
	/**
	 * ...
	 */
	struct cudaGraphicsResource *_depthResource;

	/******************************** METHODS *********************************/
	
	/**
	 * ...
	 */
	void cuda_Init();
	/**
	 * ...
	 */
	void cuda_Destroy();

	/**
	 * ...
	 *
	 * @param fs ...
	 */
	void initFrameObjects(const uint2 &fs);
	/**
	 * ...
	 */
	void deleteFrameObjects();

	/**
	 * ...
	 *
	 * @param modelMatrix ...
	 * @param viewMatrix ...
	 * @param projectionMatrix ...
	 */
	virtual void doRender(const float4x4 &modelMatrix, const float4x4 &viewMatrix, const float4x4 &projectionMatrix, const int4& pViewport );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	uint2 _frameSize;

	ProducerType *gpuProducer;

	bool _dynamicUpdate;
	uint _currentTime;

	float4 _userParam;

	uint _maxVolTreeDepth;

	//uint3 volTreeRootAddressGPU;

	//////////////////////////////////////////
	GvCore::Array3DGPULinear<uchar4> *d_inFrameColor;
	GvCore::Array3DGPULinear<float> *d_inFrameDepth;
	GvCore::Array3DGPULinear<uchar4> *d_outFrameColor;
	GvCore::Array3DGPULinear<float> *d_outFrameDepth;

	GvCore::Array3DGPULinear<uchar4> *d_rayOutputColor;
	GvCore::Array3DGPULinear<float4> *d_rayOutputNormal;

	///////////////////////
	//Restart info
	GvCore::Array3DGPULinear<float> *d_rayBufferTmin;		// 1 per ray
	GvCore::Array3DGPULinear<float> *d_rayBufferT;			// 1 per ray
	GvCore::Array3DGPULinear<int> *d_rayBufferMaskedAt;		// 1 per ray
	GvCore::Array3DGPULinear<int> *d_rayBufferStackIndex;	// 1 per tile (i.e. rendering tile)
	GvCore::Array3DGPULinear<uint> *d_rayBufferStackVals;	// BVH_TRAVERSAL_STACK_SIZE per tile

public:

	bool debugDisplayTimes;
	//Debug options
	int2 currentDebugRay;

	void clearCache();

	bool &dynamicUpdateState(){
		return dynamicUpdate;
	}

	void setUserParam(const float4 &up){
		_userParam=up;
	}
	float4 &getUserParam(){
		return _userParam;
	}

	uint getMaxVolTreeDepth(){
		return _maxVolTreeDepth;
	}
	void setMaxVolTreeDepth(uint maxVolTreeDepth){
		_maxVolTreeDepth = maxVolTreeDepth;
	}

	void nextFrame(){
		_currentTime++;
	}
};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#if 0
#include "BvhTreeRenderer.hcu"
#else
#include "BVHRenderer_kernel.hcu"
#endif
#include "BvhTreeRenderer.inl"

#endif // !_BVHTREERENDERER_H_
