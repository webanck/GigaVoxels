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

#ifndef _SAMPLECORE_H_
#define _SAMPLECORE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <cuda_runtime.h>
#include <vector_types.h>

// Loki
#include <loki/Typelist.h>

// GigaVoxels
#include <GvCore/vector_types_ext.h>
#include <GvUtils/GvForwardDeclarationHelper.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// Forward references
namespace GvCore
{
	//template<
	//	class DataTList, template<typename> class DataArrayType,
	//	class NodeTileRes, class BrickRes, uint BorderSize >
	//struct VolumeTree;

	//template< uint r >
	//struct StaticRes1D;

	template< typename T >
	class Array3DGPULinear;

	//template< typename T >
	//class Array3DGPUTex;
}

template< class DataTList >
struct BvhTree;

template < typename BvhTreeType >
class BvhTreeCache;

// Producer
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
class GPUTriangleProducerBVH;

//template < typename NodeRes, typename BrickRes, uint BorderSize, typename VolumeTreeType >
//class SphereProducer;

// Shaders
//class SphereShader;

// Renderers
template< typename BvhTreeType, typename BvhTreeCacheType, typename ProducerType >
class BvhTreeRenderer;

//template<
//	class VolTreeType, class NodeResolution,
//	class BrickResolution, uint BorderSize,
//	class GPUProducer, class SampleShader >
//class RendererCUDA;

// Defines the type list representing the content of one voxel
typedef Loki::TL::MakeTypelist< float4, uchar4 >::Result DataType;

// Defines the size of a node tile
//typedef gigavoxels::StaticRes1D< 2 > NodeRes;

// Defines the size of a brick
//typedef gigavoxels::StaticRes1D< 8 > BrickRes;

// Defines the size of the border around a brick
//enum { BrickBorderSize = 1 };

// Defines the total size of a brick
//typedef gigavoxels::StaticRes1D<8 + 2 * BrickBorderSize> RealBrickRes;

// Defines the type of structure we want to use. Array3DGPUTex is the type of array used 
// to store the bricks.
typedef BvhTree< DataType > BvhTreeType;

// Defines the type of the producer
//typedef SphereShader ShaderType;

// Defines the type of the cache.
typedef BvhTreeCache< BvhTreeType > BvhTreeCacheType;

// Defines the type of the shader
typedef GPUTriangleProducerBVH<	BvhTreeType, 32, BvhTreeCacheType >	ProducerType;

// Defines the type of the renderer we want to use.
typedef BvhTreeRenderer< BvhTreeType, BvhTreeCacheType, ProducerType > RendererType;

//typedef RendererCUDA< VolumeTreeType,
//	NodeRes, BrickRes, BrickBorderSize,
//	ProducerType, ShaderType >					RendererType;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class SampleCore
 *
 * @brief The SampleCore class provides...
 *
 * ...
 */
class SampleCore
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	SampleCore();

	/**
	 * Destructor
	 */
	~SampleCore();

	/**
	 * ...
	 */
	void init();
	/**
	 * ...
	 */
	void draw();
	/**
	 * ...
	 *
	 * @param width ...
	 * @param height ...
	 */
	void resize( int width, int height );

	/**
	 * ...
	 */
	void clearCache();

	/**
	 * ...
	 */
	void toggleDisplayOctree();
	/**
	 * ...
	 */
	void toggleDynamicUpdate();
	/**
	 * ...
	 *
	 * @param mode ...
	 */
	void togglePerfmonDisplay( uint mode );

	/**
	 * ...
	 */
	void incMaxVolTreeDepth();
	/**
	 * ...
	 */
	void decMaxVolTreeDepth();

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
	 * ...
	 */
	BvhTreeType			*mBvhTree;
	/**
	 * ...
	 */
	BvhTreeCacheType	*mBvhTreeCache;
	/**
	 * ...
	 */
	RendererType		*mBvhTreeRenderer;
	/**
	 * ...
	 */
	ProducerType		*mProducer;

	/**
	 * ...
	 */
	GLuint				mColorBuffer;
	/**
	 * ...
	 */
	GLuint				mDepthBuffer;

	/**
	 * ...
	 */
	GLuint				mColorTex;
	/**
	 * ...
	 */
	GLuint				mDepthTex;

	/**
	 * ...
	 */
	GLuint				mFrameBuffer;

	/**
	 * ...
	 */
	struct cudaGraphicsResource	*mColorResource;
	/**
	 * ...
	 */
	struct cudaGraphicsResource	*mDepthResource;

	/**
	 * ...
	 */
	int					mWidth;
	/**
	 * ...
	 */
	int					mHeight;

	/**
	 * ...
	 */
	bool				mDisplayOctree;
	/**
	 * ...
	 */
	uint				mDisplayPerfmon;
	/**
	 * ...
	 */
	uint				mMaxVolTreeDepth;

	/******************************** METHODS *********************************/

};

#endif // !_SAMPLECORE_H_
