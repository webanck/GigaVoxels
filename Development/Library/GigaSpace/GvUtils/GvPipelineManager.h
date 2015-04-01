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

#ifndef _SAMPLECORE_H_
#define _SAMPLECORE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <cuda_runtime.h>
#include <vector_types.h>
//#include <helper_math.h>

// Loki
#include <loki/Typelist.h>

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
	template< uint r >
	struct StaticRes1D;
}

namespace GvStructure
{
	template<
		class DataTList, class NodeTileRes,
		class BrickRes, uint BorderSize >
	struct GvVolumeTree;

	template <
		typename VolumeTreeType, typename ProducerType,
		typename NodeTileRes, typename BrickFullRes >
	class GvDataProductionManager;
}

namespace GvRendering
{	
	template<
		typename VolumeTreeType, typename VolumeTreeCacheType,
		typename ProducerType, typename SampleShader >
	class GvRendererCUDA;
}

// Producers
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
class ProducerLoad;

// Shaders
class ShaderLoad;

// Defines the type list representing the content of one voxel
typedef Loki::TL::MakeTypelist< uchar4 >::Result DataType;

// Defines the size of a node tile
typedef GvCore::StaticRes1D< 2 > NodeRes;

// Defines the size of a brick
typedef GvCore::StaticRes1D< 8 > BrickRes;

// Defines the size of the border around a brick
enum { BrickBorderSize = 1 };

// Defines the total size of a brick
typedef GvCore::StaticRes1D<8 + 2 * BrickBorderSize> RealBrickRes;

// Defines the type of the producer
typedef ShaderLoad ShaderType;

// Defines the type of the shader
typedef ProducerLoad< DataType,
	NodeRes, BrickRes, BrickBorderSize >		ProducerType;

// Defines the type of structure we want to use.
typedef GvStructure::GvVolumeTree< DataType,
	NodeRes, BrickRes, BrickBorderSize >		VolumeTreeType;

// Defines the type of the cache we want to use.
typedef GvStructure::GvDataProductionManager<
	VolumeTreeType, ProducerType,
	NodeRes, RealBrickRes >						VolumeTreeCacheType;

// Defines the type of the renderer we want to use.
typedef GvRendering::GvRendererCUDA<
	VolumeTreeType, VolumeTreeCacheType,
	ProducerType, ShaderType >					VolumeTreeRendererType;

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
	 * Clear the cache
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
	VolumeTreeType			*mVolumeTree;
	/**
	 * ...
	 */
	VolumeTreeCacheType		*mVolumeTreeCache;
	/**
	 * ...
	 */
	VolumeTreeRendererType	*mVolumeTreeRenderer;
	/**
	 * ...
	 */
	ProducerType			*mProducer;

	/**
	 * ...
	 */
	GLuint					mColorBuffer;
	/**
	 * ...
	 */
	GLuint					mDepthBuffer;

	/**
	 * ...
	 */
	GLuint					mColorTex;
	/**
	 * ...
	 */
	GLuint					mDepthTex;

	/**
	 * ...
	 */
	GLuint					mFrameBuffer;

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
	int				mWidth;
	/**
	 * ...
	 */
	int				mHeight;

	/**
	 * ...
	 */
	bool			mDisplayOctree;
	/**
	 * ...
	 */
	uint			mDisplayPerfmon;
	/**
	 * ...
	 */
	uint			mMaxVolTreeDepth;

	/******************************** METHODS *********************************/

};

#endif // !_SAMPLECORE_H_
