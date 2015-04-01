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

#ifndef _GV_ARRAY_3D_GPU_TEX_H_
#define _GV_ARRAY_3D_GPU_TEX_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// OpenGL
#include <GL/glew.h>

// Cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Cuda SDK
#include <helper_math.h>

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/vector_types_ext.h"
#include "GvCore/Array3DKernelTex.h"

// STL
#include <string>

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
namespace GvCore
{
	//template< typename T > class Array3DKernelTex;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{

/** 
 * @class Array3DGPUTex
 *
 * @brief The Array3DGPUTex class provides features to manipulate device memory array.
 *
 * @ingroup GvCore
 *
 * 3D Array manipulation class located in device texture memory. It is not the same as linear memory array.
 * Textures and surfaces should be bound to array in order to read/write data.
 */
template< typename T >
class Array3DGPUTex
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Defines the type of the associated kernel array
	 */
	typedef Array3DKernelTex< T > KernelArrayType;

	/**
	 * Enumeration used to define array in normal or graphics interoperability mode
	 */
	enum ArrayOptions
	{
		GLInteroperability = 1
	};

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param res resolution
	 * @param options options
	 */
	Array3DGPUTex( const uint3& res, uint options = 0 );

	/**
	 * Constructor
	 *
	 * @param res resolution
	 * @param cfd channel format descriptor
	 */
	Array3DGPUTex( const uint3& res, cudaChannelFormatDesc cfd );
	
	/**
	 * Destructor
	 */
	virtual ~Array3DGPUTex();

	/**
	 * Get the resolution
	 *
	 * @return the resolution
	 */
	uint3 getResolution() const;

	/**
	 * Bind texture to array
	 *
	 * @param symbol device texture symbol
	 * @param texRefName texture reference name
	 * @param normalizedAccess Type of access
	 * @param filterMode Type of filtering mode
	 * @param addressMode Type of address mode
	 */
	void bindToTextureReference( const void* symbol, const char* texRefName, bool normalizedAccess, cudaTextureFilterMode filterMode, cudaTextureAddressMode addressMode );

	/**
	 * Unbind texture to array
	 *
	 * @param texRefName texture reference name
	 */
	void unbindTexture( const char* texRefName );

	/**
	 * Bind surface to array
	 *
	 * @param surfRefName device surface symbol
	 * @param surfRefName surface reference name
	 */
	void bindToSurfaceReference( const void* pSurfaceReferenceSymbol, const char* surfRefName );

	/**
	 * Get the associated device-side object
	 *
	 * @return the associated device-side object
	 */
	Array3DKernelTex< T > getDeviceArray();

	/**
	 * Get the internal device memory array
	 *
	 * @return the internal device memory array
	 */
	cudaArray* getCudaArray();

	/**
	 * Get the associated graphics library handle if graphics library interoperability is used
	 *
	 * @return the associated graphics library handle
	 */
	GLuint getBufferName() const;

	/**
	 * Map the associated graphics resource if graphics library interoperability is used
	 */
	void mapResource();

	/**
	 * Unmap the associated graphics resource if graphics library interoperability is used
	 */
	void unmapResource();

	/**
	 * Get the associated CUDA graphics resource
	 *
	 * return the associated CUDA graphics resource
	 */
	cudaGraphicsResource* getGraphicsResource();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Flags used to define array in normal or graphics interoperability mode
	 */
	uint _arrayOptions;

	/**
	 * Array resolution (i.e. dimension)
	 */
	uint3 _resolution;

	/**
	 * ...
	 */
	uint2 _atlasRes;

	/**
	 * Underlying device memory array
	 */
	cudaArray* _dataArray;

	/**
	 * Underlying channel format descriptor
	 */
	cudaChannelFormatDesc _channelFormatDesc;

	/**
	 * Associated graphics library handle if graphics library interoperability is used
	 */
	GLuint _bufferObject;

	/**
	 * Associated graphics resource if graphics library interoperability is used
	 */
	struct cudaGraphicsResource* _bufferResource;

	/**
	 * Bounded texture reference name (if any)
	 */
	std::string _textureReferenceName;

	/**
	 * Bounded device texture symbol (if any)
	 */
	const void* _textureSymbol;

	/******************************** METHODS *********************************/

	/**
	 * Helper method to allocate internal device memory array
	 *
	 * @param res resolution
	 * @param channelFormatDesc channel format descriptor
	 */
	void allocArray( const uint3& res, cudaChannelFormatDesc channelFormatDesc );

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
	Array3DGPUTex( const Array3DGPUTex& );

	/**
	 * Copy operator forbidden.
	 */
	Array3DGPUTex& operator=( const Array3DGPUTex& );
	
};

} //namespace GvCore

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{
	/**
	 * ...
	 *
	 * @param dstarray ...
	 * @param posDst ...
	 * @param resDst ...
	 * @param h_srcptr ...
	 */
	template< typename T >
	inline void memcpyArray( Array3DGPUTex< T >* dstarray, uint3 posDst, uint3 resDst, const T* h_srcptr );

} //namespace GvCore

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "Array3DGPUTex.inl"

#endif // !_GV_ARRAY_3D_GPU_TEX_H_
