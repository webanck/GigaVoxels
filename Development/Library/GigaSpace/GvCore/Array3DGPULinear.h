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

#ifndef _ARRAY3DGPULINEAR_H_
#define _ARRAY3DGPULINEAR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

// OpenGL
#include <GL/glew.h>

// Cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

// Cuda SDK
#include <helper_math.h>

// GigaVoxels
#include "GvCore/vector_types_ext.h"
#include "GvCore/Array3DKernelLinear.h"
#include "GvCore/Array3D.h"

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

namespace GvCore
{

/** 
 * @class Array3DGPULinear
 *
 * @brief The Array3DGPULinear class provides...
 *
 * @ingroup GvCore
 *
 * 3D Array located in GPU linear memory manipulation class.
 */
template< typename T >
class Array3DGPULinear
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * ...
	 */
	enum ArrayOptions
	{
		GLInteroperability = 1
	};

	/**
	 * Defines the type of the associated kernel array
	 */
	typedef Array3DKernelLinear< T > KernelArrayType;

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor.
	 * Create a 3D array of the given resolution using GPU linear memory.
	 *
	 * @param res ...
	 * @param options ...
	 */
	Array3DGPULinear( const uint3& res, uint options = 0 );

	/**
	 * Constructor.
	 * Create a 3D array of the given resolution.
	 *
	 * @param data ...
	 * @param res ...
	 */
	Array3DGPULinear( T* data, const uint3& res );

	/**
	 * Destructor.
	 */
 	~Array3DGPULinear();

	/**
	 * ...
	 *
	 * @return ...
	 */
	uint3 getResolution() const;
	/**
	 * ...
	 *
	 * @return ...
	 */
	size_t getNumElements() const;
	/**
	 * ...
	 *
	 * @return ...
	 */
	size_t getMemorySize() const;

	/**
	 * ...
	 *
	 * @param pos ...
	 *
	 * @return ...
	 */
	T* getPointer( const uint3& pos ) const;
	/**
	 * ...
	 *
	 * @param offset ...
	 *
	 * @return ...
	 */
	T* getPointer( size_t offset ) const;
	/**
	 * ...
	 *
	 * @return ...
	 */
	T* getPointer() const;

	/**
	 * ...
	 *
	 * @param pos ...
	 *
	 * @return ...
	 */
	const T* getConstPointer( const uint3& pos ) const;
	/**
	 * ...
	 *
	 * @param address ...
	 *
	 * @return ...
	 */
	const T* getConstPointer( const size_t& address ) const;

	/**
	 * ...
	 *
	 * @param dptr ...
	 */
	void manualSetDataStorage( T* dptr );
	/**
	 * ...
	 *
	 * @return
	 */
	T** getDataStoragePtrAddress();

	/*void zero();*/

	/**
	 *  Fill array with a value
	 *
	 * @param v value
	 */
	void fill( int v );

	/**
	 * Fill array asynchronously with a value
	 *
	 * @param v value
	 */
	void fillAsync( int v );

	///
	///GPU related stuff
	///

	/**
	 * ...
	 *
	 * @return ...
	 */
	KernelArrayType getDeviceArray() const;
	
	/**
	 * ...
	 *
	 * @return ...
	 */
	cudaPitchedPtr getCudaPitchedPtr() const;
	
	/**
	 * ...
	 *
	 * @return ...
	 */
	cudaExtent getCudaExtent() const;

	/**
	 * Map the associated graphics resource
	 */
	void mapResource();

	/**
	 * Unmap the associated graphics resource
	 */
	void unmapResource();

	/**
	 * Get the associated OpenGL handle
	 *
	 * @return the associated OpenGL buffer
	 */
	GLuint getBufferName() const;

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

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/**
	 * ...
	 */
	enum ArrayPrivateOptions
	{
		SharedData = 0x80000000
	};

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	T* _data;

	/**
	 * ...
	 */
	uint3 _resolution;

	/**
	 * ...
	 */
	size_t _pitch;

	/**
	 * ...
	 */
	uint _arrayOptions;

	/**
	 * ...
	 */
	GLuint _bufferObject;

	/**
	 * The associated CUDA graphics resource
	 */
	struct cudaGraphicsResource* _bufferResource;
	
	/******************************** METHODS *********************************/

	/**
	 * ...
	 *
	 * @param position ...
	 *
	 * @return ...
	 */
	uint3 getSecureIndex( uint3 position ) const;

};

} // namespace GvCore

namespace GvCore
{
	/**
	 * ...
	 *
	 * @param dstarray ...
	 * @param h_srcptr ...
	 */
	template< typename T >
	void memcpyArray( Array3DGPULinear< T >* dstarray, T* h_srcptr );

	/**
	 * ...
	 *
	 * @param dstarray ...
	 * @param h_srcptr ...
	 * @param numElems ...
	 */
	template< typename T >
	void memcpyArray( Array3DGPULinear< T >* dstarray, T* h_srcptr, uint numElems );

	/**
	 * ...
	 *
	 * @param dstarray ...
	 * @param srcarray ...
	 */
	template< typename T >
	void memcpyArray( Array3DGPULinear< T >* dstarray, Array3D< T >* srcarray );

	/**
	 * ...
	 *
	 * @param dstarray ...
	 * @param srcarray ...
	 */
	template< typename T >
	void memcpyArray( Array3D< T >* dstarray, Array3DGPULinear< T >* srcarray, cudaStream_t pStream = NULL );

	/**
	 * ...
	 *
	 * @param dstarray ...
	 * @param srcarray ...
	 * @param numElems ...
	 */
	template< typename T >
	void memcpyArray( Array3D< T >* dstarray, Array3DGPULinear< T >* srcarray, uint numElems );

} // namespace GvCore

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "Array3DGPULinear.inl"

/******************************************************************************
 ************************** INSTANTIATION SECTION *****************************
 ******************************************************************************/

namespace GvCore
{
	//GIGASPACE_TEMPLATE_EXPORT template class GIGASPACE_EXPORT Array3DGPULinear< uint >;
	//GIGASPACE_TEMPLATE_EXPORT template class GIGASPACE_EXPORT Array3DGPULinear< uchar4 >;
	//GIGASPACE_TEMPLATE_EXPORT template class GIGASPACE_EXPORT Array3DGPULinear< float >;
	//GIGASPACE_TEMPLATE_EXPORT template class GIGASPACE_EXPORT Array3DGPULinear< float4 >;
}

#endif
