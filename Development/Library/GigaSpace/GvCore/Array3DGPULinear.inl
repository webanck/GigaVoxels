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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvError.h"

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvCore
{
	
/******************************************************************************
 * Constructor.
 * Create a 3D array of the given resolution using GPU linear memory.
 *
 * @param res ...
 * @param options ...
 ******************************************************************************/
template< typename T >
inline Array3DGPULinear< T >::Array3DGPULinear( const uint3& res, uint options )
:	_data( NULL )
{
	_resolution = res;
	_arrayOptions = options & ~SharedData;

	if ( _arrayOptions & (uint)GLInteroperability )
	{
		// Allocate the buffer with OpenGL
		glGenBuffers( 1, &_bufferObject );
		glBindBuffer( GL_TEXTURE_BUFFER, _bufferObject );
		glBufferData( GL_TEXTURE_BUFFER, getMemorySize(), NULL, GL_DYNAMIC_DRAW );
		//glMakeBufferResidentNV( GL_TEXTURE_BUFFER, GL_READ_WRITE );
		//glGetBufferParameterui64vNV( GL_TEXTURE_BUFFER, GL_BUFFER_GPU_ADDRESS_NV, &_bufferAddress );
		glBindBuffer( GL_TEXTURE_BUFFER, 0 );
		GV_CHECK_GL_ERROR();

		// Register it inside Cuda
		GV_CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer( &_bufferResource, _bufferObject, cudaGraphicsRegisterFlagsNone ) );
		mapResource();
	}
	else
	{
		GV_CUDA_SAFE_CALL( cudaMalloc( (void**)&_data, getMemorySize() ) );
	}

	_pitch = _resolution.x * sizeof( T );
}

/******************************************************************************
 * Constructor.
 * Create a 3D array of the given resolution.
 *
 * @param data ...
 * @param res ...
 ******************************************************************************/
template< typename T >
inline Array3DGPULinear< T >::Array3DGPULinear( T* data, const uint3& res )
:	_data( NULL )
{
	_resolution = res;

	_data = data;
	_pitch = _resolution.x * sizeof( T );

	_arrayOptions = (uint)SharedData;
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
template< typename T >
inline Array3DGPULinear< T >::~Array3DGPULinear()
{
	if ( _arrayOptions & (uint)GLInteroperability )
	{
		unmapResource();
		GV_CUDA_SAFE_CALL( cudaGraphicsUnregisterResource( _bufferResource ) );

		// TO DO : need a glDeleteBuffers() ?
	}

	if ( _data && !( _arrayOptions & (uint)SharedData ) )
	{
		GV_CUDA_SAFE_CALL( cudaFree( _data ) );
		_data = NULL;
	}
}

/******************************************************************************
 * ...
 *
 * @return ...
 ******************************************************************************/
template< typename T >
inline uint3 Array3DGPULinear< T >::getResolution() const
{
	return _resolution;
}

/******************************************************************************
 * ...
 *
 * @return ...
 ******************************************************************************/
template< typename T >
inline size_t Array3DGPULinear< T >::getNumElements() const
{
	return _resolution.x * _resolution.y * _resolution.z;
}

/******************************************************************************
 * ...
 *
 * @return ...
 ******************************************************************************/
template< typename T >
inline size_t Array3DGPULinear< T >::getMemorySize() const
{
	return getNumElements() * sizeof( T );
}

/******************************************************************************
 * ...
 *
 * @param pos ...
 *
 * @return ...
 ******************************************************************************/
template< typename T >
inline T* Array3DGPULinear< T >::getPointer( const uint3& pos ) const
{
	return &_data[ pos.x + pos.y * _resolution.x + pos.z * _resolution.x * _resolution.y ];
}

/******************************************************************************
 * ...
 *
 * @param offset ...
 *
 * @return ...
 ******************************************************************************/
template< typename T >
inline T* Array3DGPULinear< T >::getPointer( size_t offset ) const
{
	return &_data[ offset ];
}

/******************************************************************************
 * ...
 *
 * @return ...
 ******************************************************************************/
template< typename T >
inline T* Array3DGPULinear< T >::getPointer() const
{
	return _data;
}

/******************************************************************************
 * ...
 *
 * @param pos ...
 *
 * @return ...
 ******************************************************************************/
template< typename T >
inline const T* Array3DGPULinear< T >::getConstPointer( const uint3& pos ) const
{
	return &_data[ pos.x + pos.y * _resolution.x + pos.z * _resolution.x * _resolution.y ];
}

/******************************************************************************
 * ...
 *
 * @param address ...
 *
 * @return ...
 ******************************************************************************/
template< typename T >
inline const T* Array3DGPULinear< T >::getConstPointer( const size_t& address ) const
{
	return &_data[ address ];
}

/******************************************************************************
 * ...
 *
 * @param dptr ...
 ******************************************************************************/
template< typename T >
inline void Array3DGPULinear< T >::manualSetDataStorage( T* dptr )
{
	_data = dptr;
}

/******************************************************************************
 * ...
 *
 * @return
 ******************************************************************************/
template< typename T >
inline T** Array3DGPULinear< T >::getDataStoragePtrAddress()
{
	return &_data;
}

/******************************************************************************
 * ...
 ******************************************************************************/
/*template< typename T >
inline void Array3DGPULinear< T >::zero()
{
this->fill(0);
}*/

/******************************************************************************
 *  Fill array with a value
 *
 * @param v value
 ******************************************************************************/
template< typename T >
inline void Array3DGPULinear< T >::fill( int v )
{
	//assert(0);	//This should not be used
	//std::cout<<"Warning: Array3DGPULinear< T >::fill is VERY slow \n";

	GV_CUDA_SAFE_CALL( cudaMemset( _data, v, getMemorySize() ) );
}

/******************************************************************************
 *  Fill array asynchrounously with a value
 *
 * @param v value
 ******************************************************************************/
template< typename T >
inline void Array3DGPULinear< T >::fillAsync( int v )
{
	//assert(0);	//This should not be used
	//std::cout<<"Warning: Array3DGPULinear< T >::fill is VERY slow \n";

	GV_CUDA_SAFE_CALL( cudaMemsetAsync( _data, v, getMemorySize() ) );
}

///
///GPU related stuff
///

/******************************************************************************
 * ...
 *
 * @return ...
 ******************************************************************************/
template< typename T >
inline Array3DKernelLinear< T > Array3DGPULinear< T >::getDeviceArray() const
{
	Array3DKernelLinear< T > kal;
	kal.init( _data, make_uint3( _resolution ), _pitch );

	return kal;
}

/******************************************************************************
 * ...
 *
 * @return ...
 ******************************************************************************/
template< typename T >
inline cudaPitchedPtr Array3DGPULinear< T >::getCudaPitchedPtr() const
{
	return make_cudaPitchedPtr( (void*)_data, _pitch, (size_t) _resolution.x, (size_t) _resolution.y );
}

/******************************************************************************
 * ...
 *
 * @return ...
 ******************************************************************************/
template< typename T >
inline cudaExtent Array3DGPULinear< T >::getCudaExtent() const
{
	return make_cudaExtent( _pitch, (size_t)_resolution.y, (size_t)_resolution.z );
}

/******************************************************************************
 * ...
 *
 * @param position ...
 *
 * @return ...
 ******************************************************************************/
template< typename T >
inline uint3 Array3DGPULinear< T >::getSecureIndex( uint3 position ) const
{
	if
		( position.x >= _resolution.x )
	{
		position.x = _resolution.x - 1;
	}

	if
		( position.y >= _resolution.y )
	{
		position.y = _resolution.y - 1;
	}

	if
		( position.z >= _resolution.z )
	{
		position.z = _resolution.z - 1;
	}

	return position;
}

/******************************************************************************
 * Map the associated graphics resource
 ******************************************************************************/
template< typename T >
inline void Array3DGPULinear< T >::mapResource()
{
	if ( _arrayOptions & (uint)GLInteroperability )
	{
		size_t bufferSize;

		GV_CUDA_SAFE_CALL( cudaGraphicsMapResources( 1, &_bufferResource, 0 ) );
		GV_CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer( (void **)&_data, &bufferSize, _bufferResource ) );
	}
}

/******************************************************************************
* Unmap the associated graphics resource
 ******************************************************************************/
template< typename T >
inline void Array3DGPULinear< T >::unmapResource()
{
	if ( _arrayOptions & (uint)GLInteroperability )
	{
		GV_CUDA_SAFE_CALL( cudaGraphicsUnmapResources( 1, &_bufferResource, 0 ) );
		_data = 0;
	}
}

/******************************************************************************
 * Get the associated OpenGL handle
 *
 * @return the associated OpenGL buffer
 ******************************************************************************/
template< typename T >
inline GLuint Array3DGPULinear< T >::getBufferName() const
{
	return _bufferObject;
}

/******************************************************************************
 * Get the associated CUDA graphics resource
 *
 * return the associated CUDA graphics resource
 ******************************************************************************/
template< typename T >
inline cudaGraphicsResource* Array3DGPULinear< T >::getGraphicsResource()
{
	return _bufferResource;
}

/******************************************************************************
 * ...
 *
 * @return ...
 ******************************************************************************/
/*template< typename T >
inline GLuint64EXT Array3DGPULinear< T >::getBufferAddress() const
{
return _bufferAddress;
}*/

/******************************************************************************
 * ...
 *
 * @param dstarray ...
 * @param h_srcptr ...
 ******************************************************************************/
template< typename T >
inline void memcpyArray( Array3DGPULinear< T >* dstarray, T* h_srcptr )
{
	GV_CUDA_SAFE_CALL( cudaMemcpy( dstarray->getPointer(), h_srcptr, dstarray->getMemorySize(), cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * ...
 *
 * @param dstarray ...
 * @param h_srcptr ...
 * @param numElems ...
 ******************************************************************************/
template< typename T >
inline void memcpyArray( Array3DGPULinear< T >* dstarray, T* h_srcptr, uint numElems )
{
	GV_CUDA_SAFE_CALL( cudaMemcpy( dstarray->getPointer(), h_srcptr, numElems * sizeof( T ), cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * ...
 *
 * @param dstarray ...
 * @param srcarray ...
 ******************************************************************************/
template< typename T >
inline void memcpyArray( Array3DGPULinear< T >* dstarray, Array3D< T >* srcarray )
{
	memcpyArray( dstarray, srcarray->getPointer() );
}

/******************************************************************************
 * ...
 *
 * @param dstarray ...
 * @param srcarray ...
 ******************************************************************************/
template< typename T >
inline void memcpyArray( Array3D< T >* dstarray, Array3DGPULinear< T >* srcarray, cudaStream_t pStream )
{
	cudaMemcpy3DParms copyParams = { 0 };

	copyParams.kind = cudaMemcpyDeviceToHost;

	copyParams.srcPtr = srcarray->getCudaPitchedPtr();
	copyParams.dstPtr = dstarray->getCudaPitchedPtr();

	copyParams.extent = make_cudaExtent( srcarray->getResolution().x * sizeof( T ), srcarray->getResolution().y, srcarray->getResolution().z );

	GV_CUDA_SAFE_CALL( cudaMemcpy3D( &copyParams ) );
	//GV_CUDA_SAFE_CALL( cudaMemcpy3DAsync( &copyParams, pStream ) );
}

/******************************************************************************
 * ...
 *
 * @param dstarray ...
 * @param srcarray ...
 * @param numElems ...
 ******************************************************************************/
template< typename T >
void memcpyArray( Array3D< T >* dstarray, Array3DGPULinear< T >* srcarray, uint numElems )
{
	cudaMemcpy3DParms copyParams = { 0 };

	copyParams.kind = cudaMemcpyDeviceToHost;

	copyParams.srcPtr = srcarray->getCudaPitchedPtr();
	copyParams.dstPtr = dstarray->getCudaPitchedPtr();

	copyParams.extent = make_cudaExtent( srcarray->getResolution().x * sizeof( T ), srcarray->getResolution().y, srcarray->getResolution().z );

	GV_CUDA_SAFE_CALL( cudaMemcpy3D( &copyParams ) );
}

/******************************************************************************
 *
 ******************************************************************************/
/*template< typename T >
inline void memcpyArray(Array3DGPUTex<T> *dstarray, const T* h_srcptr)
{
cudaMemcpy3DParms copyParams = { 0 };

copyParams.kind = cudaMemcpyDeviceToHost;

copyParams.srcPtr
= make_cudaPitchedPtr(h_srcptr, dstarray->getResolution().x
* sizeof(T), dstarray->getResolution().x, dstarray->getResolution().y);
copyParams.dstPtr = dstarray->getCudaPitchedPtr();

copyParams.extent = make_cudaExtent(dstarray->getResolution());

GV_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));

}*/

/******************************************************************************
 *
 ******************************************************************************/
//template< typename T >
//inline void memcpyArray(Array3DGPUTex<T> *dstarray, const T* h_srcptr)
//{
//	cudaMemcpy3DParms copyParams = { 0 };
//
//	copyParams.kind = cudaMemcpyHostToDevice;
//
//	copyParams.srcPtr = make_cudaPitchedPtr((void*)h_srcptr, dstarray->getResolution().x* sizeof(T),	dstarray->getResolution().x, dstarray->getResolution().y);
//	copyParams.srcPos	= make_cudaPos(0, 0, 0);
//
//	copyParams.dstArray = dstarray->getCudaArray();
//	copyParams.dstPos	= make_cudaPos(0, 0, 0);
//	copyParams.extent   = make_cudaExtent(dstarray->getResolution());
//
//
//	GV_CUDA_SAFE_CALL(cudaMemcpy3D(&copyParams));
//}

} // namespace GvCore
