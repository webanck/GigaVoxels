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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/TemplateHelpers.h"
#include "GvCore/TypeHelpers.h"
#include "GvCore/GvError.h"

// System
#include <cstring>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvCore
{

/******************************************************************************
 * Constructor
 *
 * @param res resolution
 * @param options options
 *
 * @TODO Modify API to handle other texture types than GL_RGBA8 : ex "float" textures for voxelization, etc...
 ******************************************************************************/
template< typename T >
inline Array3DGPUTex< T >::Array3DGPUTex( const uint3& res, uint options )
:	_dataArray( NULL )
,	_textureReferenceName()
{
	_arrayOptions = options;

	if ( _arrayOptions & static_cast< uint >( GLInteroperability ) )
	{
		_resolution = res;
		_channelFormatDesc = cudaCreateChannelDesc< T >();

		// Allocate the buffer with OpenGL
		glGenTextures( 1, &_bufferObject );
		
		glBindTexture( GL_TEXTURE_3D, _bufferObject );
		
		glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
		glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

		const char* type = typeToString< T >();
		if ( strcmp( type, "float" ) != 0 )
		{
			if ( strcmp( type, "half4" ) == 0 )
			{
				//glTexImage3D( GL_TEXTURE_3D, 0, GL_RGBA32F, res.x, res.y, res.z, 0, GL_RGBA, GL_FLOAT, NULL );
				glTexImage3D( GL_TEXTURE_3D, 0, GL_RGBA16F, res.x, res.y, res.z, 0, GL_RGBA, GL_FLOAT, NULL );
			}
			else
			{
				glTexImage3D( GL_TEXTURE_3D, 0, GL_RGBA8, res.x, res.y, res.z, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL );
			}
		}
		else
		{
			// "float" type
			glTexImage3D( GL_TEXTURE_3D, 0, GL_R32F, res.x, res.y, res.z, 0, GL_RED, GL_FLOAT, NULL );
		}

		glBindTexture( GL_TEXTURE_3D, 0 );
		
		GV_CUDA_SAFE_CALL( cudaGraphicsGLRegisterImage( &_bufferResource, _bufferObject, GL_TEXTURE_3D, cudaGraphicsRegisterFlagsSurfaceLoadStore ) );

		// Register it inside Cuda
		//cudaGraphicsGLRegisterBuffer( &_bufferResource, _bufferObject, cudaGraphicsMapFlagsNone );
		
		mapResource();
	}
	else
	{
		allocArray( res, cudaCreateChannelDesc< T >() );
	}
}

/******************************************************************************
 * Constructor
 *
 * @param res resolution
 * @param cfd channel format descriptor
 ******************************************************************************/
template< typename T >
inline Array3DGPUTex< T >::Array3DGPUTex( const uint3& res, cudaChannelFormatDesc channelFormatDesc )
{
	allocArray( res, channelFormatDesc );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename T >
inline Array3DGPUTex< T >::~Array3DGPUTex()
{
	unbindTexture( _textureReferenceName.c_str() );

	if ( _arrayOptions & (uint)GLInteroperability )
	{
		unmapResource();
		GV_CUDA_SAFE_CALL( cudaGraphicsUnregisterResource( _bufferResource ) );
	}

	GV_CUDA_SAFE_CALL( cudaFreeArray( _dataArray ) );
	_dataArray = NULL;
}

/******************************************************************************
 * Get the resolution
 *
 * @return the resolution
 ******************************************************************************/
template< typename T >
inline uint3 Array3DGPUTex< T >::getResolution() const
{
	return _resolution;
}

/******************************************************************************
 * Bind texture to array
 *
 * @param symbol device texture symbol
 * @param texRefName texture reference name
 * @param normalizedAccess Type of access
 * @param filterMode Type of filtering mode
 * @param addressMode Type of address mode
 ******************************************************************************/
template< typename T >
inline void Array3DGPUTex< T >::bindToTextureReference( const void* pTextureReferenceSymbol, const char* texRefName, bool normalizedAccess, cudaTextureFilterMode filterMode, cudaTextureAddressMode addressMode )
{
	std::cout << "bindToTextureReference : " << texRefName << std::endl;

	_textureReferenceName = std::string( texRefName );

	textureReference* texRefPtr = NULL;
	GV_CUDA_SAFE_CALL( cudaGetTextureReference( (const textureReference **)&texRefPtr, pTextureReferenceSymbol ) );

	// Update internal storage
	_textureSymbol = pTextureReferenceSymbol;

	texRefPtr->normalized = normalizedAccess; // Access with normalized texture coordinates
	texRefPtr->filterMode = filterMode;
	texRefPtr->addressMode[ 0 ] = addressMode; // Wrap texture coordinates
	texRefPtr->addressMode[ 1 ] = addressMode;
	texRefPtr->addressMode[ 2 ] = addressMode;

	// Bind array to 3D texture
	GV_CUDA_SAFE_CALL( cudaBindTextureToArray( (const textureReference *)texRefPtr, _dataArray, &_channelFormatDesc ) );
}

/******************************************************************************
 * Unbind texture to array
 *
 * @param texRefName texture reference name
 ******************************************************************************/
template< typename T >
inline void Array3DGPUTex< T >::unbindTexture( const char* texRefName )
{
	std::cout << "unbindTexture : " << texRefName << std::endl;

	textureReference* texRefPtr = NULL;
	GV_CUDA_SAFE_CALL( cudaGetTextureReference( (const textureReference **)&texRefPtr, _textureSymbol ) );
	
	if ( texRefPtr != NULL )
	{
		GV_CUDA_SAFE_CALL( cudaUnbindTexture( static_cast< const textureReference* >( texRefPtr ) ) );
	}
}

/******************************************************************************
 * Bind surface to array
 *
 * @param surfRefName device surface symbol
 * @param surfRefName surface reference name
 ******************************************************************************/
template< typename T >
inline void Array3DGPUTex< T >::bindToSurfaceReference( const void* pSurfaceReferenceSymbol, const char* surfRefName )
{
	std::cout << "bindToSurfaceReference : " << surfRefName << std::endl;

	const surfaceReference* surfRefPtr;
	GV_CUDA_SAFE_CALL( cudaGetSurfaceReference( &surfRefPtr, pSurfaceReferenceSymbol ) );
	GV_CUDA_SAFE_CALL( cudaBindSurfaceToArray( surfRefPtr, _dataArray, &_channelFormatDesc ) );
}

/******************************************************************************
 * Get the associated device-side object
 *
 * @return the associated device-side object
 ******************************************************************************/
template< typename T >
inline Array3DKernelTex< T > Array3DGPUTex< T >::getDeviceArray()
{
	Array3DKernelTex< T > kat;
	return kat;
}

/******************************************************************************
 * Get the internal device memory array
 *
 * @return the internal device memory array
 ******************************************************************************/
template< typename T >
inline cudaArray* Array3DGPUTex< T >::getCudaArray()
{
	return _dataArray;
}

/******************************************************************************
 * Get the associated graphics library handle if graphics library interoperability is used
 *
 * @return the associated graphics library handle
 ******************************************************************************/
template< typename T >
inline GLuint Array3DGPUTex< T >::getBufferName() const
{
	return _bufferObject;
}

/******************************************************************************
 * Map the associated graphics resource if graphics library interoperability is used
 ******************************************************************************/
template< typename T >
inline void Array3DGPUTex< T >::mapResource()
{
	if ( _arrayOptions & (uint)GLInteroperability )
	{
		GV_CUDA_SAFE_CALL( cudaGraphicsMapResources( 1, &_bufferResource, 0 ) );
		//cudaGraphicsResourceGetMappedPointer((void **)&_dataArray, &bufferSize, _bufferResource);
		GV_CUDA_SAFE_CALL( cudaGraphicsSubResourceGetMappedArray( &_dataArray, _bufferResource, 0, 0 ) );
	}
}

/******************************************************************************
 * Unmap the associated graphics resource if graphics library interoperability is used
 ******************************************************************************/
template< typename T >
inline void Array3DGPUTex< T >::unmapResource()
{
	if ( _arrayOptions & (uint)GLInteroperability )
	{
		GV_CUDA_SAFE_CALL( cudaGraphicsUnmapResources( 1, &_bufferResource, 0 ) );
		_dataArray = 0;
	}
}

/******************************************************************************
 * Helper method to allocate internal device memory array
 *
 * @param res resolution
 * @param channelFormatDesc channel format descriptor
 ******************************************************************************/
template< typename T >
inline void Array3DGPUTex< T >::allocArray( const uint3& res, cudaChannelFormatDesc channelFormatDesc )
{
	_resolution = res;
	_channelFormatDesc = channelFormatDesc;

	GV_CUDA_SAFE_CALL( cudaMalloc3DArray( &_dataArray, &channelFormatDesc, make_cudaExtent( _resolution ), cudaArraySurfaceLoadStore ) );
}

/******************************************************************************
 * Get the associated CUDA graphics resource
 *
 * return the associated CUDA graphics resource
 ******************************************************************************/
template< typename T >
inline cudaGraphicsResource* Array3DGPUTex< T >::getGraphicsResource()
{
	return _bufferResource;
}

} //namespace GvCore

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvCore
{

/******************************************************************************
 * ...
 *
 * @param dstarray ...
 * @param posDst ...
 * @param resDst ...
 * @param h_srcptr ...
 ******************************************************************************/
template< typename T >
inline void memcpyArray( Array3DGPUTex< T >* dstarray, uint3 posDst, uint3 resDst, const T* h_srcptr )
{
	cudaMemcpy3DParms copyParams = { 0 };

	copyParams.kind = cudaMemcpyHostToDevice;

	copyParams.srcPtr = make_cudaPitchedPtr( (void*)h_srcptr, resDst.x * sizeof( T ), resDst.x, resDst.y );
	copyParams.srcPos = make_cudaPos( 0, 0, 0 );
	
	copyParams.dstArray = dstarray->getCudaArray();
	copyParams.dstPos	= make_cudaPos( posDst.x, posDst.y, posDst.z );
	copyParams.extent   = make_cudaExtent( resDst );
		
	GV_CUDA_SAFE_CALL( cudaMemcpy3D( &copyParams ) );
}

/******************************************************************************
 * Copy a Array3DGPUTex device array to a Array3D  host array
 ******************************************************************************/
template< typename T >
inline void memcpyArray( Array3D< T >* pDestinationArray, Array3DGPUTex< T >* pSourceArray )
{
	cudaMemcpy3DParms copyParams = { 0 };

	copyParams.kind = cudaMemcpyDeviceToHost;

	copyParams.dstPtr = pDestinationArray->getCudaPitchedPtr();
	copyParams.srcPos = make_cudaPos( 0, 0, 0 );
	
	copyParams.srcArray = pSourceArray->getCudaArray();
	copyParams.dstPos	= make_cudaPos( 0, 0, 0 );

	copyParams.extent   = make_cudaExtent( pDestinationArray->getResolution() );
		
	GV_CUDA_SAFE_CALL( cudaMemcpy3D( &copyParams ) );
}

// Explicit instantiations
//TEMPLATE_INSTANCIATE_CLASS_TYPES(Array3DGPUTex);

// memcpyArray : Explicit instanciations
//template void memcpyArray<uchar4>(Array3DGPUTex<uchar4> *dstarray, uint3 posDst, uint3 resDst, const uchar4* h_srcptr);

} //namespace GvCore
