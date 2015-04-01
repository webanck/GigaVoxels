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

#include "GvRendering/GvGraphicsResource.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvError.h"

// Cuda
#include <cuda_gl_interop.h>

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvRendering;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvGraphicsResource::GvGraphicsResource()
:	_graphicsBuffer( 0 )
,	_graphicsResource( NULL )
,	_type( eUndefinedType )
,	_access( eNone )
,	_mappedAddressType( eUndefinedMappedAddressType )
,	_mappedAddress( NULL )
,	_textureOffset( 0 )
,	_id( 0 )
,	_isRegistered( false )
,	_isMapped( false )
,	_flags( 0 )
,	_memoryType( eUndefinedMemoryType )
{
}

/******************************************************************************
 * Desconstructor
 ******************************************************************************/
GvGraphicsResource::~GvGraphicsResource()
{
	if ( _isMapped )
	{
		unmap();
	}
	
	if ( _isRegistered )
	{
		unregister();
	}
}

/******************************************************************************
 * Initiliaze
 ******************************************************************************/
void GvGraphicsResource::initialize()
{
	// TO DO ...
	assert( false );
}

/******************************************************************************
 * Finalize
 ******************************************************************************/
void GvGraphicsResource::finalize()
{
	// TO DO ...
	assert( false );
}

/******************************************************************************
 * Finalize
 ******************************************************************************/
void GvGraphicsResource::reset()
{
	// TO DO ...
	assert( false );
}

/******************************************************************************
 * Registers an OpenGL buffer object.
 ******************************************************************************/
cudaError_t GvGraphicsResource::registerBuffer( GLuint pBuffer, unsigned int pFlags )
{
	// Registers an OpenGL buffer object
	cudaError_t error = cudaGraphicsGLRegisterBuffer( &_graphicsResource, pBuffer, pFlags );
	assert( error == cudaSuccess );

	_graphicsBuffer = pBuffer;
	_flags = pFlags;

	_isRegistered = true;
	_memoryType = eDevicePointer;
	_type = eBuffer;
	switch ( pFlags )
	{
		case cudaGraphicsRegisterFlagsNone:
			_access = eReadWrite;
			break;

		case cudaGraphicsRegisterFlagsReadOnly:
			_access = eRead;
			break;

		case cudaGraphicsRegisterFlagsWriteDiscard:
			_access = eWrite;
			break;

		default:
			_access = eNone;
			break;
	}
	_mappedAddressType = ePointer;
	
	return error;
}

/******************************************************************************
 * Register an OpenGL texture or renderbuffer object.
 ******************************************************************************/
cudaError_t GvGraphicsResource::registerImage( GLuint pImage, GLenum pTarget, unsigned int pFlags )
{
	// target must match the type of the object, and must be one of GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE,
	// GL_TEXTURE_CUBE_MAP, GL_TEXTURE_3D, GL_TEXTURE_2D_ARRAY, or GL_RENDERBUFFER.

	// Register an OpenGL texture or renderbuffer object
	cudaError_t error = cudaGraphicsGLRegisterImage( &_graphicsResource, pImage, pTarget, pFlags );
	assert( error == cudaSuccess );
	
	_graphicsBuffer = pImage;
	_flags = pFlags;

	_isRegistered = true;
	_memoryType = eCudaArray;
	_type = eImage;
	switch ( pFlags )
	{
		case cudaGraphicsRegisterFlagsNone:
			_access = eReadWrite;
			break;

		case cudaGraphicsRegisterFlagsReadOnly:
			_access = eRead;
			break;

		case cudaGraphicsRegisterFlagsWriteDiscard:
			_access = eWrite;
			break;

		case cudaGraphicsRegisterFlagsSurfaceLoadStore:
			_access = eReadWrite;
			break;

		case cudaGraphicsRegisterFlagsTextureGather:
			_access = eRead;	// not sure ?
			break;

		default:
			_access = eNone;
			break;
	}
	//_mappedAddressType = eTexture / eSurface; // TO DO ...
		
	return error;
}

/******************************************************************************
 * Unregisters a graphics resource for access by CUDA.
 ******************************************************************************/
cudaError_t GvGraphicsResource::unregister()
{
	// Unregisters a graphics resource for access by CUDA
	cudaError_t error = cudaGraphicsUnregisterResource( _graphicsResource );
	assert( error == cudaSuccess );

	_graphicsBuffer = 0;
	_flags = 0;

	_isRegistered = false;
	_type = eUndefinedType;
	
	return error;
}

/******************************************************************************
 * Map graphics resources for access by CUDA.
 ******************************************************************************/
cudaError_t GvGraphicsResource::map()
{
	cudaStream_t stream = 0;
	cudaError_t error = cudaGraphicsMapResources( 1, &_graphicsResource, stream );
	assert( error == cudaSuccess );

	_isMapped = true;
			
	return error;
}

/******************************************************************************
 * Unmap graphics resources.
 ******************************************************************************/
cudaError_t GvGraphicsResource::unmap()
{
	cudaStream_t stream = 0;
	cudaError_t error = cudaGraphicsUnmapResources( 1, &_graphicsResource, stream );
	assert( error == cudaSuccess );

	_isMapped = false;
		
	return error;
}

/******************************************************************************
 * Get an device pointer through which to access a mapped graphics resource.
 * Get an array through which to access a subresource of a mapped graphics resource.
 ******************************************************************************/
void* GvGraphicsResource::getMappedAddress()
{
	// TESTER si mappé d'abord ?
	// ...

	void* mappedAddress = NULL;
	cudaError_t error;

	if ( _type == eBuffer )
	{
		void* devicePointer = NULL;
		size_t size = 0;
		error = getMappedPointer( &devicePointer, &size );
		assert( error == cudaSuccess );
		mappedAddress = devicePointer;
	}
	else if ( _type == eImage )
	{
		cudaArray* imageArray = NULL;
		unsigned int arrayIndex = 0;
		unsigned int mipLevel = 0;
		error = getMappedArray( &imageArray, arrayIndex, mipLevel );
		assert( error == cudaSuccess );
		mappedAddress = imageArray;
	}
	else
	{
		// TO DO
		// Handle error
		// ...
		assert( false );
	}

	return mappedAddress;
}

/******************************************************************************
 * Tell wheter or not the associated CUDA graphics resource has already been registered
 *
 * @return a flag telling wheter or not the associated CUDA graphics resource has already been registered
 ******************************************************************************/
bool GvGraphicsResource::isRegistered() const
{
	return _isRegistered;
}
