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

#ifndef _GV_GRAPHICS_RESOURCE_H_
#define _GV_GRAPHICS_RESOURCE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

// OpenGL
#include <GL/glew.h>

// Cuda
#include <driver_types.h>

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

namespace GvRendering
{

/** 
 * @class GvGraphicsResource
 *
 * @brief The GvGraphicsResource class provides interface to handle
 * graphics resources from graphics libraries like OpenGL (or DirectX),
 * in the CUDA memory context.
 *
 * Some resources from OpenGL may be mapped into the address space of CUDA,
 * either to enable CUDA to read data written by OpenGL, or to enable CUDA
 * to write data for consumption by OpenGL.
 */
class GIGASPACE_EXPORT GvGraphicsResource
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Enumeration of the graphics IO slots
	 * connected to the GigaVoxles engine
	 * (i.e. color and depth inputs/outputs)
	 */
	enum Access
	{
		eNone,
		eRead,
		eWrite,
		eReadWrite,
	};

	/**
	 * Enumeration of the graphics IO types
	 * connected to the GigaVoxles engine
	 * (i.e. color and depth inputs/outputs)
	 */
	enum Type
	{
		eUndefinedType = -1,
		eBuffer,
		eImage,
		eNbTypes
	};

	/**
	 * Enumeration of the graphics IO types
	 * connected to the GigaVoxles engine
	 * (i.e. color and depth inputs/outputs)
	 */
	enum MappedAddressType
	{
		eUndefinedMappedAddressType,
		//eNone,
		ePointer,
		eTexture,
		eSurface,
		eNbMappedAddressTypes
	};

	/**
	 * Memory type
	 */
	enum MemoryType
	{
		eUndefinedMemoryType = -1,
		eDevicePointer,
		eCudaArray,
		eNbMemoryTypes
	};

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvGraphicsResource();

	/**
	 * Destructor
	 */
	 virtual ~GvGraphicsResource();

	/**
	 * Initiliaze
	 */
	void initialize();

	/**
	 * Finalize
	 */
	void finalize();

	/**
	 * Reset
	 */
	void reset();

	/**
	 * Registers an OpenGL buffer object.
	  */
	cudaError_t registerBuffer( GLuint pBuffer, unsigned int pFlags );

	/**
	 * Register an OpenGL texture or renderbuffer object.
	 */
	cudaError_t registerImage( GLuint pImage, GLenum pTarget, unsigned int pFlags );

	/**
	 * Unregisters a graphics resource for access by CUDA.
	 */
	cudaError_t unregister();

	/**
	 * Map graphics resources for access by CUDA.
	 */
	cudaError_t map();

	/**
	 * Unmap graphics resources for access by CUDA.
	 */
	cudaError_t unmap();

	/**
	 * Get an device pointer through which to access a mapped graphics resource.
	 * Get an array through which to access a subresource of a mapped graphics resource.
	 */
	void* getMappedAddress();

	/**
	 * ...
	 */
	inline MemoryType getMemoryType() const;

	/**
	 * ...
	 */
	inline MappedAddressType getMappedAddressType() const;

	/**
	 * ...
	 */
	MappedAddressType _mappedAddressType;

	/**
	 * Tell wheter or not the associated CUDA graphics resource has already been registered
	 *
	 * @return a flag telling wheter or not the associated CUDA graphics resource has already been registered
	 */
	bool isRegistered() const;

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * OpenGL buffer that will be registered in the GigaVoxels engine
	 */
	GLuint _graphicsBuffer;

	/**
	 * CUDA graphics resource associated to registered OpenGL buffer
	 */
	cudaGraphicsResource* _graphicsResource;

	/**
	 * CUDA graphics resource types associated to OpenGL buffers (i.e buffer or image)
	 */
	Type _type;

	/**
	 * ...
	 */
	Access _access;

	///**
	// * ...
	// */
	//MappedAddressType _mappedAddressType;

	/**
	 * CUDA graphics resource mapped address associated to registered OpenGL buffers
	 */
	void* _mappedAddress;

	/**
	 * Offset (in texel unit) to apply during texture fetches
	 */
	size_t _textureOffset;

	/**
	 * Indentifier
	 */
	unsigned int _id;

	/**
	 * ...
	 */
	bool _isRegistered;

	/**
	 * ...
	 */
	bool _isMapped;

	/**
	 * ...
	 */
	unsigned int _flags;

	/**
	 * Memory type
	 */
	MemoryType _memoryType;

	/******************************** METHODS *********************************/

	/**
	 * ...
	 */
	inline cudaError_t getMappedPointer( void** pDevicePointer, size_t* pSize );

	/**
	 * ...
	 */
	inline cudaError_t getMappedArray( cudaArray** pArray, unsigned int pArrayIndex, unsigned int pMipLevel );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} // namespace GvRendering

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvGraphicsResource.inl"

#endif
