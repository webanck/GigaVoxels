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

// Cuda
#include <cuda_runtime.h>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvRendering
{
	
/******************************************************************************
 * ...
 ******************************************************************************/
inline cudaError_t GvGraphicsResource::getMappedPointer( void** pDevicePointer, size_t* pSize )
{
	return cudaGraphicsResourceGetMappedPointer( pDevicePointer, pSize, _graphicsResource );
}

/******************************************************************************
 * ...
 ******************************************************************************/
inline cudaError_t GvGraphicsResource::getMappedArray( cudaArray** pArray, unsigned int pArrayIndex, unsigned int pMipLevel )
{
	return cudaGraphicsSubResourceGetMappedArray( pArray, _graphicsResource, pArrayIndex, pMipLevel );
}

/******************************************************************************
 * ...
 ******************************************************************************/
inline GvGraphicsResource::MemoryType GvGraphicsResource::getMemoryType() const
{
	return _memoryType;
}

/******************************************************************************
 * ...
 ******************************************************************************/
inline GvGraphicsResource::MappedAddressType GvGraphicsResource::getMappedAddressType() const
{
	return _mappedAddressType;
}

} // namespace GvRendering
