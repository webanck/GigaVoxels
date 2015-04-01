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
#include "GvStructure/GvVolumeTreeKernel.h"

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvCore
{

	/******************************************************************************
	 * Get the value at given position
	 *
	 * @param position position
	 *
	 * @return the value at given position
	 ******************************************************************************/
	template< typename T >
	template< uint channel >
	__device__
	__forceinline__ T Array3DKernelTex< T >::get( const uint3& position ) const
	{
		T data;

#if (__CUDA_ARCH__ >= 200)
		// FIXME : better way to do this ?
		switch ( channel )
		{
		case 0:
			surf3Dread( &data, CreateGPUPoolChannelSurfaceReferenceName( 0 ), position.x * sizeof( T ), position.y, position.z );
			break;

		case 1:
			surf3Dread( &data, CreateGPUPoolChannelSurfaceReferenceName( 1 ), position.x * sizeof( T ), position.y, position.z );
			break;

		case 2:
			surf3Dread( &data, CreateGPUPoolChannelSurfaceReferenceName( 2 ), position.x * sizeof( T ), position.y, position.z );
			break;

		case 3:
			surf3Dread( &data, CreateGPUPoolChannelSurfaceReferenceName( 3 ), position.x * sizeof( T ), position.y, position.z );
			break;

		case 4:
			surf3Dread( &data, CreateGPUPoolChannelSurfaceReferenceName( 4 ), position.x * sizeof( T ), position.y, position.z );
			break;

		default:
			assert( false );	// TO DO : handle this.
			break;
		}
#endif
		return (data);
	}

	/******************************************************************************
	 * Set the value at given position
	 *
	 * @param position position
	 * @param val the value to write
	 ******************************************************************************/
	template< typename T >
	template< uint channel >
	__device__
	__forceinline__ void Array3DKernelTex< T >::set( const uint3& position, T val )
	{
#if (__CUDA_ARCH__ >= 200)
		// FIXME : better way to do this ?
		switch ( channel )
		{
		case 0:
            surf3Dwrite( val, CreateGPUPoolChannelSurfaceReferenceName( 0 ), position.x * sizeof( T ), position.y, position.z );
            break;

		case 1:
            surf3Dwrite( val, CreateGPUPoolChannelSurfaceReferenceName( 1 ), position.x * sizeof( T ), position.y, position.z );
			break;

		case 2:
            surf3Dwrite( val, CreateGPUPoolChannelSurfaceReferenceName( 2 ), position.x * sizeof( T ), position.y, position.z );
			break;

		case 3:
            surf3Dwrite( val, CreateGPUPoolChannelSurfaceReferenceName( 3 ), position.x * sizeof( T ), position.y, position.z );
			break;

		case 4:
            surf3Dwrite( val, CreateGPUPoolChannelSurfaceReferenceName( 4 ), position.x * sizeof( T ), position.y, position.z );
			break;

		default:
			assert( false );	// TO DO : handle this.
			break;
		}
#endif
	}

} // namespace GvCore
