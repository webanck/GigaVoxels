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

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvCore
{

/******************************************************************************
 * Inititialize.
 *
 * @param pData pointer on data
 * @param pRes resolution
 * @param pPitch pitch
 ******************************************************************************/
template< typename T >
inline void Array3DKernelLinear< T >::init( T* pData, const uint3& pRes, size_t pPitch )
{
	_resolution = pRes;
	_data = pData;
	_pitch = pPitch;
	_pitchxy = _resolution.x * _resolution.y;
}

/******************************************************************************
 * Get the resolution.
 *
 * @return the resolution
 ******************************************************************************/
template< typename T >
__device__
__forceinline__ uint3 Array3DKernelLinear< T >::getResolution() const
{
	return _resolution;
}

/******************************************************************************
 * Get the memory size.
 *
 * @return the memory size
 ******************************************************************************/
template< typename T >
__device__
__forceinline__ size_t Array3DKernelLinear< T >::getMemorySize() const
{
	return __uimul( __uimul( __uimul( _resolution.x, _resolution.y ), _resolution.z ), sizeof( T ) );
}

/******************************************************************************
 * Get the value at a given 1D address.
 *
 * @param pAddress a 1D address
 *
 * @return the value at the given address
 ******************************************************************************/
template< typename T >
__device__
__forceinline__ /*const*/ T Array3DKernelLinear< T >::get( uint pAddress ) const
{
	return _data[ pAddress ];
}

/******************************************************************************
 * Get the value at a given 2D position.
 *
 * @param pPosition a 2D position
 *
 * @return the value at the given position
 ******************************************************************************/
template< typename T >
__device__
__forceinline__ /*const*/ T Array3DKernelLinear< T >::get( const uint2& pPosition ) const
{
	return _data[ getOffset( pPosition ) ];
}

/******************************************************************************
 * Get the value at a given 3D position.
 *
 * @param pPosition a 3D position
 *
 * @return the value at the given position
 ******************************************************************************/
template< typename T >
__device__
__forceinline__ /*const*/ T Array3DKernelLinear< T >::get( const uint3& pPosition ) const
{
	return _data[ getOffset( pPosition ) ];
}

/******************************************************************************
 * Get the value at a given 1D address in a safe way.
 * Bounds are checked and address is modified if needed (as a clamp).
 *
 * @param pAddress a 1D address
 *
 * @return the value at the given address
 ******************************************************************************/
template< typename T >
__device__
__forceinline__ /*const*/ T Array3DKernelLinear< T >::getSafe( uint pAddress ) const
{
	uint numelem = _pitchxy * _resolution.z;
	if
		( pAddress >= numelem )
	{
		pAddress = numelem - 1;
	}

	return _data[ pAddress ];
}

/******************************************************************************
 * Get the value at a given 3D position in a safe way.
 * Bounds are checked and position is modified if needed (as a clamp).
 *
 * @param pPosition a 3D position
 *
 * @return the value at the given position
 ******************************************************************************/
template< typename T >
__device__
__forceinline__ /*const*/ T Array3DKernelLinear< T >::getSafe( uint3 pPosition ) const
{
	pPosition = getSecureIndex( pPosition );

	return _data[ getOffset( pPosition ) ];
}

/******************************************************************************
 * Get a pointer on data at a given 1D address.
 *
 * @param pAddress a 1D address
 *
 * @return the pointer at the given address
 ******************************************************************************/
template< typename T >
__device__
__forceinline__ T* Array3DKernelLinear< T >::getPointer( uint pAddress )
{
	return _data + pAddress;
}

/******************************************************************************
 * Set the value at a given 1D address in the data array.
 *
 * @param pAddress a 1D address
 * @param pVal a value
 ******************************************************************************/
template< typename T >
__device__
__forceinline__ void Array3DKernelLinear< T >::set( const uint pAddress, T pVal )
{
	_data[ pAddress ] = pVal;
}

/******************************************************************************
 * Set the value at a given 2D position in the data array.
 *
 * @param pPosition a 2D position
 * @param pVal a value
 ******************************************************************************/
template< typename T >
__device__
__forceinline__ void Array3DKernelLinear< T >::set( const uint2& pPosition, T pVal )
{
	_data[ getOffset( pPosition ) ] = pVal;
}

/******************************************************************************
 * Set the value at a given 3D position in the data array.
 *
 * @param pPosition a 3D position
 * @param pVal a value
 ******************************************************************************/
template< typename T >
__device__
__forceinline__ void Array3DKernelLinear< T >::set( const uint3& pPosition, T pVal )
{
	_data[ getOffset( pPosition ) ] = pVal;
}

/******************************************************************************
 * Helper function used to get the corresponding index array at a given
 * 3D position in a safe way.
 * Position is checked and modified if needed (as a clamp).
 *
 * @param pPosition a 3D position
 *
 * @return the corresponding index array at the given 3D position
 ******************************************************************************/
template< typename T >
__device__
__forceinline__ uint3 Array3DKernelLinear< T >::getSecureIndex( uint3 pPosition ) const
{
	if ( pPosition.x >= _resolution.x )
	{
		pPosition.x = _resolution.x - 1;
	}

	if ( pPosition.y >= _resolution.y )
	{
		pPosition.y = _resolution.y - 1;
	}

	if ( pPosition.z >= _resolution.z )
	{
		pPosition.z = _resolution.z - 1;
	}

	return pPosition;
}

/******************************************************************************
 * Helper function used to get the offset in the 1D linear data array
 * given a 3D position.
 *
 * @param pPosition a 3D position
 *
 * @return the corresponding offset in the 1D linear data array
 ******************************************************************************/
template< typename T >
__device__
__forceinline__ uint Array3DKernelLinear< T >::getOffset( const uint3& pPosition ) const
{
	//return position.x + position.y * _resolution.x + position.z * _pitchxy;
	return pPosition.x + __uimul( pPosition.y, _resolution.x ) + __uimul( pPosition.z, _pitchxy );
}

/******************************************************************************
 * Helper function used to get the offset in the 1D linear data array
 * given a 2D position.
 *
 * @param pPosition a 2D position
 *
 * @return the corresponding offset in the 1D linear data array
 ******************************************************************************/
template< typename T >
__device__
__forceinline__ uint Array3DKernelLinear< T >::getOffset( const uint2& pPosition ) const
{
	//return pPosition.x + pPosition.y * _resolution.x ;
	return pPosition.x + __uimul( pPosition.y, _resolution.x ) ;
}

} // namespace GvCore
