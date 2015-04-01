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
 * Return the resolution as a uint3.
 *
 * @return the resolution
 ******************************************************************************/
template< uint Trx, uint Try, uint Trz >
__host__ __device__
inline uint3 StaticRes3D< Trx, Try, Trz >::get()
{
	return make_uint3( x, y, z );
}

/******************************************************************************
 * Return the resolution as a float3.
 *
 * @return the resolution
 ******************************************************************************/
template< uint Trx, uint Try, uint Trz >
__host__ __device__
inline float3 StaticRes3D< Trx, Try, Trz >::getFloat3()
{
	return make_float3( static_cast< float >( x ), static_cast< float >( y ), static_cast< float >( z ) );
}

/******************************************************************************
 * Return the number of elements
 *
 * @return the number of elements
 ******************************************************************************/
template< uint Trx, uint Try, uint Trz >
__host__ __device__
inline uint StaticRes3D< Trx, Try, Trz >::getNumElements()
{
	return x * y * z;
}

/******************************************************************************
 *
 ******************************************************************************/
//template< uint Trx, uint Try, uint Trz >
//__host__ __device__
//inline uint StaticRes3D< Trx, Try, Trz >::getNumElementsLog2()
//{
//	return Log2< x * y * z >::value;
//}

/******************************************************************************
 * Return the log2(resolution) as an uint3.
 *
 * @return the log2(resolution)
 ******************************************************************************/
template< uint Trx, uint Try, uint Trz >
__host__ __device__
inline uint3 StaticRes3D< Trx, Try, Trz >::getLog2()
{
	return make_uint3( Log2< x >::value, Log2< y >::value, Log2< z >::value );
}

/******************************************************************************
 * Convert a three-dimensionnal value to a linear value.
 *
 * @param pValue The 3D value to convert
 *
 * @return the 1D linearized converted value
 ******************************************************************************/
template< uint Trx, uint Try, uint Trz >
__host__ __device__
inline uint StaticRes3D< Trx, Try, Trz >::toFloat1( uint3 pValue )
{
	if ( xIsPOT && yIsPOT && zIsPOT )
	{
		return pValue.x | ( pValue.y << xLog2 ) | ( pValue.z << ( xLog2 + yLog2 ) );
	}
	else
	{
		return pValue.x + pValue.y * x + pValue.z * x * y;
	}
}

/******************************************************************************
 * Convert a linear value to a three-dimensionnal value.
 *
 * @param pValue The 1D value to convert
 *
 * @return the 3D converted value
 ******************************************************************************/
template< uint Trx, uint Try, uint Trz >
__host__ __device__
inline uint3 StaticRes3D< Trx, Try, Trz >::toFloat3( uint pValue )
{
	if ( xIsPOT && yIsPOT && zIsPOT )
	{
		/*r.x = n & xLog2;
		r.y = (n >> xLog2) & yLog2;
		r.z = (n >> (xLog2 + yLog2)) & zLog2;*/
		return make_uint3( pValue & xLog2, ( pValue >> xLog2 ) & yLog2, ( pValue >> ( xLog2 + yLog2 ) ) & zLog2 );
	}
	else
	{
		/*r.x = n % x;
		r.y = (n / x) % y;
		r.z = (n / (x * y)) % z;*/
		return make_uint3( pValue % x, ( pValue / x ) % y, ( pValue / ( x * y ) ) % z );
	}
}

} // namespace GvCore
