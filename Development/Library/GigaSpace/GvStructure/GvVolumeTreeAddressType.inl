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

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvStructure
{

/******************************************************************************
 * ...
 ******************************************************************************/
__host__ __device__
inline VolTreeNodeAddress::PackedAddressType VolTreeNodeAddress::packAddress( const uint3& address )
{
	return address.x;
}

/******************************************************************************
 * ...
 ******************************************************************************/
__host__ __device__
inline VolTreeNodeAddress::PackedAddressType VolTreeNodeAddress::packAddress( uint address )
{
	return address;
}

/******************************************************************************
 * ...
 ******************************************************************************/
__host__ __device__
inline uint3 VolTreeNodeAddress::unpackAddress( VolTreeNodeAddress::PackedAddressType address )
{
	return make_uint3( address & 0x3FFFFFFF, 0, 0 );
}

/******************************************************************************
 * ...
 ******************************************************************************/
__host__ __device__
inline bool VolTreeNodeAddress::isNull( uint pa )
{
	return (pa & packedMask) == 0;
}

} // namespace GvStructure


/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvStructure
{

/******************************************************************************
 * ...
 ******************************************************************************/
__host__ __device__
inline VolTreeBrickAddress::PackedAddressType VolTreeBrickAddress::packAddress( const uint3& address )
{
	return (address.x << 20 | address.y << 10 | address.z);
}

/******************************************************************************
 * ...
 ******************************************************************************/
__host__ __device__
inline uint3 VolTreeBrickAddress::unpackAddress( VolTreeBrickAddress::PackedAddressType address )
{
	uint3 res;

	res.x = (address & 0x3FF00000) >> 20;
	res.y = (address & 0x000FFC00) >> 10;
	res.z = (address & 0x000003FF);

	return res;
}

/******************************************************************************
 * ...
 ******************************************************************************/
__host__ __device__
inline bool VolTreeBrickAddress::isNull( uint pa )
{
	return (pa & packedMask) == 0;
}

} // namespace GvStructure
