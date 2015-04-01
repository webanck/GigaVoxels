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
 * Unpack a node address
 *
 * @param pAddress ...
 *
 * @return ...
 ******************************************************************************/
__host__ __device__
inline uint3 GvNode::unpackNodeAddress( const uint pAddress )
{
	uint3 res;

	res.x = pAddress & 0x3FFFFFFF;
	res.y = 0;
	res.z = 0;

	return res;
}

/******************************************************************************
 * Pack a node address
 *
 * @param pAddress ...
 *
 * @return ...
 ******************************************************************************/
__host__ __device__
inline uint GvNode::packNodeAddress( const uint3 pAddress )
{
	return pAddress.x;
}

/******************************************************************************
 * Unpack a brick address
 *
 * @param pAddress ...
 *
 * @return ...
 ******************************************************************************/
__host__ __device__
inline uint3 GvNode::unpackBrickAddress( const uint pAddress )
{
	uint3 res;

	res.x = ( pAddress & 0x3FF00000 ) >> 20;
	res.y = ( pAddress & 0x000FFC00 ) >> 10;
	res.z = ( pAddress & 0x000003FF );

	return res;
}

/******************************************************************************
 * Pack a brick address
 *
 * @param pAddress ...
 *
 * @return ...
 ******************************************************************************/
__host__ __device__
inline uint GvNode::packBrickAddress( const uint3 pAddress )
{
	return ( pAddress.x << 20 | pAddress.y << 10 | pAddress.z );
}

/******************************************************************************
 * Set the child nodes address
 *
 * @param dpcoord ...
 ******************************************************************************/
__host__ __device__
inline void GvNode::setChildAddress( const uint3 dpcoord )
{
	setChildAddressEncoded( packNodeAddress( dpcoord ) );
}

/******************************************************************************
 * Get the child nodes address
 *
 * @return ...
 ******************************************************************************/
__host__ __device__
inline uint3 GvNode::getChildAddress() const
{
	return unpackNodeAddress( childAddress );
}

/******************************************************************************
 * Set the child nodes encoded address
 *
 * @param addr ...
 ******************************************************************************/
__host__ __device__
inline void GvNode::setChildAddressEncoded( uint addr )
{
	childAddress = ( childAddress & 0x40000000 ) | ( addr & 0x3FFFFFFF );
}

/******************************************************************************
 * Get the child nodes encoded address
 *
 * @return ...
 ******************************************************************************/
__host__ __device__
inline uint GvNode::getChildAddressEncoded() const
{
	return childAddress;
}

/******************************************************************************
 * Tell wheter or not the node has children
 *
 * @return a flag telling wheter or not the node has children
 ******************************************************************************/
__host__ __device__
inline bool GvNode::hasSubNodes() const
{
	return ( ( childAddress & 0x3FFFFFFF ) != 0 );
}

/******************************************************************************
 * Flag the node as beeing terminal or not
 *
 * @param pFlag a flag telling wheter or not the node is terminal
 ******************************************************************************/
__host__ __device__
inline void GvNode::setTerminal( bool pFlag )
{
	if ( pFlag )
	{
		childAddress = ( childAddress | 0x80000000 );
	}
	else
	{
		childAddress = ( childAddress & 0x7FFFFFFF );
	}
}

/******************************************************************************
 * Tell wheter or not the node is terminal
 *
 * @return a flag telling wheter or not the node is terminal
 ******************************************************************************/
__host__ __device__
inline bool GvNode::isTerminal() const
{
	return ( ( childAddress & 0x80000000 ) != 0 );
}

/******************************************************************************
 * Set the brick address
 *
 * @param dpcoord ...
 ******************************************************************************/
__host__ __device__
inline void GvNode::setBrickAddress( const uint3 dpcoord )
{
	setBrickAddressEncoded( packBrickAddress( dpcoord ) );
}

/******************************************************************************
 * Get the brick address
 *
 * @return ...
 ******************************************************************************/
__host__ __device__
inline uint3 GvNode::getBrickAddress() const
{
	return unpackBrickAddress( brickAddress );
}

/******************************************************************************
 * Set the brick encoded address
 *
 * @param addr ...
 ******************************************************************************/
__host__ __device__
inline void GvNode::setBrickAddressEncoded( const uint addr )
{
	brickAddress = addr;
	setStoreBrick();
}

/******************************************************************************
 * Get the brick encoded address
 *
 * @return ...
 ******************************************************************************/
__host__ __device__
inline uint GvNode::getBrickAddressEncoded() const
{
	return brickAddress;
}

/******************************************************************************
 * Flag the node as containg data or not
 *
 * @param pFlag a flag telling wheter or not the node contains data
 ******************************************************************************/
__host__ __device__
inline void GvNode::setStoreBrick()
{
	childAddress = childAddress | 0x40000000;
}

/******************************************************************************
 * Tell wheter or not the node is a brick
 *
 * @return a flag telling wheter or not the node is a brick
 ******************************************************************************/
__host__ __device__
inline bool GvNode::isBrick() const
{
	return ( ( childAddress & 0x40000000 ) != 0 );
}

/******************************************************************************
 * Tell wheter or not the node has a brick,
 * .i.e the node is a brick and its brick address is not null.
 *
 * @return a flag telling wheter or not the node has a brick
 ******************************************************************************/
__host__ __device__
inline bool GvNode::hasBrick() const
{
	return ( brickAddress != 0 ) && ( ( childAddress & 0x40000000 ) != 0 );
}

/******************************************************************************
 * Tell wheter or not the node is initializated
 *
 * @return a flag telling wheter or not the node is initializated
 ******************************************************************************/
__host__ __device__
inline bool GvNode::isInitializated() const
{
	return ( ( childAddress != 0 ) || ( brickAddress != 0 ) );
}

} // namespace GvStructure
