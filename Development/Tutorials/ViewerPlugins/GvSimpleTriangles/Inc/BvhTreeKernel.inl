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

/******************************************************************************
 * Get data from data pool given a channel and an address
 *
 * @param ... ...
 *
 * @return ...
 ******************************************************************************/
template< class TDataTypeList >
template< int TChannelIndex >
__device__
inline typename GvCore::DataChannelType< TDataTypeList, TChannelIndex >::Result BvhTreeKernel< TDataTypeList >
::getVertexData( uint pAddress )
{
	return _dataPool.getChannel( Loki::Int2Type< TChannelIndex >() ).get( pAddress );
}

/******************************************************************************
 * ...
 *
 * @param ... ...
 * @param ... ...
 ******************************************************************************/
template< class TDataTypeList >
__device__
inline void BvhTreeKernel< TDataTypeList >
::fetchBVHNode( VolTreeBVHNodeUser& resnode, uint pAddress )
{
	VolTreeBVHNode tempNodeUnion;

	/*for ( uint i = 0; i < VolTreeBVHNodeStorageUINT::numWords; i++ )
	{
		tempNodeUnion.storageUINTNode.words[ i ] = tex1Dfetch( volumeTreeBVHTexLinear, pAddress * VolTreeBVHNodeStorageUINT::numWords + i );
	}*/
	tempNodeUnion.storageUINTNode = _volumeTreeBVHArray.get( pAddress );

	resnode	= tempNodeUnion.userNode;
}

/******************************************************************************
 * ...
 *
 * @param ... ...
 * @param ... ...
 * @param ... ...
 ******************************************************************************/
template< class TDataTypeList >
__device__
inline void BvhTreeKernel< TDataTypeList >
::parallelFetchBVHNode( uint Pid, VolTreeBVHNodeUser& resnode, uint pAddress )
{
	// Shared Memory declaration
	__shared__ VolTreeBVHNode tempNodeUnion;

	if ( Pid < VolTreeBVHNodeStorageUINT::numWords )
	{
		//
		//tempNodeUnion.storageUINTNode.words[Pid] =k_volumeTreeBVHArray.get(pAddress).words[Pid];
#if 1
		uint* arrayAddress = (uint*)_volumeTreeBVHArray.getPointer( 0 );
		tempNodeUnion.storageUINTNode.words[ Pid ] = arrayAddress[ pAddress * VolTreeBVHNodeStorageUINT::numWords + Pid ];
#else
		tempNodeUnion.storageUINTNode.words[Pid] =tex1Dfetch(volumeTreeBVHTexLinear, pAddress*VolTreeBVHNodeStorageUINT::numWords+Pid);
#endif
	}

	// Thread synchronization
	__syncthreads();

	///resnode	=tempNodeUnion.userNode;

	if ( Pid == 0 )
	{
		resnode	= tempNodeUnion.userNode;
	}

	// Thread synchronization
	__syncthreads();
}

/******************************************************************************
 * ...
 *
 * @param ... ...
 * @param ... ...
 * @param ... ...
 ******************************************************************************/
template< class TDataTypeList >
__device__
inline void BvhTreeKernel< TDataTypeList >
::parallelFetchBVHNodeTile( uint Pid, VolTreeBVHNodeUser* resnodetile, uint pAddress )
{
	// Shared memory
	__shared__ VolTreeBVHNode tempNodeUnion;

	if ( Pid < VolTreeBVHNodeStorageUINT::numWords * 2 )
	{
		uint* arrayAddress = (uint*)_volumeTreeBVHArray.getPointer( 0 );
		uint* resnodetileUI = (uint*)resnodetile;

		resnodetileUI[ Pid ] =  arrayAddress[ pAddress * VolTreeBVHNodeStorageUINT::numWords + Pid ];
	}

	// Thread synchronization
	__syncthreads();
}

/******************************************************************************
 * ...
 *
 * @param ... ...
 * @param ... ...
 ******************************************************************************/
template< class TDataTypeList >
__device__
inline void BvhTreeKernel< TDataTypeList >
::writeBVHNode( const VolTreeBVHNodeUser& node, uint pAddress )
{
	VolTreeBVHNode tempNodeUnion;
	tempNodeUnion.userNode = node;

	// Update 
	_volumeTreeBVHArray.set( pAddress, tempNodeUnion.storageUINTNode );
}

/******************************************************************************
 * ...
 *
 * @param ... ...
 * @param ... ...
 * @param ... ...
 ******************************************************************************/
template< class TDataTypeList >
__device__
inline void BvhTreeKernel< TDataTypeList >
::parallelWriteBVHNode( uint Pid, const VolTreeBVHNodeStorage& node, uint pAddress )
{
	//Warning, no checking on Pid

	VolTreeBVHNodeStorage* storageArrayPtr = (VolTreeBVHNodeStorage*)_volumeTreeBVHArray.getPointer();

	storageArrayPtr[ pAddress ].words[ Pid ] = node.words[ Pid ];
}
