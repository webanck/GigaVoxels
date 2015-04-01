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
#include "GvCore/GPUPool.h"
#include "GvUtils/GvDataLoader.h"

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvUtils
{

/******************************************************************************
 * Constructor
 *
 * @param pCaller Reference on a brick producer
 * @param pIndexValue Index value
 * @param pBlockMemSize Block memory size
 * @param pLevel level of resolution
 * @param pDataPool reference on a data pool
 * @param pOffsetInPool offset in the referenced data pool
 ******************************************************************************/
template< typename TDataTypeList >
inline GvBrickLoaderChannelInitializer< TDataTypeList >
::GvBrickLoaderChannelInitializer( GvDataLoader< TDataTypeList >* pCaller,
									unsigned int pIndexValue, unsigned int pBlockMemSize, unsigned int pLevel,
									GvCore::GPUPoolHost< GvCore::Array3D, TDataTypeList >* pDataPool, size_t pOffsetInPool )
:	_caller( pCaller )
,	_indexValue( pIndexValue )
,	_blockMemorySize( pBlockMemSize )
,	_level( pLevel )
,	_dataPool( pDataPool )
,	_offsetInPool( pOffsetInPool )
{
}

/******************************************************************************
 * Concrete method used to read a brick
 *
 * @param Loki::Int2Type< channel > index of the channel (i.e. color, normal, etc...)
 ******************************************************************************/
template< typename TDataTypeList >
template< int TChannelIndex >
inline void GvBrickLoaderChannelInitializer< TDataTypeList >
::run( Loki::Int2Type< TChannelIndex > )
{
	// Type definition of the channel's data type at given channel index.
	typedef typename Loki::TL::TypeAt< TDataTypeList, TChannelIndex >::Result ChannelType;

	// Retrieve the data array associated to the data pool at given channel index.
	GvCore::Array3D< ChannelType >* dataArray = _dataPool->template getChannel< TChannelIndex >();

	// Ask the referenced brick producer to read a brick
	_caller->template readBrick< ChannelType >( TChannelIndex, _indexValue, _blockMemorySize, _level, dataArray, _offsetInPool );
}

} // namespace GvUtils
