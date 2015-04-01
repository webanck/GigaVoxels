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

#ifndef _GV_BRICK_LOADER_CHANNEL_INITIALIZER_H_
#define _GV_BRICK_LOADER_CHANNEL_INITIALIZER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Gigavoxels
#include "GvCore/Array3D.h"

// Loki
#include <loki/Typelist.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GigaVoxels
namespace GvCore
{
	template
	<
		template< typename > class THostArray, class TList
	>
	class GPUPoolHost;
}

namespace GvUtils
{
	template< typename TDataTypeList >
	class GvDataLoader;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvUtils
{

/** 
 * @struct GvBrickLoaderChannelInitializer
 *
 * @brief The GvBrickLoaderChannelInitializer struct provides a generalized functor
 * to read a brick of voxels.
 *
 * GvBrickLoaderChannelInitializer is used with GvDataLoader to read a brick from HOST.
 */
template< typename TDataTypeList >
struct GvBrickLoaderChannelInitializer
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Reference on a brick producer
	 */
	GvDataLoader< TDataTypeList >* _caller;

	/**
	 * Index value
	 */
	unsigned int _indexValue;

	/**
	 * Block memory size
	 */
	unsigned int _blockMemorySize;

	/**
	 * Level of resolution
	 */
	unsigned int _level;

	/**
	 * Reference on a data pool
	 */
	GvCore::GPUPoolHost< GvCore::Array3D, TDataTypeList >* _dataPool;

	/**
	 * Offset in the referenced data pool
	 */
	size_t _offsetInPool;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param pCaller Reference on a brick producer
	 * @param pIndexValue Index value
	 * @param pBlockMemSize Block memory size
	 * @param pLevel level of resolution
	 * @param pDataPool reference on a data pool
	 * @param pOffsetInPool offset in the referenced data pool
	 */
	inline GvBrickLoaderChannelInitializer( GvDataLoader< TDataTypeList >* pCaller,
											unsigned int pIndexValue, unsigned int pBlockMemSize, unsigned int pLevel,
											GvCore::GPUPoolHost< GvCore::Array3D, TDataTypeList >* pDataPool, size_t pOffsetInPool );

	/**
	 * Concrete method used to read a brick
	 *
	 * @param Loki::Int2Type< TChannelIndex > index of the channel (i.e. color, normal, etc...)
	 */
	template< int TChannelIndex >
	inline void run( Loki::Int2Type< TChannelIndex > );

	/**************************************************************************
	**************************** PROTECTED SECTION ***************************
	**************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	***************************** PRIVATE SECTION ****************************
	**************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

};

} // namespace GvUtils

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvBrickLoaderChannelInitializer.inl"

#endif
