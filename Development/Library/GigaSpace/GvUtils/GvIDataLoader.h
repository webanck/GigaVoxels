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

#ifndef _GV_I_DATA_LOADER_H_
#define _GV_I_DATA_LOADER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/Array3D.h"

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

namespace GvUtils
{

/** 
 * @class GvIDataLoader
 *
 * @brief The GvIDataLoader class provides...
 *
 * Interface of all Volume Producers.
 */
template< typename TDataTypeList >
class GvIDataLoader
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Enumeration of different types of a region.
	 */
	enum VPRegionInfo
	{
		VP_CONST_REGION,
		VP_NON_CONST_REGION,
		VP_UNKNOWN_REGION
	};

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Destructor
	 */
	virtual ~GvIDataLoader();

	/**
	 * Helper function used to determine the type of regions in the data structure.
	 * The data structure is made of regions containing data, empty or constant regions.
	 *
	 * Retrieve the node and associated brick located in this region of space,
	 * and depending of its type, if it contains data, load it.
	 *
	 * @param pPosition position of a region of space
	 * @param pSize size of a region of space
	 * @param pBrickPool data cache pool. This is where all data reside for each channel (color, normal, etc...)
	 * @param pOffsetInPool offset in the brick pool
	 *
	 * @return the type of the region (.i.e returns constantness information for that region)
	 */
	inline virtual VPRegionInfo getRegion( const float3& pPosition, const float3& pSize, GvCore::GPUPoolHost< GvCore::Array3D, TDataTypeList >* pBrickPool, size_t pOffsetInPool );

	/**
	 * Provides constantness information about a region. Resolution is here for compatibility. TODO:Remove resolution.
	 *
	 * @param pPosition position of a region of space
	 * @param pSize size of a region of space
	 *
	 * @return the type of the region (.i.e returns constantness information for that region)
	 */
	inline virtual VPRegionInfo getRegionInfo( const float3& pPosition, const float3& pSize/*, T *constValueOut = NULL*/ );

	/**
	 * Retrieve the node located in a region of space,
	 * and get its information (i.e. address containing its data type region).
	 *
	 * @param pPosition position of a region of space
	 * @param pSize size of a region of space
	 *
	 * @return the node encoded information
	 */
	inline virtual uint getRegionInfoNew( const float3& pPosition, const float3& pSize );

	/**
	 * Provides the size of the smallest features the producer can generate.
	 *
	 * @return the size of the smallest features the producer can generate.
	 */
	inline virtual float3 getFeaturesSize() const;

	/**
	 * Set the region resolution.
	 *
	 * @param pResolution resolution
	 *
	 * @return ...
	 */
	inline virtual int setRegionResolution( const uint3& pResolution );

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

	/******************************** METHODS *********************************/

};

} // namespace GvUtils

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvIDataLoader.inl"

#endif
