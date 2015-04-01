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

#ifndef VOLUMEPRODUCER_H
#define VOLUMEPRODUCER_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/Array3D.h>

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

/** 
 * @class VolumeProducer
 *
 * @brief The VolumeProducer class provides...
 *
 * Interface of all Volume Producers.
 */
template< typename T >
class VolumeProducer
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Enumeration of the different types of regions
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
	 * Provides a region in space at the resolution of outVol array. Returns constantness information for that region.
	 *
	 * @param pPos ...
	 * @param pSize ...
	 * @param pOutVol ...
	 * @param pOffsetInPool ...
	 *
	 * @return ...
	 */
	virtual VPRegionInfo getRegion( const float3& pPos, const float3& pSize, GvCore::GPUPoolHost< GvCore::Array3D, T >* pOutVol, size_t pOffsetInPool )
	{
		return VP_UNKNOWN_REGION;
	}

	/**
	 * Provides constantness information about a region. Resolution is here for compatibility.
	 * TODO: Remove resolution.
	 *
	 * @param pPosf ...
	 * @param pSizef ...
	 *
	 * @return ...
	 */
	virtual VPRegionInfo getRegionInfo( const float3& pPosf, const float3& pSizef/*, T* constValueOut = NULL*/ )
	{
		return VP_UNKNOWN_REGION;
	}

	/**
	 * ...
	 *
	 * @param pPosf ...
	 * @param pSizef ...
	 *
	 * @return ...
	 */
	virtual uint getRegionInfoNew( const float3& pPosf, const float3& pSizef )
	{
		return 0;
	}

	/**
	 * Provides the size of the smallest features the producer can generate.
	 *
	 * @return ...
	 */
	virtual float3 getFeaturesSize() const
	{
		return make_float3( 0.0f );
	}

	/**
	 * Destructor
	 */
	virtual ~VolumeProducer()
	{
	}

	/**
	 * Initialize used region resolution. Automatically called when the producer is connected to gigavoxels.
	 * Return 0 if succeed.
	 *
	 * @param pRes ...
	 *
	 * @return ...
	 */
	virtual int setRegionResolution( const uint3& pRes )
	{
		return 0;
	}

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

#endif
