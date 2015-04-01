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

namespace GvUtils
{

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TDataTypeList >
GvIDataLoader< TDataTypeList >
::~GvIDataLoader()
{
}

/******************************************************************************
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
 ******************************************************************************/
template< typename TDataTypeList >
inline GvIDataLoader< TDataTypeList >::VPRegionInfo GvIDataLoader< TDataTypeList >
::getRegion( const float3& pPosition, const float3& pSize, GvCore::GPUPoolHost< GvCore::Array3D, TDataTypeList >* pBrickPool, size_t pOffsetInPool )
{
	return VP_UNKNOWN_REGION;
}

/******************************************************************************
 * Provides constantness information about a region. Resolution is here for compatibility. TODO:Remove resolution.
 *
 * @param pPosition position of a region of space
 * @param pSize size of a region of space
 *
 * @return the type of the region (.i.e returns constantness information for that region)
 ******************************************************************************/
template< typename TDataTypeList >
inline GvIDataLoader< TDataTypeList >::VPRegionInfo GvIDataLoader< TDataTypeList >
::getRegionInfo( const float3& pPosition, const float3& pSize/*, T *constValueOut = NULL*/ )
{
	return VP_UNKNOWN_REGION;
}

/******************************************************************************
 * Retrieve the node located in a region of space,
 * and get its information (i.e. address containing its data type region).
 *
 * @param pPosition position of a region of space
 * @param pSize size of a region of space
 *
 * @return the node encoded information
 ******************************************************************************/
template< typename TDataTypeList >
inline uint GvIDataLoader< TDataTypeList >
::getRegionInfoNew( const float3& pPosition, const float3& pSize )
{
	return 0;
}

/******************************************************************************
 * Provides the size of the smallest features the producer can generate.
 *
 * @return the size of the smallest features the producer can generate.
 ******************************************************************************/
template< typename TDataTypeList >
inline float3 GvIDataLoader< TDataTypeList >
::getFeaturesSize() const
{
	return make_float3( 0.f );
}

/******************************************************************************
 * Set the region resolution.
 *
 * @param pResolution resolution
 *
 * @return ...
 ******************************************************************************/
template< typename TDataTypeList >
inline int GvIDataLoader< TDataTypeList >
::setRegionResolution( const uint3& pResolution )
{
	return 0;
}

} // namespace GvUtils
