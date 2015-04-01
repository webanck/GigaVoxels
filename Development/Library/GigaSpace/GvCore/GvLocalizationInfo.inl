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

// Localization Code

/******************************************************************************
 * Set the localization code value
 *
 * @param plocalizationCode The localization code value
 ******************************************************************************/
__host__ __device__
inline void GvLocalizationCode::set( uint3 plocalizationCode )
{
	_localizationCode = plocalizationCode;
}

/******************************************************************************
 * Get the localization code value
 *
 * @return The localization code value
 ******************************************************************************/
__host__ __device__
inline uint3 GvLocalizationCode::get() const
{
	return _localizationCode;
}

/******************************************************************************
 * Given the current localization code and an offset in a node tile
 *
 * @param pOffset The offset in a node tile
 *
 * @return ...
 ******************************************************************************/
template< typename TNodeTileResolution >
__host__ __device__
inline GvLocalizationCode GvLocalizationCode::addLevel( uint3 pOffset ) const
{
	uint3 localizationCode;
	localizationCode.x = _localizationCode.x << TNodeTileResolution::xLog2 | pOffset.x;
	localizationCode.y = _localizationCode.y << TNodeTileResolution::yLog2 | pOffset.y;
	localizationCode.z = _localizationCode.z << TNodeTileResolution::zLog2 | pOffset.z;

	GvLocalizationCode result;
	result.set( localizationCode );

	return result;
}

/******************************************************************************
 * ...
 *
 * @return ...
 ******************************************************************************/
template< typename TNodeTileResolution >
__host__ __device__
inline GvLocalizationCode GvLocalizationCode::removeLevel() const
{
	uint3 localizationCode;
	localizationCode.x = _localizationCode.x >> TNodeTileResolution::xLog2;
	localizationCode.y = _localizationCode.y >> TNodeTileResolution::yLog2;
	localizationCode.z = _localizationCode.z >> TNodeTileResolution::zLog2;

	GvLocalizationCode result;
	result.set( localizationCode );

	return result;
}

} // namespace GvCore

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvCore
{

// Localization Depth

/******************************************************************************
 * Get the localization depth value
 *
 * @return The localization depth value
 ******************************************************************************/
__host__ __device__
inline uint GvLocalizationDepth::get() const
{
	return _localizationDepth;
}

/******************************************************************************
 * Set the localization depth value
 *
 * @param pLocalizationDepth The localization depth value
 ******************************************************************************/
__host__ __device__
inline void GvLocalizationDepth::set( uint pLocalizationDepth )
{
	_localizationDepth = pLocalizationDepth;
}

/******************************************************************************
 * ...
 *
 * @return ...
 ******************************************************************************/
__host__ __device__
inline GvLocalizationDepth GvLocalizationDepth::addLevel() const
{
	GvLocalizationDepth result;
	result.set( _localizationDepth + 1 );

	return result;
}

/******************************************************************************
 * ...
 *
 * @return ...
 ******************************************************************************/
__host__ __device__
inline GvLocalizationDepth GvLocalizationDepth::removeLevel() const
{
	GvLocalizationDepth result;
	result.set( _localizationDepth - 1 );

	return result;
}

} // namespace GvCore
