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

/******************************************************************************
 * Get the light intensity
 *
 * @return the light intensity
 ******************************************************************************/
inline const float3& Light::getIntensity() const
{
	return _intensity;
}

/******************************************************************************
 * Set the light intensity
 *
 * @param pValue the light intensity
 ******************************************************************************/
inline void Light::setIntensity( const float3& pValue )
{
	_intensity = pValue;
}

/******************************************************************************
 * Get the light direction
 *
 * @return the light direction
 ******************************************************************************/
inline const float3& Light::getDirection() const
{
	return _direction;
}

/******************************************************************************
 * Set the light direction
 *
 * @param pValue the light direction
 ******************************************************************************/
inline void Light::setDirection( const float3& pValue )
{
	_direction = pValue;
}

/******************************************************************************
 * Get the light position
 *
 * @return the light position
 ******************************************************************************/
inline const float4& Light::getPosition() const
{
	return _position;
}

/******************************************************************************
 * Set the light position
 *
 * @param pValue the light position
 ******************************************************************************/
inline void Light::setPosition( const float4& pValue )
{
	_position = pValue;
}
