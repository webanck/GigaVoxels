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

namespace VolumeData
{

/******************************************************************************
 * Get the size of the data.
 * Resolution of data (4 components 3D float data) [size is identic on each dimension]
 *
 * @return the data size
 ******************************************************************************/
inline int vdCube3D4::getSize() const
{
	return _size;
};

/******************************************************************************
 * Get the buffer of data
 *
 * @return the pointer on the data buffer
 ******************************************************************************/
inline float* vdCube3D4::getData()
{
	return _data;
};

/******************************************************************************
 * Get the data at given 3D indexed position and component in the buffer of data
 *
 * @param pX x component of 3D indexed position of data
 * @param pY y component of 3D indexed position of data
 * @param pZ z component of 3D indexed position of data
 * @param pComponent component of the data (data has 4 components)
 *
 * @return a reference on the data
 ******************************************************************************/
inline float& vdCube3D4::get( int pX, int pY, int pZ, int pComponent )
{
	return _data[ pComponent + ( pX + ( pY + pZ * _size ) * _size ) * 4 ];
};

}
