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

namespace GvCache
{

/******************************************************************************
 * Update timestamp usage information of an element (node tile or brick)
 * with current time (i.e. current rendering pass)
 * given its address in its corresponding pool (node or brick).
 *
 * @param pElemAddress The address of the element for which we want to update usage information
 ******************************************************************************/
template< class ElementRes, class AddressType >
__device__
__forceinline__ void GvCacheManagerKernel< ElementRes, AddressType >::setElementUsage( uint pElemAddress )
{
	uint elemOffset;
	if ( ElementRes::xIsPOT )
	{
		elemOffset = pElemAddress >> ElementRes::xLog2;
	}
	else
	{
		elemOffset = pElemAddress / ElementRes::x;
	}

	// Update time stamp array with current time (i.e. the time of the current rendering pass)
	_timeStampArray.set(elemOffset, k_currentTime);
}

/******************************************************************************
 * Update timestamp usage information of an element (node tile or brick)
 * with current time (i.e. current rendering pass)
 * given its address in its corresponding pool (node or brick).
 *
 * @param pElemAddress The address of the element for which we want to update usage information
 ******************************************************************************/
template< class ElementRes, class AddressType >
__device__
__forceinline__ void GvCacheManagerKernel< ElementRes, AddressType >::setElementUsage( uint3 pElemAddress )
{
	uint3 elemOffset;
	if ( ElementRes::xIsPOT && ElementRes::yIsPOT && ElementRes::zIsPOT )
	{
		elemOffset.x = pElemAddress.x >> ElementRes::xLog2;
		elemOffset.y = pElemAddress.y >> ElementRes::yLog2;
		elemOffset.z = pElemAddress.z >> ElementRes::zLog2;
	}
	else
	{
		elemOffset = pElemAddress / ElementRes::get();
	}

	// Update time stamp array with current time (i.e. the time of the current rendering pass)
	_timeStampArray.set(elemOffset, k_currentTime);
}

} // namespace GvCache
