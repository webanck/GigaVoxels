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

/******************************************************************************
 * Constructor
 *
 * @param pDerived The user class used to implement the interface of a provider
 * kernel object.
 ******************************************************************************/
template< uint TId, typename TDerived >
inline GvIProviderKernel< TId, TDerived >::GvIProviderKernel( TDerived& pDerived )
:	mDerived( pDerived )
{
}

/******************************************************************************
 * Produce data on device.
 *
 * Producing data mecanism works element by element (node tile or brick) depending on the channel.
 *
 * In the function, user has to produce data for a node tile or a brick of voxels.
 * For a node tile, user has to defined regions (i.e nodes) where lies data, constant values,
 * etc...
 * For a brick, user has to produce data (i.e voxels) at for each of the channels
 * user had previously defined (color, normal, density, etc...)
 *
 * @param pGpuPool The device side pool (nodes or bricks)
 * @param pRequestID The current processed element coming from the data requests list (a node tile or a brick)
 * @param pProcessID Index of one of the elements inside a node tile or a voxel bricks
 * @param pNewElemAddress The address at which to write the produced data in the pool
 * @param pParentLocInfo The localization info used to locate an element in the pool
 *
 * @return A feedback value that the user can return.
 * @to do Verify the action/need of the return value (see the Page Table Kernel).
 ******************************************************************************/
template< uint TId, typename TDerived >
template< typename TGPUPoolKernelType >
__device__
__forceinline__ uint GvIProviderKernel< TId, TDerived >
::produceData( TGPUPoolKernelType& pGpuPool, uint pRequestID, uint pProcessID,
				uint3 pNewElemAddress, const GvLocalizationInfo& pParentLocInfo )
{
	return mDerived.produceData( pGpuPool, pRequestID, pProcessID,
								pNewElemAddress, pParentLocInfo, Loki::Int2Type< TId >() );
}

} // namespace GvCore
