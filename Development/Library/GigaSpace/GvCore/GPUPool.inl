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
#include "GvCore/GvError.h"

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvCore
{

/////////////// GPU Pool Kernel ////////////////

/******************************************************************************
 * Set the value at a given position in the pool.
 *
 * @param pos position the pool
 * @param val value
 ******************************************************************************/
template< template< typename > class KernelArray, class TList >
template< uint i, typename ST >
__device__
__forceinline__ void GPUPoolKernel< KernelArray, TList >
::setValue( const uint3& pos, ST val )
{
	typename Loki::TL::TypeAt< TList, i >::Result res;
	convert_type( val, res );

	// Retrieve the channel at index i
	typename GPUPool_TypeAtChannel< KernelArray, TList, GPUPoolChannelUnitValue, i >::Result& channel = getChannel( Loki::Int2Type< i >() );

	// Write data in the channel (i.e. its associated surface)
	channel.set< i >( pos, res );
}

/******************************************************************************
 * Get the value at a given position in the pool.
 *
 * @param pos position the pool
 * @param val value
 ******************************************************************************/
template< template< typename > class KernelArray, class TList >
template< uint i, typename ST >
__device__
__forceinline__ ST GPUPoolKernel< KernelArray, TList >
::getValue( const uint3& pos )
{
	typename Loki::TL::TypeAt< TList, i >::Result res;

	// Retrieve the channel at index i
	typename GPUPool_TypeAtChannel< KernelArray, TList, GPUPoolChannelUnitValue, i >::Result& channel = getChannel( Loki::Int2Type< i >() );

	// Write data in the channel (i.e. its associated surface)
	res = channel.get< i >( pos );

	ST val;
	convert_type( res, val );
	return val;
}

/////////////// GPU Pool host ////////////////

/******************************************************************************
 * Constructor
 *
 * @param pResolution resolution
 * @param pOptions options
 ******************************************************************************/
template< template< typename > class HostArray, class TList >
GPUPoolHost< HostArray, TList >
::GPUPoolHost( const uint3& pResolution, uint pOptions )
:	_resolution( pResolution )
,	_options( pOptions )
{
	// After User chooses all channels for color, normal, density, etc... data have to be allocated on device.
	// ChannelAllocator is a helper struct used to allocate theses data in a pool.
	GPUPoolHost< HostArray, TList >::ChannelAllocator channelAllocator( this, pResolution, pOptions );
	StaticLoop< ChannelAllocator, GPUPool< HostArray, TList, GPUPoolChannelUnitPointer >::numChannels - 1 >::go( channelAllocator );

	// Retrieve and initialize all device-side channel arrays
	GPUPoolHost< HostArray, TList >::ChannelInitializer channelInitializer( this );
	StaticLoop< ChannelInitializer, GPUPool< HostArray, TList, GPUPoolChannelUnitPointer >::numChannels - 1>::go( channelInitializer );

	GV_CHECK_CUDA_ERROR( "GPUPoolHost:GPUPoolHost" );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< template< typename > class HostArray, class TList >
GPUPoolHost< HostArray, TList >
::~GPUPoolHost()
{
	// Free memory of all data channels in the associated pool
	GPUPoolHost< HostArray, TList >::GvChannelDesallocator channelDesallocator( this );
	StaticLoop< GvChannelDesallocator, GPUPool< HostArray, TList, GPUPoolChannelUnitPointer >::numChannels - 1 >::go( channelDesallocator );
	
	GV_CHECK_CUDA_ERROR( "GPUPoolHost:~GPUPoolHost" );
}

/******************************************************************************
 * Get the device-side associated object
 *
 * @return the device-side associated object
 ******************************************************************************/
template< template< typename > class HostArray, class TList >
typename GPUPool_KernelPoolFromHostPool< HostArray, TList >::Result& GPUPoolHost< HostArray, TList >
::getKernelPool()
{
	return gpuPoolKernel;
}

/******************************************************************************
 * ...
 *
 * @param Loki::Int2Type< poolName > ...
 * @param normalizedResult ...
 * @param normalizedAccess ...
 * @param filterMode ...
 * @param addressMode ...
 ******************************************************************************/
template< template< typename > class HostArray, class TList >
template< int poolName >
void GPUPoolHost< HostArray, TList >
::bindPoolToTextureReferences( Loki::Int2Type< poolName >, bool normalizedResult, bool normalizedAccess, cudaTextureFilterMode filterMode, cudaTextureAddressMode addressMode )
{
	BindToTexRef< poolName > tempFunctor( this, normalizedResult, normalizedAccess, filterMode, addressMode );
	StaticLoop< BindToTexRef< poolName >, GPUPool< HostArray, TList, GPUPoolChannelUnitPointer >::numChannels - 1 >::go( tempFunctor );
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< template< typename > class HostArray, class TList >
void GPUPoolHost< HostArray, TList >
::bindPoolToSurfaceReferences()
{
	BindToSurfRef tempFunctor( this );
	StaticLoop< BindToSurfRef, GPUPool< HostArray, TList, GPUPoolChannelUnitPointer >::numChannels - 1 >::go( tempFunctor );
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< template< typename > class HostArray, class TList >
typename GPUPoolHost< HostArray, TList >::KernelPoolType GPUPoolHost< HostArray, TList >
::getKernelObject()
{
	return gpuPoolKernel;
}

} // namespace GvCore
