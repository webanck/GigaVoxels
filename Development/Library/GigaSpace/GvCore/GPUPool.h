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

#ifndef GVPOOL_H
#define GVPOOL_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/DataTypeList.h"
#include "GvCore/Array3D.h"
#include "GvCore/GvCUDATexHelpers.h"

// Cuda
//
// NOTE : the CUDA #include <host_defines.h> MUST be placed before the LOKI #include <loki/HierarchyGenerators.h>,
// because LOKI has been modified by adding the CUDA __host__ and __device__ specifiers in one of its class.
#include <host_defines.h>

// Loki
#include <loki/Typelist.h>
#include <loki/HierarchyGenerators.h>
#include <loki/TypeManip.h>
#include <loki/NullType.h>

// System
#include <cassert>

// GigaVoxels
#include "GvStructure/GvVolumeTreeKernel.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/
GPUPoolSurfaceReferences( 0 )
GPUPoolSurfaceReferences( 1 )
GPUPoolSurfaceReferences( 2 )
GPUPoolSurfaceReferences( 3 )
GPUPoolSurfaceReferences( 4 )
//GPUPoolSurfaceReferences( 5 )
//GPUPoolSurfaceReferences( 6 )
//GPUPoolSurfaceReferences( 7 )


/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// Gigavoxel
namespace GvCore
{
	template< template< typename > class ArrayType, class TList, template< typename > class ChannelUnit, uint i >
	struct GPUPool_TypeAtChannel;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{

/** 
 * @class GPUPool
 *
 * @brief The GPUPool class provides...
 *
 * @ingroup GvCore
 *
 * 3D Array manipulation class.
 */
template< template< typename > class ArrayType, class TList, template< typename > class ChannelUnit >
class GPUPool
{
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Enumeration used to store the number of elements in the list.
	 */
	enum
	{
		numChannels = Loki::TL::Length< TList >::value
	};

	/**
	 * ...
	 */
	typedef typename ReplaceTypeTemplated< TList, ArrayType >::Result ChannelsTList;

	/**
	 * ...
	 */
	typedef Loki::GenScatterHierarchy< typename GPUPool::ChannelsTList, ChannelUnit > ChannelsType;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	ChannelsType channels;

	/******************************** METHODS *********************************/

	/**
	 * ...
	 */
	template< uint i >
	struct TypeAtChannel
	{
		typedef typename ChannelUnit< typename Loki::TL::TypeAt< ChannelsTList, i >::Result >::StorageType Result;
		//typedef typename Loki::TL::TypeAt<ChannelsTList, i>::Result Result;
	};
	/**
	 * ...
	 */
	template< uint i >
	struct TypeAtChannelNoPointer
	{
		typedef typename Loki::TL::TypeAt< ChannelsTList, i >::Result Result;
	};

	/**
	 * ...
	 *
	 * @return ...
	 */
	template< uint i >
	__host__ __device__
	inline typename TypeAtChannel< i >::Result& getChannel()
	{
		return ( Loki::FieldHelper< ChannelsType, i >::Do( channels ) ).value_;
	}

	//Workaround for gcc problems compiling the other version
	/**
	 * ...
	 *
	 * @param valtype ...
	 *
	 * @return ...
	 */
	template< int i >
	__host__ __device__
	inline typename TypeAtChannel< i >::Result& getChannel( Loki::Int2Type< i > valtype )
	{
		return ( Loki::FieldHelper< ChannelsType, i >::Do( channels ) ).value_;
	}

	/**
	 * Get the pool resolution.
	 *
	 * @return the pool resolution
	 */
	__host__ __device__
	const uint3 getResolution() const
	{
		return getChannel( Loki::Int2Type< 0 >() )->getResolution();
	}

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

} // namespace GvCore

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{

//Workaround for gcc templates compilation problems
template< template< typename > class ArrayType, class TList, template< typename > class ChannelUnit, uint i >
struct GPUPool_TypeAtChannel
{
	typedef typename ChannelUnit< typename Loki::TL::TypeAt< typename GPUPool< ArrayType, TList, ChannelUnit >::ChannelsTList, i >::Result >::StorageType Result;
	//typedef typename Loki::TL::TypeAt<typename GPUPool<ArrayType, TList, ChannelUnit>::ChannelsTList, i>::Result Result;
};

/**
 * ...
 *
 * @param pool ...
 *
 * @return ...
 */
template< uint i, template< typename > class ArrayType, class TList, template< typename > class ChannelUnit >
__device__ __host__
typename GPUPool_TypeAtChannel< ArrayType, TList, ChannelUnit, i >::Result& GPUPool_getChannel( GPUPool< ArrayType, TList, ChannelUnit >& pool )
{
	return ( Loki::FieldHelper< typename GPUPool< ArrayType, TList, ChannelUnit >::ChannelsType, i >::Do( pool.channels ) ).value_;
}

} // namespace GvCore

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{

/** 
 * @struct GPUPoolChannelUnitValue
 *
 * @brief The GPUPoolChannelUnitValue struct provides...
 *
 * ...
 */
template< class T >
struct GPUPoolChannelUnitValue
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	/**
	 * ...
	 */
	typedef T StorageType;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	StorageType value_;

	/******************************** METHODS *********************************/

	/**
	 * ...
	 *
	 * @return ...
	 */
	operator StorageType&()
	{
		return value_;
	}

	/**
	 * ...
	 *
	 * @return ...
	 */
	operator const StorageType&() const
	{
		return value_;
	}

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

} // namespace GvCore

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{

/** 
 * @struct GPUPoolChannelUnitPointer
 *
 * @brief The GPUPoolChannelUnitPointer struct provides...
 *
 * ...
 */
template< class T >
struct GPUPoolChannelUnitPointer
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition of a pointer on this class template type.
	 */
	typedef T* StorageType;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Stored value.
	 * Value is a pointer on the class template type.
	 */
	StorageType value_;

	/******************************** METHODS *********************************/

	/**
	 * Get the stored value.
	 *
	 * @return the storage type
	 */
	operator StorageType&()
	{
		return value_;
	}

	/**
	 * ...
	 *
	 * @return ...
	 */
	operator const StorageType&() const
	{
		return value_;
	}

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

} // namespace GvCore

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{

///////////////GPU Pool Kernel////////////////
/** 
 * @class GPUPoolKernel
 *
 * @brief The GPUPoolKernel class provides...
 *
 * ...
 */
template< template< typename > class KernelArray, class TList >
class GPUPoolKernel : public GPUPool< KernelArray, TList, GPUPoolChannelUnitValue >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Set the value at a given position in the pool.
	 *
	 * @param pos position the pool
	 * @param val value
	 */
	template< uint i, typename ST >
	__device__
	__forceinline__ void setValue( const uint3& pos, ST val );

	/**
	 * Get the value at a given position in the pool.
	 *
	 * @param pos position the pool
	 * @param val value
	 */
	template< uint i, typename ST >
	__device__
	__forceinline__ ST getValue( const uint3& pos );

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

} // namespace GvCore

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{

/////////////// GPU Pool host ////////////////

//To be specialized for different Array3D types. Would be better per array but partial specialisation did not work for this
template< template< typename > class HostArray, class TList >
struct GPUPool_KernelPoolFromHostPool;

/** 
 * @class GPUPoolKernel
 *
 * @brief The GPUPoolKernel class provides...
 *
 * Host side GPU Pool. It instanciate automatically Arrays whose constructors have to be on the form (dim3 size, uint options).
 */
template< template< typename > class HostArray, class TList >
class GPUPoolHost : public GPUPool< HostArray, TList, GPUPoolChannelUnitPointer >
{
	// TODO: remove templated name, transform into sampler name passed when asked for sampler attachement.

	// Have to use functors
	/** 
	 * @struct ChannelAllocator
	 *
	 * @brief The ChannelAllocator struct provides a way to allocate data of user defined channels.
	 *
	 * After user chooses all channels for color, normal, density, etc... data have to be allocated on device.
	 * ChannelAllocator is a helper struct used to allocate theses data in a pool.
	 */
	struct ChannelAllocator
	{

		/**************************************************************************
		 ***************************** PUBLIC SECTION *****************************
		 **************************************************************************/

		/******************************* ATTRIBUTES *******************************/

		/**
		 * Resolution
		 */
		uint3 _resolution;

		/**
		 * Options
		 */
		uint _options;

		/**
		 * Host pool
		 */
		GPUPoolHost< HostArray, TList >* _pool;

		/******************************** METHODS *********************************/

		/**
		 * Constructor
		 *
		 * @param pPool host pool
		 * @param pResolution resolution
		 * @param pOptions options
		 */
		ChannelAllocator( GPUPoolHost< HostArray, TList >* pPool, const uint3& pResolution, uint pOptions )
		:	_resolution( pResolution )
		,	_options( pOptions )
		,	_pool( pPool )
		{
		}

		/**
		 * Generic functor's main method to allocate user-defined data in the data pool given its channel index
		 *
		 * @param Loki::Int2Type< i > channel index
		 */
		template< int i >
		inline void run( Loki::Int2Type< i > )
		{
			typename GPUPool_TypeAtChannel< HostArray, TList, GPUPoolChannelUnitPointer, i >::Result& P = _pool->getChannel( Loki::Int2Type< i >() );
			//typename GPUPool_TypeAtChannel<HostArray, TList, GPUPoolChannelUnitPointer, i>::Result &P	= GPUPool_getChannel< i >( _pool->getGPUPool() );

			P = new typename GPUPool_TypeAtChannel< HostArray, TList, GPUPoolChannelUnitValue, i >::Result( _resolution, _options );
		}

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

	}; // struct ChannelAllocator

	/** 
	 * @struct GvChannelDesallocator
	 *
	 * @brief The GvChannelDesallocator struct provides a way to desallocate data of user defined channels.
	 *
	 * This helper struct is used to release all host and device resources used by data in a pool.
	 */
	struct GvChannelDesallocator
	{

		/**************************************************************************
		 ***************************** PUBLIC SECTION *****************************
		 **************************************************************************/

		/******************************* ATTRIBUTES *******************************/

		/**
		 * Associated host pool
		 */
		GPUPoolHost< HostArray, TList >* _poolHost;

		/******************************** METHODS *********************************/

		/**
		 * Constructor.
		 *
		 * @param pPoolHost the associated host pool
		 */
		GvChannelDesallocator( GPUPoolHost< HostArray, TList >* pPoolHost )
		:	_poolHost( pPoolHost )
		{
		}

		/**
		 * Generalized functor function used to desallocate data in the associated pool.
		 *
		 * @param Loki::Int2Type< i > channel index
		 */
		template< int i >
		inline void run( Loki::Int2Type< i > )
		{
			// Retrieve channel's data
			typename GPUPool_TypeAtChannel< HostArray, TList, GPUPoolChannelUnitPointer, i >::Result& P = _poolHost->getChannel( Loki::Int2Type< i >() );
		
			// Destroy data
			delete P;
			P = NULL;
		}

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

	}; // struct GvChannelDesallocator

	/** 
	 * @struct ChannelInitializer
	 *
	 * @brief The ChannelInitializer struct provides...
	 *
	 * ...
	 */
	struct ChannelInitializer
	{

		/**************************************************************************
		 ***************************** PUBLIC SECTION *****************************
		 **************************************************************************/

		/******************************* ATTRIBUTES *******************************/

		/**
		 * Host pool
		 */
		GPUPoolHost< HostArray, TList >* gph;

		/******************************** METHODS *********************************/

		/**
		 * Constructor
		 *
		 * @param 
		 */
		ChannelInitializer( GPUPoolHost< HostArray, TList >* p )
		:	gph( p )
		{
		}

		/**
		 * ...
		 *
		 * @param Loki::Int2Type< i > ...
		 */
		template< int i >
		inline void run( Loki::Int2Type< i > )
		{
			gph->gpuPoolKernel.getChannel( Loki::Int2Type< i >() ) = gph->getChannel( Loki::Int2Type< i >() )->getDeviceArray();
		}

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

	}; // struct ChannelInitializer

	/** 
	 * @struct BindToTexRef
	 *
	 * @brief The BindToTexRef struct provides a generalized functor to bind an array to a texture.
	 *
	 * Data in pools are stored in array allocated on device. These arrays are then bound to 3D textures.
	 */
	template< int poolName >
	struct BindToTexRef
	{

		/**************************************************************************
		 ***************************** PUBLIC SECTION *****************************
		 **************************************************************************/

		/******************************* ATTRIBUTES *******************************/

		/**
		 * Host pool.
		 * GPU pool host from which textures will be bound.
		 */
		GPUPoolHost< HostArray, TList >* gph;

		/**
		 * ...
		 */
		bool normalizedResult;

		/**
		 * Flag to tell of texture is access in normalized way or not
		 */
		bool normalizedAccess;

		/**
		 * Filter mode
		 */
		cudaTextureFilterMode filterMode;

		/**
		 * Address mode
		 */
		cudaTextureAddressMode addressMode;

		/******************************** METHODS *********************************/

		/**
		 * Constructor.
		 *
		 * @param p GPU pool host from which textures will be bound
		 * @param nr ...
		 * @param na flag to tell of texture is access in normalized way or not
		 * @param fm filter mode
		 * @param am address mode
		 */
		BindToTexRef( GPUPoolHost< HostArray, TList >* p, bool nr, bool na, cudaTextureFilterMode fm, cudaTextureAddressMode am )
			:	gph( p )
			,	normalizedResult( nr )
			,	normalizedAccess( na )
			,	filterMode( fm )
			,	addressMode( am )
		{
		}

		/**
		 * Generalized functor method used to bound textures.
		 *
		 * @param Loki::Int2Type< i > channel
		 */
		template< int i >
		inline void run( Loki::Int2Type< i > )
		{
			if ( normalizedResult && !( IsFloatFormat< typename DataChannelType< TList, i >::Result >::value ) )
			{
				CUDATexHelpers_BindGPUArrayToTexRef( Loki::Int2Type< poolName >(), Loki::Int2Type< i >(), Loki::Int2Type< 3 >(),
													typename DataChannelType< TList, i >::Result(), Loki::Int2Type< cudaReadModeNormalizedFloat >(),
													gph->getChannel( Loki::Int2Type< i >() ), normalizedAccess, filterMode, addressMode );
			}
			else
			{
				CUDATexHelpers_BindGPUArrayToTexRef( Loki::Int2Type< poolName >(), Loki::Int2Type< i >(), Loki::Int2Type< 3 >(),
													typename DataChannelType< TList, i >::Result(), Loki::Int2Type< cudaReadModeElementType >(),
													gph->getChannel( Loki::Int2Type< i >() ), normalizedAccess, filterMode, addressMode );
			}
		}

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

	}; // struct BindToTexRef

	/** 
	 * @struct BindToSurfRef
	 *
	 * @brief The BindToSurfRef struct provides...
	 *
	 * ...
	 */
	struct BindToSurfRef
	{

		/**************************************************************************
		 ***************************** PUBLIC SECTION *****************************
		 **************************************************************************/

		/******************************* ATTRIBUTES *******************************/

		/******************************** METHODS *********************************/

		/**
		 * Host pool
		 */
		GPUPoolHost< HostArray, TList >* gph;

		/**
		 * Constructor
		 *
		 * @param p ...
		 */
		BindToSurfRef( GPUPoolHost< HostArray, TList >* p )
		:	gph( p )
		{
		}

		/**
		 * ...
		 *
		 * @param Loki::Int2Type<i > ...
		 */
		template< int i >
		inline void run( Loki::Int2Type< i > )
		{
			CUDATexHelpers_BindGPUArrayToSurfRef( Loki::Int2Type< i >(), gph->getChannel( Loki::Int2Type< i >() ) );
		}

	}; // struct BindToSurfRef

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition.
	 * ...
	 */
	typedef typename GPUPool_KernelPoolFromHostPool< HostArray, TList >::Result KernelPoolType;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Device-side associated object
	 */
	KernelPoolType gpuPoolKernel;

	/**
	 * Resolution
	 */
	uint3 _resolution;

	/**
	 * Options
	 */
	uint _options;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param res resolution
	 * @param options options
	 */
	GPUPoolHost( const uint3& res, uint options = 0 );

	/**
	 * Destructor
	 */
	virtual ~GPUPoolHost();

	/**
	 * Get the device-side associated object
	 *
	 * @return the device-side associated object
	 */
	typename GPUPool_KernelPoolFromHostPool< HostArray, TList >::Result& getKernelPool();

	/**
	 * Bind pool to textures.
	 * Currently, 3D textures are used to read data in the pool.
	 *
	 * @param Loki::Int2Type< poolName > ...
	 * @param normalizedResult ...
	 * @param normalizedAccess ...
	 * @param filterMode ...
	 * @param addressMode ...
	 */
	template< int poolName >
	void bindPoolToTextureReferences( Loki::Int2Type< poolName >,
									bool normalizedResult, bool normalizedAccess,
									cudaTextureFilterMode filterMode, cudaTextureAddressMode addressMode );

	/**
	 * Bind pool to surfaces.
	 * Currently, 3D surfaces are used to write data in the pool.
	 */
	void bindPoolToSurfaceReferences();

	// TO DO : check if it is the same method that getKernelPool() ? Seems to be identical except the "&" ?
	/**
	 * Get the device-side object.
	 *
	 * @return the associated device-side object
	 */
	KernelPoolType getKernelObject();

}; // class GPUPoolHost

} // namespace GvCore

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GPUPool.inl"

#endif
