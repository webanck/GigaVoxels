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

#ifndef GVDATATYPELIST_H
#define GVDATATYPELIST_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/vector_types_ext.h"
#include "GvCore/TypeHelpers.h"

// Cuda
//
// NOTE : the CUDA #include <host_defines.h> MUST be placed before the LOKI #include <loki/HierarchyGenerators.h>,
// because LOKI has been modified by adding the CUDA __host__ and __device__ specifiers in one of its class.
#include <host_defines.h>
#include <vector_types.h>

// Loki
#include <loki/Typelist.h>
#include <loki/HierarchyGenerators.h>
#include <loki/TypeManip.h>
#include <loki/NullType.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/


namespace GvCore
{

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @struct DataChannelType
 *
 * @brief The DataChannelType struct provides an access to a particular type from a list of types.
 *
 * @ingroup GvCore
 *
 * Given an index into a list if types, it returns the type at the specified index from the list.
 */
template< class TList, unsigned int index >
struct DataChannelType
{
	/**
	 * Type definition.
	 * Result is the type at the specified index from the list.
	 */
	typedef typename Loki::TL::TypeAt< TList, index >::Result Result;
};

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/**
 * Functor used to serialize/deserialize data pool of a data structure
 */
template< typename TDataTypeList >
struct GvDataTypeInspector
{
	/**
	 * List of data types
	 */
	std::vector< std::string > _dataTypes;

	/**
	 * Generalized functor method used to bound textures.
	 *
	 * @param Loki::Int2Type< i > channel
	 */
	template< int TIndex >
	inline void run( Loki::Int2Type< TIndex > )
	{
		typedef typename GvCore::DataChannelType< TDataTypeList, TIndex >::Result VoxelType;

		//std::cout << "-\t" << GvCore::typeToString< GvCore::DataChannelType< TDataTypeList, TIndex > >() << std::endl;
		const char* type = GvCore::typeToString< VoxelType >();
		_dataTypes.push_back( type );
	}
};

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @struct DataNumChannels
 *
 * @brief The DataNumChannels struct provides the way the access the number of elements in a list of types.
 *
 * @ingroup GvCore
 *
 * Given a list of types, it returns the number of elements in the list.
 */
template< class TList >
struct DataNumChannels
{
	/**
	 * Enumeration definition.
	 * value is equal to the number of elements in the list if types.
	 */
	enum
	{
		value = Loki::TL::Length< TList >::value
	};
};

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @struct DataChannelSize
 *
 * @brief The DataChannelSize struct provides...
 *
 * @ingroup GvCore
 *
 * ...
 */
template< class TList, int index >
struct DataChannelSize
{
	/**
	 * ...
	 */
	enum
	{
		value = sizeof( typename Loki::TL::TypeAt< TList, index >::Result )
	};
};

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @struct DataTotalChannelSize
 *
 * @brief The DataTotalChannelSize struct provides...
 *
 * @ingroup GvCore
 *
 * ...
 */
template< class TList, int index = GvCore::DataNumChannels< TList >::value - 1 >
struct DataTotalChannelSize
{
	/**
	 * ...
	 */
	enum
	{
		value = DataChannelSize< TList, index >::value + DataChannelSize< TList, index - 1 >::value
	};
};

/** 
 * DataTotalChannelSize struct specialization
 */
template< class TList >
struct DataTotalChannelSize< TList, 0 >
{
	/**
	 * ...
	 */
	enum
	{
		value = DataChannelSize< TList, 0 >::value
	};
};

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @struct DataChannelUnitValue
 *
 * @brief The DataChannelUnitValue struct provides...
 *
 * @ingroup GvCore
 *
 * ...
 */
template< class T >
struct DataChannelUnitValue
{
	/**
	 * ...
	 */
	typedef T StorageType;

	/**
	 * ...
	 */
	StorageType value_;
   /* operator StorageType&() { return value_; }
	operator const StorageType&() const { return value_; }*/
};

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @struct DataStruct
 *
 * @brief The DataStruct struct provides...
 *
 * @ingroup GvCore
 *
 * ...
 */
template< class TList >
struct DataStruct
{
	/**
	 * ...
	 */
	typedef Loki::GenScatterHierarchy< TList, DataChannelUnitValue > Result;
};

/////////Type reconition///////

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @struct IsFloatFormat
 *
 * @brief The IsFloatFormat struct provides...
 *
 * @ingroup GvCore
 *
 * ...
 */
template< class T >
struct IsFloatFormat
{
	/**
	 * ...
	 */
	enum
	{
		value = 0
	};
};

/**
 * IsFloatFormat struct specialization
 */
template<>
struct IsFloatFormat< float >
{
	/**
	 * ...
	 */
	enum
	{
		value = 1
	};
};

/**
 * IsFloatFormat struct specialization
 */
template<>
struct IsFloatFormat< float2 >
{
	/**
	 * ...
	 */
	enum
	{
		value = 1
	};
};

/**
 * IsFloatFormat struct specialization
 */
template<>
struct IsFloatFormat< float3 >
{
	/**
	 * ...
	 */
	enum
	{
		value = 1
	};
};

/**
 * IsFloatFormat struct specialization
 */
template<>
struct IsFloatFormat< float4 >
{
	/**
	 * ...
	 */
	enum
	{
		value = 1
	};
};

/**
 * IsFloatFormat struct specialization
 */
template<>
struct IsFloatFormat< half4 >
{
	/**
	 * ...
	 */
	enum
	{
		value = 1
	};
};

////////////////////////////////
//////ReplaceTypeTemplated//////
////////////////////////////////

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @struct ReplaceTypeTemplated
 *
 * @brief The ReplaceTypeTemplated struct provides...
 *
 * @ingroup GvCore
 *
 * Replacement with type templated by replaced type
 */
template< class TList, template< typename > class RT >
struct ReplaceTypeTemplated;

/**
 * ReplaceTypeTemplated struct specialization
 */
template< template< typename > class RT >
struct ReplaceTypeTemplated< Loki::NullType, RT >
{
	typedef Loki::NullType Result;
};

/**
 * ReplaceTypeTemplated struct specialization
 */
template< class T, class Tail, template< typename > class RT >
struct ReplaceTypeTemplated< Loki::Typelist< T, Tail >, RT >
{
	typedef typename Loki::Typelist< RT< T >, typename ReplaceTypeTemplated< Tail, RT >::Result > Result;
};

////////////////////////////////
///////////Static loop//////////
////////////////////////////////

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class StaticLoop
 *
 * @brief The StaticLoop class provides...
 *
 * @ingroup GvCore
 *
 * ...
 */
template< class SFunctor, int i >
class StaticLoop
{

public:

	/**
	 * ...
	 *
	 * @param f ...
	 */
	inline static void go( SFunctor& f )
	{
		StaticLoop< SFunctor, i - 1 >::go( f );
		f.run( Loki::Int2Type< i >() );
	}

};

/**
 * StaticLoop class specialization
 */
template< class SFunctor >
class StaticLoop< SFunctor, -1 >
{

public:

	/**
	 * ...
	 *
	 * @param f ...
	 */
	inline static void go( SFunctor& f )
	{
	}

};

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class StaticLoopCallStatic
 *
 * @brief The StaticLoopCallStatic class provides...
 *
 * @ingroup GvCore
 *
 * ...
 */
template< class SFunctor, int i >
class StaticLoopCallStatic
{

public:

	/**
	 * ...
	 */
	__device__ __host__
	inline static void go()
	{
		SFunctor::run( Loki::Int2Type< i >() );
		StaticLoop< SFunctor, i - 1 >::go();
	}
};

/**
 * StaticLoopCallStatic class specialization
 */
template< class SFunctor >
class StaticLoopCallStatic< SFunctor, -1 >
{

public:

	/**
	 * ...
	 */
	__device__ __host__
	inline static void go()
	{
	}

};

} //namespace GvCore

#endif
