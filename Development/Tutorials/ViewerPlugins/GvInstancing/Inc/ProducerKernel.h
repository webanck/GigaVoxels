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

#ifndef _PRODUCER_KERNEL_H_
#define _PRODUCER_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvUtils/GvUtils.h>
//#include <Array3DKernelLinear.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Volume producer load's max number of levels
 *
 * Level corresponds to the resolution of data on disk
 */
#define GPUVPLD_MAX_NUM_LEVELS 16

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// Project
template< typename TDataStructureType, typename TGPUPoolKernelType, int channel >
class ProducerKernel_ChannelLoad;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class ProducerKernel
 *
 * @brief The ProducerKernel class provides the mecanisms to produce data
 * on device following data requests emitted during N-tree traversal (rendering).
 *
 * It is the main user entry point to produce data from GPU, for instance,
 * procedurally generating data (apply noise patterns, etc...).
 *
 * This class implements the mandatory functions of the GvIProviderKernel base class.
 *
 * @param DataTList Data type list
 * @param NodeRes Node resolution
 * @param BrickFullRes Brick full resolution
 * @param DataCachePoolKernelType Device-side associated's data cache pool
 */
template< typename TDataStructureType >
class ProducerKernel
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	// TODO: makes dependant on voxel type !

	/** @name Mandatory type definitions
	 *
	 * Due to the use of templated classes,
	 * type definitions are used to ease use development.
	 * Some are aloso required to follow the GigaVoxels pipeline flow sequences.
	 */
	///@{

	/**
	 * MACRO
	 * 
	 * Useful and required type definition for producer kernels
	 * - it is used to access the DataStructure typedef passed in argument
	 *
	 * @param TDataStructureType a data structure type (should be the template parameter of a Producer Kernel)
	 */
	GV_MACRO_PRODUCER_KERNEL_REQUIRED_TYPE_DEFINITIONS( TDataStructureType )

	/**
	 * CUDA block dimension used for nodes production (kernel launch)
	 */
	typedef GvCore::StaticRes3D< 32, 1, 1 > NodesKernelBlockSize;

	/**
	 * CUDA block dimension used for bricks production (kernel launch)
	 */
	typedef GvCore::StaticRes3D< 16, 8, 1 > BricksKernelBlockSize;

	///@}

	/**
	 * Defines the data type list
	 */
	typedef typename TDataStructureType::DataTypeList DataTList;

	/**
	 * Brick full resolution
	 */
	typedef GvCore::StaticRes3D
	<
		BrickRes::x + 2 * BorderSize,
		BrickRes::y + 2 * BorderSize,
		BrickRes::z + 2 * BorderSize
	>
	BrickFullRes;

	/**
	 * Brick pool kernel type
	 */
	typedef GvCore::GPUPoolKernel< GvCore::Array3DKernelLinear, DataTList >	DataCachePoolKernelType;

	/**
	 * Brick voxel alignment
	 */
	enum
	{
		BrickVoxelAlignment = GvCore::IDivUp< BrickFullRes::x * BrickFullRes::y * BrickFullRes::z, 32 >::value * 32
	};

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Max depth
	 */
	uint _maxDepth;

	/**
	 * DEVICE-side associated HOST data cache pool
	 */
	DataCachePoolKernelType _cpuDataCachePool;

	/**
	 * DEVICE-side associated HOST nodes cache
	 */
	GvCore::Array3DKernelLinear< uint >	_cpuNodesCache;

	/******************************** METHODS *********************************/

	/**
	 * Initialize the producer
	 * 
	 * @param volumeTreeKernel Reference on a volume tree data structure
	 */
	inline void initialize( DataStructureKernel& pDataStructure );

	/**
	 * Inititialize
	 *
	 * @param maxdepth max depth
	 * @param nodescache nodes cache
	 * @param datacachepool data cache pool
	 */
	inline void init( uint maxdepth, const GvCore::Array3DKernelLinear< uint >& nodescache, const DataCachePoolKernelType& datacachepool );

	/**
	 * Produce data on device.
	 * Implement the produceData method for the channel 0 (nodes)
	 *
	 * Producing data mecanism works element by element (node tile or brick) depending on the channel.
	 *
	 * In the function, user has to produce data for a node tile or a brick of voxels :
	 * - for a node tile, user has to defined regions (i.e nodes) where lies data, constant values,
	 * etc...
	 * - for a brick, user has to produce data (i.e voxels) at for each of the channels
	 * user had previously defined (color, normal, density, etc...)
	 *
	 * @param pGpuPool The device side pool (nodes or bricks)
	 * @param pRequestID The current processed element coming from the data requests list (a node tile or a brick)
	 * @param pProcessID Index of one of the elements inside a node tile or a voxel bricks
	 * @param pNewElemAddress The address at which to write the produced data in the pool
	 * @param pParentLocInfo The localization info used to locate an element in the pool
	 * @param Loki::Int2Type< 0 > corresponds to the index of the node pool
	 *
	 * @return A feedback value that the user can return.
	 */
	template< typename GPUPoolKernelType >
	__device__
	inline uint produceData( GPUPoolKernelType& nodePool, uint requestID, uint processID,
							uint3 newElemAddress, const GvCore::GvLocalizationInfo& parentLocInfo,
							Loki::Int2Type< 0 > );

	/**
	 * Produce data on device.
	 * Implement the produceData method for the channel 1 (bricks)
	 *
	 * Producing data mecanism works element by element (node tile or brick) depending on the channel.
	 *
	 * In the function, user has to produce data for a node tile or a brick of voxels :
	 * - for a node tile, user has to defined regions (i.e nodes) where lies data, constant values,
	 * etc...
	 * - for a brick, user has to produce data (i.e voxels) at for each of the channels
	 * user had previously defined (color, normal, density, etc...)
	 *
	 * @param pGpuPool The device side pool (nodes or bricks)
	 * @param pRequestID The current processed element coming from the data requests list (a node tile or a brick)
	 * @param pProcessID Index of one of the elements inside a node tile or a voxel bricks
	 * @param pNewElemAddress The address at which to write the produced data in the pool
	 * @param pParentLocInfo The localization info used to locate an element in the pool
	 * @param Loki::Int2Type< 1 > corresponds to the index of the brick pool
	 *
	 * @return A feedback value that the user can return.
	 */
	template< typename TGPUPoolKernelType >
	__device__
	inline uint produceData( TGPUPoolKernelType& pDataPool, uint pRequestID, uint pProcessID,
							uint3 pNewElemAddress, const GvCore::GvLocalizationInfo& pParentLocInfo,
							Loki::Int2Type< 1 > );

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

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Data Structure device-side associated object
	 *
	 * Note : use this element if you need to sample data in cache
	 */
	//DataStructureKernel _dataStructureKernel;

	/******************************** METHODS *********************************/

};

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class ProducerKernel_ChannelLoad
 *
 * @brief The ProducerKernel_ChannelLoad class provides the mecanisms to produce data
 * on device following data requests emitted during N-tree traversal (rendering)
 * for a single channel (as color, normal, etc...).
 *
 * It is a helper class used by the main DEVICE producer class ProducerKernel.
 * 
 * @param DataTList Data type list
 * @param NodeTileRes Node tile resolution
 * @param BrickFullRes Brick full resolution
 * @param DataCachePoolKernelType Device-side associated's data cache pool
 * @param GPUPoolKernelType Device-side data pool (i.e. brick of voxels)
 * @param channel index of the channel
 */
template< typename TDataStructureType, typename TGPUPoolKernelType, int channel >
class ProducerKernel_ChannelLoad
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/**
	 * Type definition of the node tile resolution
	 */
	typedef typename TDataStructureType::NodeTileResolution NodeRes;

	/**
	 * Type definition of the brick resolution
	 */
	typedef typename TDataStructureType::BrickResolution BrickRes;

	/**
	 * Enumeration to define the brick border size
	 */
	enum
	{
		BorderSize = TDataStructureType::BrickBorderSize
	};

	/**
	 * Defines the data type list
	 */
	typedef typename TDataStructureType::DataTypeList DataTList;

	/**
	 * Brick full resolution
	 */
	typedef GvCore::StaticRes3D
	<
		BrickRes::x + 2 * BorderSize,
		BrickRes::y + 2 * BorderSize,
		BrickRes::z + 2 * BorderSize
	>
	BrickFullRes;

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Produce data at the specified channel
	 *
	 * @param gpuVPLK reference on the volume producer load kernel
	 * @param dataPool the data pool in which to write data
	 * @param elemAddress The address at which to write the produced data in the pool
	 * @param parentLocInfo The localization info used to locate an element in the pool
	 * @param pRequestID The current processed element coming from the data requests list (a brick)
	 * @param pProcessID Index of one of the elements inside a voxel bricks
	 */
	__device__
	inline static bool produceDataChannel( ProducerKernel< TDataStructureType >& gpuVPLK,
										TGPUPoolKernelType& dataPool, uint3 elemAddress, const GvCore::GvLocalizationInfo& parentLocInfo, uint requestID, uint processID );

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

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * ProducerKernel_ChannelLoad struct specialization
 */
template< typename TDataStructureType, typename TGPUPoolKernelType >
class ProducerKernel_ChannelLoad< TDataStructureType, TGPUPoolKernelType, -1 >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Produce data at the specified channel
	 *
	 * @param gpuVPLK reference on the volume producer load kernel
	 * @param dataPool the data pool in which to write data
	 * @param elemAddress The address at which to write the produced data in the pool
	 * @param parentLocInfo The localization info used to locate an element in the pool
	 * @param pRequestID The current processed element coming from the data requests list (a brick)
	 * @param pProcessID Index of one of the elements inside a voxel bricks
	 */
	__device__
	static inline bool produceDataChannel( ProducerKernel< TDataStructureType >& gpuVPLK,
											TGPUPoolKernelType& dataPool, uint3 elemAddress, const GvCore::GvLocalizationInfo& parentLocInfo, uint requestID, uint processID );

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

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "ProducerKernel.inl"

#endif // !_PRODUCER_KERNEL_H_
