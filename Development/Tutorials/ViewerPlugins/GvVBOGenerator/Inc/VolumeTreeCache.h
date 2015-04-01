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

#ifndef _VOLUME_TREE_CACHE_H_
#define _VOLUME_TREE_CACHE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

// System
#include <iostream>

// Cuda
#include <vector_types.h>

// Thrust
#include <thrust/device_vector.h>
#include <thrust/copy.h>

// GigaVoxels
#include <GvCore/Array3D.h>
#include <GvCore/Array3DGPULinear.h>
#include <GvCore/RendererTypes.h>
#include <GvCore/GPUPool.h>
#include <GvCore/StaticRes3D.h>
#include <GvCore/GvPageTable.h>
#include <GvCore/GvIProvider.h>
#include <GvRendering/GvRendererHelpersKernel.h>
#include <GvCore/GPUVoxelProducer.h>
#include <GvCore/GvLocalizationInfo.h>
#include <GvCache/GvCacheManager.h>
#include <GvPerfMon/GvPerformanceMonitor.h>
#include <GvStructure/GvVolumeTreeAddressType.h>
#include <GvStructure/GvDataProductionManager.h>
//#include <GvStructure/GvVolumeTreeCacheKernel.h>

#if USE_CUDPP_LIBRARY
	// cudpp
	#include <cudpp.h>
#endif

// Project
#include "CacheManager.h"
#include "VolumeTreeCacheKernel.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

//// VBO
//namespace GvUtils
//{
//	class GvProxyGeometryHandler;
//}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/
	
	/** 
	 * @class VolumeTreeCache
	 *
	 * @brief The VolumeTreeCache class provides the concept of cache.
	 *
	 * This class implements the cache mechanism for the VolumeTree data structure.
	 * As device memory is limited, it is used to store the least recently used element in memory.
	 * It is responsible to handle the data requests list generated during the rendering process.
	 * (ray-tracing - N-tree traversal).
	 * Request are then sent to producer to load or produced data on the host or on the device.
	 *
	 * @param TDataStructureType The volume tree data structure (nodes and bricks)
	 * @param ProducerType The associated user producer (host or device)
	 * @param NodeTileRes The user defined node tile resolution
	 * @param BrickFullRes The user defined brick resolution
	 */
	template< typename TDataStructureType >
	class VolumeTreeCache : public GvStructure::GvDataProductionManager< TDataStructureType >
	{

		/**************************************************************************
		 ***************************** PUBLIC SECTION *****************************
		 **************************************************************************/

	public:

		/****************************** INNER TYPES *******************************/

		/**
		 * Type definition of the parent class
		 */
		typedef GvStructure::GvDataProductionManager< TDataStructureType > ParentClass;

		/**
		 * Type definition of the node tile resolution
		 */
		typedef typename TDataStructureType::NodeTileResolution NodeTileRes;

		/**
		 * Type definition for the bricks cache manager
		 */
		typedef CacheManager
		<
			1, typename ParentClass::BrickFullRes, GvStructure::VolTreeBrickAddress, GvCore::Array3DGPULinear< uint >, typename ParentClass::BrickPageTableType, TDataStructureType
		>
		VBOCacheManagerType;

		/**
		 * Type definition for the associated device-side object
		 */
		typedef VolumeTreeCacheKernel
		<
			typename ParentClass::NodeTileResLinear, typename ParentClass::BrickFullRes, GvStructure::VolTreeNodeAddress, GvStructure::VolTreeBrickAddress
		>
		VBOVolumeTreeCacheKernelType;

		/******************************* ATTRIBUTES *******************************/

		/**
		 * VBO
		 */
		//GvUtils::GvProxyGeometryHandler* _vbo;

		/******************************** METHODS *********************************/

		/**
		 * Constructor
		 *
		 * @param pDataStructure a pointer to the data structure.
		 * @param gpuprod a pointer to the user's producer.
		 * @param nodepoolres the 3d size of the node pool.
		 * @param brickpoolres the 3d size of the brick pool.
		 * @param graphicsInteroperability Graphics interoperabiliy flag
		 */
		VolumeTreeCache( TDataStructureType* pDataStructure, uint3 nodepoolres, uint3 brickpoolres/*, GvUtils::GvProxyGeometryHandler* _vbo*/, uint graphicsInteroperability = 0 );

		/**
		 * Destructor
		 */
		virtual ~VolumeTreeCache();

		/**
		 * This method is called before the rendering process. We just clear the request buffer.
		 */
		virtual void preRenderPass();

		/**
		 * This method is called after the rendering process. She's responsible for processing requests.
		 *
		 * @return the number of requests processed.
		 *
		 * @todo Check wheter or not the inversion call of updateTimeStamps() with manageUpdates() has side effects
		 */
		virtual uint handleRequests();

		/**
		 * This method destroy the current N-tree and clear the caches.
		 */
		virtual void clearCache();

		/**
		 * Get the associated device-side object
		 *
		 * @return The device-side object
		 */
		inline VBOVolumeTreeCacheKernelType getVBOKernelObject() const;

		/**
		 * Get the VBO cache manager
		 *
		 * @return the VBO cache manager
		 */
		inline const VBOCacheManagerType* getVBOCacheManager() const;

		/**
		 * Get the VBO cache manager
		 *
		 * @return the VBO cache manager
		 */
		inline VBOCacheManagerType* editVBOCacheManager();

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

		/**
		 * The associated device-side object
		 */
		VBOVolumeTreeCacheKernelType _vboVolumeTreeCacheKernel;

		/**
		 * VBO cache manager
		 */
		VBOCacheManagerType* _vboCacheManager;

		/******************************** METHODS *********************************/

	};


/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "VolumeTreeCache.inl"

#endif
