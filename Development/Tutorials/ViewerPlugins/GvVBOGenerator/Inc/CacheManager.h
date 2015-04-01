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

#ifndef _CACHE_MANAGER_H_
#define _CACHE_MANAGER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GvCoreConfig.h>
#include <GvPerfMon/GvPerformanceMonitor.h>
#include <GvCache/GvCacheManager.h>

// CUDA
#include <vector_types.h>

// CUDA SDK
#include <helper_math.h>

// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// GigaVoxels
#include <GvCache/GvCacheManagerKernel.h>
#include <GvCore/Array3D.h>
#include <GvCore/Array3DGPULinear.h>
#include <GvCore/functional_ext.h>
#include <GvCache/GvCacheManager.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

//// GigaVoxels
//namespace GvUtils
//{
//	class GvProxyGeometryHandler;
//}

// Project
class ParticleSystem;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
* @class CacheManager
*
* @brief The CacheManager class provides...
*
* @ingroup ...
*
* This class is used to manage a cache on the GPU
*
* Aide PARAMETRES TEMPLATES :
* dans VolumeTreeCache.h :
* - PageTableArrayType == Array3DGPULinear< uint >
* - PageTableType == PageTableNodes< ... Array3DKernelLinear< uint > ... > ou PageTableBricks< ... >
* - GPUProviderType == IProvider< 1, GPUProducer > ou bien avec 0
*/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename TDataStructureType >
class CacheManager : public GvCache::GvCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition of the parent class
	 */
	typedef GvCache::GvCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType > ParentClass;

	/**
	 * Type definition for the GPU side associated object
	 *
	 * @todo pass this parameter as a template parameter in order to be able to overload this component easily
	 */
	typedef typename ParentClass::KernelType KernelType;

	/**
	 * Data structure
	 */
	TDataStructureType* _dataStructure;

	/**
	 * VBO
	 */
	//GvUtils::GvProxyGeometryHandler* _vbo;

		/**
	 * Particle system
	 */
	ParticleSystem* _particleSystem;
	
	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param cachesize
	 * @param pageTableArray
	 * @param graphicsInteroperability
	 */
	CacheManager( uint3 cachesize, PageTableArrayType* pageTableArray, /*GvUtils::GvProxyGeometryHandler* _vbo,*/ uint graphicsInteroperability = 0 );

	/**
	 * Destructor
	 */
	virtual ~CacheManager();

	/**
	 * Update VBO
	 */
	uint updateVBO( bool manageUpdatesOnly );

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
	 * VBO ...
	 */
	thrust::device_vector< uint >* _d_vboBrickList;

	/**
	 * VBO ...
	 */
	thrust::device_vector< uint >* _d_vboIndexOffsetList;
	
#if USE_CUDPP_LIBRARY
	/**
	 * CUDPP vbo SCAN PLAN
	 *
	 * A plan is a data structure containing state and intermediate storage space that CUDPP uses to execute algorithms on data.
	 */
	 CUDPPHandle _vboScanPlan;
	 uint _vboScanPlanSize;
#endif

	/******************************** METHODS *********************************/

	/**
	 * Get a CUDPP plan given a number of elements to be processed.
	 *
	 * @param pSize The maximum number of elements to be processed
	 *
	 * @return a handle on the plan
	 */
	CUDPPHandle getVBOScanPlan( uint pSize );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "CacheManager.inl"

#endif
