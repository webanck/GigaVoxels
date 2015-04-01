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

#ifndef _BVHTREECACHEMANAGER_H_
#define _BVHTREECACHEMANAGER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// CUDA
#include <cutil.h>
#include <cutil_math.h>

#include <vector_types.h>

#include "RendererBVHTrianglesCommon.h"

// Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// cudpp
#include <cudpp.h>

//#include "BvhTreeCache.hcu"

// GigaVoxels
#include <GvPerfMon/GvPerformanceMonitor.h>

#include "BvhTreeCacheManager.hcu"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

// Protect null reference
static const uint cNumLockedElements = 1;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class GPUCacheManager
 *
 * @brief The GPUCacheManager class provides ...
 *
 * @param ElementRes ...
 * @param ProviderType ...
 */
template< typename ElementRes, typename ProviderType >
class GPUCacheManager
{
	
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

#if USE_SYNTHETIC_INFO
	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< uchar4 >* d_SyntheticCacheStateBufferArray;
#endif

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GPUCacheManager( uint3 cachesize, uint3 elemsize );
	
	/**
	 * Destructor
	 */
	~GPUCacheManager();

	/**
	 * ...
	 *
	 * @return ...
	 */
	uint getNumElements()
	{
		return _elemsCacheSize.x * _elemsCacheSize.y * _elemsCacheSize.z;
	}

	/**
	 * ...
	 *
	 * @return ...
	 */
	GvCore::Array3DGPULinear< uint >* getdTimeStampArray()
	{
		return d_TimeStampArray;
	}

	/**
	 * ...
	 *
	 * @return ...
	 */
	thrust::device_vector< uint >* getTimeStampsElemAddressList()
	{
		return d_elemAddressList;
	}

	/**
	 * Update symbols
	 */
	void updateSymbols();

	/**
	 * Set the associated producer
	 *
	 * @param provider
	 */
	void setProvider( ProviderType* provider );

	/**
	  * Handle requests
	 *
	 * @param updateList input list of node addresses that have been updated with "subdivision" or "load/produce" requests
	 * @param numUpdateElems maximum number of elements to process
	 * @param updateMask a unique given type of requests to take into account
	 * @param maxNumElems ...
	 * @param numValidNodes ...
	 * @param gpuPool associated pool (nodes or bricks)
	 *
	 * @return ... 
	 */
	template< typename GPUPoolType >
	uint genericWrite( uint* updateList, uint numUpdateElems, uint updateMask,
		uint maxNumElems, uint numValidNodes, const GPUPoolType& gpuPool );

	/**
	 * ...
	 */
	// Return number of elements not used
	uint updateTimeStamps();

	/**
	 * Clear cache
	 */
	void clearCache();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Associated producer
	 */
	ProviderType* mProvider;

	/******************************** METHODS *********************************/

	/**
	 * Create the list of nodes that will be concerned by the data production management
	 *
	 * @param inputList input list of node addresses that have been updated with "subdivision" or "load/produce" requests
	 * @param inputNumElem maximum number of elements to process
	 * @param testFlag a unique given type of requests to take into account
	 *
	 * @return the number of requests that the manager will have to handle
	 */
	uint createUpdateList( uint* inputList, uint inputNumElem, uint testFlag );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	uint3 _cacheSize;

	/**
	 * ...
	 */
	uint3 _elemsCacheSize;

	/**
	 * Array of time stamps
	 *
	 * For each elements of the cache, it contains an associated time.
	 * During rendering, used elements are flagged with the current frame number.
	 */
	GvCore::Array3DGPULinear< uint >* d_TimeStampArray;

	/**
	 * List of masks of "unused elements" at current time (.i.e. frame)
	 */
	thrust::device_vector< uint >* d_TempMaskList;

	/**
	 * List of masks of "used elements" at current time (.i.e. frame)
	 */
	thrust::device_vector< uint >* d_TempMaskList2; //for cudpp approach

	/**
	 * The final list of elements (nodes or bricks) where data production management
	 * will be able to store its newly produced elements.
	 */
	thrust::device_vector< uint >* d_elemAddressList;
	
	/**
	 * Temporary list used to create the list of elements where to write new elements
	 * of the data production management.
	 * It is associated to [ d_elemAddressList ].
	 */
	thrust::device_vector< uint >* d_elemAddressListTmp;

	/**
	 * Temporary list used to store the masks of nodes whose request corresponds to a given type.
	 * It is associated to [ d_UpdateCompactList ].
	 */
	thrust::device_vector< uint >* d_TempUpdateMaskList;

	/**
	 * The final list of nodes whose request corresponds to a given type
	 *
	 * These will be the nodes concerned by the data production management
	 */
	thrust::device_vector< uint >* d_UpdateCompactList;

	// CUDPP
	/**
	 * ...
	 */
	size_t* d_numElementsPtr;
	
	// CUDPP
	/**
	 * ...
	 */
	CUDPPHandle scanplan;

	/******************************** METHODS *********************************/

};

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename ElementRes, typename ProviderType >
GPUCacheManager< ElementRes, ProviderType >
::~GPUCacheManager()
{
	//CUDA_SAFE_CALL( cudaFree(d_numElementsPtr) );
	//CUDPPResult result = cudppDestroyPlan (scanplan);
	//if (CUDPP_SUCCESS != result) {
	//	printf("Error destroying CUDPPPlan\n");
	//	exit(-1);
	//}
}

/******************************************************************************
 * Update symbols
 ******************************************************************************/
template< typename ElementRes, typename ProviderType >
void GPUCacheManager< ElementRes, ProviderType >
::updateSymbols()
{
	// Update time stamps buffer
	CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_VTC_TimeStampArray, (&d_TimeStampArray->getDeviceArray()),
		sizeof( d_TimeStampArray->getDeviceArray() ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Generate the final list of elements (nodes or bricks) where data production management
 * will be able to store its newly produced elements.
 *
 * Resulting list will be placed in [ d_elemAddressList ]
 ******************************************************************************/
//Return number of elements not used
template< typename ElementRes, typename ProviderType >
uint GPUCacheManager< ElementRes, ProviderType >
::updateTimeStamps()
{
	uint numElemsNotUsed = 0;

	if ( true/* || this->lastNumLoads>0*/ )
	{
 		//uint numElemsNotUsed = 0;
		//uint numElemsNotUsed = 0;

		uint cacheNumElems = getNumElements();

		//std:cout << "manageUpdatesOnly " << (int)manageUpdatesOnly << "\n";

		uint activeNumElems = cacheNumElems;
		//TODO: re-enable manageUpdatesOnly !
		/*if ( manageUpdatesOnly )
			activeNumElems = this->lastNumLoads;*/
		uint inactiveNumElems = cacheNumElems - activeNumElems;

		uint numElemToSort = activeNumElems;
		
		if ( numElemToSort > 0 )
		{
			uint sortingStartPos = 0;

#if GPUCACHE_BENCH_CPULRU==0

			CUDAPM_START_EVENT( gpucache_updateTimeStamps_createMask );

			// Create masks in a single pass
			dim3 blockSize( 64, 1, 1 );
			uint numBlocks = iDivUp( numElemToSort, blockSize.x );
			dim3 gridSize = dim3( std::min( numBlocks, 65535U ) , iDivUp( numBlocks, 65535U ), 1 );
			
			//---------------------------------------------------
			///// TEST Pascal
			//---------------------------------------------------
			numElemToSort -= 1;
			//---------------------------------------------------

			// Generate an error with CUDA 3.2
			CacheManagerFlagTimeStampsSP/*<ElementRes, AddressType>*/
				<<<gridSize, blockSize, 0>>>(/*d_cacheManagerKernel, */numElemToSort,
					thrust::raw_pointer_cast(&(*d_elemAddressList)[sortingStartPos]),
					thrust::raw_pointer_cast(&(*d_TempMaskList)[0]),
					thrust::raw_pointer_cast(&(*d_TempMaskList2)[0]));

			CUT_CHECK_ERROR("CacheManagerFlagTimeStampsSP");
			CUDAPM_STOP_EVENT(gpucache_updateTimeStamps_createMask);

			thrust::device_vector<uint>::const_iterator elemAddressListFirst = d_elemAddressList->begin();
			thrust::device_vector<uint>::const_iterator elemAddressListLast = d_elemAddressList->begin() + numElemToSort;
			thrust::device_vector<uint>::iterator elemAddressListTmpFirst = d_elemAddressListTmp->begin();

			// Stream compaction to collect non-used elements at the beginning
			CUDAPM_START_EVENT(gpucache_updateTimeStamps_threadReduc1);
# if USE_CUDPP_LIBRARY
			cudppCompact (scanplan,
				thrust::raw_pointer_cast(&(*d_elemAddressListTmp)[inactiveNumElems]), d_numElementsPtr,
				thrust::raw_pointer_cast(&(*d_elemAddressList)[sortingStartPos]),
				thrust::raw_pointer_cast(&(*d_TempMaskList)[0]),
				numElemToSort);

			size_t numElemsNotUsedST;
			// Get number of elements
			CUDA_SAFE_CALL( cudaMemcpy( &numElemsNotUsedST, d_numElementsPtr, sizeof(size_t), cudaMemcpyDeviceToHost) );
			numElemsNotUsed=(uint)numElemsNotUsedST + inactiveNumElems;
# else // USE_CUDPP_LIBRARY
			size_t numElemsNotUsedST = thrust::copy_if(
				elemAddressListFirst,
				elemAddressListLast,
				d_TempMaskList->begin(),
				elemAddressListTmpFirst + inactiveNumElems,
				GvCore::not_equal_to_zero<uint>()) - (elemAddressListTmpFirst + inactiveNumElems);

			numElemsNotUsed=(uint)numElemsNotUsedST + inactiveNumElems;
# endif // USE_CUDPP_LIBRARY
			CUDAPM_STOP_EVENT(gpucache_updateTimeStamps_threadReduc1);

			// Stream compaction to collect used elements at the end
			CUDAPM_START_EVENT(gpucache_updateTimeStamps_threadReduc2);
# if USE_CUDPP_LIBRARY
			cudppCompact (scanplan,
				thrust::raw_pointer_cast(&(*d_elemAddressListTmp)[numElemsNotUsed]), d_numElementsPtr,
				thrust::raw_pointer_cast(&(*d_elemAddressList)[sortingStartPos]),
				thrust::raw_pointer_cast(&(*d_TempMaskList2)[0]),
				numElemToSort);
# else // USE_CUDPP_LIBRARY
			thrust::copy_if(
				elemAddressListFirst,
				elemAddressListLast,
				d_TempMaskList2->begin(),
				elemAddressListTmpFirst + numElemsNotUsed,
				GvCore::not_equal_to_zero<uint>());
# endif // USE_CUDPP_LIBRARY
			CUDAPM_STOP_EVENT(gpucache_updateTimeStamps_threadReduc2);
#else

			memcpyArray(cpuTimeStampArray, d_TimeStampArray);
			
			uint curDstPos=0;

			CUDAPM_START_EVENT(gpucachemgr_updateTimeStampsCPU);
			//Copy unused
			for(uint i=0; i<cacheNumElems; ++i){
				uint elemAddressEnc=(*cpuTimeStampsElemAddressList)[i];
				uint3 elemAddress;
				elemAddress=AddressType::unpackAddress(elemAddressEnc);

				if(cpuTimeStampArray->get(elemAddress)!=GPUCacheManager_currentTime){
					(*cpuTimeStampsElemAddressList2)[curDstPos]=elemAddressEnc;
					curDstPos++;
				}
			}
			numElemsNotUsed=curDstPos;

			//copy used
			for(uint i=0; i<cacheNumElems; ++i){
				uint elemAddressEnc=(*cpuTimeStampsElemAddressList)[i];
				uint3 elemAddress;
				elemAddress=AddressType::unpackAddress(elemAddressEnc);

				if(cpuTimeStampArray->get(elemAddress)==GPUCacheManager_currentTime){
					(*cpuTimeStampsElemAddressList2)[curDstPos]=elemAddressEnc;
					curDstPos++;
				}
			}
			CUDAPM_STOP_EVENT(gpucachemgr_updateTimeStampsCPU);

			thrust::copy(cpuTimeStampsElemAddressList2->begin(), cpuTimeStampsElemAddressList2->end(), d_elemAddressListTmp->begin());
#endif
		}
		else
		{ //if( numElemToSort>0 )
			numElemsNotUsed = cacheNumElems;
		}

		////swap buffers////
		thrust::device_vector< uint >* tmpl = d_elemAddressList;
		d_elemAddressList = d_elemAddressListTmp;
		d_elemAddressListTmp = tmpl;

#if CUDAPERFMON_CACHE_INFO==1
		{
			uint *usedPageList=thrust::raw_pointer_cast( &(*d_elemAddressList)[numElemsNotUsed] );

			uint numPageUsed=getNumElements()-numElemsNotUsed;

			if(numPageUsed>0){
				dim3 blockSize(128, 1, 1);
				uint numBlocks=iDivUp(numPageUsed, blockSize.x);
				dim3 gridSize=dim3( std::min( numBlocks, 32768U) , iDivUp(numBlocks,32768U), 1);

				SyntheticInfo_Update_PageUsed< ElementRes, AddressType >
					<<<gridSize, blockSize, 0>>>(
					d_CacheStateBufferArray->getPointer(), numPageUsed, usedPageList, elemsCacheSize);

				CUT_CHECK_ERROR("SyntheticInfo_Update_PageUsed");

				// update counter
				numPagesUsed=numPageUsed;
			}
		}
#endif
		//this->lastNumLoads=0;
	}

	return numElemsNotUsed;
}

/******************************************************************************
 * Clear cache
 ******************************************************************************/
template< typename ElementRes, typename ProviderType >
void GPUCacheManager< ElementRes, ProviderType >
::clearCache()
{
	////Init
	//thrust::host_vector<uint> tmpelemaddress;

	//uint3 pos;
	//for(pos.x=0; pos.x<elemsCacheSize.x; pos.x++){

	//	tmpelemaddress.push_back(pos.x);
	//}

	//thrust::copy(tmpelemaddress.begin()+1, tmpelemaddress.end(), d_TimeStampsElemAddressList->begin());

	//thrust::fill(d_TempMaskList->begin(), d_TempMaskList->end(), (uint) 0);
	//thrust::fill(d_TempMaskList2->begin(), d_TempMaskList2->end(), (uint) 0);

	////thrust::fill(d_TimeStampsElemAddressList2->begin(), d_TimeStampsElemAddressList2->end(), (uint) 0);
	//d_TimeStampArray->fill(0);
}

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "BvhTreeCacheManager.inl"

#endif // !_BVHTREECACHEMANAGER_H_
