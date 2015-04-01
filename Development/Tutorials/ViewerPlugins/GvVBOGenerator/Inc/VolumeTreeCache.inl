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

//#ifndef _VOLUME_TREE_CACHE_INL_
//#define _VOLUME_TREE_CACHE_INL_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda SDK
#include <helper_math.h>

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvPerfMon/GvPerformanceMonitor.h"
#include "GvCore/functional_ext.h"
#include "GvCore/GvError.h"
//#include <GvUtils/GvProxyGeometryHandler.h>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

	/******************************************************************************
	 * Constructor
	 *
	 * @param pDataStructure a pointer to the data structure.
	 * @param gpuprod a pointer to the user's producer.
	 * @param nodepoolres the 3d size of the node pool.
	 * @param brickpoolres the 3d size of the brick pool.
	 * @param graphicsInteroperability ...
	 ******************************************************************************/
	template< typename TDataStructureType >
	VolumeTreeCache< TDataStructureType >
	::VolumeTreeCache( TDataStructureType* pDataStructure, uint3 nodepoolres, uint3 brickpoolres/*, GvUtils::GvProxyGeometryHandler* _vbo*/, uint graphicsInteroperability )
	:	GvStructure::GvDataProductionManager< TDataStructureType >( pDataStructure, nodepoolres, brickpoolres, graphicsInteroperability )
	,	_vboCacheManager( NULL )
	//,	_vbo( NULL )
	{
		// Cache managers creation : nodes and bricks
		_vboCacheManager = new VBOCacheManagerType( this->_brickPoolRes, this->_dataStructure->_dataArray/*, _vbo*/, graphicsInteroperability );

		// TEST VBO ----
		//_vbo = _vboCacheManager->_vbo;
		//--------------
		
		// The creation of the localization arrays should be moved here!
		_vboCacheManager->_pageTable->locCodeArray = this->_dataStructure->_localizationCodeArray;
		_vboCacheManager->_pageTable->locDepthArray = this->_dataStructure->_localizationDepthArray;
		_vboCacheManager->_pageTable->getKernel().childArray = this->_dataStructure->_childArray->getDeviceArray();
		_vboCacheManager->_pageTable->getKernel().dataArray = this->_dataStructure->_dataArray->getDeviceArray();
		_vboCacheManager->_pageTable->getKernel().locCodeArray = this->_dataStructure->_localizationCodeArray->getDeviceArray();
		_vboCacheManager->_pageTable->getKernel().locDepthArray = this->_dataStructure->_localizationDepthArray->getDeviceArray();
		_vboCacheManager->_totalNumLoads = 0;
		_vboCacheManager->_lastNumLoads = 0;
		_vboCacheManager->_dataStructure = this->_dataStructure;

		_vboVolumeTreeCacheKernel._updateBufferArray = this->_dataProductionManagerKernel._updateBufferArray;
		_vboVolumeTreeCacheKernel._nodeCacheManager = this->_dataProductionManagerKernel._nodeCacheManager;
		_vboVolumeTreeCacheKernel._brickCacheManager = this->_dataProductionManagerKernel._brickCacheManager;
		_vboVolumeTreeCacheKernel._vboCacheManager = this->_vboCacheManager->getKernelObject();
	}

	/******************************************************************************
	 * Destructor
	 ******************************************************************************/
	template< typename TDataStructureType >
	VolumeTreeCache< TDataStructureType >
	::~VolumeTreeCache()
	{
		// Delete cache manager (nodes and bricks)
		delete _vboCacheManager;
	}

	/******************************************************************************
	 * This method is called before the rendering process. We just clear the request buffer.
	 ******************************************************************************/
	template< typename TDataStructureType >
	void VolumeTreeCache< TDataStructureType >
	::preRenderPass()
	{
		CUDAPM_START_EVENT( gpucache_preRenderPass );

		// Clear subdiv pool
		this->_updateBufferArray->fill( 0 );

		// Number of requests cache has handled
		this->_nbNodeSubdivisionRequests = 0;
		this->_nbBrickLoadRequests = 0;

	#if CUDAPERFMON_CACHE_INFO==1
		_nodesCacheManager->_d_CacheStateBufferArray->fill( 0 );
		_nodesCacheManager->_numPagesUsed = 0;
		_nodesCacheManager->_numPagesWrited = 0;

		_bricksCacheManager->_d_CacheStateBufferArray->fill( 0 );
		_bricksCacheManager->_numPagesUsed = 0;
		_bricksCacheManager->_numPagesWrited = 0;
	#endif

		//---------------------------------------
		// TO DO : clear vbo used flags ?
		//_vboCacheManager->
		//---------------------------------------

		CUDAPM_STOP_EVENT( gpucache_preRenderPass );
	}

	/******************************************************************************
	 * This method is called after the rendering process. She's responsible for processing requests.
	 *
	 * @return the number of requests processed.
	 ******************************************************************************/
	template< typename TDataStructureType >
	uint VolumeTreeCache< TDataStructureType >
	::handleRequests()
	{
		// Generate the requests buffer
		//
		// Collect and compact update informations for both nodes and bricks
		CUDAPM_START_EVENT( dataProduction_manageRequests );
		uint nbRequests = this->manageUpdates();
		CUDAPM_STOP_EVENT( dataProduction_manageRequests );

		// Stop post-render pass if no request
	//	if ( nbRequests > 0 )
	//	{
			// Update time stamps
			CUDAPM_START_EVENT( cache_updateTimestamps );
			this->updateTimeStamps();
			CUDAPM_STOP_EVENT( cache_updateTimestamps );

			//-------------------------------------------------------------------------------------------------------------
			//
			// BEGIN : VBO Generation
			//
			this->_numBricksNotInUse = _vboCacheManager->updateVBO( this->_intraFramePass );
			//
			// END : VBO Generation
			//
			//-------------------------------------------------------------------------------------------------------------

			// Handle requests :

			// [ 1 ] - Handle the "subdivide nodes" requests
			CUDAPM_START_EVENT( producer_nodes );
			//uint numSubDiv = manageSubDivisions( nbRequests );
			this->_nbNodeSubdivisionRequests = this->manageSubDivisions( nbRequests );
			CUDAPM_STOP_EVENT( producer_nodes );

			//  [ 2 ] - Handle the "load/produce bricks" requests
			CUDAPM_START_EVENT( producer_bricks );
			//if ( numSubDiv < nbRequests )
			if ( this->_nbNodeSubdivisionRequests < nbRequests )
			{
				this->_nbBrickLoadRequests = this->manageDataLoadGPUProd( nbRequests );
			}
			CUDAPM_STOP_EVENT( producer_bricks );
		//}

		return nbRequests;
	}

	/******************************************************************************
	 * This method destroy the current N-tree and clear the caches.
	 ******************************************************************************/
	template< typename TDataStructureType >
	void VolumeTreeCache< TDataStructureType >
	::clearCache()
	{
		// Launch Kernel
		dim3 blockSize( 32, 1, 1 );
		dim3 gridSize( 1, 1, 1 );
		GvStructure::ClearVolTreeRoot<<< gridSize, blockSize >>>( this->_dataStructure->volumeTreeKernel, NodeTileRes::getNumElements() );

		GV_CHECK_CUDA_ERROR( "ClearVolTreeRoot" );

		// Reset nodes cache manager
		this->_nodesCacheManager->clearCache();
		this->_nodesCacheManager->_totalNumLoads = 2;
		this->_nodesCacheManager->_lastNumLoads = 1;

		// Reset bricks cache manager
		this->_bricksCacheManager->clearCache();
		this->_bricksCacheManager->_totalNumLoads = 0;
		this->_bricksCacheManager->_lastNumLoads = 0;

		// VBO
		_vboCacheManager->clearCache();
		_vboCacheManager->_totalNumLoads = 0;
		_vboCacheManager->_lastNumLoads = 0;
	}

	/******************************************************************************
	 * Get the associated device-side object
	 *
	 * @return The device-side object
	 ******************************************************************************/
	template< typename TDataStructureType >
	inline VolumeTreeCache< TDataStructureType >
	::VBOVolumeTreeCacheKernelType VolumeTreeCache< TDataStructureType >::getVBOKernelObject() const
	{
		return _vboVolumeTreeCacheKernel;
	}

	/******************************************************************************
	 * Get the bricks cache manager
	 *
	 * @return the bricks cache manager
	 ******************************************************************************/
	template< typename TDataStructureType >
	inline const VolumeTreeCache< TDataStructureType >::VBOCacheManagerType*
	VolumeTreeCache< TDataStructureType >::getVBOCacheManager() const
	{
		return _vboCacheManager;
	}

	/******************************************************************************
	 * Get the bricks cache manager
	 *
	 * @return the bricks cache manager
	 ******************************************************************************/
	template< typename TDataStructureType >
	inline VolumeTreeCache< TDataStructureType >::VBOCacheManagerType*
	VolumeTreeCache< TDataStructureType >::editVBOCacheManager()
	{
		return _vboCacheManager;
	}
