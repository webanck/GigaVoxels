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
#include <GvCore/GvError.h>
//#include <GvUtils/GvProxyGeometryHandler.h>
#include <GvCache/GvCacheManagerResources.h>

// Project
#include "VolumeTreeCacheKernel.h"
//#include "CacheManagerKernel.h"
#include "ParticleSystem.h"

// Thrust
#include <thrust/reduce.h>
#include <thrust/functional.h>

// CUDA - NSight
//#define GV_NSIGHT_PROLIFING
#ifdef GV_NSIGHT_PROLIFING
	#include <nvToolsExtCuda.h>
#endif

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * GvKernel_UpdateVBO kernel
 *
 * This kernel update the VBO by dumping all used bricks content (i.e. points)
 *
 * @param pVBO VBO to update
 * @param pNbBricks number of bricks to process
 * @param pBrickAddressList list of brick addresses in cache (used to retrive positions where to fetch data)
 * @param pNbPointsList list of points inside each brick
 * @param pVboIndexOffsetList list of number of points for each used bricks
 * @param pDataStructure data structure in cache where to fecth data
 ******************************************************************************/
__global__
void KERNEL_UpdateVBO_CacheManager( float3* pVBO, const uint pNbPoints, const unsigned int nbFrame )
{
	// Retrieve global data index
	uint lineSize = __uimul( blockDim.x, gridDim.x );
	uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );

	// Check bounds
	if ( elem < pNbPoints )
	{
		const float x = 0.5f + 0.5f * cosf( nbFrame * 0.01f * 2.f * 3.141592f * ( (float)elem / (float)pNbPoints ) );
		const float y = 0.5f + 0.5f * sinf( nbFrame * 0.001f * 2.f * 3.141592f * ( (float)elem / (float)pNbPoints ) );
		const float z = 0.5f + 0.5f * cosf( nbFrame * 0.05f * 2.f * 3.141592f * ( (float)elem / (float)pNbPoints ) );

		// Write to output global memory
		pVBO[ elem ] = make_float3( x, y, z );
	}
}

/******************************************************************************
 * Constructor
 *
 * @param cachesize ...
 * @param pageTableArray ...
 * @param graphicsInteroperability ...
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename TDataStructureType >
CacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType, TDataStructureType >
::CacheManager( uint3 cachesize, PageTableArrayType* pageTableArray, /*GvUtils::GvProxyGeometryHandler* pVBO,*/ uint graphicsInteroperability )
:	GvCache::GvCacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType >( cachesize, pageTableArray, graphicsInteroperability )
,	_dataStructure( NULL )
//,	_vbo( pVBO )
//,	_vbo( NULL )
,	_particleSystem( NULL )
,	_vboScanPlan( 0 )
,	_vboScanPlanSize( 0 )
{
	////****************************************************************************
	//delete _d_TimeStampArray;
	//_d_TimeStampArray = new GvCore::Array3DGPULinear< uint >( make_uint3( 1000000, 1, 1 ) );
	//_d_TimeStampArray->fill( 0 );
	//delete _d_elemAddressList;
	//delete _d_elemAddressListTmp;
	//_d_elemAddressList = new thrust::device_vector< uint >( 1000000 );
	//_d_elemAddressListTmp = new thrust::device_vector< uint >( 1000000 );
	//thrust::host_vector< uint > tmpelemaddress( 1000000 );
	//uint3 pos;
	//uint index = 0;
	//for ( pos.z = 0; pos.z < 100; pos.z++ )
	//for ( pos.y = 0; pos.y < 100; pos.y++ )
	//for ( pos.x = 0; pos.x < 100; pos.x++ )
	//{
	//	tmpelemaddress[ index ] = AddressType::packAddress( pos );
	//	index++;
	//}
	//thrust::copy( tmpelemaddress.begin(), tmpelemaddress.end(), _d_elemAddressList->begin() );
	////delete _d_TempMaskList;
	////delete _d_TempMaskList2;
	//_d_TempMaskList = new thrust::device_vector< uint >( 1000000 );
	//_d_TempMaskList2 = new thrust::device_vector< uint >( 1000000 );
	//thrust::fill( _d_TempMaskList->begin(), _d_TempMaskList->end(), 0 );
	//thrust::fill( _d_TempMaskList2->begin(), _d_TempMaskList2->end(), 0 );
	//// changer la taille pour avoir la taille de 1.000.000
	//// => flaguer les bricks et seulement relire ce tableau => pour la génération du VBO
	//// problème, il faut également avoir le nombre ?? Pour ne lire que les N 1ères brikcs used...
	////****************************************************************************

	_d_vboBrickList = new thrust::device_vector< uint >( this->_numElements );
	
	_d_vboIndexOffsetList = new thrust::device_vector< uint >( this->_numElements );

#if USE_CUDPP_LIBRARY
	const uint3& pageTableRes = this->_d_pageTableArray->getResolution();
	uint cudppNumElem = std::max( pageTableRes.x * pageTableRes.y * pageTableRes.z, this->_numElements );
	_vboScanPlan = getVBOScanPlan( cudppNumElem );
#endif

	//this->_vbo = new GvUtils::GvProxyGeometryHandler();
	//this->_vbo->initialize();
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename TDataStructureType >
CacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType, TDataStructureType >
::~CacheManager()
{
	// TO DO
	// delete objects ...

	//delete _vbo;
}



/******************************************************************************
 * Update the list of available elements according to their timestamps.
 * Unused and recycled elements will be placed first.
 *
 * @param manageUpdatesOnly ...
 *
 * @return the number of available elements
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename TDataStructureType >
inline uint CacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType, TDataStructureType >
::updateVBO( bool manageUpdatesOnly )
{

	CUDAPM_START_EVENT( gpucache_update_VBO );

	uint numElemToSort = this->getNumElements();

	if ( numElemToSort > 0 )
	{
		uint sortingStartPos = 0;

		
		CUDAPM_START_EVENT( gpucache_update_VBO_createMask );

		// Create masks in a single pass
		dim3 blockSize( 64, 1, 1 );
		uint numBlocks = iDivUp( numElemToSort, blockSize.x );
		dim3 gridSize = dim3( std::min( numBlocks, 65535U ) , iDivUp( numBlocks, 65535U ), 1 );

		////-------------------------------------------------------
		//std::cout << "\n----------------------------------- updateVBO - numElemToSort : " << numElemToSort << " -----------------------------------" << std::endl;
		//// DEBUG
		//std::cout << "elemAddressList CONTENT" << std::endl;
		//for ( int i = 0; i < 10; i++ )
		//{
		//	const uint packedElemAddress= (*_d_elemAddressList)[ i ];
		//	const uint3 elemAddress = GvStructure::VolTreeBrickAddress::unpackAddress( packedElemAddress );
		//	std::cout << "_d_elemAddressList[ " << i << " ] = " << (*_d_elemAddressList)[ i ] << " - [ " << elemAddress.x << " " << elemAddress.y << " " << elemAddress.z << " ]" << std::endl;
		//}
		////-------------------------------------------------------
		
		//-------------------------------------------------------------------------------------------------------------
		// This kernel creates the usage mask list of used and non used elements (in current rendering pass) in a single pass
		//
		// Note : Generate an error with CUDA 3.2
		//
		// [ 1 ]
		//
		//GvCache::CacheManagerFlagTimeStampsSP< ElementRes, AddressType >
		//	<<< gridSize, blockSize, 0 >>>( _d_cacheManagerKernel, numElemToSort,
		//	thrust::raw_pointer_cast( &(*_d_elemAddressList)[ sortingStartPos ] ),
		//	thrust::raw_pointer_cast( &(*_d_TempMaskList)[ 0 ] )/*resulting mask list of non-used elements*/,
		//	thrust::raw_pointer_cast( &(*_d_TempMaskList2)[ 0 ] )/*resulting mask list of used elements*/ );
		//GvKernel_FlagTimeStampsSP< ElementRes, AddressType >

#ifdef GV_NSIGHT_PROLIFING
		nvtxRangeId_t idTest_01 = nvtxRangeStartA( "FLAG used-unused nodes" );
#endif

		GvKernel_FlagTimeStampsSP< KernelType >
			<<< gridSize, blockSize, 0 >>>( this->_d_cacheManagerKernel, numElemToSort,
			thrust::raw_pointer_cast( &(*this->_d_elemAddressList)[ sortingStartPos ] ),
			thrust::raw_pointer_cast( &(*this->_d_TempMaskList)[ 0 ] )/*resulting mask list of non-used elements*/,
			thrust::raw_pointer_cast( &(*this->_d_TempMaskList2)[ 0 ] )/*resulting mask list of used elements*/ );
		GV_CHECK_CUDA_ERROR( "CacheManager::CacheManagerFlagTimeStampsSP" );

#ifdef GV_NSIGHT_PROLIFING
		GV_CUDA_SAFE_CALL( cudaDeviceSynchronize() );
		nvtxRangeEnd( idTest_01 );
#endif
		
		CUDAPM_STOP_EVENT( gpucache_update_VBO_createMask );

		CUDAPM_START_EVENT( gpucache_update_VBO_compaction );

		thrust::device_vector< uint >::const_iterator elemAddressListFirst = this->_d_elemAddressList->begin();
		thrust::device_vector< uint >::const_iterator elemAddressListLast = this->_d_elemAddressList->begin() + numElemToSort;
		thrust::device_vector< uint >::iterator elemAddressListTmpFirst = this->_d_elemAddressListTmp->begin();

		//-------------------------------------------------------------------------------------------------------------
		//
		// BEGIN : VBO Generation
		//
		// [ 2 ]
		//
		
#ifdef GV_NSIGHT_PROLIFING
		//nvtxRangeId_t idTest_02 = nvtxRangeStartA( "Compact used nodes" );
		nvtxEventAttributes_t idTest_02_eventAttrib = {0};
		idTest_02_eventAttrib.version = NVTX_VERSION;
		idTest_02_eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
		idTest_02_eventAttrib.colorType = NVTX_COLOR_ARGB;
		idTest_02_eventAttrib.color = 0xFF0000FF;/*COLOR_BLUE*/;
		idTest_02_eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
		idTest_02_eventAttrib.message.ascii = "Compact used nodes";
		nvtxRangeId_t idTest_02 = nvtxRangeStartEx( &idTest_02_eventAttrib );
#endif
		
		cudppCompact( /*handle to CUDPPCompactPlan*/this->_scanplan,
			/* OUT : compacted output */thrust::raw_pointer_cast( &(*this->_d_elemAddressListTmp)[ 0 ] ),
			/* OUT :  number of elements valid flags in the d_isValid input array */this->_d_numElementsPtr,
			/* input to compact */thrust::raw_pointer_cast( &(*this->_d_elemAddressList)[ 0 ] ),
			/* which elements in input are valid */thrust::raw_pointer_cast( &(*this->_d_TempMaskList2)[ 0 ] ),		// masks list of "used" elements
			/* nb of elements in input */numElemToSort );
		
#ifdef GV_NSIGHT_PROLIFING
		GV_CUDA_SAFE_CALL( cudaDeviceSynchronize() );
		nvtxRangeEnd( idTest_02 );
#endif
		
		// Get number of elements
		size_t nbUsedBricks;
		
#ifdef GV_NSIGHT_PROLIFING
		nvtxRangeId_t idTest_03 = nvtxRangeStartA( "Copy nb used nodes" );
#endif
		
		GV_CUDA_SAFE_CALL( cudaMemcpy( &nbUsedBricks, this->_d_numElementsPtr, sizeof( size_t ), cudaMemcpyDeviceToHost ) );
		
#ifdef GV_NSIGHT_PROLIFING
		GV_CUDA_SAFE_CALL( cudaDeviceSynchronize() );
		nvtxRangeEnd( idTest_03 );
#endif

		CUDAPM_STOP_EVENT( gpucache_update_VBO_compaction );
		//--------------
		//
		// END : VBO Generation
		//
		//-------------------------------------------------------------------------------------------------------------


		CUDAPM_START_EVENT( gpucache_update_VBO_nb_pts );
		//-------------------------------------------------------------------------------------------------------------
		//
		// [ 3 ] - Read nb points for each used bricks
		//
		//void GvKernel_ReadVboNbPoints( uint* pNbPointsList, const uint pNbBricks, const uint* pBrickAddressList, TDataStructureKernelType pDataStructure )
		//GvKernel_ReadVboNbPoints< TDataStructureType ><<< gridSize, blockSize, 0 >>>(
		const uint myNbUsedBricks = static_cast< const uint >( nbUsedBricks );
		if ( myNbUsedBricks < 1 )
		{
			return 0;
		}
		//std::cout << "\nNB USED BRICKS : " << myNbUsedBricks << std::endl;

		dim3 blockSizeStep3( 64, 1, 1 );
		uint numBlocksStep3 = iDivUp( myNbUsedBricks, blockSizeStep3.x );
		dim3 gridSizeStep3 = dim3( std::min( numBlocksStep3, 65535U ) , iDivUp( numBlocksStep3, 65535U ), 1 );
		
#ifdef GV_NSIGHT_PROLIFING
		nvtxRangeId_t idTest_04 = nvtxRangeStartA( "Read nb points" );
#endif
		
		GvKernel_ReadVboNbPoints< typename TDataStructureType::VolTreeKernelType ><<< gridSizeStep3, blockSizeStep3, 0 >>>(
			/*OUTPUT*/static_cast< uint* >( thrust::raw_pointer_cast( &(*_d_vboBrickList)[ 0 ] ) ),	// number of points for each used brick
			///*IN*/static_cast< const uint >( nbUsedBricks ),
			myNbUsedBricks,
			/*IN*/static_cast< const uint* >( thrust::raw_pointer_cast( &(*this->_d_elemAddressListTmp)[ 0 ] ) ),
			/*IN*/_dataStructure->volumeTreeKernel );
		GV_CHECK_CUDA_ERROR( "CacheManager::GvKernel_ReadVboNbPoints" );
		
#ifdef GV_NSIGHT_PROLIFING
		GV_CUDA_SAFE_CALL( cudaDeviceSynchronize() );
		nvtxRangeEnd( idTest_04 );
#endif
		
		//-------------------------------------------------------------------------------------------------------------

		//-------------------------------------------------------------------------------------------------------------
		// Intermediate : compute total number of points
		//thrust::device_vector< uint >* _d_vboBrickList;
		
#ifdef GV_NSIGHT_PROLIFING
		nvtxRangeId_t idTest_05 = nvtxRangeStartA( "Total nb points" );
#endif
		
		uint sum = thrust::reduce( (*_d_vboBrickList).begin(), (*_d_vboBrickList).begin() + myNbUsedBricks, static_cast< uint >( 0 ), thrust::plus< uint >() );
		//cudppReduce (CUDPP_ADD,&sum, thrust::raw_pointer_cast(&_d_vboBrickList[0]),myNbUsedBricks);

#ifdef GV_NSIGHT_PROLIFING
		GV_CUDA_SAFE_CALL( cudaDeviceSynchronize() );
		nvtxRangeEnd( idTest_05 );
#endif
		
		//std::cout << "TOTAL number of points : " << sum << std::endl;
		// update VBO with this value to render it !!!
		//_vbo->_nbPoints = sum;
		_particleSystem->_nbRenderablePoints = sum;
		
		///////////////////////////////////////////////////////
		// TEST : 
		///////////////////////////////////////////////////////
		/*if ( sum > 9500000 )
		{
			std::cout << "TOTAL number of points : " << sum << std::endl;
		}*/
		if ( sum > 999999 )
		{
			_particleSystem->_nbRenderablePoints = 999999;
		}

		CUDAPM_STOP_EVENT( gpucache_update_VBO_nb_pts );

		//-------------------------------------------------------------------------------------------------------------

		//-------------------------------------------------------------------------------------------------------------
		//
		// [ 4 ] - Parallel prefix sum
		//
		// Generate an array containing, for each brick, the index offset to use in order to dump points in the VBO
		//

		CUDAPM_START_EVENT( gpucache_update_VBO_parallel_prefix_sum );
		
#ifdef GV_NSIGHT_PROLIFING
		nvtxRangeId_t idTest_06 = nvtxRangeStartA( "Scan VBO offset" );
#endif
		
		cudppScan( /*handle to CUDPPCompactPlan*/_vboScanPlan,
			/* OUT : compacted output */thrust::raw_pointer_cast( &(*_d_vboIndexOffsetList)[ 0 ] ),
			/* input to scan */thrust::raw_pointer_cast( &(*_d_vboBrickList)[ 0 ] ),
			/* nb of elements in input */nbUsedBricks );
		GV_CHECK_CUDA_ERROR( "CacheManager::cudppScan" );

#ifdef GV_NSIGHT_PROLIFING
		GV_CUDA_SAFE_CALL( cudaDeviceSynchronize() );
		nvtxRangeEnd( idTest_06 );
#endif

		//// DEBUG
		//std::cout << "- number of points for each used brick" << std::endl;
		//for ( int i = 0; i < nbUsedBricks; i++ )
		//{
		//	std::cout << "_d_vboBrickList[ " << i << " ] = " << (*_d_vboBrickList)[ i ] << std::endl;
		//}
		//std::cout << "- for each used brick, the index offset to use in order to dump points in the VBO" << std::endl;
		//for ( int i = 0; i < nbUsedBricks; i++ )
		//{
		//	std::cout << "_d_vboIndexOffsetList[ " << i << " ] = " << (*_d_vboIndexOffsetList)[ i ] << std::endl;
		//}
		//-------------------------------------------------------------------------------------------------------------
		CUDAPM_STOP_EVENT( gpucache_update_VBO_parallel_prefix_sum );
		CUDAPM_START_EVENT( gpucache_update_VBO_update_VBO );


		//-------------------------------------------------------------------------------------------------------------
		//
		// [ 5 ] - Update VBO content (memory dump)
		//
		// Map graphics resource
		//cudaStream_t stream = 0;
		//cudaError_t error = cudaGraphicsMapResources( 1, &_vbo->_d_vertices, stream );
		cudaError_t error = _particleSystem->getGraphicsResource()->map();
		assert( error == cudaSuccess );
		// Get graphics resource's mapped address
		float3* vboDevicePointer = NULL;
		//size_t size = 0;
		//error = cudaGraphicsResourceGetMappedPointer( (void**)&vboDevicePointer, &size, _vbo->_d_vertices );
		vboDevicePointer = reinterpret_cast< float3* >( _particleSystem->getGraphicsResource()->getMappedAddress() );
		//assert( error == cudaSuccess );
		// Update VBO
		//GvKernel_UpdateVBO< TDataStructureType::VolTreeKernelType ><<< gridSize, blockSize, 0 >>>(

#ifdef GV_NSIGHT_PROLIFING
		nvtxRangeId_t idTest_07 = nvtxRangeStartA( "Fill VBO" );
#endif
		// OPTIM --------------------
		//printf("%u\n",sum);
		if  (sum!=0) 
		{
			

			dim3 blockSizeStep4( 256, 1, 1 );
			uint numBlocksStep4 = iDivUp( sum, blockSizeStep4.x );
			dim3 gridSizeStep4 = dim3( std::min( numBlocksStep4, 65535U ) , 1, 1 );
			// --------------------------

			GvKernel_UpdateVBO< typename TDataStructureType::VolTreeKernelType ><<< gridSizeStep4, blockSizeStep4, 0 >>>(
				vboDevicePointer,	// VBO to update
				/*IN*/nbUsedBricks,
				/*IN*/thrust::raw_pointer_cast( &(*this->_d_elemAddressListTmp)[ 0 ] ), // addresses of each used bricks
				/*IN*/thrust::raw_pointer_cast( &(*_d_vboBrickList)[ 0 ] ),	// number of points for each used brick
				/*IN*/thrust::raw_pointer_cast( &(*_d_vboIndexOffsetList)[ 0 ] ),
				/*IN*/_dataStructure->volumeTreeKernel );	// data structure to fetch data
			GV_CHECK_CUDA_ERROR( "CacheManager::GvKernel_UpdateVBO" );
		}
#ifdef GV_NSIGHT_PROLIFING
		GV_CUDA_SAFE_CALL( cudaDeviceSynchronize() );
		nvtxRangeEnd( idTest_07 );
#endif

		// Unmap graphics resource
		//error = cudaGraphicsUnmapResources( 1, &_vbo->_d_vertices, stream );
		error = _particleSystem->getGraphicsResource()->unmap();
		assert( error == cudaSuccess );
		CUDAPM_STOP_EVENT( gpucache_update_VBO_update_VBO );
		////-------------------------------------------------------------------------------------------------------------

		//// Render VBO
		//_vbo->render();
		
		//	// --------------------------------------------------------------
	//	// DEBUG vbo
	//	//
	//	cudaDeviceSynchronize();
	//	//
	//	glBindBuffer( GL_ARRAY_BUFFER, _vbo->_vertexBuffer );
	//	GLfloat* vertexBufferData = static_cast< GLfloat* >( glMapBuffer( GL_ARRAY_BUFFER, GL_READ_ONLY ) );
	//	if ( vertexBufferData != NULL )
	//	{
	//		int j = 0;
	//		for ( int i = 0; i < _vbo->_nbPoints; i++ )
	//		{
	//			//printf( "\nvertexBufferData[ %d ] = ( %f, %f, %f, %f )", i, vertexBufferData[ j ], vertexBufferData[ j + 1 ], vertexBufferData[ j + 2 ], vertexBufferData[ j + 3 ] );
	//			//j += 4;
	//			printf( "\nvertexBufferData[ %d ] = ( %f, %f, %f )", i, vertexBufferData[ j ], vertexBufferData[ j + 1 ], vertexBufferData[ j + 2 ] );
	//			j += 3;
	//		}
	//	}
	//	glBindBuffer( GL_ARRAY_BUFFER, 0 );
	//// --------------------------------------------------------------
		
		//-------------------------------------------------------------------------------------------------------------
		//
		// [ 6 ] - Render VBO
		//
		//_vbo->render();
		//-------------------------------------------------------------------------------------------------------------

		////-------------------------------------------------------------------------------------------------------------
		//static unsigned int nbFrame = 0;
		//// Update VBO content
		//// Map graphics resource
		////cudaStream_t stream = 0;
		///*cudaError_t*/ error = cudaGraphicsMapResources( 1, &_vbo->_d_vertices, stream );
		//assert( error == cudaSuccess );
		//// Get graphics resource's mapped address
		///*float3* */vboDevicePointer = NULL;
		///*size_t */size = 0;
		//error = cudaGraphicsResourceGetMappedPointer( (void**)&vboDevicePointer, &size, _vbo->_d_vertices );
		//assert( error == cudaSuccess );
		//// Update VBO
		//gridSize = dim3 ( 1, 1, 1 );
		//blockSize = dim3( 1024, 1, 1 );
		//KERNEL_UpdateVBO_CacheManager<<< gridSize, blockSize >>>( vboDevicePointer, 1000, nbFrame );
		//GV_CHECK_CUDA_ERROR( "KERNEL_UpdateVBO_CacheManager" );
		//// Unmap graphics resource
		//error = cudaGraphicsUnmapResources( 1, &_vbo->_d_vertices, stream );
		//assert( error == cudaSuccess );
		//// Render VBO
		//_vbo->_nbPoints = 1000;
		//_vbo->render();
		//// Update frame nb
		//nbFrame++;
		////-------------------------------------------------------------------------------------------------------------

		// Swap buffers
	//	thrust::device_vector< uint >* tmpl = _d_elemAddressList;
	//	_d_elemAddressList = _d_elemAddressListTmp;
	//	_d_elemAddressListTmp = tmpl;
	}

	CUDAPM_STOP_EVENT( gpucache_update_VBO );
	return 0;
}


/******************************************************************************
 * Get a CUDPP plan given a number of elements to be processed.
 *
 * @param pSize The maximum number of elements to be processed
 *
 * @return a handle on the plan
 ******************************************************************************/
template< unsigned int TId, typename ElementRes, typename AddressType, typename PageTableArrayType, typename PageTableType, typename TDataStructureType >
inline CUDPPHandle CacheManager< TId, ElementRes, AddressType, PageTableArrayType, PageTableType, TDataStructureType >
::getVBOScanPlan( uint pSize )
{
	if ( _vboScanPlanSize < pSize )
	{
		if ( _vboScanPlanSize > 0 )
		{
			if ( cudppDestroyPlan( _vboScanPlan ) != CUDPP_SUCCESS )
			{
				printf( "warning: cudppDestroyPlan() failed!\n ");
			}
			
		/*	if ( cudppDestroy( _cudppLibrary ) != CUDPP_SUCCESS )
			{
				printf( "warning: cudppDestroy() failed!\n" );
			}*/

			_vboScanPlanSize = 0;
		}

		// Creates an instance of the CUDPP library, and returns a handle.
		//cudppCreate( &_cudppLibrary );	// pas bon !!!!!!!!!!!!!!!!

		// Create a CUDPP plan.
		//
		// A plan is a data structure containing state and intermediate storage space that CUDPP uses to execute algorithms on data.
		// A plan is created by passing to cudppPlan() a CUDPPConfiguration that specifies the algorithm, operator, datatype, and options.
		// The size of the data must also be passed to cudppPlan(), in the numElements, numRows, and rowPitch arguments.
		// These sizes are used to allocate internal storage space at the time the plan is created.
		// The CUDPP planner may use the sizes, options, and information about the present hardware to choose optimal settings.
		// Note that numElements is the maximum size of the array to be processed with this plan.
		// That means that a plan may be re-used to process (for example, to sort or scan) smaller arrays.
		//
		// ---- configuration struct specifying algorithm and options 
		CUDPPConfiguration config;
		config.op = CUDPP_ADD;
		config.datatype = CUDPP_UINT;
		config.algorithm = CUDPP_SCAN;
		config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
		// ---- pointer to an opaque handle to the internal plan
		_vboScanPlan = 0;
		// ---- create the CUDPP plan.
		CUDPPResult result = cudppPlan( GvCache::GvCacheManagerResources::getCudppLibrary(), &_vboScanPlan,
										config,
										pSize,		// The maximum number of elements to be processed
										1,			// The number of rows (for 2D operations) to be processed
										0 );		// The pitch of the rows of input data, in elements

		if ( CUDPP_SUCCESS != result )
		{
			printf( "Error creating VBO CUDPPPlan\n" );
			exit( -1 );			// TO DO : remove this exit and use exception ?
		}

		_vboScanPlanSize = pSize;
	}

	return _vboScanPlan;
}
