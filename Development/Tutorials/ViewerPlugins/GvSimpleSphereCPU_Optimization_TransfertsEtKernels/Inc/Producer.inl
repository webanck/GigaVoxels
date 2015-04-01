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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GvIProviderKernel.h>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor.
 * Initialize all buffers.
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
Producer< TDataStructureType, TDataProductionManager >
::Producer()
:	GvUtils::GvSimpleHostProducer< ProducerKernel< TDataStructureType >, TDataStructureType, TDataProductionManager >()
,	requestListCode( NULL )
,	requestListDepth( NULL )
,	requestListCodeDevice( NULL )
,	requestListDepthDevice( NULL )
,	nodesBuffer( NULL )
,	bricksPool( NULL )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
Producer< TDataStructureType, TDataProductionManager >
::~Producer()
{
	finalize();
}

/******************************************************************************
 * Initialize
 *
 * @param pDataStructure data structure
 * @param pDataProductionManager data production manager
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::initialize( TDataStructureType* pDataStructure, TDataProductionManager* pDataProductionManager )
{
	// Call parent class
	GvUtils::GvSimpleHostProducer< ProducerKernel< TDataStructureType >, TDataStructureType, TDataProductionManager >::initialize( pDataStructure, pDataProductionManager );

	// This two buffers will contains the localization and the depth of the requested elements.
	/*requestListCode = new GvCore::GvLocalizationInfo::CodeType[ nbMaxRequests ];
	requestListDepth = new GvCore::GvLocalizationInfo::DepthType[ nbMaxRequests ];*/
	requestListCode = new GvCore::GvLocalizationInfo::CodeType[ pDataProductionManager->getMaxNbNodeSubdivisions() ];
	requestListDepth = new GvCore::GvLocalizationInfo::DepthType[ pDataProductionManager->getMaxNbNodeSubdivisions() ];
	requestListCode2 = new GvCore::GvLocalizationInfo::CodeType[ pDataProductionManager->getMaxNbBrickLoads() ];
	requestListDepth2 = new GvCore::GvLocalizationInfo::DepthType[ pDataProductionManager->getMaxNbBrickLoads() ];

	// The following two buffers are their equivalents on GPU
	/*requestListCodeDevice = new thrust::device_vector< GvCore::GvLocalizationInfo::CodeType >( nbMaxRequests );
	requestListDepthDevice = new thrust::device_vector< GvCore::GvLocalizationInfo::DepthType >( nbMaxRequests );*/
	requestListCodeDevice = new thrust::device_vector< GvCore::GvLocalizationInfo::CodeType >( pDataProductionManager->getMaxNbNodeSubdivisions() );
	requestListDepthDevice = new thrust::device_vector< GvCore::GvLocalizationInfo::DepthType >( pDataProductionManager->getMaxNbNodeSubdivisions() );
	requestListCodeDevice2 = new thrust::device_vector< GvCore::GvLocalizationInfo::CodeType >( pDataProductionManager->getMaxNbBrickLoads() );
	requestListDepthDevice2 = new thrust::device_vector< GvCore::GvLocalizationInfo::DepthType >( pDataProductionManager->getMaxNbBrickLoads() );

	// This 1D array will contain the nodes produced by the CPU
	//
	// Note : memory is mapped
	nodesBuffer = new GvCore::Array3D< uint >( make_uint3( nbMaxRequests * NodeRes::numElements, 1, 1 ), 2 );

	// This 1D pool will contain an array for each voxel's field
	//
	// Fields have defined as color and normal (see SampleCore.h file): typedef Loki::TL::MakeTypelist< uchar4, half4 >::Result DataType;
	//
	// Note : memory is mapped
	size_t voxelSize = GvCore::DataTotalChannelSize< DataTList >::value;
	size_t brickSize = voxelSize * static_cast< size_t >( 1000 );
	bricksPool = new BricksPool( make_uint3( nbMaxRequests * brickSize, 1, 1 ), 2 );

	//////////////////////////////////////////////////
	_data1 = new uchar4[ nbMaxRequests * brickSize ];
	_data2 = new half4[ nbMaxRequests * brickSize ];
	//////////////////////////////////////////////////

	// GPU producer initialization
	// It copies references to the nodes buffer and the bricks pool GPU equivalents
	this->_kernelProducer.init( nodesBuffer->getDeviceArray(), bricksPool->getKernelPool() );
}

/******************************************************************************
 * Finalize
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::finalize()
{
	delete[] requestListCode;
	delete[] requestListDepth;

	delete requestListCodeDevice;
	delete requestListDepthDevice;

	delete nodesBuffer;
	delete bricksPool;
}

/******************************************************************************
 * Implement the produceData method for the channel 0 (nodes).
 * This method is called by the cache manager when you have to produce data for a given pool.
 *
 * @param pNumElems the number of elements you have to produce.
 * @param nodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
 * @param gpuPool the pool for which we need to produce elements.
 * @param pageTable the page table associated to the pool
 * @param Loki::Int2Type< 0 > id of the channel
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::produceData( uint pNumElems,
				thrust::device_vector< uint >* pNodesAddressCompactList,
				thrust::device_vector< uint >* pElemAddressCompactList,
				Loki::Int2Type< 0 > )
{
	// Wrap kernel producer.
	// This is mandatory when calling cache helper to write into cache (see below).
	GvCore::GvIProviderKernel< 0, KernelProducerType > kernelProvider( this->_kernelProducer );

	// Define kernel block size
	//const uint3 kernelBlockSize = KernelProducerType::NodesKernelBlockSize::get();
	//const dim3 blockSize( kernelBlockSize.x, kernelBlockSize.y, kernelBlockSize.z );
	const dim3 blockSize( 32, 1, 1 );

	// Retrieve raw pointers from device_vectors
	uint* nodesAddressList = thrust::raw_pointer_cast( &(*pNodesAddressCompactList)[0] );
	uint* elemAddressList = thrust::raw_pointer_cast( &(*pElemAddressCompactList)[0] );

	// Retrieve raw pointers from device_vectors
	GvCore::GvLocalizationInfo::CodeType* locCodeList = thrust::raw_pointer_cast( &(*requestListCodeDevice)[0] );
	GvCore::GvLocalizationInfo::DepthType* locDepthList = thrust::raw_pointer_cast( &(*requestListDepthDevice)[0] );

	//---------------------------------------------------------------
	if ( pNumElems > 0 )
	{
		// Create the lists containing localization and depth of each element.
		this->_nodePageTable->createLocalizationLists( pNumElems, nodesAddressList, requestListCodeDevice, requestListDepthDevice );
		// Retrieve localization info lists from device to host
		cudaMemcpyAsync( requestListCode, locCodeList, pNumElems * sizeof( GvCore::GvLocalizationInfo::CodeType ), cudaMemcpyDeviceToHost );
		cudaMemcpy( requestListDepth, locDepthList, pNumElems * sizeof( GvCore::GvLocalizationInfo::DepthType ), cudaMemcpyDeviceToHost );
	}
	//---------------------------------------------------------------

	// Iterates through all elements
	while ( pNumElems > 0 )
	{
		uint numRequests = min( pNumElems, nbMaxRequests );

		// Create the lists containing localization and depth of each element.
		//this->_nodePageTable->createLocalizationLists( numRequests, nodesAddressList, requestListCodeDevice, requestListDepthDevice );

		// Produce nodes on host
		// and write resulting nodes info to the memory mapped nodes buffer
		//produceNodes( numRequests, locCodeList, locDepthList );
		produceNodes( numRequests, requestListCode, requestListDepth );

		// Write into cache
		//
		// This stage launches the device producer to write the produced nodes to the node pool
		this->_cacheHelper.template genericWriteIntoCache< NodeTileResLinear >( numRequests, nodesAddressList, elemAddressList, this->_nodePool, kernelProvider, this->_nodePageTable, blockSize );

		// Update
		pNumElems			-= numRequests;
		nodesAddressList	+= numRequests;
		elemAddressList		+= numRequests;
		////////////////////////////
		requestListCode += numRequests;
		requestListDepth += numRequests;
		////////////////////////////
	}
}

/******************************************************************************
 * Implement the produceData method for the channel 1 (bricks).
 * This method is called by the cache manager when you have to produce data for a given pool.
 *
 * @param pNumElems the number of elements you have to produce.
 * @param pNodesAddressCompactList a list containing the addresses of the numElems nodes concerned.
 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
 * @param gpuPool the pool for which we need to produce elements.
 * @param pageTable the page table associated to the pool
 * @param Loki::Int2Type< 1 > id of the channel
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::produceData( uint pNumElems,
				thrust::device_vector< uint >* pNodesAddressCompactList,
				thrust::device_vector< uint >* pElemAddressCompactList,
				Loki::Int2Type< 1 > )
{
	// Wrap kernel producer.
	// This is mandatory when calling cache helper to write into cache (see below).
	GvCore::GvIProviderKernel< 1, KernelProducerType > kernelProvider( this->_kernelProducer );
	
	// Define kernel block size
	/*const uint3 kernelBlockSize = KernelProducerType::BricksKernelBlockSize::get();
	const dim3 blockSize( kernelBlockSize.x, kernelBlockSize.y, kernelBlockSize.z );*/
	const dim3 blockSize( 16, 8, 1 );
	
	// Retrieve raw pointers from device_vectors
	uint* nodesAddressList = thrust::raw_pointer_cast( &(*pNodesAddressCompactList)[0] );
	uint* elemAddressList = thrust::raw_pointer_cast( &(*pElemAddressCompactList)[0] );

	// Retrieve raw pointers from device_vectors
	GvCore::GvLocalizationInfo::CodeType* locCodeList = thrust::raw_pointer_cast( &(*requestListCodeDevice)[0] );
	GvCore::GvLocalizationInfo::DepthType* locDepthList = thrust::raw_pointer_cast( &(*requestListDepthDevice)[0] );
	//---------------------------------------------------------------
	//GvCore::GvLocalizationInfo::CodeType* locCodeList = thrust::raw_pointer_cast( &(*requestListCodeDevice2)[0] );
	//GvCore::GvLocalizationInfo::DepthType* locDepthList = thrust::raw_pointer_cast( &(*requestListDepthDevice2)[0] );
	//---------------------------------------------------------------
	//if ( pNumElems > 0 )
	//{
	//	// Create the lists containing localization and depth of each element.
	//	this->_dataPageTable->createLocalizationLists( pNumElems, nodesAddressList, requestListCodeDevice2, requestListDepthDevice2 );
	//	// Retrieve localization info lists from device to host
	//	cudaMemcpyAsync( requestListCode2, locCodeList, pNumElems * sizeof( GvCore::GvLocalizationInfo::CodeType ), cudaMemcpyDeviceToHost );
	//	cudaMemcpy( requestListDepth2, locDepthList, pNumElems * sizeof( GvCore::GvLocalizationInfo::DepthType ), cudaMemcpyDeviceToHost );
	//}
	//---------------------------------------------------------------

	// Iterates through all elements
	while ( pNumElems > 0 )
	{
		uint numRequests = min( pNumElems, nbMaxRequests );

		// Create the lists containing localization and depth of each element.
		//
		// Once filled, these lists will be passsed to the HOST producer to retrieve localization info of requested elements
		this->_dataPageTable->createLocalizationLists( numRequests, nodesAddressList, requestListCodeDevice, requestListDepthDevice );

		// Produce bricks on host
		// and write resulting brick's voxels data to the memory mapped data buffer
		//
		// We pass the previously filled localization info on device
		produceBricks( numRequests, locCodeList, locDepthList );
		//produceBricks( numRequests, requestListCode2, requestListDepth2 );

		// Write into cache
		//
		// This stage launches the device producer to write the produced bricks to the data pool
		this->_cacheHelper.template genericWriteIntoCache< BrickFullRes >( numRequests, nodesAddressList, elemAddressList, this->_dataPool, kernelProvider, this->_dataPageTable, blockSize );

		// Update
		pNumElems			-= numRequests;
		nodesAddressList	+= numRequests;
		elemAddressList		+= numRequests;
		////////////////////////////
		//requestListCode2 += numRequests;
		//requestListDepth2 += numRequests;
		////////////////////////////
	}
}

/******************************************************************************
 * Test if a point is in the unit sphere centered at [0,0,0]
 *
 * @param pPoint the point to test
 *
 * @return a flag to tell wheter or not the point is in the sphere
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline bool Producer< TDataStructureType, TDataProductionManager >
::isInSphere( const float3& pPoint ) const
{
	if ( dot( pPoint, pPoint ) < 1.0f )
	{
		return true;
	}

	return false;
}

/******************************************************************************
 * Produce nodes
 *
 * Node production is associated to node subdivision to refine data.
 * With the help of an oracle, user has to tell what is inside each subregion
 * of its children.
 *
 * @param pNbElements number of elements to process (i.e. nodes)
 * @param pRequestListCodePtr localization code list on device
 * @param pRequestListDepthPtr localization depth list on device
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::produceNodes( const uint pNbElements, const GvCore::GvLocalizationInfo::CodeType* pRequestListCodePtr, const GvCore::GvLocalizationInfo::DepthType* pRequestListDepthPtr )
{
	// Retrieve localization info lists from device to host
	//cudaMemcpy( requestListCode, pRequestListCodePtr, pNbElements * sizeof( GvCore::GvLocalizationInfo::CodeType ), cudaMemcpyDeviceToHost );
	//cudaMemcpyAsync( requestListCode, pRequestListCodePtr, pNbElements * sizeof( GvCore::GvLocalizationInfo::CodeType ), cudaMemcpyDeviceToHost );
	//cudaMemcpy( requestListDepth, pRequestListDepthPtr, pNbElements * sizeof( GvCore::GvLocalizationInfo::DepthType ), cudaMemcpyDeviceToHost );

	// Iterates through all elements (i.e. nodes)
	for ( uint i = 0; i < pNbElements; ++i )
	{
		// Get current node's localization info
		/*GvCore::GvLocalizationCode::ValueType parentLocCode = requestListCode[ i ].get();
		GvCore::GvLocalizationDepth::ValueType parentLocDepth = requestListDepth[ i ].get();*/
		GvCore::GvLocalizationCode::ValueType parentLocCode = pRequestListCodePtr[ i ].get();
		GvCore::GvLocalizationDepth::ValueType parentLocDepth = pRequestListDepthPtr[ i ].get();

		// To subdivide node and refine data, go to next level of resolution (i.e. its children)
		uint locDepth = parentLocDepth + 1;

		// Get the voxel's resolution at the child level
		uint3 levelRes = getLevelResolution( locDepth );

		// Iterate through current node's children
		uint3 nodeOffset;
		uint nodeOffsetLinear = 0;
		for ( nodeOffset.z = 0; nodeOffset.z < NodeRes::z; ++nodeOffset.z )
		{
			for ( nodeOffset.y = 0; nodeOffset.y < NodeRes::y; ++nodeOffset.y )
			{
				for ( nodeOffset.x = 0; nodeOffset.x < NodeRes::x; ++nodeOffset.x )
				{
					uint3 locCode = parentLocCode * NodeRes::get() + nodeOffset;

					// Convert the localization to a region
					float3 nodePos = make_float3( locCode * BrickRes::get() ) / make_float3( levelRes );
					float3 nodeSize = make_float3( BrickRes::get() ) / make_float3( levelRes );

					// Work in the range [-1.0; 1.0]
					float3 brickPos = 2.0f * nodePos - 1.0f;
					float3 brickSize = 2.0f * nodeSize;

					float3 q000 = brickPos;
					float3 q001 = make_float3( q000.x + brickSize.x,	q000.y,					q000.z );
					float3 q010 = make_float3( q000.x,					q000.y + brickSize.y,	q000.z );
					float3 q011 = make_float3( q000.x + brickSize.x,	q000.y + brickSize.y,	q000.z );
					float3 q100 = make_float3( q000.x,					q000.y,					q000.z + brickSize.z );
					float3 q101 = make_float3( q000.x + brickSize.x,	q000.y,					q000.z + brickSize.z );
					float3 q110 = make_float3( q000.x,					q000.y + brickSize.y,	q000.z + brickSize.z );
					float3 q111 = make_float3( q000.x + brickSize.x,	q000.y + brickSize.y,	q000.z + brickSize.z );

					uint nodeInfo = 0;

					// Check if we are inside the sphere
					if ( isInSphere(q000) || isInSphere(q001) || isInSphere(q010) || isInSphere(q011) ||
						isInSphere(q100) || isInSphere(q101) || isInSphere(q110) || isInSphere(q111) )
					{
						nodeInfo = 0xFFFFFFFF;
					}

					// Write the node info to the memory mapped nodes buffer
					nodesBuffer->get( i * NodeRes::numElements + nodeOffsetLinear ) = nodeInfo;

					nodeOffsetLinear++;
				}
			}
		}
	}
}

/******************************************************************************
 * Produce bricks
 *
 * Brick production is associated to fill brick with voxels.
 *
 * @param pNbElements number of elements to process (i.e. bricks)
 * @param pRequestListCodePtr localization code list on device
 * @param pRequestListDepthPtr localization depth list on device
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::produceBricks( const uint numElements, const GvCore::GvLocalizationInfo::CodeType* pRequestListCodePtr, const GvCore::GvLocalizationInfo::DepthType* pRequestListDepthPtr )
{
	// Retrieve localization info lists from device to host
	//cudaMemcpy( requestListCode, pRequestListCodePtr, numElements * sizeof( GvCore::GvLocalizationInfo::CodeType ), cudaMemcpyDeviceToHost );
	//cudaMemcpyAsync( requestListCode, pRequestListCodePtr, numElements * sizeof( GvCore::GvLocalizationInfo::CodeType ), cudaMemcpyDeviceToHost );
	//cudaMemcpy( requestListDepth, pRequestListDepthPtr, numElements * sizeof( GvCore::GvLocalizationInfo::DepthType ), cudaMemcpyDeviceToHost );

	// Brick's resolution, including the border
	uint3 brickRes = BrickRes::get() + make_uint3( 2 * BorderSize );

	// Iterates through all elements (i.e. bricks)
	for ( uint i = 0; i < numElements; ++i )
	{
		// Get current brick's localization info
		/*GvCore::GvLocalizationCode::ValueType locCode = requestListCode[ i ].get();
		GvCore::GvLocalizationDepth::ValueType locDepth = requestListDepth[ i ].get();*/
		GvCore::GvLocalizationCode::ValueType locCode = pRequestListCodePtr[ i ].get();
		GvCore::GvLocalizationDepth::ValueType locDepth = pRequestListDepthPtr[ i ].get();
	
		// Get the voxel's resolution at the child level
		uint3 levelRes = getLevelResolution( locDepth );
		float3 levelResInv = make_float3( 1.0f ) / make_float3( levelRes );

		// Convert the localization to a region
		float3 nodePos = make_float3( locCode * BrickRes::get() ) * levelResInv;
		float3 nodeSize = make_float3( BrickRes::get() ) * levelResInv;

		// Position of the brick (same as the position of the node minus the border)
		float3 brickPos = nodePos - make_float3( BorderSize ) * levelResInv;

		// Iterate through current brick's voxels
		uint3 brickOffset;
		uint brickOffsetLinear = 0;
		for ( brickOffset.z = 0; brickOffset.z < brickRes.z; ++brickOffset.z )
		{
			for ( brickOffset.y = 0; brickOffset.y < brickRes.y; ++brickOffset.y )
			{
				for ( brickOffset.x = 0; brickOffset.x < brickRes.x; ++brickOffset.x )
				{
					// Position of the current voxel's center (relative to the brick)
					float3 voxelPosInBrick = ( make_float3( brickOffset ) + 0.5f ) / make_float3( levelRes );
					// Position of the current voxel's center (absolute, in [0.0; 1.0] range)
					float3 voxelPosInTree = brickPos + voxelPosInBrick;
					// Position of the current voxel's center (scaled to the range [-1.0; 1.0])
					float3 posF = 2.0f * voxelPosInTree - 1.0f;

					float4 voxelColor = make_float4( 1.0f, 0.0f, 0.0f, 0.0f );
					float4 voxelNormal = make_float4( normalize( posF ), 1.0f );

					// If the voxel is located inside the unit sphere
					if ( isInSphere( posF ) )
					{
						voxelColor.w = 1.0f;
					}

					voxelColor.x *= voxelColor.w;
					voxelColor.y *= voxelColor.w;
					voxelColor.z *= voxelColor.w;

					typedef typename GvCore::DataChannelType< DataTList, 0 >::Result ColorType;
					typedef typename GvCore::DataChannelType< DataTList, 1 >::Result NormalType;

					ColorType color;
					NormalType normal;

					convert_type( voxelColor, color );
					convert_type( voxelNormal, normal );

					// Write the brick's voxel data to the memory mapped data buffer
					//bricksPool->getChannel( Loki::Int2Type< 0 >() )->get( i * 1000 + brickOffsetLinear ) = color;
				//	bricksPool->getChannel( Loki::Int2Type< 1 >() )->get( i * 1000 + brickOffsetLinear ) = normal;
					
					////////////////////////////////////////////
					_data1[ i * 1000 + brickOffsetLinear ] = color;
					_data2[ i * 1000 + brickOffsetLinear ] = normal;
					////////////////////////////////////////////

					brickOffsetLinear++;
				}
			}
		}

		////////////////////////////////////////////
	//	memcpy( bricksPool->getChannel( Loki::Int2Type< 0 >() )->getPointer( i * 1000 ), &_data1[ i * 1000 ], 1000 * sizeof( uchar4 ) );
	//	memcpy( bricksPool->getChannel( Loki::Int2Type< 1 >() )->getPointer( i * 1000 ), &_data2[ i * 1000 ], 1000 * sizeof( half4 ) );
		////////////////////////////////////////////
	}

	////////////////////////////////////////////
	if ( numElements > 0 )
	{
		memcpy( bricksPool->getChannel( Loki::Int2Type< 0 >() )->getPointer(), _data1, numElements * 1000 * sizeof( uchar4 ) );
		memcpy( bricksPool->getChannel( Loki::Int2Type< 1 >() )->getPointer(), _data2, numElements * 1000 * sizeof( half4 ) );
	}
	////////////////////////////////////////////
}

/******************************************************************************
 * Helper function used to retrieve the number of voxels at a given level of resolution
 *
 * @param pLevel level of resolution
 *
 * @return the number of voxels at given level of resolution
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline uint3 Producer< TDataStructureType, TDataProductionManager >
::getLevelResolution( const uint pLevel ) const
{
	return make_uint3( 1 << pLevel ) * BrickRes::get();
}
