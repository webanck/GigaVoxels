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

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor.
 * Initialize all buffers.
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
Producer< DataTList, NodeRes, BrickRes, BorderSize >
::Producer()
{
	// This two buffers will contains the localization and the depth of the requested elements.
	requestListCode = new GvCore::GvLocalizationInfo::CodeType[ nbMaxRequests ];
	requestListDepth = new GvCore::GvLocalizationInfo::DepthType[ nbMaxRequests ];
	// The following two buffers are their equivalents on GPU
	requestListCodeDevice = new thrust::device_vector< GvCore::GvLocalizationInfo::CodeType >( nbMaxRequests );
	requestListDepthDevice = new thrust::device_vector< GvCore::GvLocalizationInfo::DepthType >( nbMaxRequests );

	// This array will contain the nodes produced by the CPU.
	nodesBuffer = new GvCore::Array3D< uint >( make_uint3( nbMaxRequests * NodeRes::numElements, 1, 1 ), 2 );
	// This pool will contain an array for each voxel's field.
	// Fields have defined as color and normal (see SampleCore.h file): typedef Loki::TL::MakeTypelist< uchar4, half4 >::Result DataType;
	size_t voxelSize = GvCore::DataTotalChannelSize< DataTList >::value;
	size_t brickSize = voxelSize * static_cast< size_t >( 1000 );
	bricksPool = new BricksPool( make_uint3( nbMaxRequests * brickSize, 1, 1 ), 2 );

	// GPU producer initialization.
	// It copies references to the nodes buffer and the bricks pool GPU equivalents
	kernelProducer.init( nodesBuffer->getDeviceArray(), bricksPool->getKernelPool() );

	// TEST
	_hasBrickDrawOneSlice = true;
	const unsigned int brickResolution = BrickRes::x;
	for ( unsigned int z = 0; z < brickResolution; z++ )
	{
		for ( unsigned int y = 0; y < brickResolution; y++ )
		{
			for ( unsigned int x = 0; x < brickResolution; x++ )
			{
				_presenceFlags[ x ][ y ][ z ] = 0;
			}
		}
	}
}

/******************************************************************************
 * Implement the produceData method for the channel 0 (nodes).
 * This method is called by the cache manager when you have to produce data for a given pool.
 *
 * @param numElems the number of elements you have to produce.
 * @param nodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
 * @param elemAddressCompactList a list containing numElems addresses where you need to store the result.
 * @param gpuPool the pool for which we need to produce elements.
 * @param pageTable the page table associated to the pool
 * @param Loki::Int2Type< 0 > id of the channel
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
template< typename ElementRes, typename GPUPoolType, typename PageTableType >
inline void Producer< DataTList, NodeRes, BrickRes, BorderSize >
::produceData( uint numElems,
			  thrust::device_vector< uint >* nodesAddressCompactList,
			  thrust::device_vector< uint >* elemAddressCompactList,
			  GPUPoolType& gpuPool, PageTableType pageTable, Loki::Int2Type< 0 > )
{
	// Wrap kernel producer.
	// This is mandatory when calling cache helper to write into cache (see below).
	GvCore::GvIProviderKernel< 0, KernelProducerType > kernelProvider( kernelProducer );
	// Set kernel block dimension (used by cache helper)
	dim3 blockSize( 32, 1, 1 );

	// Retrieve raw pointers from device_vectors
	uint* nodesAddressList = thrust::raw_pointer_cast( &(*nodesAddressCompactList)[0] );
	uint* elemAddressList = thrust::raw_pointer_cast( &(*elemAddressCompactList)[0] );

	// Retrieve raw pointers from device_vectors
	GvCore::GvLocalizationInfo::CodeType* locCodeList = thrust::raw_pointer_cast( &(*requestListCodeDevice)[0] );
	GvCore::GvLocalizationInfo::DepthType* locDepthList = thrust::raw_pointer_cast( &(*requestListDepthDevice)[0] );

	// Iterates through all elements
	while ( numElems > 0 )
	{
		uint numRequests = min( numElems, nbMaxRequests );

		// Create the lists containing localization and depth of each element.
		pageTable->createLocalizationLists( numRequests, nodesAddressList,
			requestListCodeDevice, requestListDepthDevice );

		// Produce nodes
		produceNodes( numRequests, locCodeList, locDepthList );

		// Write into cache
		cacheHelper.genericWriteIntoCache< ElementRes >( numRequests, nodesAddressList,
			elemAddressList, gpuPool, kernelProvider, pageTable, blockSize );

		// Update
		numElems			-= numRequests;
		nodesAddressList	+= numRequests;
		elemAddressList		+= numRequests;
	}
}

/******************************************************************************
 * Implement the produceData method for the channel 1 (bricks).
 * This method is called by the cache manager when you have to produce data for a given pool.
 *
 * @param numElems the number of elements you have to produce.
 * @param nodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
 * @param elemAddressCompactList a list containing numElems addresses where you need to store the result.
 * @param gpuPool the pool for which we need to produce elements.
 * @param pageTable the page table associated to the pool
 * @param Loki::Int2Type< 1 > id of the channel
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
template< typename ElementRes, typename GPUPoolType, typename PageTableType >
inline void Producer< DataTList, NodeRes, BrickRes, BorderSize >
::produceData( uint numElems,
			  thrust::device_vector< uint >* nodesAddressCompactList,
			  thrust::device_vector< uint >* elemAddressCompactList,
			  GPUPoolType& gpuPool, PageTableType pageTable, Loki::Int2Type< 1 > )
{
	// Wrap kernel producer.
	// This is mandatory when calling cache helper to write into cache (see below).
	GvCore::GvIProviderKernel< 1, KernelProducerType > kernelProvider( kernelProducer );
	// Set kernel block dimension (used by cache helper)
	dim3 blockSize( 16, 8, 1 );
	
	// Retrieve raw pointers from device_vectors
	uint* nodesAddressList = thrust::raw_pointer_cast( &(*nodesAddressCompactList)[0] );
	uint* elemAddressList = thrust::raw_pointer_cast( &(*elemAddressCompactList)[0] );

	// Retrieve raw pointers from device_vectors
	GvCore::GvLocalizationInfo::CodeType* locCodeList = thrust::raw_pointer_cast( &(*requestListCodeDevice)[0] );
	GvCore::GvLocalizationInfo::DepthType* locDepthList = thrust::raw_pointer_cast( &(*requestListDepthDevice)[0] );

	// Iterates through all elements
	while ( numElems > 0 )
	{
		uint numRequests = min( numElems, nbMaxRequests );

		// Create the lists containing localization and depth of each element.
		pageTable->createLocalizationLists( numRequests, nodesAddressList,
			requestListCodeDevice, requestListDepthDevice );

		// Produce bricks
		produceBricks( numRequests, locCodeList, locDepthList );

		// Write into cache
		cacheHelper.genericWriteIntoCache< ElementRes >( numRequests, nodesAddressList,
			elemAddressList, gpuPool, kernelProvider, pageTable, blockSize );

		// Update
		numElems			-= numRequests;
		nodesAddressList	+= numRequests;
		elemAddressList		+= numRequests;
	}
}

/******************************************************************************
 * Test if a point is in the unit sphere centered at [0,0,0]
 *
 * @param pPoint the point to test
 *
 * @return a flag to tell wheter or not the point is in the sphere
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
inline bool Producer< DataTList, NodeRes, BrickRes, BorderSize >
::isInSphere( const float3& pPoint ) const
{
	if
		( dot( pPoint, pPoint ) < 1.0f )
	{
		return true;
	}

	return false;
}

/******************************************************************************
 * ...
 *
 * @param numElements ...
 * @param requestListCodePtr ...
 * @param requestListDepthPtr ...
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
inline void Producer< DataTList, NodeRes, BrickRes, BorderSize >
::produceNodes( uint numElements, GvCore::GvLocalizationInfo::CodeType* requestListCodePtr, GvCore::GvLocalizationInfo::DepthType* requestListDepthPtr )
{
	// Copy lists to the CPU
	cudaMemcpy( requestListCode, requestListCodePtr, numElements * sizeof( GvCore::GvLocalizationInfo::CodeType ), cudaMemcpyDeviceToHost );
	cudaMemcpy( requestListDepth, requestListDepthPtr, numElements * sizeof( GvCore::GvLocalizationInfo::DepthType ), cudaMemcpyDeviceToHost );

	// Iterates through all elements
	for ( uint i = 0; i < numElements; ++i )
	{
		GvCore::GvLocalizationCode::ValueType parentLocCode = requestListCode[ i ].get();
		GvCore::GvLocalizationDepth::ValueType parentLocDepth = requestListDepth[ i ].get();

		uint locDepth = parentLocDepth + 1;

		// Get the voxel's resolution at the child level
		uint3 levelRes = getLevelResolution( locDepth );

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
					float3 nodePos = make_float3( locCode * BrickRes::get()) / make_float3( levelRes );
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

					//// Check if we are inside the sphere
					//if ( isInSphere(q000) || isInSphere(q001) || isInSphere(q010) || isInSphere(q011) ||
					//	isInSphere(q100) || isInSphere(q101) || isInSphere(q110) || isInSphere(q111) )
					//{
					//	nodeInfo = 0xFFFFFFFF;
					//}

					if ( _hasBrickDrawOneSlice )
					{
						if ( locCode.z == 0 )
						{
							nodeInfo = 0xFFFFFFFF;
						}
					}
					else
					{
						nodeInfo = 0xFFFFFFFF;
					}

					// Write the current node to the buffer
					nodesBuffer->get( i * NodeRes::numElements + nodeOffsetLinear ) = nodeInfo;

					nodeOffsetLinear++;
				}
			}
		}
	}
}

/******************************************************************************
 * ...
 *
 * @param numElements ...
 * @param requestListCodePtr ...
 * @param requestListDepthPtr ...
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
inline void Producer< DataTList, NodeRes, BrickRes, BorderSize >
::produceBricks( uint numElements, GvCore::GvLocalizationInfo::CodeType* requestListCodePtr, GvCore::GvLocalizationInfo::DepthType* requestListDepthPtr )
{
	// Copy lists to the CPU
	cudaMemcpy( requestListCode, requestListCodePtr, numElements * sizeof( GvCore::GvLocalizationInfo::CodeType ), cudaMemcpyDeviceToHost );
	cudaMemcpy( requestListDepth, requestListDepthPtr, numElements * sizeof( GvCore::GvLocalizationInfo::DepthType ), cudaMemcpyDeviceToHost );

	// Brick's resolution, including the border
	uint3 brickRes = BrickRes::get() + make_uint3( 2 * BorderSize );

	for ( uint i = 0; i < numElements; ++i )
	{
		GvCore::GvLocalizationCode::ValueType locCode = requestListCode[ i ].get();
		GvCore::GvLocalizationDepth::ValueType locDepth = requestListDepth[ i ].get();

		// Get the voxel's resolution at the child level
		uint3 levelRes = getLevelResolution( locDepth );
		float3 levelResInv = make_float3( 1.0f ) / make_float3( levelRes );

		// Convert the localization to a region
		float3 nodePos = make_float3( locCode * BrickRes::get() ) * levelResInv;
		float3 nodeSize = make_float3( BrickRes::get() ) * levelResInv;

		// Position of the brick (same as the position of the node minus the border)
		float3 brickPos = nodePos - make_float3( BorderSize ) * levelResInv;

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

					//// If the voxel is located inside the unit sphere
					//if ( isInSphere( posF ) )
					//{
					//	voxelColor.w = 1.0f;
					//}
					if ( ( brickOffset.x >= 1 && brickOffset.x <= 8 ) && ( brickOffset.y >= 1 && brickOffset.y <= 8 ) && ( brickOffset.z >= 1 && brickOffset.z <= 8 ) )
					{
						if ( _presenceFlags[ brickOffset.x - 1 ][ brickOffset.y - 1 ][ brickOffset.z - 1 ] == 1 )
						{
							voxelColor.w = 1.0f;
						}
					}

					voxelColor.x *= voxelColor.w;
					voxelColor.y *= voxelColor.w;
					voxelColor.z *= voxelColor.w;

					typedef typename GvCore::DataChannelType< DataTList, 0 >::Result ColorType;
					//typedef typename GvCore::DataChannelType< DataTList, 1 >::Result NormalType;

					ColorType color;
					//NormalType normal;

					convert_type( voxelColor, color );
				//	convert_type( voxelNormal, normal );

					bricksPool->getChannel( Loki::Int2Type< 0 >() )->get( i * 1000 + brickOffsetLinear ) = color;
				//	bricksPool->getChannel( Loki::Int2Type< 1 >() )->get( i * 1000 + brickOffsetLinear ) = normal;

					brickOffsetLinear++;
				}
			}
		}
	}
}

/******************************************************************************
 * Helpers
 *
 * @param level ...
 *
 * @return ...
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
inline uint3 Producer< DataTList, NodeRes, BrickRes, BorderSize >
::getLevelResolution( uint level )
{
	return make_uint3( 1 << level ) * BrickRes::get();
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
bool Producer< DataTList, NodeRes, BrickRes, BorderSize >
::hasBrickDrawOneSlice() const
{
	return _hasBrickDrawOneSlice;
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
void Producer< DataTList, NodeRes, BrickRes, BorderSize >
::setBrickDrawOneSlice( bool pFlag )
{
	_hasBrickDrawOneSlice = pFlag;
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
void Producer< DataTList, NodeRes, BrickRes, BorderSize >
::setBrickPresenceFlags( unsigned int pBrickPresenceFlags[][ 8 ][ 8 ] )
{
	const unsigned int brickResolution = BrickRes::x;

	for ( unsigned int z = 0; z < brickResolution; z++ )
	{
		for ( unsigned int y = 0; y < brickResolution; y++ )
		{
			for ( unsigned int x = 0; x < brickResolution; x++ )
			{
				_presenceFlags[ x ][ y ][ z ] = pBrickPresenceFlags[ x ][ y ][ z ];
			}
		}
	}

	// TEST -----
	int test = 0;
	for ( unsigned int z = 0; z < brickResolution; z++ )
	{
		for ( unsigned int y = 0; y < brickResolution; y++ )
		{
			for ( unsigned int x = 0; x < brickResolution; x++ )
			{
				if ( _presenceFlags[ x ][ y ][ z ] == 1 )
				{
					test++;
				}
			}
		}
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
void Producer< DataTList, NodeRes, BrickRes, BorderSize >
::clearCache()
{
	// The following two buffers are their equivalents on GPU
	requestListCodeDevice->clear();
	requestListDepthDevice->clear();

	// This array will contain the nodes produced by the CPU.
	nodesBuffer->fill( 0 );
	//bricksPool->getChannel( Loki::Int2Type< 0 >() )->fill( 0 );
}
