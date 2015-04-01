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
#include <GvStructure/GvVolumeTreeAddressType.h>

// Thrust
#include <thrust/copy.h>

// STL
#include <vector>
#include <iostream>

// glm
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform2.hpp>
#include <glm/gtx/projection.hpp>
#include <glm/gtc/type_ptr.hpp>

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
,	_nodesBuffer( NULL )
,	_requestListCode( NULL )
,	_requestListDepth( NULL )
,	_requestListCodeDevice( NULL )
,	_requestListDepthDevice( NULL )
,	_requestListAddress( NULL )
,	_voxelizationProgram( 0 )
,	_scene( 6 )
,	mWidth( 0 )
,	mHeight( 0 )
,	mDepth( 0 )
,	mFbo( 0 )
,	mDepthTex( 0 )
,	mTex2D( 0 )
,	_emptyData( 0 )
,	_atomicCounter( 0 )
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
	_requestListCode = new GvCore::GvLocalizationInfo::CodeType[ nbMaxRequests ];
	_requestListDepth = new GvCore::GvLocalizationInfo::DepthType[ nbMaxRequests ];

	_requestListAddress = new uint[ nbMaxRequests ];

	// The following two buffers are their equivalents on GPU
	_requestListCodeDevice = new thrust::device_vector< GvCore::GvLocalizationInfo::CodeType >( nbMaxRequests );
	_requestListDepthDevice = new thrust::device_vector< GvCore::GvLocalizationInfo::DepthType >( nbMaxRequests );

	// This 1D array will contain the nodes produced by the CPU
	//
	// Note : memory is mapped
	_nodesBuffer = new GvCore::Array3D< uint >( make_uint3( nbMaxRequests * NodeRes::numElements, 1, 1 ), 2 );

	// GPU producer initialization
	// It copies references to the nodes buffer pool to GPU equivalent
	this->_kernelProducer.init( _nodesBuffer->getDeviceArray() );

	// Init Scene 
	//TEST : comment next ligne
	_scene.init( (char*)"Data/3DModels/bunny.obj" ); //TO DO : have to be a parameter of the producer

	// Init dimension 
	mWidth = 10; 
	mHeight = 10; 
	mDepth = 10;

	// Init depth buffer
	glGenTextures( 1, &mDepthTex );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, mDepthTex );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexImage2D( GL_TEXTURE_RECTANGLE_EXT, 0, GL_DEPTH24_STENCIL8_EXT, mWidth, mHeight, 0, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, NULL );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
	GV_CHECK_GL_ERROR();

	// Init texture 2D 
	glGenTextures( 1, &mTex2D );
	glBindTexture( GL_TEXTURE_2D, mTex2D);
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F, mWidth, mHeight, 0, GL_RGB, GL_FLOAT, NULL );
	GV_CHECK_GL_ERROR();

	// Init fbo 
	glGenFramebuffers( 1, &mFbo );
	glBindFramebuffer( GL_FRAMEBUFFER, mFbo);
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mTex2D, 0 );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, mDepthTex, 0 );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, mDepthTex, 0 );

	if ( glCheckFramebufferStatus( GL_FRAMEBUFFER ) != GL_FRAMEBUFFER_COMPLETE )
	{
		std::cout << " ERROR : Problem with a framebuffer \n";
	}

	// Init shader programs
	_voxelizationProgram = GvUtils::GvShaderManager::createShaderProgram( "Data/Shaders/Voxelization/voxelization_VS.glsl", "Data/Shaders/Voxelization/voxelization_GS.glsl", "Data/Shaders/Voxelization/voxelization_FS.glsl" );
	GvUtils::GvShaderManager::linkShaderProgram( _voxelizationProgram );
	
	// Init atomic counter buffer
	glGenBuffers( 1, &_atomicCounter );
	glBindBuffer( GL_ATOMIC_COUNTER_BUFFER, _atomicCounter );
	glBufferData( GL_ATOMIC_COUNTER_BUFFER, sizeof( GLuint ) , NULL, GL_DYNAMIC_DRAW );
	glBindBuffer( GL_ATOMIC_COUNTER_BUFFER, 0 );
	GV_CHECK_GL_ERROR();

	// Init empty data pbo
	std::vector< float > emptyData = std::vector< float >( 10 * 10 * 10, 0 );
	glGenBuffers( 1, &_emptyData );
	glBindBuffer( GL_PIXEL_PACK_BUFFER, _emptyData );
	glBufferData( GL_PIXEL_PACK_BUFFER, 10 * 10 * 10 * sizeof( float ) , &emptyData[0], GL_STATIC_COPY );
	glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
	GV_CHECK_GL_ERROR();
}

/******************************************************************************
 * Finalize
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::finalize()
{
	delete[] _requestListCode;
	delete[] _requestListDepth;

	delete[] _requestListAddress;

	delete _requestListCodeDevice;
	delete _requestListDepthDevice;

	delete _nodesBuffer;
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
	const uint3 kernelBlockSize = KernelProducerType::NodesKernelBlockSize::get();
	const dim3 blockSize( kernelBlockSize.x, kernelBlockSize.y, kernelBlockSize.z );
	//const dim3 blockSize( 32, 1, 1 );

	// Retrieve raw pointers from device_vectors
	uint* nodesAddressList = thrust::raw_pointer_cast( &(*pNodesAddressCompactList)[0] );
	uint* elemAddressList = thrust::raw_pointer_cast( &(*pElemAddressCompactList)[0] );

	// Retrieve raw pointers from device_vectors
	GvCore::GvLocalizationInfo::CodeType* locCodeList = thrust::raw_pointer_cast( &(*_requestListCodeDevice)[0] );
	GvCore::GvLocalizationInfo::DepthType* locDepthList = thrust::raw_pointer_cast( &(*_requestListDepthDevice)[0] );

	// Iterates through all elements
	while ( pNumElems > 0 )
	{
		uint numRequests = min( pNumElems, nbMaxRequests );

		// Create the lists containing localization and depth of each element.
		this->_nodePageTable->createLocalizationLists( numRequests, nodesAddressList, _requestListCodeDevice, _requestListDepthDevice );

		// Produce nodes on host
		// and write resulting nodes info to the memory mapped nodes buffer
		produceNodes( numRequests, locCodeList, locDepthList );

		// Write into cache
		//
		// This stage launches the device producer to write the produced nodes to the node pool
		this->_cacheHelper.template genericWriteIntoCache< NodeTileResLinear >( numRequests, nodesAddressList, elemAddressList, this->_nodePool, kernelProvider, this->_nodePageTable, blockSize );

		// Update
		pNumElems			-= numRequests;
		nodesAddressList	+= numRequests;
		elemAddressList		+= numRequests;
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
	const uint3 kernelBlockSize = KernelProducerType::BricksKernelBlockSize::get();
	const dim3 blockSize( kernelBlockSize.x, kernelBlockSize.y, kernelBlockSize.z );
	//const dim3 blockSize( 16, 8, 1 );
	
	// Retrieve raw pointers from device_vectors
	uint* nodesAddressList = thrust::raw_pointer_cast( &(*pNodesAddressCompactList)[0] );
	uint* elemAddressList = thrust::raw_pointer_cast( &(*pElemAddressCompactList)[0] );

	// Retrieve raw pointers from device_vectors
	GvCore::GvLocalizationInfo::CodeType* locCodeList = thrust::raw_pointer_cast( &(*_requestListCodeDevice)[0] );
	GvCore::GvLocalizationInfo::DepthType* locDepthList = thrust::raw_pointer_cast( &(*_requestListDepthDevice)[0] );

	// Iterates through all elements
	while ( pNumElems > 0 )
	{
		uint numRequests = min( pNumElems, nbMaxRequests );

		// Create the lists containing localization and depth of each element.
		//
		// Once filled, these lists will be passsed to the HOST producer to retrieve localization info of requested elements
		this->_dataPageTable->createLocalizationLists( numRequests, nodesAddressList, _requestListCodeDevice, _requestListDepthDevice );

		// Produce bricks on host
		// and write resulting brick's voxels data to the memory mapped data buffer
		//
		// We pass the previously filled localization info on device
		produceBricks( numRequests, locCodeList, locDepthList, elemAddressList );

		// Write into cache
		//
		// This stage launches the device producer to write the produced bricks to the data pool
		this->_cacheHelper.template genericWriteIntoCache< BrickFullRes >( numRequests, nodesAddressList, elemAddressList, this->_dataPool, kernelProvider, this->_dataPageTable, blockSize );

		// Update
		pNumElems			-= numRequests;
		nodesAddressList	+= numRequests;
		elemAddressList		+= numRequests;
	}
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
	cudaMemcpy( _requestListCode, pRequestListCodePtr, pNbElements * sizeof( GvCore::GvLocalizationInfo::CodeType ), cudaMemcpyDeviceToHost );
	cudaMemcpy( _requestListDepth, pRequestListDepthPtr, pNbElements * sizeof( GvCore::GvLocalizationInfo::DepthType ), cudaMemcpyDeviceToHost );

	// Iterates through all elements (i.e. nodes)
	for ( uint i = 0; i < pNbElements; ++i )
	{
		// Get current node's localization info
		GvCore::GvLocalizationCode::ValueType parentLocCode = _requestListCode[ i ].get();
		GvCore::GvLocalizationDepth::ValueType parentLocDepth = _requestListDepth[ i ].get();

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

					// We work in the range [0.0; 1.0]

					// We only use pre-computed information
					uint nodeInfo = _scene.intersectMesh( locDepth, locCode );

					//uint nodeInfo = intersectMesh( nodePos, nodeSize.x, nodeSize.y, nodeSize.z, locDepth, locCode );

					// Write the node info to the memory mapped nodes buffer
					_nodesBuffer->get( i * NodeRes::numElements + nodeOffsetLinear ) = nodeInfo;

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
 * @param pElemAddressListPtr localization depth list on device
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::produceBricks( const uint numElements, const GvCore::GvLocalizationInfo::CodeType* pRequestListCodePtr, const GvCore::GvLocalizationInfo::DepthType* pRequestListDepthPtr, const uint* pElemAddressListPtr )
{
	// TODO : use asynchronous data transfert

	// Retrieve localization info lists from device to host
	cudaMemcpy( _requestListCode, pRequestListCodePtr, numElements * sizeof( GvCore::GvLocalizationInfo::CodeType ), cudaMemcpyDeviceToHost );
	cudaMemcpy( _requestListDepth, pRequestListDepthPtr, numElements * sizeof( GvCore::GvLocalizationInfo::DepthType ), cudaMemcpyDeviceToHost );

	// Retrieve brick addresses in cache (coming from the node pool)
	cudaMemcpy( _requestListAddress, pElemAddressListPtr, numElements * sizeof( uint ), cudaMemcpyDeviceToHost );

	// Brick's resolution, including the border
	uint3 brickRes = BrickRes::get() + make_uint3( 2 * BorderSize );

	// Unmap GigaVoxels graphics resources from CUDA environment in order to be used by OpenGL
	this->_dataStructure->_dataPool->getChannel( Loki::Int2Type< 0 >() )->unmapResource();

	// Iterates through all elements (i.e. bricks)
	for ( uint i = 0; i < numElements; ++i )
	{
		// Get current brick's localization info
		GvCore::GvLocalizationCode::ValueType locCode = _requestListCode[ i ].get();
		GvCore::GvLocalizationDepth::ValueType locDepth = _requestListDepth[ i ].get();

		// Get brick index offset in cache (i.e. data pool)
		//
		// Note : cache is indexed by voxels not bricks
		uint3 addressBrick = GvStructure::VolTreeBrickAddress::unpackAddress( _requestListAddress[ i ] );

		// Get the voxel's resolution at the child level
		uint3 levelRes = getLevelResolution( locDepth );	// Maximum number of voxels at given level of resolution (in each dimension)
		float3 levelResInv = make_float3( 1.0f ) / make_float3( levelRes );	// Size of one voxel

		// Convert localization info to 3D world region
		float3 nodePos = make_float3( locCode * BrickRes::get() ) * levelResInv;	// bottom left corner
		float3 nodeSize = make_float3( BrickRes::get() ) * levelResInv;

		// Position of the brick
		//
		// Beware : same as the position of the node but take into account borders
		float3 brickPos = nodePos - make_float3( BorderSize ) * levelResInv;

		// Size of the brick ( same as the size of the node plus the border )
		float3 brickSize = nodeSize + 2 * make_float3( BorderSize ) * levelResInv;

		// Iterate through current brick's voxels and fill data in cache (i.e data pool)
		uint3 brickOffset;
		uint brickOffsetLinear = 0;
		intersectMesh( brickPos, brickSize.x, brickSize.y, brickSize.z, locDepth, locCode, addressBrick );
	}

	// Map GigaVoxels graphics resources in order to be used by CUDA environment
	this->_dataStructure->_dataPool->getChannel( Loki::Int2Type< 0 >() )->mapResource();
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

/******************************************************************************
 * Intersect a brick with a mesh
 *
 * This method is used to fill the data pool (i.e. each voxel data)
 *
 * @param pBrickPos position of the brick (same as the position of the node minus the border)
 * @param pXSize x size of the brick ( same as the size of the node plus the border )
 * @param pYSize y size of the brick ( same as the size of the node plus the border )
 * @param pZSize z size of the brick ( same as the size of the node plus the border )
 * @param pDepth depth localization info of the brick
 * @param pLocCode code localization info of the brick
 * @param pAddressBrick address in cache where to write result of production
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::intersectMesh( float3 pBrickPos, float pXSize, float pYSize, float pZSize, unsigned int pDepth, uint3 pLocCode, uint3 pAddressBrick )
{
	// Push the server attribute stack
	//
	// GL_VIEWPORT_BIT ...
	// GL_ENABLE_BIT ...
	// GL_COLOR_BUFFER_BIT ...
	glPushAttrib( GL_VIEWPORT_BIT | GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT );

	// Initialize viewport
	glViewport( 0, 0, mWidth, mHeight ); // DO IT HERE ???

	// Disable depth test
	//
	// ...
	glDisable( GL_DEPTH_TEST );

	// Disable culling
	//
	// ...
	glDisable( GL_CULL_FACE );

	// Don't write color
	glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE );

	// WORK WITH MATRIX LIKE THAT ?? IS THERE AN OTHER WAY TO DO IT??
	// TO DO : don't compute matrix like that !!! and not here maybe...

	// Set orthographic projection
	//
	// - orthographic projection is required to do voxelization
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();

	// Setup camera 
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();
	
	// Compute projection matrices
	//
	// This method computes change-of-basis matrices to project along the 3 axis.
	// Those matrices are the multiplication of the openGL modelViewMatrix with projectionMatrix
	// after a call to gluLookAt(...) and glortho(...).
	//float4x4 projectionMatX;
	//float4x4 projectionMatY;
	//float4x4 projectionMatZ;
	//MatrixHelper::projectionMatrix( pBrickPos, pXSize, pYSize, pZSize, projectionMatX, projectionMatY, projectionMatZ );
	//-------------------------------
	glm::mat4 projectionMatrix;
	glm::mat4 modelViewMatrix;
	glm::mat4 xAxisMVP;
	glm::mat4 yAxisMVP;
	glm::mat4 zAxisMVP;
	modelViewMatrix = glm::lookAt(	/*eye*/glm::vec3( pBrickPos.x + pXSize, pBrickPos.y, pBrickPos.z ), 
									/*center*/glm::vec3( pBrickPos.x, pBrickPos.y, pBrickPos.z ), 
									/*up*/glm::vec3( 0.0f, 0.0f, 1.0f ) );
	projectionMatrix = glm::ortho(	/*left*/0.f, /*right*/pYSize,
									/*bottom*/0.f, /*top*/pZSize,
									/*near*/0.f, /*far*/pXSize );
	xAxisMVP = projectionMatrix * modelViewMatrix;
	modelViewMatrix = glm::lookAt(	/*eye*/glm::vec3( pBrickPos.x , pBrickPos.y + pYSize, pBrickPos.z ), 
									/*center*/glm::vec3( pBrickPos.x, pBrickPos.y, pBrickPos.z ), 
									/*up*/glm::vec3( 1.0f, 0.0f, 0.0f ) );
	projectionMatrix = glm::ortho( 0.f, pZSize, 0.f, pXSize, 0.f, pYSize );
	yAxisMVP = projectionMatrix * modelViewMatrix;
	modelViewMatrix = glm::lookAt(	/*eye*/glm::vec3( pBrickPos.x , pBrickPos.y , pBrickPos.z + pZSize ), 
									/*center*/glm::vec3( pBrickPos.x, pBrickPos.y, pBrickPos.z ), 
									/*up*/glm::vec3( 0.0f, 1.0f, 0.0f ) );
	projectionMatrix = glm::ortho( 0.f, pXSize, 0.f, pYSize, 0.f, pZSize );
	zAxisMVP = projectionMatrix * modelViewMatrix;
	//-------------------------------
		
	// Compute brick matrix
	//
	// ...
	//float4x4 brickMatrix = MatrixHelper::brickBaseMatrix( pBrickPos, pXSize, pYSize, pZSize );
	glm::mat4 brickMatrix;
	glm::mat4 transMatrix = glm::translate(
		glm::mat4( 1.0f ),
		glm::vec3( - ( 1.0f / pXSize ) * pBrickPos.x, - ( 1.0f / pYSize ) * pBrickPos.y, - ( 1.0f / pZSize ) * pBrickPos.z )
		);
	brickMatrix = glm::scale(  // Scale first
		transMatrix,              // Translate second
		glm::vec3( 1.0f / pXSize, 1.0f / pYSize, 1.0f / pZSize )
		);

	// Specify values of uniform matrix variables for shader program object
	//glProgramUniformMatrix4fvEXT( _voxelizationProgram, glGetUniformLocation( _voxelizationProgram, "uProjectionMatX" ), 1, GL_FALSE, projectionMatX._array );
	//glProgramUniformMatrix4fvEXT( _voxelizationProgram, glGetUniformLocation( _voxelizationProgram, "uProjectionMatY" ), 1, GL_FALSE, projectionMatY._array );
	//glProgramUniformMatrix4fvEXT( _voxelizationProgram, glGetUniformLocation( _voxelizationProgram, "uProjectionMatZ" ), 1, GL_FALSE, projectionMatZ._array );
	glProgramUniformMatrix4fvEXT( _voxelizationProgram, glGetUniformLocation( _voxelizationProgram, "uProjectionMatX" ), 1, GL_FALSE, glm::value_ptr( xAxisMVP ) );
	glProgramUniformMatrix4fvEXT( _voxelizationProgram, glGetUniformLocation( _voxelizationProgram, "uProjectionMatY" ), 1, GL_FALSE, glm::value_ptr( yAxisMVP ) );
	glProgramUniformMatrix4fvEXT( _voxelizationProgram, glGetUniformLocation( _voxelizationProgram, "uProjectionMatZ" ), 1, GL_FALSE, glm::value_ptr( zAxisMVP ) );
	//glProgramUniformMatrix4fvEXT( _voxelizationProgram, glGetUniformLocation( _voxelizationProgram, "uBrickMatrix" ), 1, GL_FALSE, brickMatrix._array );
	glProgramUniformMatrix4fvEXT( _voxelizationProgram, glGetUniformLocation( _voxelizationProgram, "uBrickMatrix" ), 1, GL_FALSE, glm::value_ptr( brickMatrix ) );

	// Reset atomic counter
	resetAtomicCounter();
	
	// Clear dataPool region
	//
	// ...
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, _emptyData );

	glBindTexture( GL_TEXTURE_3D, this->_dataStructure->_dataPool->getChannel( Loki::Int2Type< 0 >() )->getBufferName() );
	glTexSubImage3D( GL_TEXTURE_3D, 0, pAddressBrick.x * 10, pAddressBrick.y * 10, pAddressBrick.z * 10, 
					10, 10, 10,
					GL_RED, GL_FLOAT, 0 );

	glBindTexture( GL_TEXTURE_3D, this->_dataStructure->_dataPool->getChannel( Loki::Int2Type< 1 >() )->getBufferName() );
	glTexSubImage3D( GL_TEXTURE_3D, 0, pAddressBrick.x * 10, pAddressBrick.y * 10, pAddressBrick.z * 10, 
					10, 10, 10,
					GL_RED, GL_FLOAT, 0 );

	glBindTexture( GL_TEXTURE_3D, this->_dataStructure->_dataPool->getChannel( Loki::Int2Type< 2 >() )->getBufferName() );
	glTexSubImage3D( GL_TEXTURE_3D, 0, pAddressBrick.x * 10, pAddressBrick.y * 10, pAddressBrick.z * 10, 
					10, 10, 10,
					GL_RED, GL_FLOAT, 0 );

	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
	glBindTexture( GL_TEXTURE_3D, 0 );
	GV_CHECK_GL_ERROR();

	// Use shader program 
	glUseProgram( _voxelizationProgram ); // DO IT HERE ???
	GV_CHECK_GL_ERROR();

	// Bind atomic counter
	glBindBufferBase( GL_ATOMIC_COUNTER_BUFFER, 0, _atomicCounter );

	// Bind fbo 
	//glBindFramebufferEXT( GL_FRAMEBUFFER_EXT , mFbo );

	glUniform1i( glGetUniformLocation( _voxelizationProgram, "uDataPoolx" ), 0 );
	glUniform1i( glGetUniformLocation( _voxelizationProgram, "uDataPooly" ), 1 );
	glUniform1i( glGetUniformLocation( _voxelizationProgram, "uDataPoolz" ), 2 );
	GV_CHECK_GL_ERROR();

	// Bind data pool to image unit
	//
	// glBindImageTexture — bind a level of a texture to an image unit
	//GLuint aux = this->_dataStructure->_dataPool->getChannel( Loki::Int2Type< 0 >() )->getBufferName();
	glBindImageTexture( 0, this->_dataStructure->_dataPool->getChannel( Loki::Int2Type< 0 >() )->getBufferName(), 
	//					0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F );
						0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F );
	glBindImageTexture( 1, this->_dataStructure->_dataPool->getChannel( Loki::Int2Type< 1 >() )->getBufferName(), 
	//					0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F );
						0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F );
	glBindImageTexture( 2, this->_dataStructure->_dataPool->getChannel( Loki::Int2Type< 2 >() )->getBufferName(), 
	//					0, GL_TRUE, 0, GL_READ_WRITE, GL_R32F );
						0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R32F );
	GV_CHECK_GL_ERROR();

	// Specify address of the current brick
	glProgramUniform3i( _voxelizationProgram, glGetUniformLocation( _voxelizationProgram, "uBrickAddress" ), pAddressBrick.x, pAddressBrick.y, pAddressBrick.z );

	// Specify the half heigth pixel, which is in our case the half size of a voxel
	// We assume here that voxels are cube
	glProgramUniform2f( _voxelizationProgram, glGetUniformLocation( _voxelizationProgram, "hPixel" ), pXSize / 20.0f, pYSize / 20.0f );

	// Draw scene
	_scene.draw( pDepth, pLocCode );
	//_scene.draw();

	// Restore old variable
	glPopMatrix();
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();

	// Pop the server attribute stack
	glPopAttrib();

	// Unbind fbo 
	//glBindFramebufferEXT( GL_FRAMEBUFFER_EXT , 0 );

	// Disable program
	glUseProgram( 0 );

	// Update Scene octree
	_scene.setOctreeNode( pDepth, pLocCode, 0, getAtomicCounter() );
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline GLuint Producer< TDataStructureType, TDataProductionManager >
::getAtomicCounter()
{
	glBindBuffer( GL_ATOMIC_COUNTER_BUFFER, _atomicCounter );

	// Declare a pointer to hold the values in the buffer
	GLuint* userCounters;
	
	// Map the buffer to userCounters
	userCounters = (GLuint*)glMapBufferRange( GL_ATOMIC_COUNTER_BUFFER,
											 0,
											 sizeof(GLuint) ,
											 GL_MAP_READ_BIT );
	GLuint count = userCounters[ 0 ];
	
	glUnmapBuffer( GL_ATOMIC_COUNTER_BUFFER );

	glBindBuffer( GL_ATOMIC_COUNTER_BUFFER, 0 );

	return count;
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::resetAtomicCounter()
{
	glBindBuffer( GL_ATOMIC_COUNTER_BUFFER, _atomicCounter );

	// Declare a pointer to hold the values in the buffer
	GLuint* userCounters;
	
	// Map the buffer, userCounters will point to the buffers data
	userCounters = (GLuint*)glMapBufferRange( GL_ATOMIC_COUNTER_BUFFER, 
											 0 , 
											 sizeof(GLuint) , 
											 GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT );

	// Set the memory to zeros, resetting the values in the buffer
	memset( userCounters, 0, sizeof( GLuint ) );
	
	// Unmap the buffer
	glUnmapBuffer( GL_ATOMIC_COUNTER_BUFFER );

	glBindBuffer( GL_ATOMIC_COUNTER_BUFFER, 0 );
}
