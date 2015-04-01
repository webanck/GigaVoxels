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

// Thrust
#include <thrust/copy.h>

//
#include <iostream>

//
#include <GvStructure/GvVolumeTreeAddressType.h>

//Vector
#include <vector>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor.
 * Initialize all buffers.
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
Producer< TDataStructureType, TDataProductionManager >::Producer()
:	GvUtils::GvSimpleHostProducer< ProducerKernel< TDataStructureType >, TDataStructureType, TDataProductionManager >()
,	mScene( 6 )
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
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::initialize( TDataStructureType* pDataStructure, TDataProductionManager* pDataProductionManager )
{
	// Call parent class
	GvUtils::GvSimpleHostProducer< ProducerKernel< TDataStructureType >, TDataStructureType, TDataProductionManager >::initialize( pDataStructure, pDataProductionManager );

	// This two buffers will contains the localization and the depth of the requested elements.
	requestListCode = new GvCore::GvLocalizationInfo::CodeType[ nbMaxRequests ];
	requestListDepth = new GvCore::GvLocalizationInfo::DepthType[ nbMaxRequests ];
	requestListAddress = new uint[ nbMaxRequests ];

	// The following two buffers are their equivalents on GPU
	requestListCodeDevice = new thrust::device_vector< GvCore::GvLocalizationInfo::CodeType >( nbMaxRequests );
	requestListDepthDevice = new thrust::device_vector< GvCore::GvLocalizationInfo::DepthType >( nbMaxRequests );

	// Init Scene 
	mScene.init( (char*)"Data/3DModels/bunny.obj" ); //TO DO : have to be a parameter of the producer

	// Init dimension
	mWidth = 10; 
	mHeight = 10; 
	mDepth = 10;

	// Create 3 textures
	glGenTextures( 3, _distanceTexture );

	// Init texture 3D 
	//
	// - 1 float component
	glBindTexture( GL_TEXTURE_3D, _distanceTexture[ 0 ] );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER );
	glTexImage3D( GL_TEXTURE_3D, 0, GL_R32F, mWidth, mHeight, mDepth, 0, GL_RED, GL_FLOAT, 0 );
	GV_CHECK_GL_ERROR();

	// Init texture 3D 
	//
	// - 1 float component
	glBindTexture( GL_TEXTURE_3D, _distanceTexture[ 1 ] );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER );
	glTexImage3D( GL_TEXTURE_3D, 0, GL_R32F, mWidth, mHeight, mDepth, 0, GL_RED, GL_FLOAT, 0 );
	GV_CHECK_GL_ERROR();

	// Init texture 3D 
	//
	// - 1 float component
	glBindTexture( GL_TEXTURE_3D, _distanceTexture[ 2 ] );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );
	glTexParameteri( GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER );
	glTexImage3D( GL_TEXTURE_3D, 0, GL_R32F, mWidth, mHeight, mDepth, 0, GL_RED, GL_FLOAT, 0 );

	// Initialize shader program
	mDistProg = GvUtils::GvShaderManager::createShaderProgram( "Data/Shaders/VoxelizationSignedDistanceField/signedDist_VS.glsl", NULL, "Data/Shaders/VoxelizationSignedDistanceField/signedDist_FS.glsl" );
	GvUtils::GvShaderManager::linkShaderProgram( mDistProg );

	// Initialize shader program
	mPotentialProg = GvUtils::GvShaderManager::createShaderProgram( "Data/Shaders/VoxelizationSignedDistanceField/potentialFunc_VS.glsl", NULL, "Data/Shaders/VoxelizationSignedDistanceField/potentialFunc_FS.glsl" );
	GvUtils::GvShaderManager::linkShaderProgram( mPotentialProg );

	// Store GLSL uniform variable locations
	mProjectionMatrixId = glGetUniformLocation( mDistProg, "uProjectionMatrix" );
	mModelViewMatrixId = glGetUniformLocation( mDistProg, "uModelViewMatrix" );
	mAxeId = glGetUniformLocation( mDistProg, "uAxe" );
	mSliceId = glGetUniformLocation( mDistProg, "uSlice" );
	mDistanceId = glGetUniformLocation( mDistProg, "uDistance" );
	mDistanceXId = glGetUniformLocation( mPotentialProg, "uDistanceX" );
	mDistanceYId = glGetUniformLocation( mPotentialProg, "uDistanceY" );
	mDistanceZId = glGetUniformLocation( mPotentialProg, "uDistanceZ" );
	mPotentialId = glGetUniformLocation( mPotentialProg, "uPotential" );
	mBrickAddressId = glGetUniformLocation( mPotentialProg, "uBrickAddress" );

	// Set value of some uniform variables
	GLint aux[ 3 ] = { 0, 1, 2 };
	glProgramUniform1iv( mDistProg, mDistanceId, 3, aux );
	glProgramUniform1i( mPotentialProg, mDistanceXId, 0 );
	glProgramUniform1i( mPotentialProg, mDistanceYId, 1 );
	glProgramUniform1i( mPotentialProg, mDistanceZId, 2 );
	glProgramUniform1i( mPotentialProg, mPotentialId, 3 );

	// Initialize empty data PBO
	std::vector< float > emptyData = std::vector< float >( 10 * 10 * 10, 1.0f );
	glGenBuffers( 1, &mEmptyData );
	glBindBuffer( GL_PIXEL_PACK_BUFFER, mEmptyData );
	glBufferData( GL_PIXEL_PACK_BUFFER, 10 * 10 * 10 * sizeof( float ), &emptyData[ 0 ], GL_STATIC_COPY );
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
	delete[] requestListCode;
	delete[] requestListDepth;

	delete[] requestListAddress;

	delete requestListCodeDevice;
	delete requestListDepthDevice;

	delete nodesBuffer;
	delete bricksPool;
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
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::produceData( uint pNumElems,
				thrust::device_vector< uint >* pNodesAddressCompactList,
				thrust::device_vector< uint >* pElemAddressCompactList,
				Loki::Int2Type< 0 > )
{
	// TO DO
	//
	// Optimize code :  HOST node producer seems to do nothing...

	// Wrap kernel producer.
	// This is mandatory when calling cache helper to write into cache (see below).
	GvCore::GvIProviderKernel< 0, KernelProducerType > kernelProvider( this->_kernelProducer );

	// Set kernel block dimension (used by cache helper)
	dim3 blockSize( 32, 1, 1 );

	// Retrieve raw pointers from device_vectors
	uint* nodesAddressList = thrust::raw_pointer_cast( &(*pNodesAddressCompactList)[0] );
	uint* elemAddressList = thrust::raw_pointer_cast( &(*pElemAddressCompactList)[0] );

	// Retrieve raw pointers from device_vectors
	GvCore::GvLocalizationInfo::CodeType* locCodeList = thrust::raw_pointer_cast( &(*requestListCodeDevice)[0] );
	GvCore::GvLocalizationInfo::DepthType* locDepthList = thrust::raw_pointer_cast( &(*requestListDepthDevice)[0] );

	// Iterates through all elements
	while ( pNumElems > 0 )
	{
		uint numRequests = min( pNumElems, nbMaxRequests );

		// Create the lists containing localization and depth of each element.
		this->_nodePageTable->createLocalizationLists( numRequests, nodesAddressList, requestListCodeDevice, requestListDepthDevice );

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
 * @param numElems the number of elements you have to produce.
 * @param nodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
 * @param elemAddressCompactList a list containing numElems addresses where you need to store the result.
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
	
	// Set kernel block dimension (used by cache helper)
	dim3 blockSize( 16, 8, 1 );
	
	// Retrieve raw pointers from device_vectors
	uint* nodesAddressList = thrust::raw_pointer_cast( &(*pNodesAddressCompactList)[0] );
	uint* elemAddressList = thrust::raw_pointer_cast( &(*pElemAddressCompactList)[0] );

	// Retrieve raw pointers from device_vectors
	GvCore::GvLocalizationInfo::CodeType* locCodeList = thrust::raw_pointer_cast( &(*requestListCodeDevice)[0] );
	GvCore::GvLocalizationInfo::DepthType* locDepthList = thrust::raw_pointer_cast( &(*requestListDepthDevice)[0] );

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
	// Not used
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
::produceBricks( const uint numElements, const GvCore::GvLocalizationInfo::CodeType* pRequestListCodePtr, const GvCore::GvLocalizationInfo::DepthType* pRequestListDepthPtr, const uint* pElemAddressListPtr )
{
	// To be able to retrieve the 3D world position associated to each node/brick we have to produce,
	// we need their "localization information" (i.e. depth and code).
	// As these information are stored on device, we first need to copy them from device to host.
	cudaMemcpy( requestListCode, pRequestListCodePtr, numElements * sizeof( GvCore::GvLocalizationInfo::CodeType ), cudaMemcpyDeviceToHost );
	cudaMemcpy( requestListDepth, pRequestListDepthPtr, numElements * sizeof( GvCore::GvLocalizationInfo::DepthType ), cudaMemcpyDeviceToHost );
	// Copy brick addresses in cache from device to host
	cudaMemcpy( requestListAddress, pElemAddressListPtr, numElements * sizeof( uint ), cudaMemcpyDeviceToHost );

	// Brick's resolution, including the border
	// - this is the number of voxels stored in a brick in cache
	uint3 brickRes = BrickRes::get() + make_uint3( 2 * BorderSize );

	// First, unmap GigaVoxels graphics resources from CUDA environment in order to be used by OpenGL
	this->_dataStructure->_dataPool->getChannel( Loki::Int2Type< 0 >() )->unmapResource();

	// Push the server attribute stack
	//
	// GL_VIEWPORT_BIT ...
	// GL_ENABLE_BIT ...
	// GL_COLOR_BUFFER_BIT ...
	glPushAttrib( GL_VIEWPORT_BIT | GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT );

	// Initialize viewport
	//
	// - viewport is aligned to one brick
	// - 1 fragment will be associated to 1 voxel
	glViewport( 0, 0, mWidth, mHeight );

	// Disable depth test
	//
	// - to find closest distance to camera plane
	//   we need to consider all triangles whichever side they lie
	glDisable( GL_DEPTH_TEST );

	// Disable culling
	//
	// - todo => say why : ...
	glDisable( GL_CULL_FACE );

	// No color ouput so deactivate it in OpenGL pipeline
	glColorMask( GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE );

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

	// Iterates through all elements (i.e. bricks)
	for ( uint i = 0; i < numElements; ++i )
	{
		// Get current brick's localization info (i.e. depth and code)
		const GvCore::GvLocalizationCode::ValueType locCode = requestListCode[ i ].get();
		const GvCore::GvLocalizationDepth::ValueType locDepth = requestListDepth[ i ].get();

		// Get brick address in cache
		const uint3 addressBrick = GvStructure::VolTreeBrickAddress::unpackAddress( requestListAddress[ i ] );

		// Get the voxel's resolution at the child level
		const uint3 levelRes = getLevelResolution( locDepth );
		const float3 levelResInv = make_float3( 1.0f ) / make_float3( levelRes );

		// Convert the localization to a region
		const float3 nodePos = make_float3( locCode * BrickRes::get() ) * levelResInv;
		const float3 nodeSize = make_float3( BrickRes::get() ) * levelResInv;

		// Position of the brick (same as the position of the node minus the border)
		const float3 brickPos = nodePos - make_float3( BorderSize ) * levelResInv;

		// Size of the brick ( same as the size of the node plus the border )
		const float3 brickSize = nodeSize + 2 * make_float3( BorderSize ) * levelResInv;

		// Compute "signed distance field" data and store it in the first channel
		produceSignedDistanceField( brickPos, brickSize.x, brickSize.y, brickSize.z, locDepth, locCode, addressBrick );
	}

	// Restore old variable
	glPopMatrix();
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();
	glPopAttrib();

	// Disable shader program
	glUseProgram( 0 );
	
	// Map GigaVoxels graphics resources in order to be used by CUDA environment
	this->_dataStructure->_dataPool->getChannel( Loki::Int2Type< 0 >() )->mapResource();
}

/******************************************************************************
 * Compute "signed distance field"
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
::produceSignedDistanceField( float3 pBrickPos, float pXSize, float pYSize, float pZSize, unsigned int pDepth, uint3 pLocCode, uint3 pAddressBrick )
{
	// [ I ] ---------------- 1st pass algorithm ----------------
	//
	// Generate a signed distance field from the mesh
	// 
	// - this is done for 1 brick
	// - we don't take into account normals
	//
	// - find the closest distance for the three axes separately

	// [ 1 ] - Activate shader program used to ...
	glUseProgram( mDistProg );
	
	// [ 2 ] - Clear "distance" textures with default value "1.0f"
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, mEmptyData );
	// clear _distanceTexture[ 0 ] texture content
	glBindTexture( GL_TEXTURE_3D, _distanceTexture[ 0 ] );
	glTexSubImage3D( GL_TEXTURE_3D/*target*/, 0/*level*/, 0/*x-offset*/, 0/*y-offset*/, 0/*z-offset*/, 
					10/*width*/, 10/*height*/, 10/*depth*/,
					GL_RED/*format*/, GL_FLOAT/*type*/, 0/*pixels*/ );
	// clear _distanceTexture[ 1 ] texture content
	glBindTexture( GL_TEXTURE_3D, _distanceTexture[ 1 ] );
	glTexSubImage3D( GL_TEXTURE_3D/*target*/, 0/*level*/, 0/*x-offset*/, 0/*y-offset*/, 0/*z-offset*/, 
					10/*width*/, 10/*height*/, 10/*depth*/,
					GL_RED/*format*/, GL_FLOAT/*type*/, 0/*pixels*/ );
	// clear _distanceTexture[ 2 ] texture content
	glBindTexture( GL_TEXTURE_3D, _distanceTexture[ 2 ] );
	glTexSubImage3D( GL_TEXTURE_3D/*target*/, 0/*level*/, 0/*x-offset*/, 0/*y-offset*/, 0/*z-offset*/, 
					10/*width*/, 10/*height*/, 10/*depth*/,
					GL_RED/*format*/, GL_FLOAT/*type*/, 0/*pixels*/ );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
	glBindTexture( GL_TEXTURE_3D, 0 );
	GV_CHECK_GL_ERROR();

	// [ 3 ] - Bind "distance" textures to image units
	glBindImageTexture( 0/*unit*/, _distanceTexture[ 0 ]/*texture*/, 
						0/*level*/, GL_TRUE/*layered*/, 0/*layer*/, GL_READ_WRITE/*access*/, GL_R32UI/*format*/ );
	glBindImageTexture( 1/*unit*/, _distanceTexture[ 1 ]/*texture*/, 
						0/*level*/, GL_TRUE/*layered*/, 0/*layer*/, GL_READ_WRITE/*access*/, GL_R32UI/*format*/ );
	glBindImageTexture( 2/*unit*/, _distanceTexture[ 2 ]/*texture*/, 
						0/*level*/, GL_TRUE/*layered*/, 0/*layer*/, GL_READ_WRITE/*access*/, GL_R32UI/*format*/ );
	GV_CHECK_GL_ERROR();

	// [ 4 ] - Fill "distance" textures with 3 axis distance information
	float4x4 projectionMatrix;
	float4x4 modelViewMatrix;
	// Iterate through max z-depth (here 10)
	for ( int z = 0; z < mDepth ; z++ )
	{	    	
		// Modify GLSL uniform variable
		//
		// - set current z-slice
		glUniform1i( mSliceId, z );
		
	    // [ a ] - Z min depth ----------------

	    // Toggle program for Z axis
	    glUniform1i( mAxeId, 2 );
	    
	    // Set up the ModelView matrix
		modelViewMatrix = MatrixHelper::lookAt( /*eye*/pBrickPos.x, pBrickPos.y, pBrickPos.z + pZSize * ( z + 0.5f ) / static_cast< float >( mDepth ), 
	     						  /*center*/pBrickPos.x, pBrickPos.y, pBrickPos.z - 1.0f, 
	    						  /*up*/0.0f, 1.0f, 0.0f );
		glUniformMatrix4fv( mModelViewMatrixId, 1, GL_FALSE/*transpose*/, modelViewMatrix._array );
		
		 // Set up the Projection matrix
		projectionMatrix = MatrixHelper::ortho( 0.f, pXSize, 0.f, pYSize, -10.f, 10.f );
		glUniformMatrix4fv( mProjectionMatrixId, 1/*count*/, GL_FALSE/*transpose*/, projectionMatrix._array );

	   	// Draw only the triangles usefull for this brick
		mScene.draw( pDepth, pLocCode );
		
	    // [ b ] - X min depth ----------------

	    // Toggle program for X axis
		glUniform1i( mAxeId, 0 );

	    // Set up the ModelView matrix
		modelViewMatrix = MatrixHelper::lookAt( /*eye*/pBrickPos.x + pXSize * ( z + 0.5f ) / static_cast< float >( mDepth ), pBrickPos.y, pBrickPos.z, 
	     						  /*center*/pBrickPos.x - 1.0f, pBrickPos.y, pBrickPos.z, 
	    						  /*up*/0.0f, 0.0f, 1.0f ); 
		glUniformMatrix4fv( mModelViewMatrixId, 1/*count*/, GL_FALSE/*transpose*/, modelViewMatrix._array );

	    // Set up the Projection matrix
	    projectionMatrix = MatrixHelper::ortho( 0.f, pYSize, 0.f, pZSize, -10.f, 10.f );
		glUniformMatrix4fv( mProjectionMatrixId, 1/*count*/, GL_FALSE/*transpose*/, projectionMatrix._array );

	   	// Draw only the triangle usefull for this brick
    	mScene.draw( pDepth, pLocCode );
	   
	    // [ c ] - Y min depth ----------------

	    // Toggle program for Y axis
		glUniform1i( mAxeId, 1 );

	   // Set up the ModelView matrix
		modelViewMatrix = MatrixHelper::lookAt( /*eye*/pBrickPos.x, pBrickPos.y + pYSize * ( z +0.5 ) / static_cast< float >( mDepth ), pBrickPos.z, 
	     						  /*center*/pBrickPos.x, pBrickPos.y - 1.0, pBrickPos.z, 
	    						  /*up*/1.0f, 0.0f, 0.0f ); 
		glUniformMatrix4fv( mModelViewMatrixId, 1/*count*/, GL_FALSE/*transpose*/, modelViewMatrix._array );

		 // Set up the Projection matrix
        projectionMatrix = MatrixHelper::ortho( 0.f, pZSize, 0.f, pXSize, -10.f, 10.f );
		glUniformMatrix4fv( mProjectionMatrixId, 1/*count*/, GL_FALSE/*transpose*/, projectionMatrix._array );

	   	// Draw only the triangles usefull for this brick
    	mScene.draw( pDepth, pLocCode );
	}

	// To debug !!
	//glFinish();

	// [ II ] ---------------- 2nd pass algorithm ----------------
	//
	// We fill the dataPool
	//
	// - generate the final "signed distance field" as the distance to the plane defined as the closest intersection along the 3 axis

	// [ 1 ] - Activate shader program used to ...
	glUseProgram( mPotentialProg );

	// [ 2 ] - Bind distance textures to image units
	//       ---- read-only mode
	glBindImageTexture( 0/*unit*/, _distanceTexture[ 0 ]/*texture*/,
						0/*level*/, GL_TRUE/*layered*/, 0/*layer*/, GL_READ_ONLY/*access*/, GL_R32F/*format*/ );
	glBindImageTexture( 1/*unit*/, _distanceTexture[ 1 ]/*texture*/, 
						0/*level*/, GL_TRUE/*layered*/, 0/*layer*/, GL_READ_ONLY/*access*/, GL_R32F/*format*/ );
	glBindImageTexture( 2/*unit*/, _distanceTexture[ 2 ]/*texture*/, 
						0/*level*/, GL_TRUE/*layered*/, 0/*layer*/, GL_READ_ONLY/*access*/, GL_R32F/*format*/ );
	GV_CHECK_GL_ERROR();

	// Bind data pool to image unit 
	glBindImageTexture( 3/*unit*/, this->_dataStructure->_dataPool->getChannel( Loki::Int2Type< 0 >() )->getBufferName()/*texture*/, 
						0/*level*/, GL_TRUE/*leyered*/, 0/*layer*/, GL_READ_WRITE/*access*/, GL_R32F/*format*/ );
	
	// Set the current "brick address in cache"
	glUniform3i( mBrickAddressId, pAddressBrick.x, pAddressBrick.y, pAddressBrick.z );

	// Draw a full quad on screen to generate fragments everywhere
	glBegin( GL_QUADS );
		glVertex3f( -1.0f, -1.0f, 0.0f );
		glVertex3f( 1.0f, -1.0f, 0.0f );
		glVertex3f( 1.0f, 1.0f, 0.0f );
		glVertex3f( -1.0f, 1.0f, 0.0f );
    glEnd();

	// Disable program
	glUseProgram( 0 );
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
