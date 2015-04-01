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
#include <GvStructure/GvNode.h>
#include <GvUtils/GvNoiseKernel.h>
#include "NodeVisitorKernel.h"
//#include <GvStructure/GvVolumeTree.h>



/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/**
 * Transfer function texture
 */
texture< float4, cudaTextureType1D, cudaReadModeElementType > transferFunctionTexture;

/**
 * Volume texture (signed distance field : 3D normal + distance)
 */
texture< float4, cudaTextureType3D, cudaReadModeElementType > volumeTex;

/******************************************************************************
 * Map a distance to a color (with a transfer function)
 *
 * @param pDistance the distance to map
 *
 * @return The color corresponding to the distance
 ******************************************************************************/
__device__
inline float4 distToColor( float pDistance )
{
	// Fetch data from transfer function
	return tex1D( transferFunctionTexture, pDistance );
}


/******************************************************************************
 * Initialize the producer
 * 
 * @param volumeTreeKernel Reference on a volume tree data structure
 ******************************************************************************/
template< typename TDataStructureType >
inline void ProducerKernel< TDataStructureType >
::initialize( DataStructureKernel& pDataStructure )
{
	_dataStructureKernel = pDataStructure;
}


/******************************************************************************
 * Get the RGBA data of distance field + noise.
 * Note : color is alpha pre-multiplied to avoid color bleeding effect.
 *
 * @param voxelPosF 3D position in the current brick
 * @param levelRes number of voxels at the current brick resolution
 *
 * @return computed RGBA color
 ******************************************************************************/
__device__
float4 getRGBA( float3 voxelPosF, uint3 levelRes ,float& coarserNoise,uint depth)
{
	// Type definition for the noise
	typedef GvUtils::GvNoiseKernel Noise;
	
	// Retrieve "normal" and "distance" from signed distance fied of user's 3D model
	float4 voxelNormalAndDist = tex3D( volumeTex, voxelPosF.x, voxelPosF.y, voxelPosF.z );
	float3 voxelNormal = normalize( make_float3( voxelNormalAndDist.x, voxelNormalAndDist.y, voxelNormalAndDist.z ) );
	float voxelDist = voxelNormalAndDist.w;
	/*
	float3 voxelNormal = voxelPosF - make_float3(0.5f);
	float voxelDist = length(voxelNormal)-0.25f;
	voxelNormal = normalize( voxelNormal);
	*/
	float4 voxelRGBA;

	// Compute color by mapping a distance to a color (with a transfer function)
	//float4 color = make_float4(1.f,0.f,0.f,1.f);
	float4 color = distToColor( clamp( 0.5f + 0.5f * voxelDist * cNoiseFrequency, 0.f, 1.f ) );
	if ( color.w > 0.f )
	{
		// De multiply color with alpha because transfer function data has been pre-multiplied when generated
		color.x /= color.w;
		color.y /= color.w;
		color.z /= color.w;
	}
	
	if (depth>=cNoiseSkip ) {
		float noiseFrequency = cNoiseFrequency * static_cast< float >( 1 << depth );
		float noiseAmplitude = cNoiseAmplitude / static_cast< float >( 1 << (depth - cNoiseSkip) );


		// Compute noise
		//coarserNoise = 0.f;
		//for (int k =0;k<depth;k++)
		//{
		//	noiseFrequency = cNoiseFrequency * static_cast< float >( 1 << k );
		//	noiseAmplitude = cNoiseAmplitude / static_cast< float >( 1 << k );

		coarserNoise =coarserNoise + noiseAmplitude * Noise::getValue( noiseFrequency * ( voxelPosF - voxelDist * voxelNormal ) );
	}
	/*}*/	

	// Compute alpha
	//voxelRGBA.w = voxelDist>=coarserNoise ? 0.f : 1.f;
	voxelRGBA.w = 	clamp( 0.5f - 0.5f * ( voxelDist + coarserNoise ) * static_cast< float >( levelRes.x ), 0.f, 1.f );

	// Pre-multiply color with alpha
	voxelRGBA.x = color.x * voxelRGBA.w;
	voxelRGBA.y = color.y * voxelRGBA.w;
	voxelRGBA.z = color.z * voxelRGBA.w;

	return voxelRGBA;
}



/******************************************************************************
 * Get the normal of distance field + noise
 *
 * @param voxelPosF 3D position in the current brick
 * @param levelRes number of voxels at the current brick resolution
 *
 * @return ...
 ******************************************************************************/
__device__
//float3 getNormal( float3 voxelPosF, uint3 levelRes ,float3& grad_noise,uint depth )
void getNormal( float3 voxelPosF, uint3 levelRes ,uint depth ,float3& normal)
{
	// Type definition for the noise
	typedef GvUtils::GvNoiseKernel Noise;

	// Retrieve "normal" and "distance" from signed distance fied of user's 3D model
	float4 voxelNormalAndDist = tex3D( volumeTex, voxelPosF.x, voxelPosF.y, voxelPosF.z );
	float3 voxelNormal = normalize( make_float3( voxelNormalAndDist.x, voxelNormalAndDist.y, voxelNormalAndDist.z ) );
	float voxelDist = voxelNormalAndDist.w;

	//float3 voxelNormal = voxelPosF - make_float3(0.5f);
	//float voxelDist = length(voxelNormal)-0.25f;
	//voxelNormal = normalize( voxelNormal);

	float eps = 0.5f / static_cast< float >( levelRes.x );
	float3 grad_noise  = make_float3(0.f);

	if (depth>=cNoiseSkip && depth>0) {
		float noiseFrequency = cNoiseFrequency * static_cast< float >( 1 << depth );
		
		float noiseAmplitude = cNoiseAmplitude / static_cast< float >( 1 << (depth -  cNoiseSkip) );

	
		// Compute symetric gradient noise
		//grad_noise = make_float3( 0.0f );
		//for ( int k = 0; k < depth; k++ )
		//{
		//	noiseFrequency = cNoiseFrequency * static_cast< float >( 1 << k );
		//	noiseAmplitude = cNoiseAmplitude / static_cast< float >( 1 << k );

		grad_noise.x =  noiseAmplitude  * Noise::getValue( noiseFrequency * ( voxelPosF + make_float3( eps, 0.0f, 0.0f ) - voxelDist * voxelNormal ) )
						-noiseAmplitude   * Noise::getValue( noiseFrequency * ( voxelPosF - make_float3( eps, 0.0f, 0.0f ) - voxelDist * voxelNormal ) );

		grad_noise.y =  noiseAmplitude   * Noise::getValue( noiseFrequency * ( voxelPosF + make_float3( 0.0f, eps, 0.0f ) - voxelDist * voxelNormal ) )
						-noiseAmplitude   * Noise::getValue( noiseFrequency * ( voxelPosF - make_float3( 0.0f, eps, 0.0f ) - voxelDist * voxelNormal ) );

		grad_noise.z =  noiseAmplitude   * Noise::getValue( noiseFrequency * ( voxelPosF + make_float3( 0.0f, 0.0f, eps ) - voxelDist * voxelNormal ) )
						-noiseAmplitude   * Noise::getValue( noiseFrequency * ( voxelPosF - make_float3( 0.0f, 0.0f, eps ) - voxelDist * voxelNormal ) );
	

		grad_noise =grad_noise* 0.5f / (eps);
		normal = ( normal + grad_noise - dot( grad_noise, voxelNormal ) * voxelNormal );
	
	} else {
	
	normal =voxelNormal;
	}


}


/******************************************************************************
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
 ******************************************************************************/
template< typename TDataStructureType >
template< typename GPUPoolKernelType >
__device__
inline uint ProducerKernel< TDataStructureType >
::produceData( GPUPoolKernelType& nodePool, uint requestID, uint processID, uint3 newElemAddress,
			  const GvCore::GvLocalizationInfo& parentLocInfo, Loki::Int2Type< 0 > )
{
	// NOTE :
	// In this method, you are inside a node tile.
	// The goal is to determine, for each node of the node tile, which type of data it holds.
	// Data type can be :
	// - a constant region,
	// - a region with data,
	// - a region where max resolution is reached.
	// So, one thread is responsible of the production of one node of a node tile.
	
	// Retrieve current node tile localization information code and depth
	const GvCore::GvLocalizationInfo::CodeType *parentLocCode = &parentLocInfo.locCode;
	const GvCore::GvLocalizationInfo::DepthType *parentLocDepth = &parentLocInfo.locDepth;

	// Process ID gives the 1D index of a node in the current node tile
	if ( processID < NodeRes::getNumElements() )
	{
		// First, compute the 3D offset of the node in the node tile
		uint3 subOffset = NodeRes::toFloat3( processID );

		// Node production corresponds to subdivide a node tile.
		// So, based on the index of the node, find the node child.
		// As we want to sudbivide the curent node, we retrieve localization information
		// at the next level
		uint3 regionCoords = parentLocCode->addLevel< NodeRes >( subOffset ).get();
		uint regionDepth = parentLocDepth->addLevel().get();

		// Create a new node for which you will have to fill its information.
		GvStructure::GvNode newnode;
		newnode.childAddress = 0;
		newnode.brickAddress = 0;

		// Call what we call an oracle that will determine the type of the region of the node accordingly
		GPUVoxelProducer::GPUVPRegionInfo nodeinfo = getRegionInfo( regionCoords, regionDepth );

		// Now that the type of the region is found, fill the new node information
		if ( nodeinfo == GPUVoxelProducer::GPUVP_CONSTANT )
		{
			newnode.setTerminal( true );
		}
		else if ( nodeinfo == GPUVoxelProducer::GPUVP_DATA )
		{
			newnode.setStoreBrick();
			newnode.setTerminal( false );
		}
		else if ( nodeinfo == GPUVoxelProducer::GPUVP_DATA_MAXRES )
		{
			newnode.setStoreBrick();
			newnode.setTerminal( true );
		}

		// Finally, write the new node information into the node pool by selecting channels :
		// - Loki::Int2Type< 0 >() points to node information
		// - Loki::Int2Type< 1 >() points to brick information
		//
		// newElemAddress.x + processID : is the adress of the new node in the node pool
		nodePool.getChannel( Loki::Int2Type< 0 >() ).set( newElemAddress.x + processID, newnode.childAddress );
		nodePool.getChannel( Loki::Int2Type< 1 >() ).set( newElemAddress.x + processID, newnode.brickAddress );

	}

	return 0;
}

/******************************************************************************
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
 ******************************************************************************/
template< typename TDataStructureType >
template< typename GPUPoolKernelType >
__device__
inline uint ProducerKernel< TDataStructureType >
::produceData( GPUPoolKernelType& dataPool, uint requestID, uint processID, uint3 newElemAddress,
			  const GvCore::GvLocalizationInfo& parentLocInfo, Loki::Int2Type< 1 > )
{
	// NOTE :
	// In this method, you are inside a brick of voxels.
	// The goal is to determine, for each voxel of the brick, the value of each of its channels.
	//
	// Data type has previously been defined by the user, as a
	// typedef Loki::TL::MakeTypelist< uchar4, half4 >::Result DataType;
	//
	// In this tutorial, we have choosen two channels containing color at channel 0 and normal at channel 1.
	
	// Retrieve current brick localization information code and depth
	const GvCore::GvLocalizationInfo::CodeType parentLocCode = parentLocInfo.locCode;
	const GvCore::GvLocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;

	// Shared memory declaration
	//
	// Each threads of a block process one and unique brick of voxels.
	// We store in shared memory common useful variables that will be used by each thread.
	__shared__ uint3 brickRes;
	__shared__ uint3 levelRes; 
	__shared__ float3 levelResInv; 
	__shared__ int3 brickPos;
	__shared__ float3 brickPosF;

	// Compute useful variables used for retrieving positions in 3D space
	brickRes = BrickRes::get();
	levelRes = make_uint3( 1 << parentLocDepth.get() ) * brickRes;
	levelResInv = make_float3( 1.0f ) / make_float3( levelRes );

	brickPos = make_int3( parentLocCode.get() * brickRes ) - BorderSize;
	brickPosF = make_float3( brickPos ) * levelResInv;

	// Real brick size (with borders)
	uint3 elemSize = BrickRes::get() + make_uint3( 2 * BorderSize );

	
	// Retrieve coaser brick position in pool
	__shared__ uint smCoaserBrickPositionInPool;
	__shared__ bool smUseCoaserLevel;
	__shared__ GvStructure::GvNode smParentNode;
	__shared__ float3 smBrickCenter;
	__shared__ uint3 smParentNodeBrickAddress;
	GvCore::GvLocalizationCode parentNodeLocalizationCode; 
	float3 brickSize; 
	float3 parentBrickSize; 
	float3 parentNodePosition;
	uint3 nodeChildCoordinates; 
	float3 brickPosition; 

	float noiseFrequency;

	if ( processID == 0 )
	{
		smUseCoaserLevel = false;
		//printf("%u\n",cNoiseSkip);
		smParentNode.childAddress = 0;
		smParentNode.brickAddress = 0;

		smBrickCenter = make_float3( parentLocCode.get() ) * ( 1.f / static_cast< float >( 1 << parentLocDepth.get() ) ) + make_float3( 0.5f * ( 1.f / static_cast< float >( 1 << parentLocDepth.get() ) ) );

		smParentNodeBrickAddress = make_uint3( 0, 0, 0 );

		if ( parentLocDepth.get() >= cNoiseSkip && parentLocDepth.get()>0)
		{
			smUseCoaserLevel = true;

			// Retrieve parent node
				
				
			NodeVisitorKernel::getNodeFather( _dataStructureKernel, smParentNode, smBrickCenter, parentLocDepth.get() - 1 );
				
			// Retrieve parent node's brick address
			smParentNodeBrickAddress = smParentNode.getBrickAddress();
		}
		//printf("Production d'une brique\n");
	}
	__syncthreads();

	//--------------------------------
	parentNodeLocalizationCode = /*parent loc code*/parentLocCode.removeLevel< NodeRes >();
	brickSize = make_float3( 1.0f ) / static_cast< float >( 1 << parentLocDepth.get() );
	parentBrickSize = 2.f * brickSize;
	parentNodePosition = make_float3( parentNodeLocalizationCode.get() ) * parentBrickSize;
	nodeChildCoordinates = make_uint3(  floorf(   (smBrickCenter - parentNodePosition ) / brickSize) );
	brickPosition = make_float3( parentLocCode.get() * BrickRes::get() ) * levelResInv;
	//--------------------------------
	
	
	
	// The original KERNEL execution configuration on the HOST has a 2D block size :
	// dim3 blockSize( 16, 8, 1 );
	//
	// Each block process one brick of voxels.
	//
	// One thread iterate in 3D space given a pattern defined by the 2D block size
	// and the following "for" loops. Loops take into account borders.
	// In fact, each thread of the current 2D block compute elements layer by layer
	// on the z axis.
	//
	// One thread process only a subset of the voxels of the brick.
	//
	// Iterate through z axis step by step as blockDim.z is equal to 1
	uint3 elemOffset;
	for ( elemOffset.z = 0; elemOffset.z < elemSize.z; elemOffset.z += blockDim.z )
	{
		for ( elemOffset.y = 0; elemOffset.y < elemSize.y; elemOffset.y += blockDim.y )
		{
			for ( elemOffset.x = 0; elemOffset.x < elemSize.x; elemOffset.x += blockDim.x )
			{
				// Compute position index
				uint3 locOffset = elemOffset + make_uint3( threadIdx.x, threadIdx.y, threadIdx.z );

				// Test if the computed position index is inside the brick (with borders)
				if ( locOffset.x < elemSize.x && locOffset.y < elemSize.y && locOffset.z < elemSize.z )
				{
					// Position of the current voxel's center (relative to the brick)
					float3 voxelPosInBrickF = ( make_float3( locOffset ) + 0.5f ) * levelResInv;
					// Position of the current voxel's center (absolute, in [0.0;1.0] range)
					float3 voxelPosF = brickPosF + voxelPosInBrickF;
					// Position of the current voxel's center (scaled to the range [-1.0;1.0])
					float3 posF = voxelPosF * 2.0f - 1.0f;

					// Retrieve already computed coarser noise value from parent node
					float coarserNoise = 0.f;
					//float4 voxelNormalAndDist = tex3D( volumeTex, voxelPosF.x, voxelPosF.y, voxelPosF.z );
					float3 coarserNormal;// =normalize(make_float3(voxelNormalAndDist.x,voxelNormalAndDist.y,voxelNormalAndDist.z));
					//printf("avantavant(%f,%f,%f)\n",coarserNormal.x,coarserNormal.y,coarserNormal.z);

					if ( smUseCoaserLevel )
					{
						
						
							// Retrieve parent brick's noise value
							float3 offsetPositionInNode = ( voxelPosF - brickPosition ) / brickSize;
						
							float3 samplePosition = 	make_float3( nodeChildCoordinates ) *( (_dataStructureKernel.brickSizeInCacheNormalized) * 0.5f) 
														+ offsetPositionInNode * ((_dataStructureKernel.brickSizeInCacheNormalized) * 0.5f) ;
					
							//printf("%f\n", _dataStructureKernel.brickSizeInCacheNormalized.x);
							float4 brickData = _dataStructureKernel.template getSampleValueTriLinear< 1 >( make_float3( smParentNodeBrickAddress.x, smParentNodeBrickAddress.y, smParentNodeBrickAddress.z ) * _dataStructureKernel.brickCacheResINV,
														samplePosition);
							coarserNoise = brickData.x;
							
							coarserNormal = make_float3(brickData.y,brickData.z,brickData.w);

					

					}
					
					// Compute data
					float4 voxelColor = getRGBA( voxelPosF, levelRes ,coarserNoise,parentLocDepth.get());
					
					//float3 voxelNormal = getNormal( voxelPosF, levelRes , grad_noise,parentLocDepth.get());
					//printf("avant(%f,%f,%f)\n",coarserNormal.x,coarserNormal.y,coarserNormal.z);

					getNormal( voxelPosF, levelRes ,parentLocDepth.get(),coarserNormal);
					//printf("apres(%f,%f,%f)\n",coarserNormal.x,coarserNormal.y,coarserNormal.z);

					// Alpha pre-multiplication used to avoid the "color bleeding" effect
				
					// Compute the new element's address
					uint3 destAddress = newElemAddress + locOffset;
					// Write the voxel's color in the first field
					dataPool.template setValue< 0 >( destAddress, voxelColor );
					// Write the voxel's normal and noise in the second field
					dataPool.template setValue< 1 >( destAddress, make_float4( coarserNoise,coarserNormal.x,coarserNormal.y,coarserNormal.z ) );

					

				}
			}
		}
	}

	return 0;
}

/******************************************************************************
 * Helper function used to determine the type of zones in the data structure.
 *
 * The data structure is made of regions containing data, empty or constant regions.
 * Besides, this function can tell if the maximum resolution is reached in a region.
 *
 * @param regionCoords region coordinates
 * @param regionDepth region depth
 *
 * @return the type of the region
 ******************************************************************************/
template< typename TDataStructureType >
__device__
inline GPUVoxelProducer::GPUVPRegionInfo ProducerKernel< TDataStructureType >
::getRegionInfo( uint3 regionCoords, uint regionDepth )
{
	// Type definition for the noise
	typedef GvUtils::GvNoiseKernel Noise;
	
	// Limit the depth.
	// Currently, 32 is the max depth of the GigaVoxels engine.
	if ( regionDepth >= 32 )
	{
		return GPUVoxelProducer::GPUVP_DATA_MAXRES;
	}

	// Shared memory declaration
	__shared__ uint3 brickRes;
	//__shared__ float3 brickSize;
	__shared__ uint3 levelRes;
	__shared__ float3 levelResInv;

	brickRes = BrickRes::get();

	levelRes = make_uint3( 1 << regionDepth ) * brickRes;
	levelResInv = make_float3( 1.f ) / make_float3( levelRes );

	int3 brickPos = make_int3(regionCoords * brickRes) - BorderSize;
	float3 brickPosF = make_float3( brickPos ) * levelResInv;

	uint3 elemSize = BrickRes::get() + make_uint3( 2 * BorderSize );
	uint3 elemOffset;

	bool isEmpty = true;

	float brickSize = 1.0f / (float)( 1 << regionDepth );
	float parentBrickSize = 2.f * brickSize;
	float3 parentNodePosition = make_float3( regionCoords.x >> 1, regionCoords.y >> 1, regionCoords.z >> 1 ) * parentBrickSize;
	 GvStructure::GvNode smParentNode;
	
	 uint3 smParentNodeBrickAddress = make_uint3( 0, 0, 0 );
	 float3 smBrickCenter ;

	

	smBrickCenter = make_float3( regionCoords ) * ( 1.f / static_cast< float >( 1 << regionDepth ) ) + make_float3( 0.5f * ( 1.f / static_cast< float >( 1 << regionDepth ) ) );
	// Retrieve parent node
	NodeVisitorKernel::getNodeFather( _dataStructureKernel, smParentNode, smBrickCenter, regionDepth - 1 );
				
	// Retrieve parent node's brick address
	smParentNodeBrickAddress = smParentNode.getBrickAddress();
		

	
	uint3 nodeChildCoordinates = make_uint3(  floorf(   (smBrickCenter - parentNodePosition ) / brickSize) );
	



	// Iterate through voxels
	for ( elemOffset.z = 0; elemOffset.z < elemSize.z && isEmpty; elemOffset.z++ )
	{
		for ( elemOffset.y = 0; elemOffset.y < elemSize.y && isEmpty; elemOffset.y++ )
		{
			for ( elemOffset.x = 0; elemOffset.x < elemSize.x && isEmpty; elemOffset.x++ )
			{
				uint3 locOffset = elemOffset;// + make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);

				if ( locOffset.x < elemSize.x && locOffset.y < elemSize.y && locOffset.z < elemSize.z )
				{
					// Position of the current voxel's center (relative to the brick)
					float3 voxelPosInBrickF = ( make_float3( locOffset ) + 0.5f ) * levelResInv;
					// Position of the current voxel's center (absolute, in [0.0;1.0] range)
					float3 voxelPosF = brickPosF + voxelPosInBrickF;
					// Retrieve parent brick's noise value
					const float3 offsetPositionInNode = ( voxelPosF - brickPosF ) / brickSize;
						
					const float3 samplePosition = 	make_float3( nodeChildCoordinates ) *( (_dataStructureKernel.brickSizeInCacheNormalized) * 0.5f) 
												+ offsetPositionInNode * ((_dataStructureKernel.brickSizeInCacheNormalized) * 0.5f) ;
					
					//printf("%f\n", _dataStructureKernel.brickSizeInCacheNormalized.x);
					const float4 brickData = _dataStructureKernel.template getSampleValueTriLinear< 1 >( make_float3( smParentNodeBrickAddress.x, smParentNodeBrickAddress.y, smParentNodeBrickAddress.z ) * _dataStructureKernel.brickCacheResINV,
												samplePosition);



					// Test opacity to determine if there is data
					float4 voxelNormalAndDist = tex3D( volumeTex, voxelPosF.x, voxelPosF.y, voxelPosF.z );

					if (regionDepth<cNoiseSkip)
					{
						if (fabs(voxelNormalAndDist.w) <= (2/(2-1.f)) *cNoiseAmplitude)
							isEmpty = false;
					} else {
						
						if ( fabs(voxelNormalAndDist.w  + brickData.x) <= cNoiseAmplitude/static_cast<float>(1<<(regionDepth-1-cNoiseSkip+1)) )
						//if ( fabs(voxelNormalAndDist.w  + dist_noise) <= cNoiseAmplitude/((2-1.f)* powf(2,static_cast<float>(regionDepth-2-cNoiseSkip+1) ) ) )
							isEmpty = false;
					}
				}
			}
		}
	}

	if ( isEmpty )
	{
		return GPUVoxelProducer::GPUVP_CONSTANT;
	}

	return GPUVoxelProducer::GPUVP_DATA;
}
