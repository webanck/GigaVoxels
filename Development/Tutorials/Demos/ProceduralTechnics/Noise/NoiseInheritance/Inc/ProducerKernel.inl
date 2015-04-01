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
#define cR 0.6f
#define cr 0.3f

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvStructure/GvNode.h>
#include <GvUtils/GvNoiseKernel.h>
#include <GvStructure/GvVolumeTreeKernel.h>
#include <GvRendering/GvNodeVisitorKernel.h>

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
template< typename TDataStructureType >
ProducerKernel< TDataStructureType >
::ProducerKernel()
{
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

	// Process ID gives the 1D index of a node in the current node tile
	if ( processID < NodeRes::getNumElements() )
	{
		// First, compute the 3D offset of the node in the node tile
		uint3 subOffset = NodeRes::toFloat3( processID );

		// Node production corresponds to subdivide a node tile.
		// So, based on the index of the node, find the node child.
		// As we want to sudbivide the curent node, we retrieve localization information
		// at the next level
		uint3 regionCoords = parentLocInfo.locCode.addLevel< NodeRes >( subOffset ).get();
		uint regionDepth = parentLocInfo.locDepth.addLevel().get();

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
	// Type definition for the noise
	typedef GvUtils::GvNoiseKernel Noise;

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
	// Compute useful variables used for retrieving positions in 3D space
	const uint3 brickRes = BrickRes::get();
	const uint3 levelRes = make_uint3( 1 << parentLocDepth.get() ) * brickRes;	// number of voxels (in each dimension)
	const float3 levelResInv = make_float3( 1.0f ) / make_float3( levelRes );	// size of a voxel (in each dimension)
	const int3 brickPos = make_int3( parentLocCode.get() * brickRes ) - BorderSize;
	const float3 brickPosF = make_float3( brickPos ) * levelResInv;

	// Real brick size (with borders)
	const uint3 elemSize = BrickRes::get() + make_uint3( 2 * BorderSize );

	// Retrieve coarser brick position in pool
	const float coeff = 1.f / static_cast< float >( 1 << parentLocDepth.get());
	const float3 smBrickCenter = make_float3( parentLocCode.get() ) * coeff + make_float3( 0.5f * coeff );
	__shared__ bool smUseCoarserLevel;

	__shared__ float3 smParentNodeBrickAddress;
	if ( processID ) {
		smUseCoarserLevel = parentLocDepth.get() > 0;

		if( smUseCoarserLevel ) {
			// Retrieve parent node
			GvStructure::GvNode smParentNode;
			uint depth = GvRendering::GvNodeVisitorKernel::getNodeFather( _dataStructureKernel, smParentNode, smBrickCenter, parentLocDepth.get() - 1 );

			smUseCoarserLevel = ( smParentNode.hasBrick() ) && ( depth == parentLocDepth.get() );

			// Retrieve parent node's brick address
			// If the brick was clear from the cache, disable the use of coarser noise level
			smParentNodeBrickAddress = make_float3( smParentNode.getBrickAddress() );
		}
	}
	__syncthreads();

	//--------------------------------
	GvCore::GvLocalizationCode parentNodeLocalizationCode = /*parent loc code*/parentLocCode.removeLevel< NodeRes >();
	const float3 brickSize = make_float3( 1.0f ) / static_cast< float >( 1 << parentLocDepth.get());
	const float3 brickSizeInv = 1.f / brickSize;
	const float3 parentBrickSize = 2.f * brickSize;
	const float3 parentNodePosition = make_float3( parentNodeLocalizationCode.get() ) * parentBrickSize;
	const float3 nodeChildCoordinates = floorf(( smBrickCenter - parentNodePosition ) * brickSizeInv );
	const float3 brickPosition = make_float3( parentLocCode.get() * BrickRes::get() ) * levelResInv;
	//--------------------------------


	const float noiseFrequency = cFrequencyCoefficient * static_cast< float >( 1u << parentLocDepth.get() );
	const float noiseAmplitude = __powf( cAmplitudeCoefficient, static_cast< float >( parentLocDepth.get() ) );
	const float crInv = 1.f / cr;

	const float3 brickSizeInCacheNormalized = _dataStructureKernel.brickSizeInCacheNormalized * 0.5f;
	const float3 brickAddress = _dataStructureKernel.brickCacheResINV * smParentNodeBrickAddress;

	// Each block process one brick of voxels.
	//
	// One thread iterate in 3D space given a pattern defined by the 2D block size
	// and the following "for" loops. Loops take into account borders.
	//
	// One thread process only a subset of the voxels of the brick.
	const unsigned int nVoxels = elemSize.x * elemSize.y * elemSize.z;
	const unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;
	for( unsigned int index = processID; index < nVoxels; index += blockSize ) {
		uint3 locOffset;
		locOffset.x = index % elemSize.x;
		locOffset.y = ( index / elemSize.x ) % elemSize.y;
		locOffset.z = index / ( elemSize.x * elemSize.y );

		// Position of the current voxel's center (relative to the brick)
		// In order to make the mip-mapping mecanism OK,
		// data values must be set at the center of voxels.
		float3 voxelPosInBrickF = ( make_float3( locOffset ) + 0.5f ) * levelResInv;
		// Position of the current voxel's center (absolute, in [0.0;1.0] range)
		float3 voxelPosF = brickPosF + voxelPosInBrickF;
		// Position of the current voxel's center (scaled to the range [-1.0;1.0])
		float3 posF = voxelPosF * 2.0f - 1.0f;

		// Retrieve already computed coarser noise value from parent node
		float noise = 0.f;
		if ( smUseCoarserLevel ) {
			// Retrieve parent brick's noise value
			const float3 offsetPositionInNode = ( voxelPosF - brickPosition ) * brickSizeInv;
			const float3 samplePosition = ( offsetPositionInNode + nodeChildCoordinates ) * brickSizeInCacheNormalized;
			const float4 brickData = _dataStructureKernel.template getSampleValueTriLinear< 2 >( brickAddress, samplePosition );

			noise = 1.f / noiseFrequency * noiseAmplitude * ( Noise::getValueT( noiseFrequency * posF )) + brickData.x;
		} else {
			// Compute the sum of noise
			float frequency = cFrequencyCoefficient;
			float amplitude = 1.f;
			for( uint level = 0; level < parentLocDepth.get() + 1; ++level ) {
				noise += 1.f / frequency * amplitude * ( Noise::getValueT( frequency * posF ));
				frequency *= 2.f;
				amplitude *= cAmplitudeCoefficient;
			}
		}

		// Color
		float4 voxelColor;

		// Normal
		float radiusInv = rsqrtf( posF.x * posF.x + posF.y * posF.y );
		float3 circleCenter = make_float3( posF.x * cR * radiusInv,
				                           posF.y * cR * radiusInv,
				                           0.f );
		float lInv = rsqrtf( dot( posF - circleCenter, posF - circleCenter ));
		float4 voxelNormal = make_float4(( posF - circleCenter ) * lInv, 0.f ); 

		// Test if the voxel is located inside the torus
		float color = 0.f;
		voxelColor.w = 0.f;
		if( lInv >= crInv ) {
			voxelColor.w = 1.f;
			color = __saturatef( noise ); 
		}
		voxelColor = make_float4( color, color, color, voxelColor.w );

		// Compute the new element's address
		uint3 destAddress = newElemAddress + locOffset;

		// Write the voxel's color in the first field
		dataPool.template setValue< 0 >( destAddress, voxelColor );

		// Write the voxel's normal in the second field
		dataPool.template setValue< 1 >( destAddress, voxelNormal );

		// Write the voxel's parent brick's noise value in the third field
		dataPool.template setValue< 2 >( destAddress, noise/*sum of noise*/ );
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
	// Limit the depth.
	// Currently, 32 is the max depth of the GigaVoxels engine.
	if ( regionDepth >= 32 )
	{
		return GPUVoxelProducer::GPUVP_DATA_MAXRES;
	}

	// Shared memory declaration
	__shared__ uint3 brickRes;
	__shared__ float3 brickSize;
	__shared__ uint3 levelRes;
	__shared__ float3 levelResInv;

	brickRes = BrickRes::get();

	levelRes = make_uint3( 1 << regionDepth ) * brickRes;
	levelResInv = make_float3( 1.f ) / make_float3( levelRes );

	//int3 brickPos = make_int3( regionCoords * brickRes ) - BorderSize;
	//float3 brickPosF = make_float3( brickPos ) * levelResInv;

	// Since we work in the range [-1;1] below, the brick size is two time bigger
	brickSize = make_float3( 1.f ) / make_float3( 1 << regionDepth ) * 2.f;

	// Build the eight brick corners of a sphere centered in [0;0;0]
	float3 q000 = make_float3( regionCoords * brickRes ) * levelResInv * 2.f - 1.f;
	float3 q001 = make_float3( q000.x + brickSize.x, q000.y,			   q000.z);
	float3 q010 = make_float3( q000.x,				 q000.y + brickSize.y, q000.z);
	float3 q011 = make_float3( q000.x + brickSize.x, q000.y + brickSize.y, q000.z);
	float3 q100 = make_float3( q000.x,				 q000.y,			   q000.z + brickSize.z);
	float3 q101 = make_float3( q000.x + brickSize.x, q000.y,			   q000.z + brickSize.z);
	float3 q110 = make_float3( q000.x,				 q000.y + brickSize.y, q000.z + brickSize.z);
	float3 q111 = make_float3( q000.x + brickSize.x, q000.y + brickSize.y, q000.z + brickSize.z);

	// Test if any of the eight brick corner lies in the sphere
	if ( intersectTorus( q000, q001, q010, q011, q100, q101, q110, q111 ) )
	{
		return GPUVoxelProducer::GPUVP_DATA;
	}

	return GPUVoxelProducer::GPUVP_CONSTANT;
}

/******************************************************************************
 * Helper class to test if a point is inside a torus centered in [0,0,0]
 *
 * @param pPoint the point to test
 *
 * @return flag to tell wheter or not the point is inside the torus
 ******************************************************************************/
template< typename TDataStructureType >
__device__
inline bool ProducerKernel< TDataStructureType >::isInTorus( const float3 pPoint )
{
	float temp = cR - sqrt( pPoint.x*pPoint.x + pPoint.y*pPoint.y );
	return temp * temp + pPoint.z*pPoint.z < cr*cr;
}

/******************************************************************************
 * Helper class to test if a cube intersect the torus
 *
 * @param q01 point 1 of the plan 0 ( the plan of lowest height )
 * @param q02 point 2 of the plan 0 ( the plan of lowest height )
 * @param q03 point 3 of the plan 0 ( the plan of lowest height )
 * @param q04 point 4 of the plan 0 ( the plan of lowest height )
 * @param q11 point 1 of the plan 1 ( the plan of lowest height )
 * @param q12 point 2 of the plan 1 ( the plan of lowest height )
 * @param q13 point 3 of the plan 1 ( the plan of lowest height )
 * @param q14 point 4 of the plan 1 ( the plan of lowest height )
 *
 * @return flag to tell whether or not the cube intersects the torus
 ******************************************************************************/
template< typename TDataStructureType >
__device__
inline bool ProducerKernel< TDataStructureType >::intersectTorus( float3 q01, float3 q02, float3 q03, float3 q04, float3 q11, float3 q12, float3 q13, float3 q14 )
{
	// Rmq : we will only try if one of the both plans intersect the torus,
	// it is enough because the torus is centered in (0.0, 0.0, 0.0) and his rotation axe is 0z.

	// We try if the first plan intersect the torus
	float3 aux = make_float3(0.0, 0.0, q01.z);
	if ( cr > aux.z ) {

		float a = cR - sqrt(cr*cr - aux.z*aux.z);
		float b = cR + sqrt(cr*cr - aux.z*aux.z);
		float d1 = length(q01 - aux);
		float d2 = length(q02 - aux);
		float d3 = length(q03 - aux);
		float d4 = length(q04 - aux);

		if ( d1 < a || d2 < a || d3 < a || d4 < a ) {
			if ( d1 > a || d2 > a || d3 > a || d4 > a ) {
				return true;
			}
		} else if ( d1 < b || d2 < b || d3 < b || d4 < b ) {
			return true;
		}
	}

	// We try if the second plan intersect the torus
	aux = make_float3(0.0, 0.0, q11.z);
	if ( cr > aux.z ) {

		float a = cR - sqrt(cr*cr - aux.z*aux.z);
		float b = cR + sqrt(cr*cr - aux.z*aux.z);
		float d1 = length(q11 - aux);
		float d2 = length(q12 - aux);
		float d3 = length(q13 - aux);
		float d4 = length(q14 - aux);

		if ( d1 < a || d2 < a || d3 < a || d4 < a ) {
			if ( d1 > a || d2 > a || d3 > a || d4 > a ) {
				return true;
			}
		} else if ( d1 < b || d2 < b || d3 < b || d4 < b ) {
			return true;
		}
	}

	return false;
}
