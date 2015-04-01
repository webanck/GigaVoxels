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
//#include <GvStructure/GvVolumeTree.h>

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/**
 * Transfer function texture
 */
texture< float4, cudaTextureType1D, cudaReadModeElementType > transferFunctionTexture;

/**
 * Volume texture (3D normal + signed distance field)
 */
texture< float4, cudaTextureType3D, cudaReadModeElementType > volumeTex;

/******************************************************************************
 * Map a distance to a color (with a transfer function)
 *
 * @param dist the distance to map
 *
 * @return The color corresponding to the distance
 ******************************************************************************/
__device__
inline float4 distToColor( float dist )
{
	return tex1D( transferFunctionTexture, dist );
}

/******************************************************************************
 * Get the RGBA data of distance field + noise
 *
 * @param p 3D position
 *
 * @return ...
 ******************************************************************************/
__device__
float4 getRGBA( float3 voxelPosF, uint3 levelRes)
{
	// Type definition for the noise
	typedef GvUtils::GvNoiseKernel Noise;

	float4 voxelNormalAndDist = tex3D( volumeTex, voxelPosF.x, voxelPosF.y, voxelPosF.z );
	float3 voxelNormal = normalize( make_float3( voxelNormalAndDist.x, voxelNormalAndDist.y, voxelNormalAndDist.z ) );

	// compute noise
	float dist_noise = 0.0f;
	for ( float frequency = cNoiseFirstFrequency; frequency < levelRes.x; frequency *= 2.f )
	{
		dist_noise += cNoiseStrength / frequency * Noise::getValue( frequency * voxelPosF );
	}

	// compute color
	float4 voxelRGBA = distToColor( clamp( 0.5f - 0.5f * ( voxelNormalAndDist.w + dist_noise ) * cNoiseFirstFrequency, 0.f, 1.f ) );

	return voxelRGBA;
}

/******************************************************************************
 * Get the normal of distance field + noise
 *
 * @param p 3D position
 *
 * @return ...
 ******************************************************************************/
__device__
float3 getNormal(float3 voxelPosF, uint3 levelRes)
{
	// Type definition for the noise
	typedef GvUtils::GvNoiseKernel Noise;

	float4 voxelNormalAndDist = tex3D( volumeTex, voxelPosF.x, voxelPosF.y, voxelPosF.z );
	float3 voxelNormal = normalize( make_float3( voxelNormalAndDist.x, voxelNormalAndDist.y, voxelNormalAndDist.z ) );

	float eps = 0.5f / (float) levelRes.x;

	// compute noise
	float3 grad_noise = make_float3(0.0f);
	for ( float frequency = cNoiseFirstFrequency; frequency < levelRes.x; frequency *= 2.f )
	{
		grad_noise.x +=  cNoiseStrength / frequency * Noise::getValue( frequency * ( voxelPosF + make_float3( eps, 0.0f, 0.0f ) ) )
						-cNoiseStrength / frequency * Noise::getValue( frequency * ( voxelPosF - make_float3( eps, 0.0f, 0.0f ) ) );
		grad_noise.y +=  cNoiseStrength / frequency * Noise::getValue( frequency * ( voxelPosF + make_float3( 0.0f, eps, 0.0f ) ) )
						-cNoiseStrength / frequency * Noise::getValue( frequency * ( voxelPosF - make_float3( 0.0f, eps, 0.0f ) ) );
		grad_noise.z +=  cNoiseStrength / frequency * Noise::getValue( frequency * ( voxelPosF + make_float3( 0.0f, 0.0f, eps ) ) )
						-cNoiseStrength / frequency * Noise::getValue( frequency * ( voxelPosF - make_float3( 0.0f, 0.0f, eps ) ) );
	}

	grad_noise *= 0.5f / eps;

	voxelNormal = normalize( voxelNormal + grad_noise - dot( grad_noise, voxelNormal ) * voxelNormal );
	return voxelNormal;
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
	//_dataStructureKernel = pDataStructure;
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
	const GvCore::GvLocalizationInfo::CodeType parentLocCode = parentLocInfo.locCode;
	const GvCore::GvLocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;

	if ( processID < NodeRes::getNumElements() )
	{
		uint3 subOffset = NodeRes::toFloat3( processID );

		uint3 regionCoords = parentLocCode.addLevel< NodeRes >( subOffset ).get();
		uint regionDepth = parentLocDepth.addLevel().get();

		GvStructure::GvNode newnode;
		newnode.childAddress = 0;
		newnode.brickAddress = 0;

		GPUVoxelProducer::GPUVPRegionInfo nodeinfo = getRegionInfo( regionCoords, regionDepth );

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

		// Write node info into the node pool
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
	const GvCore::GvLocalizationInfo::CodeType parentLocCode = parentLocInfo.locCode;
	const GvCore::GvLocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;

	//printf( "\nDepth = %d", parentLocDepth.get() );

	__shared__ uint3 brickRes;
	__shared__ uint3 levelRes; 
	__shared__ float3 levelResInv; 
	__shared__ int3 brickPos;
	__shared__ float3 brickPosF;

	brickRes = BrickRes::get();
	levelRes = make_uint3( 1 << parentLocDepth.get() ) * brickRes;
	levelResInv = make_float3( 1.f ) / make_float3( levelRes );

	brickPos = make_int3( parentLocCode.get() * brickRes) - BorderSize;
	brickPosF = make_float3( brickPos ) * levelResInv;

	uint3 elemSize = BrickRes::get() + make_uint3( 2 * BorderSize );
	uint3 elemOffset;

	for ( elemOffset.z = 0; elemOffset.z < elemSize.z; elemOffset.z += blockDim.z )
	{
		for ( elemOffset.y = 0; elemOffset.y < elemSize.y; elemOffset.y += blockDim.y )
		{
			for ( elemOffset.x = 0; elemOffset.x < elemSize.x; elemOffset.x += blockDim.x )
			{
				uint3 locOffset = elemOffset + make_uint3( threadIdx.x, threadIdx.y, threadIdx.z );

				if ( locOffset.x < elemSize.x && locOffset.y < elemSize.y && locOffset.z < elemSize.z )
				{
					// Position of the current voxel's center (relative to the brick)
					float3 voxelPosInBrickF = ( make_float3( locOffset ) + 0.5f ) * levelResInv;
					// Position of the current voxel's center (absolute, in [0.0;1.0] range)
					float3 voxelPosF = brickPosF + voxelPosInBrickF;

					// Compute data
					float4 voxelColor = getRGBA( voxelPosF, levelRes );
					float3 voxelNormal = getNormal( voxelPosF, levelRes );					

					// Compute the new element's address
					uint3 destAddress = newElemAddress + locOffset;
					// Write the voxel's color in the first field
					dataPool.template setValue< 0 >( destAddress, voxelColor );
					// Write the voxel's normal in the second field
					dataPool.template setValue< 1 >( destAddress, make_float4( voxelNormal, 0.f ) );
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
 * @param regionDepth region deptj
 *
 * @return the type of the region
 ******************************************************************************/
template< typename TDataStructureType >
__device__
inline GPUVoxelProducer::GPUVPRegionInfo ProducerKernel< TDataStructureType >
::getRegionInfo( uint3 regionCoords, uint regionDepth )
{
	//if (regionDepth <= 4)
	//return GPUVoxelProducer::GPUVP_DATA;

	// Limit the depth
	if ( regionDepth >= 32 )
	{
		return GPUVoxelProducer::GPUVP_DATA_MAXRES;
	}

	//const GvCore::GvLocalizationInfo::CodeType parentLocCode = parentLocInfo.locCode;
	//const GvCore::GvLocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;

	uint3 brickRes;
	uint3 levelRes;
	float3 levelResInv;
	int3 brickPos;
	float3 brickPosF;

	brickRes = BrickRes::get();
	levelRes = make_uint3( 1 << regionDepth ) * brickRes;
	levelResInv = make_float3( 1.f ) / make_float3( levelRes );

	brickPos = make_int3( regionCoords * brickRes ) - BorderSize;
	brickPosF = make_float3( brickPos ) * levelResInv;

	uint3 elemSize = BrickRes::get() + make_uint3( 2 * BorderSize );
	uint3 elemOffset;

	bool isEmpty = true;

	float brickSize = 1.0f / (float)( 1 << regionDepth );

	for ( elemOffset.z = 0; elemOffset.z < elemSize.z && isEmpty; elemOffset.z++ )
	{
		for ( elemOffset.y = 0; elemOffset.y < elemSize.y  && isEmpty; elemOffset.y++ )
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

					float4 voxelColor = getRGBA( voxelPosF, levelRes );

					if ( voxelColor.w > 0.0f )
					{
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
