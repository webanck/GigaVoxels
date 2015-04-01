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

/******************************************************************************
 * ...
 ******************************************************************************/
__device__
float getDensity( const float3 posInWorld )
{
	// Type definition for the noise
	typedef GvUtils::GvNoiseKernel Noise;

	//posInWorld.x += 4.f;

	//const int PI = 3.141592f;
	//return cosf(posInWorld.x * PI);
	float density = -posInWorld.y;

	density -= 0.5f;
	density += 0.5000f * Noise::getValue( 1.f * posInWorld.x, 1.f * posInWorld.y, 1.f * posInWorld.z );
	density += 0.2500f * Noise::getValue( 2.f * posInWorld.x, 2.f * posInWorld.y, 2.f * posInWorld.z );
	density += 0.1250f * Noise::getValue( 4.f * posInWorld.x, 4.f * posInWorld.y, 4.f * posInWorld.z );
	density += 0.0625f * Noise::getValue( 8.f * posInWorld.x, 8.f * posInWorld.y, 8.f * posInWorld.z );

	return density;
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
	//parentLocDepth++;

	if ( processID < NodeRes::getNumElements() )
	{
		uint3 subOffset;
		subOffset.x = processID & NodeRes::xLog2;
		subOffset.y = (processID >> NodeRes::xLog2) & NodeRes::yLog2;
		subOffset.z = (processID >> (NodeRes::xLog2 + NodeRes::yLog2)) & NodeRes::zLog2;

		uint3 regionCoords = parentLocCode->addLevel<NodeRes>(subOffset).get();
		uint regionDepth = parentLocDepth->addLevel().get();

		GvStructure::GvNode newnode;
		newnode.childAddress=0;

		GPUVoxelProducer::GPUVPRegionInfo nodeinfo = getRegionInfo( regionCoords, regionDepth );

		if ( nodeinfo == GPUVoxelProducer::GPUVP_CONSTANT )
		{
		//	newnode.data.setValue(0.0f);
		//	newnode.setStoreValue();
			newnode.setTerminal(true);
		}
		else if ( nodeinfo == GPUVoxelProducer::GPUVP_DATA )
		{
		//	newnode.data.brickAddress = 0;
			newnode.setStoreBrick();
			newnode.setTerminal( false );
		}
		else if ( nodeinfo == GPUVoxelProducer::GPUVP_DATA_MAXRES )
		{
		//	newnode.data.brickAddress = 0;
			newnode.setStoreBrick();
			newnode.setTerminal( true );
		}

		// Write node info into the node pool
		nodePool.getChannel(Loki::Int2Type<0>()).set(newElemAddress.x + processID, newnode.childAddress);
		//nodePool.getChannel(Loki::Int2Type<1>()).set(newElemAddress.x + processID, newnode.data.brickAddress);
		nodePool.getChannel(Loki::Int2Type<1>()).set(newElemAddress.x + processID, newnode.brickAddress);
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
	// typedef Loki::TL::MakeTypelist< half4 >::Result DataType;
	//
	// In this tutorial, we have choosen one channel containing color at channel 0.

	// Retrieve current brick localization information code and depth
	const GvCore::GvLocalizationInfo::CodeType parentLocCode = parentLocInfo.locCode;
	const GvCore::GvLocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;

	//parentLocDepth++;

	__shared__ uint3 brickRes;
	__shared__ uint3 levelRes; 
	__shared__ float3 levelResInv; 
	__shared__ int3 brickPos;
	__shared__ float3 brickPosF;

	brickRes = BrickRes::get();
	levelRes = make_uint3(1 << parentLocDepth.get()) * brickRes;
	levelResInv = make_float3(1.0f) / make_float3(levelRes);

	brickPos = make_int3(parentLocCode.get() * brickRes) - BorderSize;
	brickPosF = make_float3(brickPos) * levelResInv;

	uint3 elemSize = BrickRes::get() + make_uint3(2 * BorderSize);
	uint3 elemOffset;

	for (elemOffset.z = 0; elemOffset.z < elemSize.z; elemOffset.z += blockDim.z)
	for (elemOffset.y = 0; elemOffset.y < elemSize.y; elemOffset.y += blockDim.y)
	for (elemOffset.x = 0; elemOffset.x < elemSize.x; elemOffset.x += blockDim.x)
	{
		uint3 locOffset = elemOffset + make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);

		if (locOffset.x < elemSize.x && locOffset.y < elemSize.y && locOffset.z < elemSize.z)
		{
			// Position of the current voxel's center (relative to the brick)
			float3 voxelPosInBrickF = (make_float3(locOffset) + 0.5f) * levelResInv;
			// Position of the current voxel's center (absolute, in [0.0;1.0] range)
			float3 voxelPosF = brickPosF + voxelPosInBrickF;
			// Position of the current voxel's center (scaled to the range [-1.0;1.0])
			float3 posF = voxelPosF * 2.0f - 1.0f;

			// the final voxel's data
			float4 data;

			// w component hold the density
			data.w = getDensity(posF);

			// xyz components holds the gradient
			float step = 1.0f / levelRes.x;

			float3 grad;
			grad.x = getDensity(posF + make_float3(step, 0.0f, 0.0f)) - getDensity(posF - make_float3(step, 0.0f, 0.0f));
			grad.y = getDensity(posF + make_float3(0.0f, step, 0.0f)) - getDensity(posF - make_float3(0.0f, step, 0.0f));
			grad.z = getDensity(posF + make_float3(0.0f, 0.0f, step)) - getDensity(posF - make_float3(0.0f, 0.0f, step));
			grad = normalize(-grad);

			// compute the new element's address
			uint3 destAddress = newElemAddress + locOffset;
			// write the voxel's data in the first field
			dataPool.template setValue< 0 >( destAddress, data );
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
	//__shared__ float3 levelRes;
	//__shared__ float3 nodeSize;

	//if (regionDepth >= 31)
	//	return GPUVoxelProducer::GPUVP_DATA_MAXRES;
	////else
	////	return GPUVoxelProducer::GPUVP_DATA;

	//levelRes = make_float3(1 << regionDepth);
	//nodeSize = make_float3(1.0f) / levelRes;

	//float3 nodePosInLocal = make_float3(regionCoords) * nodeSize;
	//float3 nodeCenterInLocal = nodePosInLocal + nodeSize / 2.0f;
	//float3 nodeCenterInWorld = nodeCenterInLocal * 2.0f - 1.0f;

	//float maxDensity = -1.f;
	//float3 offset;

	//// get the minimal distance
	//for (offset.z = -1.0f; offset.z <= 1.0f; offset.z += 1.0f)
	//for (offset.y = -1.0f; offset.y <= 1.0f; offset.y += 1.0f)
	//for (offset.x = -1.0f; offset.x <= 1.0f; offset.x += 1.0f)
	//{
	//	maxDensity = max(maxDensity, getDensity(nodeCenterInWorld + offset * nodeSize));
	//}

	////if (maxDensity > 0.0f)
	//return GPUVoxelProducer::GPUVP_DATA;
	////else
	////	return GPUVoxelProducer::GPUVP_CONSTANT;
	//parentLocDepth++;

	__shared__ uint3 brickRes;
	__shared__ uint3 levelRes; 
	__shared__ float3 levelResInv; 
	__shared__ int3 brickPos;
	__shared__ float3 brickPosF;

	brickRes = BrickRes::get();
	levelRes = make_uint3(1 << regionDepth) * brickRes;
	levelResInv = make_float3(1.0f) / make_float3(levelRes);

	brickPos = make_int3(regionCoords * brickRes) - BorderSize;
	brickPosF = make_float3(brickPos) * levelResInv;

	uint3 elemSize = BrickRes::get() + make_uint3(2 * BorderSize);
	uint3 elemOffset;

	float maxDensity = 0.f;

	for (elemOffset.z = 0; elemOffset.z < elemSize.z; elemOffset.z += blockDim.z)
	for (elemOffset.y = 0; elemOffset.y < elemSize.y; elemOffset.y += blockDim.y)
	for (elemOffset.x = 0; elemOffset.x < elemSize.x; elemOffset.x += blockDim.x)
	{
		uint3 locOffset = elemOffset + make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);

		if (locOffset.x < elemSize.x && locOffset.y < elemSize.y && locOffset.z < elemSize.z)
		{
			// Position of the current voxel's center (relative to the brick)
			float3 voxelPosInBrickF = (make_float3(locOffset) + 0.5f) * levelResInv;
			// Position of the current voxel's center (absolute, in [0.0;1.0] range)
			float3 voxelPosF = brickPosF + voxelPosInBrickF;
			// Position of the current voxel's center (scaled to the range [-1.0;1.0])
			float3 posF = voxelPosF * 2.0f - 1.0f;

			// the final voxel's data
			maxDensity = max(maxDensity, getDensity(posF));
		}
	}

	if ( maxDensity > 0.f )
	{
		return GPUVoxelProducer::GPUVP_DATA;
	}
	else
	{
		return GPUVoxelProducer::GPUVP_CONSTANT;
	}
}
