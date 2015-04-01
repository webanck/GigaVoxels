/*
 * Copyright (C) 2011 Fabrice Neyret <Fabrice.Neyret@imag.fr>
 * Copyright (C) 2011 Cyril Crassin <Cyril.Crassin@icare3d.org>
 * Copyright (C) 2011 Morgan Armand <morgan.armand@gmail.com>
 *
 * This file is part of Gigavoxels.
 *
 * Gigavoxels is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Gigavoxels is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Gigavoxels.  If not, see <http://www.gnu.org/licenses/>.
 */

__device__
inline bool isInCube(float3 p)
{
	return (fabsf(p.x) <= 0.5f && fabsf(p.y) <= 0.5f && fabsf(p.z) <= 0.5f);
}

template < typename NodeRes, typename BrickRes, uint BorderSize, typename VolTreeKernelType >
template < typename GPUPoolKernelType >
__device__
inline uint CubeProducerKernel<NodeRes, BrickRes, BorderSize, VolTreeKernelType>
::produceData(GPUPoolKernelType &nodePool, uint requestID, uint processID, uint3 newElemAddress,
			  const LocalizationInfo &parentLocInfo, Loki::Int2Type<0>)
{
	const LocalizationInfo::CodeType parentLocCode = parentLocInfo.locCode;
	const LocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;

	if (processID < NodeRes::getNumElements())
	{
		uint3 subOffset = NodeRes::toFloat3(processID);

		uint3 regionCoords = parentLocCode.addLevel<NodeRes>(subOffset).get();
		uint regionDepth = parentLocDepth.addLevel().get();

		gigavoxels::OctreeNode newnode;
		newnode.childAddress=0;
		newnode.brickAddress = 0;

		GPUVoxelProducer::GPUVPRegionInfo nodeinfo = getRegionInfo(regionCoords, regionDepth);

		if (nodeinfo == GPUVoxelProducer::GPUVP_CONSTANT)
		{
			newnode.setTerminal(true);
		}
		else if (nodeinfo == GPUVoxelProducer::GPUVP_DATA)
		{
			newnode.setStoreBrick();
			newnode.setTerminal(false);
		}
		else if (nodeinfo == GPUVoxelProducer::GPUVP_DATA_MAXRES)
		{
			newnode.setStoreBrick();
			newnode.setTerminal(true);
		}

		// Write node info into the node pool
		nodePool.getChannel(Loki::Int2Type<0>()).set(newElemAddress.x + processID, newnode.childAddress);
		nodePool.getChannel(Loki::Int2Type<1>()).set(newElemAddress.x + processID, newnode.brickAddress);
	}

	return (0);
}

template < typename NodeRes, typename BrickRes, uint BorderSize, typename VolTreeKernelType >
template < typename GPUPoolKernelType >
__device__
inline uint CubeProducerKernel<NodeRes, BrickRes, BorderSize, VolTreeKernelType>
::produceData(GPUPoolKernelType &dataPool, uint requestID, uint processID, uint3 newElemAddress,
			  const LocalizationInfo &parentLocInfo, Loki::Int2Type<1>)
{
	const LocalizationInfo::CodeType parentLocCode = parentLocInfo.locCode;
	const LocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;

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
	{
		for (elemOffset.y = 0; elemOffset.y < elemSize.y; elemOffset.y += blockDim.y)
		{
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

					float4 voxelColor = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
					float4 voxelNormal = make_float4(normalize(posF), 1.0f);

					// if the voxel is located inside the unit sphere
					if (isInCube(posF))
					{
						voxelColor.w = 1.0f;
					}

					// alpha pre-multiplication
					voxelColor.x *= voxelColor.w;
					voxelColor.y *= voxelColor.w;
					voxelColor.z *= voxelColor.w;

					// compute the new element's address
					uint3 destAddress = newElemAddress + locOffset;
					// write the voxel's color in the first field
					dataPool.template setValue<0>(destAddress, voxelColor);
					// write the voxel's normal in the second field
					dataPool.template setValue<1>(destAddress, voxelNormal);
				}
			}
		}
	}

	return (0);
}

template < typename NodeRes, typename BrickRes, uint BorderSize, typename VolTreeKernelType >
__device__
inline GPUVoxelProducer::GPUVPRegionInfo CubeProducerKernel<NodeRes, BrickRes, BorderSize, VolTreeKernelType>
::getRegionInfo(uint3 regionCoords, uint regionDepth)
{
	// Limit the depth
	if (regionDepth >= 32)
	{
		return GPUVoxelProducer::GPUVP_DATA_MAXRES;
	}

	__shared__ uint3 brickRes;
	__shared__ float3 brickSize;
	__shared__ uint3 levelRes;
	__shared__ float3 levelResInv;

	brickRes = BrickRes::get();

	levelRes = make_uint3(1 << regionDepth) * brickRes;
	levelResInv = make_float3(1.0f) / make_float3(levelRes);

	int3 brickPos = make_int3(regionCoords * brickRes) - BorderSize;
	float3 brickPosF = make_float3(brickPos) * levelResInv;

	// since we work in the range [-1;1] below, the brick size is two time bigger
	brickSize = make_float3(1.0f) / make_float3(1 << regionDepth) * 2.0f;

	// test if any of the eight brick corner lie in the sphere
	float3 q000 = make_float3(regionCoords * brickRes) * levelResInv * 2.0f - 1.0f;
	float3 q001 = make_float3(q000.x + brickSize.x,	q000.y,					q000.z);
	float3 q010 = make_float3(q000.x,				q000.y + brickSize.y,	q000.z);
	float3 q011 = make_float3(q000.x + brickSize.x,	q000.y + brickSize.y,	q000.z);
	float3 q100 = make_float3(q000.x,				q000.y,					q000.z + brickSize.z);
	float3 q101 = make_float3(q000.x + brickSize.x,	q000.y,					q000.z + brickSize.z);
	float3 q110 = make_float3(q000.x,				q000.y + brickSize.y,	q000.z + brickSize.z);
	float3 q111 = make_float3(q000.x + brickSize.x,	q000.y + brickSize.y,	q000.z + brickSize.z);

	if (isInSphere(q000) || isInSphere(q001) || isInSphere(q010) || isInSphere(q011) ||
		isInSphere(q100) || isInSphere(q101) || isInSphere(q110) || isInSphere(q111))
		return GPUVoxelProducer::GPUVP_DATA;

	return GPUVoxelProducer::GPUVP_CONSTANT;
}
