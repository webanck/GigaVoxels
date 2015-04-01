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

#ifndef _CUBEPRODUCER_H_
#define _CUBEPRODUCER_H_

#include <IProvider.h>
#include <IProviderKernel.h>
#include <GPUCacheHelper.h>

#include "CubeProducer.hcu"

template <typename NodeRes, typename BrickRes, uint BorderSize, typename VolumeTreeType>
class CubeProducer
	: public gigavoxels::IProvider< 0, CubeProducer<NodeRes, BrickRes, BorderSize, VolumeTreeType> >
	, public gigavoxels::IProvider< 1, CubeProducer<NodeRes, BrickRes, BorderSize, VolumeTreeType> >
{
public:

	//! Typedef the kernel part of the shader
	typedef CubeProducerKernel< NodeRes, BrickRes, BorderSize,
		typename VolumeTreeType::VolTreeKernelType >				KernelProducerType;

	//! Implement the produceData method for the channel 0 (nodes)
	template < typename ElementRes, typename GPUPoolType, typename PageTableType >
	inline void produceData(uint numElems,
		thrust::device_vector<uint> *nodesAddressCompactList,
		thrust::device_vector<uint> *elemAddressCompactList,
		GPUPoolType gpuPool, PageTableType pageTable, Loki::Int2Type<0>)
	{
		gigavoxels::IProviderKernel<0, KernelProducerType> kernelProvider(kernelProducer);

		dim3 blockSize(32, 1, 1);

		uint *nodesAddressList = thrust::raw_pointer_cast(&(*nodesAddressCompactList)[0]);
		uint *elemAddressList = thrust::raw_pointer_cast(&(*elemAddressCompactList)[0]);

		cacheHelper.genericWriteIntoCache<ElementRes>(numElems, nodesAddressList,
			elemAddressList, gpuPool, kernelProvider, pageTable, blockSize);
	}

	//! Implement the produceData method for the channel 1 (bricks)
	template < typename ElementRes, typename GPUPoolType, typename PageTableType >
	inline void produceData(uint numElems,
		thrust::device_vector<uint> *nodesAddressCompactList,
		thrust::device_vector<uint> *elemAddressCompactList,
		GPUPoolType gpuPool, PageTableType pageTable, Loki::Int2Type<1>)
	{
		gigavoxels::IProviderKernel<1, KernelProducerType> kernelProvider(kernelProducer);

		dim3 blockSize(16, 8, 1);

		uint *nodesAddressList = thrust::raw_pointer_cast(&(*nodesAddressCompactList)[0]);
		uint *elemAddressList = thrust::raw_pointer_cast(&(*elemAddressCompactList)[0]);

		cacheHelper.genericWriteIntoCache<ElementRes>(numElems, nodesAddressList,
			elemAddressList, gpuPool, kernelProvider, pageTable, blockSize);
	}

private:
	GPUCacheHelper		cacheHelper;
	KernelProducerType	kernelProducer;
};

#endif // !_CUBEPRODUCER_H_