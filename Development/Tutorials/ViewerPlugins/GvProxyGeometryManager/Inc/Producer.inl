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
 * This method is called by the cache manager when you have to produce data for a given pool.
 * Implement the produceData method for the channel 0 (nodes)
 *
 * @param pNumElems the number of elements you have to produce.
 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
 * @param pGpuPool the pool for which we need to produce elements.
 * @param pPageTable the page table associated to the pool
 * @param Loki::Int2Type< 0 > corresponds to the index of the node pool
 ******************************************************************************/
template< typename NodeRes, typename BrickRes, uint BorderSize, typename VolumeTreeType >
template< typename ElementRes, typename GPUPoolType, typename PageTableType >
inline void Producer< NodeRes, BrickRes, BorderSize, VolumeTreeType >
::produceData( uint pNumElems,
				thrust::device_vector< uint >* pNodesAddressCompactList,
				thrust::device_vector< uint >* pElemAddressCompactList,
				GPUPoolType& pGpuPool,
				PageTableType pPageTable,
				Loki::Int2Type< 0 > )
{
	// Initialize the device-side producer
	GvCore::GvIProviderKernel< 0, KernelProducerType > kernelProvider( _kernelProducer );

	// Define kernel block size
	dim3 blockSize( 32, 1, 1 );

	// Retrieve updated addresses
	uint* nodesAddressList = thrust::raw_pointer_cast( &(*pNodesAddressCompactList)[ 0 ] );
	uint* elemAddressList = thrust::raw_pointer_cast( &(*pElemAddressCompactList)[ 0 ] );

	// Call cache helper to write into cache
	_cacheHelper.genericWriteIntoCache< ElementRes >( pNumElems, nodesAddressList,
								elemAddressList, pGpuPool, kernelProvider, pPageTable, blockSize );
}

/******************************************************************************
 * This method is called by the cache manager when you have to produce data for a given pool.
 * Implement the produceData method for the channel 1 (bricks)
 *
 * @param pNumElems the number of elements you have to produce.
 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
 * @param pGpuPool the pool for which we need to produce elements.
 * @param pPageTable the page table associated to the pool
 * @param Loki::Int2Type< 1 > corresponds to the index of the brick pool
 ******************************************************************************/
template< typename NodeRes, typename BrickRes, uint BorderSize, typename VolumeTreeType >
template< typename ElementRes, typename GPUPoolType, typename PageTableType >
inline void Producer< NodeRes, BrickRes, BorderSize, VolumeTreeType >
::produceData( uint pNumElems,
				thrust::device_vector< uint >* pNodesAddressCompactList,
				thrust::device_vector< uint >* pElemAddressCompactList,
				GPUPoolType& pGpuPool,
				PageTableType pPageTable,
				Loki::Int2Type< 1 > )
{
	// Initialize the device-side producer
	GvCore::GvIProviderKernel< 1, KernelProducerType > kernelProvider( _kernelProducer );

	// Define kernel block size
	dim3 blockSize( 16, 8, 1 );

	// Retrieve updated addresses
	uint* nodesAddressList = thrust::raw_pointer_cast( &(*pNodesAddressCompactList)[ 0 ] );
	uint* elemAddressList = thrust::raw_pointer_cast( &(*pElemAddressCompactList)[ 0 ] );

	// Call cache helper to write into cache
	_cacheHelper.genericWriteIntoCache< ElementRes >( pNumElems, nodesAddressList,
								elemAddressList, pGpuPool, kernelProvider, pPageTable, blockSize );
}
