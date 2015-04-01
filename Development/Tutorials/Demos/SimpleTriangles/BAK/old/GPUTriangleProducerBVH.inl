/******************************************************************************
 * ...
 *
 * @param numElems ...
 * @param nodesAddressCompactList ...
 * @param elemAddressCompactList ...
 * @param gpuPool ...
 * @param pageTable ...
 ******************************************************************************/
template < typename MeshVertexTList, uint DataPageSize, typename BvhTreeType >
template< typename ElementRes, typename GPUPoolType, typename PageTableType >
inline void GPUTriangleProducerBVH< MeshVertexTList, DataPageSize, BvhTreeType >
::produceData( uint numElems,
		thrust::device_vector< uint > *nodesAddressCompactList,
		thrust::device_vector< uint > *elemAddressCompactList,
		GPUPoolType gpuPool, PageTableType pageTable, Loki::Int2Type< 0 > )
{
	// NOT USED ATM
}

/******************************************************************************
 * ...
 *
 * @param numElems ...
 * @param nodesAddressCompactList ...
 * @param elemAddressCompactList ...
 * @param gpuPool ...
 * @param pageTable ...
 ******************************************************************************/
template < typename MeshVertexTList, uint DataPageSize, typename BvhTreeType >
template< typename ElementRes, typename GPUPoolType, typename PageTableType >
inline void GPUTriangleProducerBVH< MeshVertexTList, DataPageSize, BvhTreeType >
::produceData( uint numElems,
		thrust::device_vector< uint > *nodesAddressCompactList,
		thrust::device_vector< uint > *elemAddressCompactList,
		GPUPoolType gpuPool, PageTableType pageTable, Loki::Int2Type< 1 > )
{
	IBvhTreeProviderKernel< 1, KernelProducerType > lKernelProvider( kernelProducer );

	// Define kernel block size
	dim3 blockSize( BVH_DATA_PAGE_SIZE, 1, 1 );

	// Retrieve updated addresses
	uint* nodesAddressList = thrust::raw_pointer_cast( &(*nodesAddressCompactList)[0] );
	uint* elemAddressList = thrust::raw_pointer_cast( &(*elemAddressCompactList)[0] );

	// Call GPU cache helper to write into cache
	_cacheHelper.genericWriteIntoCache< ElementRes >( numElems, nodesAddressList,
		elemAddressList, _bvhTree, gpuPool, lKernelProvider, pageTable, blockSize );
}
