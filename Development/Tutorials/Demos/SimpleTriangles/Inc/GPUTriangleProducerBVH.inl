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

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 *
 * @param baseFileName ...
 ******************************************************************************/
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::GPUTriangleProducerBVH()
:	GvCore::GvProvider< TDataStructureType, TDataProductionManager >()
,	 _nodesBuffer( NULL )
,	 _dataBuffer( NULL )
,	_kernelProducer()
,	_bvhTrianglesManager( NULL )
,	_cacheHelper()
,	_filename()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::~GPUTriangleProducerBVH()
{
}

/******************************************************************************
 * Initialize
 *
 * @param pDataStructure data structure
 * @param pDataProductionManager data production manager
 ******************************************************************************/
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
inline void GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::initialize( TDataStructureType* pDataStructure, TDataProductionManager* pDataProductionManager )
{
	assert( ! _filename.empty() );

	// Call parent class
	GvCore::GvProvider< TDataStructureType, TDataProductionManager >::initialize( pDataStructure, pDataProductionManager );

	// Triangles manager's initialization
	_bvhTrianglesManager = new BVHTrianglesManager< DataTypeList, DataPageSize >();

#if 1
	//_bvhTrianglesManager->loadPowerPlant( baseFileName );
	_bvhTrianglesManager->loadMesh( _filename );
	//_bvhTrianglesManager->saveRawMesh( baseFileName );
#else
	_bvhTrianglesManager->loadRawMesh( baseFileName );
#endif

	_bvhTrianglesManager->generateBuffers( 2 );
	_nodesBuffer = _bvhTrianglesManager->getNodesBuffer();
	_dataBuffer = _bvhTrianglesManager->getDataBuffer();

	// Producer's device_side associated object initialization
	_kernelProducer.init( (VolTreeBVHNodeStorage*)_nodesBuffer->getGPUMappedPointer(), _dataBuffer->getKernelPool() );
}

/******************************************************************************
 * Finalize
 ******************************************************************************/
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
inline void GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::finalize()
{
}

/******************************************************************************
 * Get the triangles manager
 *
 * @return the triangles manager
 ******************************************************************************/
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
BVHTrianglesManager< typename GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >::DataTypeList, DataPageSize >* GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::getBVHTrianglesManager()
{
	return _bvhTrianglesManager;
}

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
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
inline void GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::produceData( uint pNumElems,
				thrust::device_vector< uint >* pNodesAddressCompactList,
				thrust::device_vector< uint >* pElemAddressCompactList,
				Loki::Int2Type< 0 > )
{
	// NOT USED ATM
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
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
inline void GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::produceData( uint pNumElems,
				thrust::device_vector< uint >* pNodesAddressCompactList,
				thrust::device_vector< uint >* pElemAddressCompactList,
				Loki::Int2Type< 1 > )
{
	// Initialize the device-side producer
	IBvhTreeProviderKernel< 1, KernelProducerType > kernelProvider( _kernelProducer );

	// Define kernel block size
	dim3 blockSize( BVH_DATA_PAGE_SIZE, 1, 1 );

	// Retrieve updated addresses
	uint* nodesAddressList = thrust::raw_pointer_cast( &(*pNodesAddressCompactList)[ 0 ] );
	uint* elemAddressList = thrust::raw_pointer_cast( &(*pElemAddressCompactList)[ 0 ] );

	// Call cache helper to write into cache
	//
	// - this function call encapsulates a kernel launch to produce data on device
	// - i.e. the associated device-side producer will call its device function "ProducerKernel::produceData< 1 >()"
	DataPoolType* pool = this->_dataPool;
	_cacheHelper.template genericWriteIntoCache< typename TDataProductionManager::DataCacheResolution >( pNumElems, nodesAddressList, elemAddressList, this->_dataStructure, pool, kernelProvider, blockSize );
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
void GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::renderGL()
{
	_bvhTrianglesManager->renderGL();
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
void GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::renderFullGL()
{
	_bvhTrianglesManager->renderFullGL();
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
void GPUTriangleProducerBVH< TDataStructureType, DataPageSize, TDataProductionManager >
::renderDebugGL()
{
	_bvhTrianglesManager->renderDebugGL();
}
