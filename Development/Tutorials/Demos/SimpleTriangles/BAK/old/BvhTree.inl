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
 * @param pNodePoolSize Cache size used to store nodes
 * @param pVertexPoolSize Cache size used to store bricks
 ******************************************************************************/
template< class DataTList >
BvhTree< DataTList >::BvhTree( uint nodePoolSize, uint vertexPoolSize )
{
	// Node pool initialization
	_nodePool = new GvCore::Array3DGPULinear< VolTreeBVHNodeUser >( dim3( nodePoolSize, 1, 1 ) );
	_nodePool->fill( 0 );

	// Data pool initialization
	dataPool = new GvCore::GPUPoolHost< GvCore::Array3DGPULinear, DataTList >( make_uint3( vertexPoolSize, 1, 1 ) );
	//dataPool->fill(0);

	// Associated device-side object initialization
	_kernelObject.dataPool = dataPool->getKernelObject();
	GvCore::Array3DGPULinear< VolTreeBVHNodeStorageUINT >* nodesArrayGPUStoragePtr = ( GvCore::Array3DGPULinear< VolTreeBVHNodeStorageUINT >* )_nodePool;
	_kernelObject._volumeTreeBVHArray = nodesArrayGPUStoragePtr->getDeviceArray();
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< class DataTList >
BvhTree<DataTList>::~BvhTree()
{
	delete _nodePool;
	delete dataPool;
	//delete vertexPosArrayGPU;
}

/******************************************************************************
 * CUDA initialization
 ******************************************************************************/
template< class DataTList >
void BvhTree< DataTList >::cuda_Init()
{
	volumeTreeBVHTexLinear.normalized = false;					  // access with normalized texture coordinates
	volumeTreeBVHTexLinear.filterMode = cudaFilterModePoint;		// nearest interpolation
	volumeTreeBVHTexLinear.addressMode[ 0 ] = cudaAddressModeClamp;   // wrap texture coordinates
	volumeTreeBVHTexLinear.addressMode[ 1 ] = cudaAddressModeClamp;
	volumeTreeBVHTexLinear.addressMode[ 2 ] = cudaAddressModeClamp;

	CUDA_SAFE_CALL( cudaBindTexture( NULL, volumeTreeBVHTexLinear, _nodePool->getPointer() ) );

	CUT_CHECK_ERROR( "BvhTree::cuda_Init end" );
}

/******************************************************************************
 * Initialize the cache
 *
 * @param pBvhTrianglesManager ...
 ******************************************************************************/
template< class DataTList >
void BvhTree< DataTList >::initCache( BVHTrianglesManager< DataTList, BVH_DATA_PAGE_SIZE >* bvhTrianglesManager )
{
	// TODO : Stream Me \o/

	// Initialize the node buffer
	memcpyArray( _nodePool, (VolTreeBVHNodeUser*)( bvhTrianglesManager->getNodeBufferArray()->getPointer() ), bvhTrianglesManager->getNodeBufferArray()->getResolution().x );
	
	//memcpyArray( dataPool->getChannel( Loki::Int2Type< 0 >() ), bvhTrianglesManager->getVertexBufferPool()->getChannel( Loki::Int2Type< 0 >() )->getPointer( 0 ) );
}

/******************************************************************************
 * Clear
 ******************************************************************************/
template < class DataTList >
void BvhTree<DataTList>::clear()
{
	/*((Array3DGPULinear<uint>*)volTreeChildArrayGPU )->fill(0);
	((Array3DGPULinear<uint>*)volTreeDataArrayGPU )->fill(0);*/
}
