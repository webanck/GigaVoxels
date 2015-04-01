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

// GigaVoxels
#include "GvCore/GvError.h"

// STL
#include <iostream>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvStructure
{

/******************************************************************************
 * Constructor.
 *
 * @param nodesCacheSize Cache size used to store nodes
 * @param bricksCacheRes Cache size used to store bricks
 * @param graphicsInteroperability Flag used for graphics interoperability
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize, typename TDataStructureKernelType >
GvVolumeTree< DataTList, NodeTileRes, BrickRes, BorderSize, TDataStructureKernelType >
::GvVolumeTree( const uint3& nodesCacheSize, const uint3& bricksCacheRes, uint graphicsInteroperability )
:	GvIDataStructure()
,	_childArray( NULL )
,	_dataArray( NULL )
,	_nodePool( NULL )
,	_dataPool( NULL )
,	_localizationCodeArray( NULL )
,	_localizationDepthArray( NULL )
,	_childArraySync( NULL )
,	_dataArraySync( NULL )
,	_showNodeHasBrickTerminal( true )
,	_showNodeHasBrickNotTerminal( true )
,	_showNodeIsBrickNotInCache( true )
,	_showNodeEmptyOrConstant( true )
,	_nodeHasBrickTerminalColor( make_float4( 1.0f, 1.0f, 0.0f, 1.0f ) )		// yellow
,	_nodeHasBrickNotTerminalColor( make_float4( 1.0f, 0.0f, 0.0f, 1.0f ) )	// red
,	_nodeIsBrickNotInCacheColor( make_float4( 0.0f, 1.0f, 0.0f, 1.0f ) )	// green
,	_nodeEmptyOrConstantColor( make_float4( 0.0f, 0.0f, 1.0f, 1.0f ) )		// blue
{
	// LOG info
	std::cout << "\nData Structure ( N3-Tree )" << std::endl;
	std::cout << "- node cache size : " << nodesCacheSize << std::endl;
	std::cout << "- bricks cache resolution : " << bricksCacheRes << std::endl;

	// Node pool initialization
	_nodePool = new GvCore::GPUPoolHost< GvCore::Array3DGPULinear, NodeTList >( nodesCacheSize, graphicsInteroperability );

	// Data pool initialization
	_dataPool = new GvCore::GPUPoolHost< GvCore::Array3DGPUTex, DataTList >( bricksCacheRes, graphicsInteroperability );

	// Helper object : it is a reference on nodes in the node pool
	_childArray = _nodePool->getChannel( Loki::Int2Type< 0 >() );
	_childArray->fill( 0 );

	// Helper object : it is a reference on bricks in the node pool
	_dataArray = _nodePool->getChannel( Loki::Int2Type< 1 >() );
	_dataArray->fill( 0 );

	//volumeTreeKernel.dataGPUPoolKernel = this->dataGPUPool->getKernelPool();
	volumeTreeKernel.brickCacheResINV = make_float3( 1.0f ) / make_float3( bricksCacheRes );	// Size of 1 voxel in the pool of bricks of voxels (3D texture)
	volumeTreeKernel._rootAddress = NodeTileRes::getNumElements();
	volumeTreeKernel.brickSizeInCacheNormalized = make_float3( BrickRes::get() ) / make_float3( bricksCacheRes );

	////Localization codes////
	uint3 nodeCacheRes = nodesCacheSize / NodeTileRes::get();

	// Localization code array initialization
	_localizationCodeArray = new GvCore::Array3DGPULinear< GvCore::GvLocalizationInfo::CodeType >( nodeCacheRes );
	_localizationCodeArray->fill( 0 );

	// Localization depth array initialization
	_localizationDepthArray = new GvCore::Array3DGPULinear< GvCore::GvLocalizationInfo::DepthType >( nodeCacheRes );
	_localizationDepthArray->fill( 0 );

	_childArraySync	= NULL;
	_dataArraySync = NULL;

	setMaxDepth( 9 ); // this call cudaMemcpyToSymbol()

	this->cuda_Init();
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize, typename TDataStructureKernelType >
GvVolumeTree< DataTList, NodeTileRes, BrickRes, BorderSize, TDataStructureKernelType >
::~GvVolumeTree()
{
	delete _nodePool;
	delete _dataPool;

	delete _localizationCodeArray;
	delete _localizationDepthArray;

	delete _childArraySync;
	delete _dataArraySync;
}

/******************************************************************************
 * Clear the volume tree information
 * It clears the node pool and its associated localization info
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize, typename TDataStructureKernelType >
void GvVolumeTree< DataTList, NodeTileRes, BrickRes, BorderSize, TDataStructureKernelType >
::clearVolTree()
{
	// Clear the child/brick addresses arrays.
	_childArray->fill( 0 );
	_dataArray->fill( 0 );

	// Clear the locations code/depth arrays.
	_localizationCodeArray->fill( 0 );
	_localizationDepthArray->fill( 0 );
}

/******************************************************************************
 * Cuda specific initialization
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize, typename TDataStructureKernelType >
void GvVolumeTree< DataTList, NodeTileRes, BrickRes, BorderSize, TDataStructureKernelType >
::cuda_Init()
{
	////VOLTREE////

#if USE_LINEAR_VOLTREE_TEX
	volumeTreeChildTexLinear.normalized = false;					  // access with normalized texture coordinates
	volumeTreeChildTexLinear.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates
	volumeTreeChildTexLinear.addressMode[1] = cudaAddressModeClamp;
	volumeTreeChildTexLinear.addressMode[2] = cudaAddressModeClamp;
	volumeTreeChildTexLinear.filterMode = cudaFilterModePoint;		// nearest interpolation

	cudaChannelFormatDesc indicesTexChannelDesc = cudaCreateChannelDesc<uint>();

	GV_CUDA_SAFE_CALL(cudaBindTexture(NULL, volumeTreeChildTexLinear,
		this->_childArray->getPointer()));

	volumeTreeDataTexLinear.normalized = false;					  // access with normalized texture coordinates
	volumeTreeDataTexLinear.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates
	volumeTreeDataTexLinear.addressMode[1] = cudaAddressModeClamp;
	volumeTreeDataTexLinear.addressMode[2] = cudaAddressModeClamp;
	volumeTreeDataTexLinear.filterMode = cudaFilterModePoint;		// nearest interpolation

	GV_CUDA_SAFE_CALL(cudaBindTexture(NULL, volumeTreeDataTexLinear,
		this->_dataArray->getPointer()));
#endif

	// Data pool

	// LOG info
	std::cout << "\nData Pool" << std::endl;
	this->_dataPool->bindPoolToTextureReferences( Loki::Int2Type< TEXDATAPOOL >(), true, true, cudaFilterModeLinear, cudaAddressModeClamp ); //true, true, ...
	this->_dataPool->bindPoolToSurfaceReferences();

	// Copy node pool's nodes to constant memory (its pointer)
	GvCore::Array3DKernelLinear< uint > h_volumeTreeChildArray = this->_childArray->getDeviceArray();
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_volumeTreeChildArray, &h_volumeTreeChildArray, sizeof( h_volumeTreeChildArray ), 0, cudaMemcpyHostToDevice ) );

	// Copy node pool's bricks to constant memory (its pointer)
	GvCore::Array3DKernelLinear< uint > h_volumeTreeDataArray = this->_dataArray->getDeviceArray();
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_volumeTreeDataArray, &h_volumeTreeDataArray, sizeof( h_volumeTreeDataArray ), 0, cudaMemcpyHostToDevice ) );

	GV_CHECK_CUDA_ERROR( "GvVolumeTree::cuda_Init end" );
}

/******************************************************************************
 * Get the node tile resolution.
 *
 * @return the node tile resolution
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize, typename TDataStructureKernelType >
const NodeTileRes& GvVolumeTree< DataTList, NodeTileRes, BrickRes, BorderSize, TDataStructureKernelType >
::getNodeTileResolution() const
{
	return _nodeTileResolution;
}

/******************************************************************************
 * Get the brick resolution (voxels).
 *
 * @param the brick resolution
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize, typename TDataStructureKernelType >
const BrickRes& GvVolumeTree< DataTList, NodeTileRes, BrickRes, BorderSize, TDataStructureKernelType >
::getBrickResolution() const
{
	return _brickResolution;
}

/******************************************************************************
 * Get the max depth of the volume tree
 *
 * @return max depth
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize, typename TDataStructureKernelType >
uint GvVolumeTree< DataTList, NodeTileRes, BrickRes, BorderSize, TDataStructureKernelType >
::getMaxDepth() const
{
	return _maxDepth;
}

/******************************************************************************
 * Set the max depth of the volume tree
 *
 * @param maxDepth Max depth
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize, typename TDataStructureKernelType >
void GvVolumeTree< DataTList, NodeTileRes, BrickRes, BorderSize, TDataStructureKernelType >
::setMaxDepth( uint maxDepth )
{
	_maxDepth = maxDepth;

	// Update CUDA memory with value
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_maxVolTreeDepth, &_maxDepth, sizeof( _maxDepth ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Debugging helpers
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize, typename TDataStructureKernelType >
void GvVolumeTree< DataTList, NodeTileRes, BrickRes, BorderSize, TDataStructureKernelType >
::render()
{
	syncDebugVolTree();

	glDisable( GL_LIGHTING );
	glDisable( GL_BLEND );
	glPolygonMode( GL_FRONT_AND_BACK , GL_LINE );
	debugDisplay( 0, make_uint3( NodeTileRes::numElements, 0, 0 ), make_float3( 0.0f ), make_float3( 1.0f ) );
	glPolygonMode( GL_FRONT_AND_BACK , GL_FILL);
}

/******************************************************************************
 * Used to display the N-tree
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize, typename TDataStructureKernelType >
void GvVolumeTree< DataTList, NodeTileRes, BrickRes, BorderSize, TDataStructureKernelType >
::syncDebugVolTree()
{
	if ( ! _childArraySync )
	{
		_childArraySync = new GvCore::Array3D< uint >( _childArray->getResolution(), GvCore::Array3D< uint >::StandardHeapMemory );
	}

	if ( ! _dataArraySync )
	{
		_dataArraySync = new GvCore::Array3D< uint >( _dataArray->getResolution(), GvCore::Array3D< uint >::StandardHeapMemory );
	}

	// TO DO : asynchronous copy
	memcpyArray( _childArraySync, _childArray );
	memcpyArray( _dataArraySync, _dataArray );

	//dumpOctree(_childArraySync->getPointer(), 8);
}

/******************************************************************************
 * Get the appearance of the N-tree (octree) of the data structure
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize, typename TDataStructureKernelType >
void GvVolumeTree< DataTList, NodeTileRes, BrickRes, BorderSize, TDataStructureKernelType >
::getDataStructureAppearance( bool& pShowNodeHasBrickTerminal, bool& pShowNodeHasBrickNotTerminal, bool& pShowNodeIsBrickNotInCache, bool& pShowNodeEmptyOrConstant,
	float4& pNodeHasBrickTerminalColor, float4& pNodeHasBrickNotTerminalColor, float4& pNodeIsBrickNotInCacheColor, float4& pNodeEmptyOrConstantColor  ) const
{
	pShowNodeHasBrickTerminal = _showNodeHasBrickTerminal;
	pShowNodeHasBrickNotTerminal = _showNodeHasBrickNotTerminal;
	pShowNodeIsBrickNotInCache = _showNodeIsBrickNotInCache;
	pShowNodeEmptyOrConstant = _showNodeEmptyOrConstant;

	pNodeHasBrickTerminalColor = _nodeHasBrickTerminalColor;
	pNodeHasBrickNotTerminalColor = _nodeHasBrickNotTerminalColor;
	pNodeIsBrickNotInCacheColor = _nodeIsBrickNotInCacheColor;
	pNodeEmptyOrConstantColor = _nodeEmptyOrConstantColor;
}
	
/******************************************************************************
 * Set the appearance of the N-tree (octree) of the data structure
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize, typename TDataStructureKernelType >
void GvVolumeTree< DataTList, NodeTileRes, BrickRes, BorderSize, TDataStructureKernelType >
::setDataStructureAppearance( bool pShowNodeHasBrickTerminal, bool pShowNodeHasBrickNotTerminal, bool pShowNodeIsBrickNotInCache, bool pShowNodeEmptyOrConstant,
	const float4& pNodeHasBrickTerminalColor, const float4& pNodeHasBrickNotTerminalColor, const float4& pNodeIsBrickNotInCacheColor, const float4& pNodeEmptyOrConstantColor )
{
	_showNodeHasBrickTerminal = pShowNodeHasBrickTerminal;
	_showNodeHasBrickNotTerminal = pShowNodeHasBrickNotTerminal;
	_showNodeIsBrickNotInCache = pShowNodeIsBrickNotInCache;
	_showNodeEmptyOrConstant = pShowNodeEmptyOrConstant;

	_nodeHasBrickTerminalColor = pNodeHasBrickTerminalColor;
	_nodeHasBrickNotTerminalColor = pNodeHasBrickNotTerminalColor;
	_nodeIsBrickNotInCacheColor = pNodeIsBrickNotInCacheColor;
	_nodeEmptyOrConstantColor = pNodeEmptyOrConstantColor;
}
	
/******************************************************************************
 * Used to display the N-tree
 *
 * @param depth Depth
 * @param address Address
 * @param pos Position
 * @param size Size
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize, typename TDataStructureKernelType >
void GvVolumeTree< DataTList, NodeTileRes, BrickRes, BorderSize, TDataStructureKernelType >
::debugDisplay( uint depth, const uint3& address, const float3& pos, const float3& size )
{
	// TO DO : avoid temp copies...
	GvNode node;
	node = getOctreeNodeSync( address );

	// If node has sub nodes AND we have not reached the max depth resolution
	if ( node.hasSubNodes() && depth < _maxDepth )
	{
		float3 subsize = size / NodeTileRes::getFloat3();

		for( int k=0; k<NodeTileRes::z; k++ )
			for( int j=0; j<NodeTileRes::y; j++ )
				for( int i=0; i<NodeTileRes::x; i++ )
				{
					float3 subpos = pos + make_float3( (float)i, (float)j, (float)k ) * subsize;
					debugDisplay( depth + 1, node.getChildAddress() + make_uint3( i + j * NodeTileRes::x + k * NodeTileRes::x * NodeTileRes::y, 0, 0 ), subpos, subsize );
				}

				// DARK GREEN
				glColor4f( 0.1f, 0.7f, 0.1f, 0.5f );
	}
	else
	{
		// If node has brick, it means that the associated brick has been produced and placed in GPU memory
		if ( node.hasBrick() )
		{
			if ( node.isTerminal() )
			{
				if ( _showNodeHasBrickTerminal )
				{
					glColor4f( _nodeHasBrickTerminalColor.x, _nodeHasBrickTerminalColor.y, _nodeHasBrickTerminalColor.z, _nodeHasBrickTerminalColor.w );
					drawCube( pos, pos + size );
				}
			}
			else
			{
				if ( _showNodeHasBrickNotTerminal )
				{
					glColor4f( _nodeHasBrickNotTerminalColor.x, _nodeHasBrickNotTerminalColor.y, _nodeHasBrickNotTerminalColor.z, _nodeHasBrickNotTerminalColor.w );
					drawCube( pos, pos + size );
				}
			}
		}
		// If node is a brick, it means that it holds data but the associated brick of voxels is not in GPU memory (i.e. not yet produced/loaded or removed from cache)
		else if ( node.isBrick() )
		{
			if ( _showNodeIsBrickNotInCache )
			{
				glColor4f( _nodeIsBrickNotInCacheColor.x, _nodeIsBrickNotInCacheColor.y, _nodeIsBrickNotInCacheColor.z, _nodeIsBrickNotInCacheColor.w );
				drawCube( pos, pos + size );
			}
		}
		else
		{
			// Empty node
			if ( _showNodeEmptyOrConstant )
			{
				glColor4f( _nodeEmptyOrConstantColor.x, _nodeEmptyOrConstantColor.y, _nodeEmptyOrConstantColor.z, _nodeEmptyOrConstantColor.w );
				drawCube( pos, pos + size );
			}
		}

		//drawCube( pos, pos + size );
	}
}

/******************************************************************************
 * Used to display the N-tree
 *
 * @param p1 Position
 * @param p2 Position
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize, typename TDataStructureKernelType >
void GvVolumeTree< DataTList, NodeTileRes, BrickRes, BorderSize, TDataStructureKernelType >
::drawCube( const float3& p1, const float3& p2 )
{
	glBegin(GL_QUADS);
	// Front Face
	glVertex3f(p1.x, p1.y, p2.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p2.z);	// Top Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p2.z);	// Top Right Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p2.z);	// Bottom Right Of The Texture and Quad

	// Back Face
	glVertex3f(p1.x, p1.y, p1.z);	// Bottom Right Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p1.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p1.z);	// Top Left Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p1.z);	// Top Right Of The Texture and Quad

	// Top Face
	glVertex3f(p1.x, p2.y, p1.z);	// Top Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p1.z);	// Top Right Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p2.z);	// Bottom Right Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p2.z);	// Bottom Left Of The Texture and Quad

	// Bottom Face
	glVertex3f(p1.x, p1.y, p1.z);	// Top Right Of The Texture and Quad
	glVertex3f(p1.x, p1.y, p2.z);	// Bottom Right Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p2.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p1.z);	// Top Left Of The Texture and Quad

	// Right face
	glVertex3f(p2.x, p1.y, p1.z);	// Bottom Right Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p2.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p2.z);	// Top Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p1.z);	// Top Right Of The Texture and Quad

	// Left Face
	glVertex3f(p1.x, p1.y, p1.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p1.z);	// Top Left Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p2.z);	// Top Right Of The Texture and Quad
	glVertex3f(p1.x, p1.y, p2.z);	// Bottom Right Of The Texture and Quad

	glEnd();
}

/******************************************************************************
 * Used to display the N-tree
 *
 * @param offset Offset
 *
 * @return ...
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize, typename TDataStructureKernelType >
GvNode GvVolumeTree< DataTList, NodeTileRes, BrickRes, BorderSize, TDataStructureKernelType >
::getOctreeNodeSync( const uint3& offset )
{
	GvNode node;
	node.childAddress = _childArraySync->get( offset );
	node.brickAddress = _dataArraySync->get( offset );

	return node;
}

/******************************************************************************
 * This method is used to serialize a pool
 *
 * @param pStream the stream from which to read
 ******************************************************************************/
template< typename TDataTypeList, unsigned int TChannelIndex, typename TDataPool >
void writeDataChannel( const TDataPool* pDataPool, std::ostream& pStream )
{
	assert( pDataPool != NULL );
	if ( pDataPool != NULL )
	{
		// Allocate temporary host array
		typedef typename GvCore::DataChannelType< TDataTypeList, TChannelIndex >::Result dataType;
		GvCore::Array3D< dataType >* dataArray = new GvCore::Array3D< dataType >( pDataPool->getChannel( Loki::Int2Type< TChannelIndex >() )->getResolution(), GvCore::Array3D< dataType >::StandardHeapMemory );
		
		// Copy data from device to host
		memcpyArray( dataArray, pDataPool->getChannel( Loki::Int2Type< TChannelIndex >() ) );
		
		// Serialize data
		pStream.write( reinterpret_cast< const char* >( dataArray->data() ), sizeof( dataType ) * dataArray->getNumElements() );
		
		// Free temporary host array
		delete dataArray;
	}
}

/**
 * Functor used to serialize/deserialize data pool of a data structure
 */
template< typename TDataTypeList, typename TDataPool >
struct GvDataPoolSerializer
{
	/**
	 * ...
	 */
	TDataPool _dataPool;

	/**
	 * Generalized functor method used to bound textures.
	 *
	 * @param Loki::Int2Type< i > channel
	 */
	template< int i >
	inline void run( Loki::Int2Type< i > )
	{
		//const TDataPool* dataPool = NULL;
		//std::ostream& stream;
		//writeDataChannel< TDataTypeList, Loki::Int2Type< i >(), TDataPool >( _dataPool, stream );
	}
};

/******************************************************************************
 * This method is called to serialize an object
 *
 * @param pStream the stream where to write
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize, typename TDataStructureKernelType >
inline void GvVolumeTree< DataTList, NodeTileRes, BrickRes, BorderSize, TDataStructureKernelType >
::write( std::ostream& pStream ) const
{
	// -------- Node pool serialization --------

	// - node info
	GvCore::Array3D< uint >* nodeTileInfo = new GvCore::Array3D< uint >( _childArray->getResolution(), GvCore::Array3D< uint >::StandardHeapMemory );
	memcpyArray( nodeTileInfo, _childArray );
	pStream.write( reinterpret_cast< const char* >( nodeTileInfo->getPointer() ), sizeof( uint ) * nodeTileInfo->getNumElements() );
	delete nodeTileInfo;

	// - data info
	GvCore::Array3D< uint >* nodeDataInfo = new GvCore::Array3D< uint >( _dataArray->getResolution(), GvCore::Array3D< uint >::StandardHeapMemory );
	memcpyArray( nodeDataInfo, _dataArray );
	pStream.write( reinterpret_cast< const char* >( nodeDataInfo->getPointer() ), sizeof( uint ) * nodeDataInfo->getNumElements() );
	delete nodeDataInfo;

	// -------- Data pool serialization --------

//	// - data
////	for ( uint i = 0; i < DataPoolType::numChannels; i++ )
////	{
//		std::cout << "Channel #" << 0 << std::endl;
//		typedef typename GvCore::DataChannelType< DataTList, 0 >::Result dataType0;
//		GvCore::Array3D< dataType0 >* dataArray0 = new GvCore::Array3D< dataType0 >( _dataPool->getChannel( Loki::Int2Type< 0 >() )->getResolution(), GvCore::Array3D< dataType0 >::StandardHeapMemory );
//		memcpyArray( dataArray0, _dataPool->getChannel( Loki::Int2Type< 0 >() ) );
//		pStream.write( reinterpret_cast< const char* >( dataArray0->getPointer() ), sizeof( dataType0 ) * dataArray0->getNumElements() );
//		delete dataArray0;
//
//		std::cout << "Channel #" << 1 << std::endl;
//		typedef typename GvCore::DataChannelType< DataTList, 1 >::Result dataType1;
//		GvCore::Array3D< dataType1 >* dataArray1 = new GvCore::Array3D< dataType1 >( _dataPool->getChannel( Loki::Int2Type< 1 >() )->getResolution(), GvCore::Array3D< dataType1 >::StandardHeapMemory );
//		memcpyArray( dataArray1, _dataPool->getChannel( Loki::Int2Type< 1 >() ) );
//		pStream.write( reinterpret_cast< const char* >( dataArray1->getPointer() ), sizeof( dataType1 ) * dataArray1->getNumElements() );
//		delete dataArray1;
////	}
	//GvDataPoolSerializer< DataTList, DataPoolType > dataPoolSerializerfunctor( this );
	//GvCore::StaticLoop< GvDataPoolSerializer< DataTList, DataPoolType >, Loki::TL::Length< DataTList >::value - 1 >::go( dataPoolSerializerfunctor );

	// -------- Localization information --------
	
	// - localization depth
	GvCore::Array3D< GvCore::GvLocalizationInfo::DepthType >* localizationDepth = new GvCore::Array3D< GvCore::GvLocalizationInfo::DepthType >( _localizationDepthArray->getResolution(), GvCore::Array3D< GvCore::GvLocalizationInfo::DepthType >::StandardHeapMemory );
	memcpyArray( localizationDepth, _localizationDepthArray );
	pStream.write( reinterpret_cast< const char* >( localizationDepth->getPointer() ), sizeof( GvCore::GvLocalizationInfo::DepthType ) * localizationDepth->getNumElements() );
	delete localizationDepth;

	// - localization code
	GvCore::Array3D< GvCore::GvLocalizationInfo::CodeType >* localizationCode = new GvCore::Array3D< GvCore::GvLocalizationInfo::CodeType >( _localizationCodeArray->getResolution(), GvCore::Array3D< GvCore::GvLocalizationInfo::CodeType >::StandardHeapMemory );
	memcpyArray( localizationCode, _localizationCodeArray );
	pStream.write( reinterpret_cast< const char* >( localizationCode->getPointer() ), sizeof( GvCore::GvLocalizationInfo::CodeType ) * localizationCode->getNumElements() );
	delete localizationCode;
}

/******************************************************************************
 * This method is called deserialize an object
 *
 * @param pStream the stream from which to read
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize, typename TDataStructureKernelType >
inline void GvVolumeTree< DataTList, NodeTileRes, BrickRes, BorderSize, TDataStructureKernelType >
::read( std::istream& pStream )
{
	std::ifstream file2( "fifthgrade.txt", std::ios::binary );
	if ( file2.is_open() )
	{
		GvCore::Array3D< uint >* nodeTileArray = new GvCore::Array3D< uint >( _childArray->getResolution(), GvCore::Array3D< uint >::StandardHeapMemory );
		file2.read( reinterpret_cast< char* >( nodeTileArray->getPointer() ), sizeof( uint ) * nodeTileArray->getNumElements() );
		//for ( size_t i = 0; i < nodeTileArray->getNumElements(); i++ )
		//{
		//	std::cout << nodeTileArray->get( i ) << std::endl;
		//}
		//_childArray->fill( 0 );
		memcpyArray( _childArray, nodeTileArray );
		delete nodeTileArray;
		file2.close();
	}
}

} // namespace GvStructure
