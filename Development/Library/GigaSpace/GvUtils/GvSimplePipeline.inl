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

// STL
#include <iostream>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvUtils
{
	
/******************************************************************************
 * Constructor
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::GvSimplePipeline()
:	GvPipeline()
,	_dataStructure( NULL )
,	_cache( NULL )
,	_renderer( NULL )
,	_nodePoolMemorySize( 0 )
,	_brickPoolMemorySize( 0 )
,	_producer( NULL )
,	_shader( NULL )
,	_clearRequested( false )
,	_dynamicUpdate( true )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::~GvSimplePipeline()
{
	// Free memory
	finalize();
}

/******************************************************************************
 * Initialize
 *
 * @param pNodePoolMemorySize Node pool memory size 
 * @param pBrickPoolMemorySize Brick pool memory size
 * @param pProducer Producer 
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline void GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::initialize( size_t pNodePoolMemorySize, size_t pBrickPoolMemorySize, TProducerType* pProducer, TShaderType* pShader, bool pUseGraphicsLibraryInteroperability )
{
	assert( pNodePoolMemorySize > 0 );
	assert( pBrickPoolMemorySize > 0 );
	assert( pProducer != NULL );
	assert( pShader != NULL );

	// Print datatype info
	printDataTypeInfo();

	// Store producer
	_producer = pProducer;
	
	// Store shader
	_shader = pShader;

	// Store global memory size of the pools
	_nodePoolMemorySize = pNodePoolMemorySize;
	_brickPoolMemorySize = pBrickPoolMemorySize;

	// Compute the resolution of the pools
	uint3 nodePoolResolution;
	uint3 brickPoolResolution;
	computePoolResolution( nodePoolResolution, brickPoolResolution );

	std::cout << "\nNode pool resolution : " << nodePoolResolution << std::endl;
	std::cout << "Brick pool resolution : " << brickPoolResolution << std::endl;

	// Retrieve the requested graphics library interoperability mode
	const unsigned int useGraphicsLibraryInteroperability = pUseGraphicsLibraryInteroperability ? 1 : 0;

	// Data structure
	_dataStructure = new DataStructureType( nodePoolResolution, brickPoolResolution, useGraphicsLibraryInteroperability );
	assert( _dataStructure != NULL );

	// Cache
	_cache = new CacheType( _dataStructure, nodePoolResolution, brickPoolResolution, useGraphicsLibraryInteroperability );
	assert( _cache != NULL );

	// Initialize the producer with the data structure
	//
	// TO DO : add a virtual base function "hasDataStructure() = false;" to GvSimpleHostProducer class
	pProducer->initialize( _dataStructure, _cache );

	// Add producer to data prouction manager
	_cache->addProducer( pProducer );
}

/******************************************************************************
 * Finalize
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline void GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::finalize()
{
	// Free memory
	delete _renderer;
	delete _cache;
	delete _dataStructure;

	// Free memory
	delete _producer;
	delete _shader;
}

/******************************************************************************
 * Launch the main GigaSpace flow sequence
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline void GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::execute()
{
	// [ 1 ] - Rendering stage
	//_renderer->render();

	// [ 2 ] - Data Production Management stage
	//_cache->handleRequests();
}

#ifndef GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_SIMPLEPIPELINE
/******************************************************************************
 * Launch the main GigaSpace flow sequence
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline void GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::execute( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	// Check if a "clear request" has been asked
	if ( _clearRequested )
	{
		CUDAPM_START_EVENT( gpucache_clear );

		// Clear the cache
		_cache->clearCache();

		// Bug [#16161] "Cache : not cleared as it should be"
		//
		// Without the following clear of the data structure node pool, artefacts should appear.
		// - it is visible in the Slisesix and ProceduralTerrain demos.
		//
		// It seems that the brick addresses of the node pool need to be reset.
		// Maybe it's a problem of "time stamp" and index of current frame (or time).
		// 
		// @todo : study this problem
		_dataStructure->clearVolTree();

		CUDAPM_STOP_EVENT( gpucache_clear );

		// Update "clear request" flag
		_clearRequested = false;
	}

	// [ 1 ] - Pre-render stage
	_cache->preRenderPass();

	// [ 2 ] - Rendering stage
	_renderer->render( pModelMatrix, pViewMatrix, pProjectionMatrix, pViewport );

	// [ 3 ] - Post-render stage (i.e. Data Production Management)
	CUDAPM_START_EVENT( dataProduction_handleRequests );
	if ( _dynamicUpdate )
	{
		_cache->_intraFramePass = false;

		// Post render pass
		// This is where requests are processed : produce or load data
		_cache->handleRequests();
	}
	CUDAPM_STOP_EVENT( dataProduction_handleRequests );
}
#else // GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_SIMPLEPIPELINE
/******************************************************************************
 * Launch the main GigaSpace flow sequence
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline void GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::execute( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	// Check if a "clear request" has been asked
	if ( _clearRequested )
	{
		CUDAPM_START_EVENT( gpucache_clear );

		// Clear the cache
		_cache->clearCache();

		// Bug [#16161] "Cache : not cleared as it should be"
		//
		// Without the following clear of the data structure node pool, artefacts should appear.
		// - it is visible in the Slisesix and ProceduralTerrain demos.
		//
		// It seems that the brick addresses of the node pool need to be reset.
		// Maybe it's a problem of "time stamp" and index of current frame (or time).
		// 
		// @todo : study this problem
		_dataStructure->clearVolTree();

		CUDAPM_STOP_EVENT( gpucache_clear );

		// Update "clear request" flag
		_clearRequested = false;
	}

	// Map resources()
	// - this function provides the synchronization guarantee that any graphics calls issued before cudaGraphicsMapResources() will complete before any subsequent CUDA work in stream begins.
	_renderer->preRender( pModelMatrix, pViewMatrix, pProjectionMatrix, pViewport );
	
	// [ 1 ] - Pre-render stage
	_cache->preRenderPass();

	// [ 2 ] - Rendering stage
	_renderer->render( pModelMatrix, pViewMatrix, pProjectionMatrix, pViewport );

	// Unmap resources()
	_renderer->postRender( pModelMatrix, pViewMatrix, pProjectionMatrix, pViewport );

	// [ 3 ] - Post-render stage (i.e. Data Production Management)
	CUDAPM_START_EVENT( dataProduction_handleRequests );
	if ( _dynamicUpdate )
	{
		_cache->_intraFramePass = false;

		// Post render pass
		// This is where requests are processed : produce or load data
		_cache->handleRequests();
	}
	CUDAPM_STOP_EVENT( dataProduction_handleRequests );
}
#endif // GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_SIMPLEPIPELINE

/******************************************************************************
 * Get the data structure
 *
 * @return The data structure
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline const GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >::DataStructureType*
GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::getDataStructure() const
{
	return _dataStructure;
}

/******************************************************************************
 * Get the data structure
 *
 * @return The data structure
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >::DataStructureType*
GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::editDataStructure()
{
	return _dataStructure;
}

/******************************************************************************
 * Get the cache
 *
 * @return The cache
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline const GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >::CacheType*
GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::getCache() const
{
	return _cache;
}

/******************************************************************************
 * Get the cache
 *
 * @return The cache
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >::CacheType*
GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::editCache()
{
	return _cache;
}

/******************************************************************************
 * Get the renderer
 *
 * @param pIndex index of the renderer
 *
 * @return The renderer
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline const GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >::RendererType*
GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::getRenderer( unsigned int pIndex ) const
{
	assert( pIndex < _renderers.size() );
	return _renderers[ pIndex ];
}

/******************************************************************************
 * Get the renderer
 *
 * @param pIndex index of the renderer
 *
 * @return The renderer
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >::RendererType*
GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::editRenderer( unsigned int pIndex )
{
	assert( pIndex < _renderers.size() );
	return _renderers[ pIndex ];
}

/******************************************************************************
 * Add a renderer
 *
 * @param pRenderer The renderer
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline void GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::addRenderer( RendererType* pRenderer )
{
	_renderers.push_back( pRenderer );

	// TO DO
	// - do it properly...
	_renderer = pRenderer;
}

/******************************************************************************
 * Remove a renderer
 *
 * @param pRenderer The renderer
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline void GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::removeRenderer( RendererType* pRenderer )
{
	// TO DO
	assert( false );
}

/******************************************************************************
 * Get the producer
 *
 * @return The producer
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline const TProducerType*
GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::getProducer() const
{
	return _producer;
}

/******************************************************************************
 * Get the producer
 *
 * @return The producer
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline TProducerType*
GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::editProducer()
{
	return _producer;
}

/******************************************************************************
 * Get the shader
 *
 * @return The shader
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline const TShaderType*
GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::getShader() const
{
	return _shader;
}

/******************************************************************************
 * Get the shader
 *
 * @return The shader
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline TShaderType*
GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::editShader()
{
	return _shader;
}

/******************************************************************************
 * Compute the resolution of the pools
 *
 * @param pNodePoolResolution Node pool resolution
 * @param pBrickPoolResolution Brick pool resolution
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
inline void GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::computePoolResolution( uint3& pNodePoolResolution, uint3& pBrickPoolResolution )
{
	assert( _nodePoolMemorySize != 0 );
	assert( _brickPoolMemorySize != 0 );
		
	// Compute the size of one element in the cache for nodes and bricks
	size_t nodeTileMemorySize = NodeTileResolution::numElements * sizeof( GvStructure::GvNode );
	size_t brickTileMemorySize = RealBrickTileResolution::numElements * GvCore::DataTotalChannelSize< DataTypeList >::value;

	// Compute how many we can fit into the given memory size
	size_t nodePoolNbElements = _nodePoolMemorySize / nodeTileMemorySize;
	size_t brickPoolNbElements = _brickPoolMemorySize / brickTileMemorySize;

	// Compute the resolution of the pools
	pNodePoolResolution = make_uint3( static_cast< uint >( floorf( powf( static_cast< float >( nodePoolNbElements ), 1.0f / 3.0f ) ) ) ) * NodeTileResolution::get();
	pBrickPoolResolution = make_uint3( static_cast< uint >( floorf( powf( static_cast< float >( brickPoolNbElements ), 1.0f / 3.0f ) ) ) ) * RealBrickTileResolution::get();
}

/******************************************************************************
 * Return the flag used to request a dynamic update mode.
 *
 * @return the flag used to request a dynamic update mode
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
bool GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::hasDynamicUpdate() const
{
	return _dynamicUpdate;
}

/******************************************************************************
 * Set the flag used to request a dynamic update mode.
 *
 * @param pFlag the flag used to request a dynamic update mode
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
void GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::setDynamicUpdate( bool pFlag )
{
	_dynamicUpdate = pFlag;

	// Update renderer state
	_renderer->setDynamicUpdate( pFlag );
}

/******************************************************************************
 * Set the flag used to request clearing the cache.
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
void GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::clear()
{
	_clearRequested = true;
}

/******************************************************************************
 * Print datatype info
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
void GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::printDataTypeInfo()
{
	std::cout << "\nVoxel datatype(s) : " << GvCore::DataNumChannels< DataTypeList >::value << " channel(s)" << std::endl;
	GvCore::GvDataTypeInspector< DataTypeList > dataTypeInspector;
	GvCore::StaticLoop< GvCore::GvDataTypeInspector< DataType >, GvCore::DataNumChannels< DataTypeList >::value - 1 >::go( dataTypeInspector );
	for ( int i = 0; i < dataTypeInspector._dataTypes.size(); i++ )
	{
		std::cout << "- " << dataTypeInspector._dataTypes[ i ] << std::endl;
	}
}

/******************************************************************************
 * This method is called to serialize an object
 *
 * @param pStream the stream where to write
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
void GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::write( std::ostream& pStream ) const
{
	// TO DO
	// ...
}

/******************************************************************************
 * This method is called deserialize an object
 *
 * @param pStream the stream from which to read
 ******************************************************************************/
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
void GvSimplePipeline< TProducerType, TShaderType, TDataStructureType, TCacheType >
::read( std::istream& pStream )
{
	// TO DO
	// ...
}

} // namespace GvUtils
