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

#ifndef _GV_SIMPLE_PIPELINE_H_
#define _GV_SIMPLE_PIPELINE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/GvIProvider.h"
#include "GvCore/vector_types_ext.h"
#include "GvCache/GvCacheHelper.h"
#include "GvUtils/GvPipeline.h"
#include "GvRendering/GvIRenderer.h"

// Thrust
#include <thrust/device_vector.h>

// STL
#include <vector>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvUtils
{

/** 
 * @class GvSimplePipeline
 *
 * @brief The GvSimplePipeline class provides the mecanisms to produce data
 * following data requests emitted during N-tree traversal (rendering).
 *
 * It is the main user entry point to produce data from CPU, for instance,
 * loading data from disk or procedurally generating data.
 *
 * This class is implements the mandatory functions of the GvIProvider base class.
 */
template< typename TProducerType, typename TShaderType, typename TDataStructureType, typename TCacheType >
class GvSimplePipeline : public GvPipeline
{
	
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Defines the size of a node tile
	 */
	typedef typename TDataStructureType::NodeTileResolution NodeTileResolution;

	/**
	 * Defines the size of a brick tile
	 */
	typedef typename TDataStructureType::BrickResolution BrickTileResolution;

	/**
	 * Defines the size of the border around a brick tile
	 */
	enum
	{
		BrickTileBorderSize = TDataStructureType::BrickBorderSize
	};

	/**
	 * Defines the total size of a brick tile (with borders)
	 */
	typedef typename TDataStructureType::FullBrickResolution RealBrickTileResolution;

	/**
	 * Defines the data type list
	 */
	typedef typename TDataStructureType::DataTypeList DataTypeList;

	/**
	 * Type defition of the data structure
	 */
	typedef TDataStructureType DataStructureType;

	/**
	 * Type defition of the cache
	 */
	typedef TCacheType CacheType;

	/**
	 * Type defition of the producer
	 */
	typedef TProducerType ProducerType;

	/**
	 * Type defition of renderers
	 */
	typedef GvRendering::GvIRenderer RendererType;
	
	/**
	 * Type defition of the shader
	 */
	typedef TShaderType ShaderType;
	
	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvSimplePipeline();

	/**
	 * Destructor
	 */
	virtual ~GvSimplePipeline();

	/**
	 * Initialize
	 *
	 * @param pNodePoolMemorySize Node pool memory size 
	 * @param pBrickPoolMemorySize Brick pool memory size
	 * @param pProducer Producer 
	 */
	virtual void initialize( size_t pNodePoolMemorySize, size_t pBrickPoolMemorySize, TProducerType* pProducer, TShaderType* pShader, bool pUseGraphicsLibraryInteroperability = false );

	/**
	 * Finalize
	 */
	virtual void finalize();

	/**
	 * Launch the main GigaSpace flow sequence
	 */
	virtual void execute();
	virtual void execute( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport );
	
	/**
	 * Get the data structure
	 *
	 * @return The data structure
	 */
	virtual const DataStructureType* getDataStructure() const;

	/**
	 * Get the data structure
	 *
	 * @return The data structure
	 */
	virtual DataStructureType* editDataStructure();

	/**
	 * Get the cache
	 *
	 * @return The cache
	 */
	virtual const CacheType* getCache() const;

	/**
	 * Get the cache
	 *
	 * @return The cache
	 */
	virtual CacheType* editCache();

	/**
	 * Get the renderer
	 *
	 * @param pIndex index of the renderer
	 *
	 * @return The renderer
	 */
	virtual const RendererType* getRenderer( unsigned int pIndex = 0 ) const;

	/**
	 * Get the renderer
	 *
	 * @param pIndex index of the renderer
	 *
	 * @return The renderer
	 */
	virtual RendererType* editRenderer( unsigned int pIndex = 0 );

	/**
	 * Add a renderer
	 *
	 * @param pRenderer The renderer
	 */
	void addRenderer( RendererType* pRenderer );

	/**
	 * Remove a renderer
	 *
	 * @param pRenderer The renderer
	 */
	void removeRenderer( RendererType* pRenderer );

	/**
	 * Set the current renderer
	 *
	 * @param pRenderer The renderer
	 */
	//void setCurrentRenderer( RendererType* pRenderer );

	/**
	 * Get the producer
	 *
	 * @return The producer
	 */
	virtual const TProducerType* getProducer() const;

	/**
	 * Get the producer
	 *
	 * @return The producer
	 */
	virtual TProducerType* editProducer();

	/**
	 * Get the shader
	 *
	 * @return The shader
	 */
	const TShaderType* getShader() const;

	/**
	 * Get the shader
	 *
	 * @return The shader
	 */
	TShaderType* editShader();

	/**
	 * Return the flag used to request a dynamic update mode.
	 *
	 * @return the flag used to request a dynamic update mode
	 */
	bool hasDynamicUpdate() const;

	/**
	 * Set the flag used to request a dynamic update mode.
	 *
	 * @param pFlag the flag used to request a dynamic update mode
	 */
	void setDynamicUpdate( bool pFlag );

	/**
	 * Set the flag used to request clearing the cache.
	 */
	void clear();

	/**
	 * Print datatype info
	 */
	void printDataTypeInfo();

	/**
	 * This method is called to serialize an object
	 *
	 * @param pStream the stream where to write
	 */
	virtual void write( std::ostream& pStream ) const;

	/**
	 * This method is called deserialize an object
	 *
	 * @param pStream the stream from which to read
	 */
	virtual void read( std::istream& pStream );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Data structure
	 */
	DataStructureType* _dataStructure;
	
	/**
	 * Cache
	 */
	CacheType* _cache;
	
	/**
	 * Renderer(s)
	 */
	RendererType* _renderer;
	std::vector< RendererType* > _renderers;

	/**
	 * Node pool memory size
	 */
	size_t _nodePoolMemorySize;

	/**
	 * Brick pool memory size
	 */
	size_t _brickPoolMemorySize;

	/**
	 * Producer
	 */
	TProducerType* _producer;

	/**
	 * Shader
	 */
	TShaderType* _shader;

	/**
	 * Flag used to request clearing the cache
	 */
	bool _clearRequested;

	/**
	 * Flag used to request a dynamic update mode.
	 *
	 * @todo explain
	 */
	bool _dynamicUpdate;

	/******************************** METHODS *********************************/

	/**
	 * Compute the resolution of the pools
	 *
	 * @param pNodePoolResolution Node pool resolution
	 * @param pBrickPoolResolution Brick pool resolution
	 */
	void computePoolResolution( uint3& pNodePoolResolution, uint3& pBrickPoolResolution );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvSimplePipeline( const GvSimplePipeline& );

	/**
	 * Copy operator forbidden.
	 */
	GvSimplePipeline& operator=( const GvSimplePipeline& );

};

} // namespace GvUtils

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvSimplePipeline.inl"

#endif // !_GV_SIMPLE_PIPELINE_H_
