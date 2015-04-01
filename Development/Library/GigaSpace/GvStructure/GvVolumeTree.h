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

#ifndef _GV_VOLUME_TREE_H_
#define _GV_VOLUME_TREE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

// Cuda
#include <vector_types.h>

// Cuda SDK
#include <helper_math.h>

// Thrust
#include <thrust/device_vector.h>

// GigaVoxels
#include "GvStructure/GvIDataStructure.h"
#include "GvCore/StaticRes3D.h"
#include "GvPerfMon/GvPerformanceMonitor.h"
#include "GvStructure/GvVolumeTreeKernel.h"		// TO DO : remove it because of template !!
#include "GvStructure/GvNode.h"
#include "GvCore/Array3D.h"
#include "GvCore/Array3DGPULinear.h"
#include "GvCore/Array3DGPUTex.h"
#include "GvCore/GPUPool.h"
#include "GvCore/GvLocalizationInfo.h"
#include "GvCore/RendererTypes.h"

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

namespace GvStructure
{

/** 
 * @struct GvVolumeTree
 *
 * @brief The GvVolumeTree struct provides a generalized N-Tree data structure
 *
 * Volume Tree encapsulates nodes and bricks data.
 *
 * - Nodes are used for space partitioning strategy. There are organized in node tiles
 * basee on their node tile resolution. Octree is the more common organization
 * (2x2x2 nodes by node tile). N-Tree represents a hierarchical structure containg
 * multi-resolution pyramid of data.
 * - Bricks are used to store user defined data as color, normal, density, etc...
 * Data is currently stored in 3D textures. In each node, we have one brick of voxels
 * based on its brick resolution (ex : 8x8x8 voxels by brick).
 *
 * Nodes and bricks are organized in pools that ara managed by a cahe mecanism.
 * LRU mecanism (Least recently Used) is used to efficiently store data in device memory.
 *
 * For each type of data defined by the user, the brick pool stores a 3D texture that
 * can be read and/or write. It corresponds to a channel in the pool.
 *
 * @param DataTList Data type list provided by the user
 * (exemple with a normal and a color by voxel : typedef Loki::TL::MakeTypelist< uchar4, half4 >::Result DataType;)
 * @param NodeTileRes Node tile resolution
 * @param NodeTileRes Brick resolution
 * @param BorderSize Brick border size (1 for the moment)
 * @param TDataStructureKernelType data structure device-side associated object (ex : GvStructure::VolumeTreeKernel< DataTList, NodeTileRes, BrickRes, BorderSize >)
 *
 * @todo : see how to handle a border superior than and if it can be useful
 */
template
<
	class DataTList, class NodeTileRes, class BrickRes, uint BorderSize,
	typename TDataStructureKernelType
>
struct GvVolumeTree : public GvIDataStructure
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	//typedef Array3DGPULinear NodeArrayType;

	/**
	 * Typedef used to describe a node type element
	 * It holds two addresses : one for node, one for brick
	 */
	typedef typename Loki::TL::MakeTypelist< uint, uint >::Result NodeTList;

	/**
	 * Typedef for the Volume Tree on GPU side
	 */
	typedef TDataStructureKernelType VolTreeKernelType;

	/**
	 * Type definition for the node tile resolution
	 */
	typedef NodeTileRes NodeTileResolution;

	/**
	 * Type definition for the brick resolution
	 */
	typedef BrickRes BrickResolution;

	/**
	 * Enumeration to define the brick border size
	 */
	enum
	{
		BrickBorderSize = BorderSize
	};

	/**
	 * Defines the total size of a brick
	 */
	typedef GvCore::StaticRes1D< BrickResolution::x + 2 * BrickBorderSize > FullBrickResolution;	// TO DO : NodeTileResolution::x (problem ?)

	/**
	 * Defines the data type list
	 */
	typedef DataTList DataTypeList;

	/**
	 * Type definition of the node pool type
	 */
	typedef GvCore::GPUPoolHost< GvCore::Array3DGPULinear, NodeTList > NodePoolType;

	/**
	 * Type definition of the data pool type
	 */
	typedef GvCore::GPUPoolHost< GvCore::Array3DGPUTex, DataTList > DataPoolType;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Helper object : it is a reference on nodes in the node pool
	 */
	GvCore::Array3DGPULinear< uint >* _childArray;

	/**
	 * Helper object : it is a reference on bricks in the node pool
	 */
	GvCore::Array3DGPULinear< uint >* _dataArray;

	/**
	 * Node pool
	 * It is implemented as an Array3DGPULinear, i.e linear memory
	 */
	NodePoolType* _nodePool;

	/**
	 * Brick pool (i.e data pool)
	 * It is implemented as an Array3DGPUTex, i.e a 3D texture
	 * There is one 3D texture for each element in the data type list DataTList defined by the user
	 */
	DataPoolType* _dataPool;

	/**
	 * Localization code array
	 *
	 * @todo The creation of the localization arrays should be moved in the Cache Management System, not in the data structure (this is cache implementation details/features)
	 */
	GvCore::Array3DGPULinear< GvCore::GvLocalizationInfo::CodeType >* _localizationCodeArray;

	/**
	 * Localization depth array
	 *
	 * @todo The creation of the localization arrays should be moved in the Cache Management System, not in the data structure (this is cache implementation details/features)
	 */
	GvCore::Array3DGPULinear< GvCore::GvLocalizationInfo::DepthType >* _localizationDepthArray;

	/**
	 * Volume tree on GPU side
	 */
	VolTreeKernelType volumeTreeKernel;

	/******************************** METHODS *********************************/

	/**
	 * Constructor.
	 *
	 * @param nodesCacheSize Cache size used to store nodes
	 * @param bricksCacheRes Cache size used to store bricks
	 * @param graphicsInteroperability Flag used for graphics interoperability
	 */
	GvVolumeTree( const uint3& nodesCacheSize, const uint3& bricksCacheRes, uint graphicsInteroperability = 0 );

	/**
	 * Destructor.
	 */
	virtual ~GvVolumeTree();

	/**
	 * Cuda specific initialization
	 */
	void cuda_Init();

	/**
	 * Clear the volume tree information
	 * It clears the node pool and its associated localization info
	 */
	void clearVolTree();

	/**
	 * Get the max depth of the volume tree
	 *
	 * @return max depth
	 */
	uint getMaxDepth() const;

	/**
	 * Set the max depth of the volume tree
	 *
	 * @param maxDepth Max depth
	 */
	void setMaxDepth( uint maxDepth );

	/**
	 * Debugging helpers
	 */
	void render();

	/**
	 * Get the node tile resolution.
	 *
	 * @return the node tile resolution
	 */
	const NodeTileRes& getNodeTileResolution() const;

	/**
	 * Get the brick resolution (voxels).
	 *
	 * @param the brick resolution
	 */
	const BrickRes& getBrickResolution() const;

	/**
	 * Get the appearance of the N-tree (octree) of the data structure
	 */
	void getDataStructureAppearance( bool& pShowNodeHasBrickTerminal, bool& pShowNodeHasBrickNotTerminal, bool& pShowNodeIsBrickNotInCache, bool& pShowNodeEmptyOrConstant,
										float4& pNodeHasBrickTerminalColor, float4& pNodeHasBrickNotTerminalColor, float4& pNodeIsBrickNotInCacheColor, float4& pNodeEmptyOrConstantColor ) const;

	/**
	 * Set the appearance of the N-tree (octree) of the data structure
	 */
	void setDataStructureAppearance( bool pShowNodeHasBrickTerminal, bool pShowNodeHasBrickNotTerminal, bool pShowNodeIsBrickNotInCache, bool pShowNodeEmptyOrConstant,
										const float4& pNodeHasBrickTerminalColor, const float4& pNodeHasBrickNotTerminalColor, const float4& pNodeIsBrickNotInCacheColor, const float4& pNodeEmptyOrConstantColor );
	
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

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Node tile resolution
	 */
	NodeTileRes _nodeTileResolution;

	/**
	 * Brick resolution (voxels)
	 */
	BrickRes _brickResolution;

	/**
	 * Used to display the N-tree
	 */
	GvCore::Array3D< uint >* _childArraySync;
		
	/**
	 * Used to display the N-tree
	 */
	GvCore::Array3D< uint >* _dataArraySync;

	/**
	 * Used to display the N-tree. This is the max possible depth of the stucture.
	 */
	uint _maxDepth;

	/**
	 * Data structure appearance
	 */
	bool _showNodeHasBrickTerminal;
	bool _showNodeHasBrickNotTerminal;
	bool _showNodeIsBrickNotInCache;
	bool _showNodeEmptyOrConstant;
	float4 _nodeHasBrickTerminalColor;
	float4 _nodeHasBrickNotTerminalColor;
	float4 _nodeIsBrickNotInCacheColor;
	float4 _nodeEmptyOrConstantColor;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	//! Debugging helpers

	/**
	 * Used to display the N-tree
	 */
	void syncDebugVolTree();

	/**
	 * Used to display the N-tree
	 *
	 * @param depth Depth
	 * @param address Address
	 * @param pos Position
	 * @param size Size
	 */
	void debugDisplay( uint depth, const uint3& address, const float3& pos, const float3& size );

	/**
	 * Used to display the N-tree
	 *
	 * @param p1 Position
	 * @param p2 Position
	 */
	void drawCube( const float3& p1, const float3& p2 );

	/**
	 * Used to display the N-tree
	 *
	 * @param offset Offset
	 *
	 * @return ...
	 */
	GvNode getOctreeNodeSync( const uint3& offset );

	/**
	 * Copy constructor forbidden.
	 */
	GvVolumeTree( const GvVolumeTree& );

	/**
	 * Copy operator forbidden.
	 */
	GvVolumeTree& operator=( const GvVolumeTree& );

};

} // namespace GvStructure

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvVolumeTree.inl"

#endif // !_GV_VOLUME_TREE_H_
