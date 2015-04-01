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

#ifndef _GV_NODE_H_
#define _GV_NODE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/vector_types_ext.h"

// CUDA
#include <host_defines.h>

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
 * @struct GvNode
 *
 * @brief The GvNode struct provides the interface to nodes of data structures (N3-tree, etc...)
 *
 * @ingroup GvStructure
 *
 * The GvNode struct holds :
 * - the address of its child nodes in cache (the address of its first child nodes organized in tiles)
 * - and the address of its associated brick of data in cache
 *
 * @todo: Rename OctreeNode as GvNode or GvDataStructureNode.
 * @todo: Rename functions isBrick() hasBrick() (old naming convention when we had constant values).
 */
struct GvNode
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Address to its child nodes
	 *
	 * It is encoded (i.e. packed)
	 */
	uint childAddress;

	/**
	 * Address to its associated brick of data
	 *
	 * It is encoded (i.e. packed)
	 */
	uint brickAddress;

#ifdef GV_USE_BRICK_MINMAX
	uint metaDataAddress;
#endif 

	/******************************** METHODS *********************************/

	/**
	 * Unpack a node address
	 *
	 * @param pAddress a packed address of a node in cache
	 *
	 * @return the associated unpacked address
	 */
	__host__ __device__
	static inline uint3 unpackNodeAddress( const uint pAddress );

	/**
	 * Pack a node address
	 *
	 * @param pAddress ...
	 *
	 * @return the associated packed address
	 */
	__host__ __device__
	static inline uint packNodeAddress( const uint3 pAddress );

	/**
	 * Unpack a brick address
	 *
	 * @param pAddress a packed address of a brick of data in cache
	 *
	 * @return the associated unpacked address
	 */
	__host__ __device__
	static inline uint3 unpackBrickAddress( const uint pAddress );

	/**
	 * Pack a brick address
	 *
	 * @param pAddress ...
	 *
	 * @return the associated packed address
	 */
	__host__ __device__
	static inline uint packBrickAddress( const uint3 pAddress );

	/** @name Child Nodes Managment
	 *
	 *  Child nodes managment methods
	 */
	///@{

	/**
	 * Set the child nodes address
	 *
	 * @param dpcoord ...
	 */
	__host__ __device__
	inline void setChildAddress( const uint3 dpcoord );

	/**
	 * Get the child nodes address
	 *
	 * @return ...
	 */
	__host__ __device__
	inline uint3 getChildAddress() const;

	/**
	 * Set the child nodes encoded address
	 *
	 * @param addr ...
	 */
	__host__ __device__
	inline void setChildAddressEncoded( uint addr );

	/**
	 * Get the child nodes encoded address
	 *
	 * @return ...
	 */
	__host__ __device__
	inline uint getChildAddressEncoded() const;

	/**
	 * Tell wheter or not the node has children
	 *
	 * @return a flag telling wheter or not the node has children
	 */
	__host__ __device__
	inline bool hasSubNodes() const;

	/**
	 * Flag the node as beeing terminal or not
	 *
	 * @param pFlag a flag telling wheter or not the node is terminal
	 */
	__host__ __device__
	inline void setTerminal( bool pFlag );

	/**
	 * Tell wheter or not the node is terminal
	 *
	 * @return a flag telling wheter or not the node is terminal
	 */
	__host__ __device__
	inline bool isTerminal() const;

	///@}

	/** @name Bricks Managment
	 *
	 *  Bricks managment methods
	 */
	///@{

	/**
	 * Set the brick address
	 *
	 * @param dpcoord ...
	 */
	__host__ __device__
	inline void setBrickAddress( const uint3 dpcoord );

	/**
	 * Get the brick address
	 *
	 * @return ...
	 */
	__host__ __device__
	inline uint3 getBrickAddress() const;

	/**
	 * Set the brick encoded address
	 *
	 * @param addr ...
	 */
	__host__ __device__
	inline void setBrickAddressEncoded( const uint addr );

	/**
	 * Get the brick encoded address
	 *
	 * @return ...
	 */
	__host__ __device__
	inline uint getBrickAddressEncoded() const;

	/**
	 * Flag the node as containg data or not
	 *
	 * @param pFlag a flag telling wheter or not the node contains data
	 */
	__host__ __device__
	inline void setStoreBrick();

	/**
	 * Tell wheter or not the node is a brick
	 *
	 * @return a flag telling wheter or not the node is a brick
	 */
	__host__ __device__
	inline bool isBrick() const;

	/**
	 * Tell wheter or not the node has a brick,
	 * .i.e the node is a brick and its brick address is not null.
	 *
	 * @return a flag telling wheter or not the node has a brick
	 */
	__host__ __device__
	inline bool hasBrick() const;

	///@}

	/** @name Initialization State
	 *
	 *  Initialization state
	 */
	///@{

	/**
	 * Tell wheter or not the node is initializated
	 *
	 * @return a flag telling wheter or not the node is initializated
	 */
	__host__ __device__
	inline bool isInitializated() const;

	///@}

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} //namespace GvStructure

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvNode.inl"

#endif // !_GV_NODE_H_
