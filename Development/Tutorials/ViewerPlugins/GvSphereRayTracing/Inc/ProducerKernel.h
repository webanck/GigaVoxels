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

#ifndef _PRODUCER_KERNEL_H_
#define _PRODUCER_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GPUVoxelProducer.h>

// CUDA
#include <cuda_runtime.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
* Spheres ray-tracing parameters
*/
__constant__ unsigned int cNbSpheres;
__constant__ unsigned int cMinLevelOfResolutionToHandle;
__constant__ unsigned int cSphereBrickIntersectionType;
__constant__ bool cGeometricCriteria;
__constant__ unsigned int cMinNbSpheresPerBrick;
__constant__ bool cAbsoluteSizeCriteria;
__constant__ bool cFixedSizeSphere;
__constant__ bool cMeanSizeOfSpheres;
__constant__ unsigned int cCoeffAbsoluteSizeCriteria;
__constant__ float cSphereRadiusFader;

/**
 * Coefficient used to approximate a brick by a sphere
 *
 * brickRadius = ( 0.5f * sqrtf( 3.f ) ) * brickWidth;
 *
 * 0.5f * sqrtf( 3.f ) = 0,86602540378443864676372317075294
 */
__device__ static const float cBrickWidth2SphereRadius = 0.866025f;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class ProducerKernel
 *
 * @brief The ProducerKernel class provides the mecanisms to produce data
 * on device following data requests emitted during N-tree traversal (rendering).
 *
 * It is the main user entry point to produce data from GPU, for instance,
 * procedurally generating data (apply noise patterns, etc...).
 *
 * This class is implements the mandatory functions of the GvIProviderKernel base class.
 *
 * @param NodeRes Node tile resolution
 * @param BrickRes Brick resolution
 * @param BorderSize Border size of bricks
 * @param VolTreeKernelType Device-side data structure
 */
template< typename TDataStructureType >
class ProducerKernel
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:
		/****************************** INNER TYPES *******************************/

	/**
	 * Data Structure device-side associated object
	 */
	typedef typename TDataStructureType::VolTreeKernelType DataStructureKernel;

	/**
	 * Type definition of the node tile resolution
	 */
	typedef typename TDataStructureType::NodeTileResolution NodeRes;

	/**
	 * Type definition of the brick resolution
	 */
	typedef typename TDataStructureType::BrickResolution BrickRes;

	/**
	 * Enumeration to define the brick border size
	 */
	enum
	{
		BorderSize = TDataStructureType::BrickBorderSize
	};

	/**
	 * CUDA block dimension used for nodes production (kernel launch)
	 */
	typedef GvCore::StaticRes3D< 32, 1, 1 > NodesKernelBlockSize;

	/**
	 * CUDA block dimension used for bricks production (kernel launch)
	 */
	//typedef GvCore::StaticRes3D< 16, 8, 1 > BricksKernelBlockSize;
	typedef GvCore::StaticRes3D< 10, 10, 10 > BricksKernelBlockSize;
	/******************************* ATTRIBUTES *******************************/

	/**
	 * Initialize the producer
	 * 
	 * @param volumeTreeKernel Reference on a volume tree data structure
	 */
	inline void initialize( DataStructureKernel& pDataStructure );


	/******************************** METHODS *********************************/

	/**
	 * Produce data on device.
	 * Implement the produceData method for the channel 0 (nodes)
	 *
	 * Producing data mecanism works element by element (node tile or brick) depending on the channel.
	 *
	 * In the function, user has to produce data for a node tile or a brick of voxels :
	 * - for a node tile, user has to defined regions (i.e nodes) where lies data, constant values,
	 * etc...
	 * - for a brick, user has to produce data (i.e voxels) at for each of the channels
	 * user had previously defined (color, normal, density, etc...)
	 *
	 * @param pGpuPool The device side pool (nodes or bricks)
	 * @param pRequestID The current processed element coming from the data requests list (a node tile or a brick)
	 * @param pProcessID Index of one of the elements inside a node tile or a voxel bricks
	 * @param pNewElemAddress The address at which to write the produced data in the pool
	 * @param pParentLocInfo The localization info used to locate an element in the pool
	 * @param Loki::Int2Type< 0 > corresponds to the index of the node pool
	 *
	 * @return A feedback value that the user can return.
	 */
	template< typename GPUPoolKernelType >
	__device__
	inline uint produceData( GPUPoolKernelType& nodePool, uint requestID, uint processID,
							uint3 newElemAddress, const GvCore::GvLocalizationInfo& parentLocInfo, Loki::Int2Type< 0 > );

	/**
	 * Produce data on device.
	 * Implement the produceData method for the channel 1 (bricks)
	 *
	 * Producing data mecanism works element by element (node tile or brick) depending on the channel.
	 *
	 * In the function, user has to produce data for a node tile or a brick of voxels :
	 * - for a node tile, user has to defined regions (i.e nodes) where lies data, constant values,
	 * etc...
	 * - for a brick, user has to produce data (i.e voxels) at for each of the channels
	 * user had previously defined (color, normal, density, etc...)
	 *
	 * @param pGpuPool The device side pool (nodes or bricks)
	 * @param pRequestID The current processed element coming from the data requests list (a node tile or a brick)
	 * @param pProcessID Index of one of the elements inside a node tile or a voxel bricks
	 * @param pNewElemAddress The address at which to write the produced data in the pool
	 * @param pParentLocInfo The localization info used to locate an element in the pool
	 * @param Loki::Int2Type< 1 > corresponds to the index of the brick pool
	 *
	 * @return A feedback value that the user can return.
	 */
	template< typename GPUPoolKernelType >
	__device__
	inline uint produceData( GPUPoolKernelType& dataPool, uint requestID, uint processID,
							uint3 newElemAddress, const GvCore::GvLocalizationInfo& parentLocInfo, Loki::Int2Type< 1 > );



	/**
	 * Set the buffer of spheres
	 *
	 * @param pSpheresBuffer the buffer of spheres (position and radius)
	 */
    inline void setPositionBuffer( float4* pSpheresBuffer );
    //inline void setPositionBuffer( std::vector<float4*> pSpheresBuffer );


	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	//GvCore::ArrayKernel3DLinear< float4 > _posBuf;
    //std::vector<float4*> _posBuf;
    float4* _posBuf;

	/**
	 * Data Structure device-side associated object
	 *
	 * Note : use this element if you need to sample data in cache
	 */
	DataStructureKernel _dataStructureKernel;

	/******************************** METHODS *********************************/

	/**
	 * Helper function used to determine the type of zones in the data structure.
	 *
	 * The data structure is made of regions containing data, empty or constant regions.
	 * Besides, this function can tell if the maximum resolution is reached in a region.
	 *
	 * @param regionCoords region coordinates
	 * @param regionDepth region depth
	 *
	 * @return the type of the region
	 */
	__device__
	inline GPUVoxelProducer::GPUVPRegionInfo getRegionInfo( uint3 regionCoords, uint regionDepth );

	/**
	 * Test the intersection between a sphere and a brick
	 *
	 * @param pSphere sphere (position and and radius)
	 * @param pBrickCenter brick center
	 * @param pBoxExtent pBrickWidth brick width
	 *
	 * @return a flag to tell wheter or not intersection occurs
	 */
	__device__
	static inline bool intersectBrick( const float4 pSphere, const float3 pBrickCenter, const float pBrickWidth );

	/**
	 * Sphere-Sphere intersection test
	 *
	 * @param pSphereCenter 1st sphere center
	 * @param pSphereRadius 1stsphere radius
	 * @param pSphere 2nd sphere (position and and radius)
	 *
	 * @return a flag to tell wheter or not intersection occurs
	 */
	__device__
	static inline bool intersectSphereSphere( const float3 pSphereCenter, const float pSphereRadius, const float4 pSphere );

	/**
	 * Sphere-Box intersection test
	 *
	 * @param pBoxCenter box center
	 * @param pBoxExtent box extent (distance from center to one side)
	 * @param pSphere sphere (position and and radius)
	 *
	 * @return a flag to tell wheter or not intersection occurs
	 */
	__device__
	static inline bool intersectSphereBox( const float3 pBoxCenter, const float pBoxExtent, const float4 pSphere );

	/**
	 * Test wheter or not geometric criteria passes
	 *
	 * Note : the node subdivision process is stopped if there is no more than a given number of spheres inside
	 *
	 * @param pNbSpheresInBrick number of spheres in a given brick
	 *
	 * @return a flag to tell wheter or not the criteria passes
	 */
	__device__
	static inline bool isGeometricCriteriaValid( const unsigned int pNbSpheresInBrick );

	/**
	 * Test wheter or not screen based criteria passes
	 *
	 * Note : the node subdivision process is stopped if ...
	 *
	 * @return a flag to tell wheter or not the criteria passes
	 */
	__device__
	static inline bool isScreenSpaceCriteriaValid();

	/**
	 * Test wheter or not absolute size criteria passes
	 *
	 * Note : the node subdivision process is stopped if ...
	 *
	 * @return a flag to tell wheter or not the criteria passes
	 */
	__device__
    static inline bool isAbsoluteSizeCriteriaValid(float sphereRadius, float brickSize);
	
};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "ProducerKernel.inl"

#endif // !_PRODUCER_KERNEL_H_
