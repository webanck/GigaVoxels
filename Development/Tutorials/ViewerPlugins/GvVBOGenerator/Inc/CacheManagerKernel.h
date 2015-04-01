///*
// * GigaVoxels is a ray-guided streaming library used for efficient
// * 3D real-time rendering of highly detailed volumetric scenes.
// *
// * Copyright (C) 2011-2012 INRIA <http://www.inria.fr/>
// *
// * Authors : GigaVoxels Team
// *
// * This program is free software: you can redistribute it and/or modify
// * it under the terms of the GNU General Public License as published by
// * the Free Software Foundation, either version 3 of the License, or
// * (at your option) any later version.
// *
// * This program is distributed in the hope that it will be useful,
// * but WITHOUT ANY WARRANTY; without even the implied warranty of
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// * GNU General Public License for more details.
// *
// * You should have received a copy of the GNU General Public License
// * along with this program.  If not, see <http://www.gnu.org/licenses/>.
// */
//
///** 
// * @version 1.0
// */
//
//#ifndef _CACHE_MANAGER_KERNEL_H_
//#define _CACHE_MANAGER_KERNEL_H_
//
///******************************************************************************
// ******************************* INCLUDE SECTION ******************************
// ******************************************************************************/
//
//// Cuda
//#include <vector_types.h>
//
//// GigaVoxels
//#include <GvCache/GvCacheManagerKernel.h>
//
///******************************************************************************
// ************************* DEFINE AND CONSTANT SECTION ************************
// ******************************************************************************/
//
///******************************************************************************
// ***************************** TYPE DEFINITION ********************************
// ******************************************************************************/
//
///******************************************************************************
// ******************************** CLASS USED **********************************
// ******************************************************************************/
//
///******************************************************************************
// ****************************** CLASS DEFINITION ******************************
// ******************************************************************************/
//
///******************************************************************************
// ***************************** KERNEL DEFINITION ******************************
// ******************************************************************************/
//
///******************************************************************************
// * GvKernel_ReadVboNbPoints kernel
// *
// * This kernel retrieve number of points contained in each used bricks
// *
// * @param pNbPointsList [out] list of points inside each brick
// * @param pNbBricks number of bricks to process
// * @param pBrickAddressList list of brick addresses in cache (used to retrive positions where to fetch data)
// * @param pDataStructure data structure in cache where to fecth data
// ******************************************************************************/
//template< class TDataStructureKernelType >
//__global__
//void GvKernel_ReadVboNbPoints( uint* pNbPointsList, const uint pNbBricks, const uint* pBrickAddressList, TDataStructureKernelType pDataStructure )
//{
//	// Retrieve global data index
//	uint lineSize = __uimul( blockDim.x, gridDim.x );
//	uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );
//
//	// Check bounds
//	if ( elem < pNbBricks )
//	{
//		//	printf( "\nGvKernel_ReadVboNbPoints : ELEM %d : dataPoolPosition = [ %d ]", elem, pBrickAddressList[ elem ] );
//
//		// Retrieve the number of points in the current brick
//		//
//		// Brick position in cache (without border)
//		//float3 dataPoolPosition = make_float3( GvStructure::GvNode::unpackBrickAddress( pBrickAddressList[ elem ] ) ) * pDataStructure.brickCacheResINV;
//		//float3 dataPoolPosition = make_float3( 20, 0, 0 ) * pDataStructure.brickCacheResINV;
//		//uint3 unpackedBrickAddress = GvStructure::GvNode::unpackBrickAddress( pBrickAddressList[ elem ] );
//		//uint3 unpackedBrickAddress = 10 * GvStructure::GvNode::unpackBrickAddress( pBrickAddressList[ elem ] );
//
//		// The adress pBrickAddressList[ elem ] is the adress of a brick in the elemAdressList 1D linearized array cache.
//		// The adress is the adress of the brick beginning with the border (there is no one border ofset)
//		uint3 unpackedBrickAddress = ( TDataStructureKernelType::BrickResolution::get() + 2 * TDataStructureKernelType::brickBorderSize ) * GvStructure::GvNode::unpackBrickAddress( pBrickAddressList[ elem ] );
//		//	printf( "\nunpackedBrickAddress = [ %d | %d | %d ]", unpackedBrickAddress.x, unpackedBrickAddress.y, unpackedBrickAddress.z );
//
//		float3 dataPoolPosition = make_float3( unpackedBrickAddress.x, unpackedBrickAddress.y, unpackedBrickAddress.z ) * pDataStructure.brickCacheResINV;
//		//	printf( "\nGvKernel_ReadVboNbPoints : dataPoolPosition = [ %f | %f | %f ]", dataPoolPosition.x, dataPoolPosition.y, dataPoolPosition.z );
//		// Shift from one voxel to be on the brick border
//		//dataPoolPosition -= pDataStructure.brickCacheResINV;
//		//const float4 nbPoints = pDataStructure.template getSampleValueTriLinear< 0 >( pBrickSampler.brickChildPosInPool - pDataStructure.brickCacheResINV,
//		//																			/*offset of 1/2 voxel to reach texel*/0.5f * pDataStructure.brickCacheResINV );
//		//	printf( "\n\tGvKernel_ReadVboNbPoints : dataPoolPosition = [ %f | %f | %f ]", dataPoolPosition.x, dataPoolPosition.y, dataPoolPosition.z );
//		const float4 nbPoints = pDataStructure.template getSampleValueTriLinear< 0 >( dataPoolPosition,
//			/*offset of 1/2 voxel to reach texel*/0.5f * pDataStructure.brickCacheResINV );
//
//		// Write to output global memory
//		pNbPointsList[ elem ] = nbPoints.x;
//
//		//	printf( "\nGvKernel_ReadVboNbPoints : nbPoints = [ %f | %f | %f | %f ]", nbPoints.x, nbPoints.y, nbPoints.z, nbPoints.w );
//	}
//}
//
///******************************************************************************
// * GvKernel_UpdateVBO kernel
// *
// * This kernel update the VBO by dumping all used bricks content (i.e. points)
// *
// * @param pVBO VBO to update
// * @param pNbBricks number of bricks to process
// * @param pBrickAddressList list of brick addresses in cache (used to retrive positions where to fetch data)
// * @param pNbPointsList list of points inside each brick
// * @param pVboIndexOffsetList list of number of points for each used bricks
// * @param pDataStructure data structure in cache where to fecth data
// ******************************************************************************/
//template< class TDataStructureKernelType >
//__global__
//void GvKernel_UpdateVBO( float4* pVBO, const uint pNbBricks, const uint* pBrickAddressList, const uint* pNbPointsList, const uint* pVboIndexOffsetList, TDataStructureKernelType pDataStructure )
//{
//	// Retrieve global data index
//	uint lineSize = __uimul( blockDim.x, gridDim.x );
//	uint elem = threadIdx.x + __uimul( blockIdx.x, blockDim.x ) + __uimul( blockIdx.y, lineSize );
//
//	// Check bounds
//	if ( elem < pNbBricks )
//	{
//		// Iterate through points
//		const uint nbPoints = pNbPointsList[ elem ];
//		const uint indexOffset = pVboIndexOffsetList[ elem ];
//
//		//	printf( "\nGvKernel_UpdateVBO : brick indx [ %d / %d ] / nb points [ %d ] / indexOffset [ %d ]", elem + 1, pNbBricks, nbPoints, indexOffset );
//
//		for ( int i = 0; i < nbPoints; ++i )
//		{
//			// Retrieve the number of points in the current brick
//			//
//			// Brick position in cache (without border)
//			//float3 dataPoolPosition = make_float3( GvStructure::GvNode::unpackBrickAddress( pBrickAddressList[ elem ] ) ) * pDataStructure.brickCacheResINV;
//			//float3 dataPoolPosition = 10.0f * make_float3( GvStructure::GvNode::unpackBrickAddress( pBrickAddressList[ elem ] ) ) * pDataStructure.brickCacheResINV;
//
//			uint3 unpackedBrickAddress = ( TDataStructureKernelType::BrickResolution::get() + 2 * TDataStructureKernelType::brickBorderSize ) * GvStructure::GvNode::unpackBrickAddress( pBrickAddressList[ elem ] );
//
//			float3 dataPoolPosition = make_float3( unpackedBrickAddress.x, unpackedBrickAddress.y, unpackedBrickAddress.z ) * pDataStructure.brickCacheResINV;
//
//			// Shift from one voxel to be on the brick border
//			//dataPoolPosition -= pDataStructure.brickCacheResINV;
//			//const float4 point = pDataStructure.template getSampleValueTriLinear< 0 >( pBrickSampler.brickChildPosInPool - pDataStructure.brickCacheResINV,
//			//	/*offset of 1/2 voxel to reach texel*/( i + 2 ) * 0.5f * pDataStructure.brickCacheResINV );
//
//			//const float4 nbPoints = pDataStructure.template getSampleValueTriLinear< 0 >( dataPoolPosition,
//			//																		/*offset of 1/2 voxel to reach texel*/( i + 2 ) * 0.5f * pDataStructure.brickCacheResINV );
//
//
//			const float4 point = pDataStructure.template getSampleValueTriLinear< 0 >( dataPoolPosition
//				, 0.5f * pDataStructure.brickCacheResINV +  make_float3( ( i + 2 ) * pDataStructure.brickCacheResINV.x, 0.f, 0.f ) );
//
//			//	printf( "\nGvKernel_UpdateVBO : position [ %d / %d ] = [ %f | %f | %f | %f ]", i + 1, nbPoints, point.x, point.y, point.z, point.w );
//
//			// Write to output global memory
//			pVBO[ indexOffset + i ] = point;
//		}
//	}
//}
//	
//#endif // !_CACHE_MANAGER_KERNEL_H_
