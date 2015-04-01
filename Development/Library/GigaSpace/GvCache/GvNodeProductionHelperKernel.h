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

#ifndef _GV_NODE_PRODUCTION_HELPER_KERNEL_H_
#define _GV_NODE_PRODUCTION_HELPER_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaSpace
#include "GvCore/GvCoreConfig.h"
#include "GvStructure/GvVolumeTreeAddressType.h"
#include "GvCore/vector_types_ext.h"
#include "GvCore/StaticRes3D.h"

/******************************************************************************
 ***************************** KERNEL DEFINITION ******************************
 ******************************************************************************/

namespace GvCache
{

/******************************************************************************
 * KERNEL GvKernel_Produce_01
 *
 * This method is a helper for writing into the cache.
 *
 * @param pNbNodes The number of elements we need to produce and write.
 * @param pNodeAddresses buffer of element addresses that producer(s) has to process (subdivide or produce/load)
 * @param pNewNodeTileIndexes buffer of available element addresses in cache where producer(s) can write
 * @param pPool The pool where we will write the produced elements.
 * @param pProvider The provider called for the production.
 * @param pPageTable page table used to retrieve localization information from element addresses
 ******************************************************************************/
template< /*typename TNodeTileRes,*/ typename TElementRes, typename TPoolType, typename TProviderType, typename TPageTableType >
__global__
// TO DO : Prolifing / Optimization
// - use "Launch Bounds" feature to profile / optimize code
// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
void GvKernel_Produce_01( const uint pNbNodes, uint* pNodeAddresses, uint* pNewNodeTileIndexes,
						    TPoolType pPool, TProviderType pProvider, TPageTableType pPageTable )
{
	// TO DO
	// - optimization : actually we are launching 1D block of 32 threads, so no need to synchronize

	typedef GvCore::StaticRes1D< 2 > NodeTileRes;
	typedef GvCore::StaticRes3D< NodeTileRes::numElements, 1, 1 > NodeTileResLinear;

	// NOTE :
	// - 1 block of threads is responsible of the production/refinement/subdiviion of 1 node
	// - 1 node is refined and resulting children are stored in 1 nodetile (contiguously in memory)
	// - 1 thread per 1 child node
	// - ex: for an octree, we use blocks of 32 threads, but only 8 threads are used (TO DO : try to enhance that)

	//typedef GvCore::StaticRes3D< TElementRes::numElements, 1, 1 > NodeTileResLinear;

	// Retrieve global indexes
	//
	// - node ID to refine
	const uint elemNum = blockIdx.x; // TO DO : ERROR !!!!!? => cause kernel is 2D
	// - index of child node (varies from [ 0 ] to [ "nb nodes in a nodetile" - 1 ])
	const uint nodeTileChildID = threadIdx.x + __uimul( threadIdx.y, blockDim.x ) + __uimul( threadIdx.z, __uimul( blockDim.x, blockDim.y ) );
	
	// Check bound
	if ( elemNum < pNbNodes )
	{
		// Shared Memory declaration
		__shared__ uint smNodeAddress;
		__shared__ /*uint*/uint3 smNewNodeTileAddress;
		__shared__ GvCore::GvLocalizationInfo smNodeLocalizationInfo;

		// Retrieve node address and its localisation info along with new element address
		//
		// Done by only one thread per block (i.e. per node)
		uint nodeTileIndex;
		uint nodeTileAddress;
		uint nodeTileOffset;
		uint3 nodeOffset;
		GvCore::GvLocalizationInfo::CodeType parentLocCode;
		GvCore::GvLocalizationInfo::DepthType parentLocDepth;
		if ( nodeTileChildID == 0 )
		{
			// Retrieve node address (the one to refine/subdivide)
			smNodeAddress = pNodeAddresses[ elemNum ] & 0x3FFFFFFF;

			// Compute nodetile address (the start address where node children will be produced)
			const /*uint*/uint3 newNodeTileIndex = GvStructure::VolTreeNodeAddress::unpackAddress( pNewNodeTileIndexes[ elemNum ] );//pNewNodeTileIndexes[ elemNum ] & 0x3FFFFFFF;
			smNewNodeTileAddress = newNodeTileIndex * NodeTileResLinear::get(); // convert into node address

			// Get the localization of the current element
			//smNodeLocalizationInfo = pPageTable.getLocalizationInfo( smNodeAddress ); // replace class with direct array access
			//-------------------------------------------------------------------------------------------------
			//-------------------------------------------------------------------------------------------------
			// Compute the address of the current node tile (and its offset in the node tile)
			// TO DO : check for modulo "%" speed
			nodeTileIndex = smNodeAddress / NodeTileRes::getNumElements();
			nodeTileAddress = nodeTileIndex * NodeTileRes::getNumElements();
			nodeTileOffset = smNodeAddress - nodeTileAddress;
			// Compute the node offset (in 3D, in the node tile)
			nodeOffset = NodeTileRes::toFloat3( nodeTileOffset );
			// Fetch associated localization infos
			parentLocCode = pPageTable.locCodeArray.get( nodeTileIndex ); // replace class with direct array access
			parentLocDepth = pPageTable.locDepthArray.get( nodeTileIndex ); // replace class with direct array access
			// Localization info initialization
			smNodeLocalizationInfo.locCode = parentLocCode.addLevel< NodeTileRes >( nodeOffset );
			smNodeLocalizationInfo.locDepth = parentLocDepth;
			//-------------------------------------------------------------------------------------------------
			//-------------------------------------------------------------------------------------------------
		}

		//-------------------------------------------------------------------------------------------------
		return;
		//-------------------------------------------------------------------------------------------------

		// Thread Synchronization
		__syncthreads();

		// Produce data
		// - call user-defined producer
		uint producerFeedback = pProvider.produceData( pPool, elemNum, nodeTileChildID, smNewNodeTileAddress, smNodeLocalizationInfo );



		






		// TO DO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		// - check if it is really safe to remove the "Thread Synchronization" __syncthreads();
		// - if yes, it should not be a problem for setPointer() code below
		// - it shoud be OK cause childArray (i.e. octree) is accessed trhough nodeAdress and not newNodeTileAdress :)

		// Update page table (who manages localization information)
		//
		// - write the new "children nodetile address" to the current refined node address
		//
		// Done by only one thread per block (i.e. per node)
		if ( nodeTileChildID == 0 )
		{
			// TO DO !!!!!
			//
			// - lots of computation has already be done in pPageTable.getLocalizationInfo()
			//
			// - idea : page table should not handle writing child adress but only localization info !! => move code ?
			//pPageTable.setPointer( smNodeAddress, smNewNodeTileAddress, /*un-used*/producerFeedback );
			
			//-------------------------------------------------------------------------------------------------
			//-------------------------------------------------------------------------------------------------
			// setPointerImpl( uint elemAddress, ElemAddressType elemPointer, /*un-used*/uint flags )

			uint packedChildAddress/*wrong name*/ = pPageTable.childArray.get( smNodeAddress );
			uint packedAddress = GvStructure::VolTreeNodeAddress::packAddress( smNewNodeTileAddress ); // for node, does nothing...

			// Update node tile's pointer
			pPageTable.childArray.set( smNodeAddress/*elemAddress*/,
				( packedChildAddress & 0x40000000 /*is that used ? CAUSE in production, only 0x3fffffff is used*/) | ( packedAddress & 0x3FFFFFFF ) );

			// Compute the address of the current node tile
			// - already done by pPageTable.getLocalizationInfo()
			//uint nodeTileIndex = smNodeAddress/*elemAddress*/ / NodeTileRes::getNumElements();
			//uint nodeTileAddress = nodeTileIndex * NodeTileRes::getNumElements();
			//uint nodeTileOffset = smNodeAddress/*elemAddress*/ - nodeTileAddress;

			// Compute the node offset
			// - already done by pPageTable.getLocalizationInfo()
			//uint3 nodeOffset = NodeTileRes::toFloat3( nodeTileOffset );

			// Fetch associated localization infos
			//
			// TO DO : already done in the production main kernel ?
			// - already done by pPageTable.getLocalizationInfo()
			//GvLocalizationInfo::CodeType parentLocCode = pPageTable.locCodeArray.get( nodeTileIndex );
			//GvLocalizationInfo::DepthType parentLocDepth = pPageTable.locDepthArray.get( nodeTileIndex );

			// Compute the address of the new node tile
			// - already done by pPageTable.getLocalizationInfo()
			//uint newNodeTileIndex = smNewNodeTileAddress/*elemPointer*/.x / ElementRes::getNumElements();
			// => ressemble à "newNodeTileIndex" déjà calculé ?
			uint newNodeTileIndex = smNewNodeTileAddress.x/*elemPointer.x*/ / NodeTileResLinear::getNumElements();
			
			// Update associated localization infos
			GvCore::GvLocalizationInfo::CodeType newLocCode = parentLocCode.addLevel< NodeTileRes >( nodeOffset );
			// => ressemble à "smNodeLocalizationInfo.locCode" déjà calculé ?
			GvCore::GvLocalizationInfo::DepthType newLocDepth = parentLocDepth.addLevel();

			pPageTable.locCodeArray.set( newNodeTileIndex, newLocCode );
			pPageTable.locDepthArray.set( newNodeTileIndex, newLocDepth );
			//-------------------------------------------------------------------------------------------------
			//-------------------------------------------------------------------------------------------------
		}
	}
}

/******************************************************************************
 * KERNEL GvKernel_Produce_01
 *
 * This method is a helper for writing into the cache.
 *
 * @param pNbNodes The number of elements we need to produce and write.
 * @param pNodeAddresses buffer of element addresses that producer(s) has to process (subdivide or produce/load)
 * @param pNewNodeTileIndexes buffer of available element addresses in cache where producer(s) can write
 * @param pPool The pool where we will write the produced elements.
 * @param pProvider The provider called for the production.
 * @param pPageTable page table used to retrieve localization information from element addresses
 ******************************************************************************/
template< /*typename TNodeTileRes,*/ typename TElementRes, typename TPoolType, typename TProviderType, typename TPageTableType >
__global__
// TO DO : Prolifing / Optimization
// - use "Launch Bounds" feature to profile / optimize code
// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
void GvKernel_Produce_02( const uint pNbNodes, uint* pNodeAddresses, uint* pNewNodeTileIndexes,
						    TPoolType pPool, TProviderType pProvider, TPageTableType pPageTable )
{
	// TO DO
	// - optimization : actually we are launching 1D block of 32 threads, so no need to synchronize

	typedef GvCore::StaticRes1D< 2 > NodeTileRes;
	typedef GvCore::StaticRes3D< NodeTileRes::numElements, 1, 1 > NodeTileResLinear;

	// NOTE :
	// - 1 block of threads is responsible of the production/refinement/subdiviion of 1 node
	// - 1 node is refined and resulting children are stored in 1 nodetile (contiguously in memory)
	// - 1 thread per 1 child node
	// - ex: for an octree, we use blocks of 32 threads, but only 8 threads are used (TO DO : try to enhance that)

	//typedef GvCore::StaticRes3D< TElementRes::numElements, 1, 1 > NodeTileResLinear;

	// Retrieve global indexes
	//
	// - node ID to refine
	const uint elemNum = blockIdx.x; // TO DO : ERROR !!!!!? => cause kernel is 2D
	// - index of child node (varies from [ 0 ] to [ "nb nodes in a nodetile" - 1 ])
	const uint nodeTileChildID = threadIdx.x + __uimul( threadIdx.y, blockDim.x ) + __uimul( threadIdx.z, __uimul( blockDim.x, blockDim.y ) );

	// Check bound
	if ( elemNum < pNbNodes )
	{
		// TO DO
		// Maybe, if no synchronization, no need to use Shared Memory also ? => local registers are faster that Shared Memory ?

		// Shared Memory declaration
		__shared__ uint smNodeAddress;
		__shared__ /*uint*/uint3 smNewNodeTileAddress;
		__shared__ GvCore::GvLocalizationInfo smNodeLocalizationInfo;

		// Retrieve node address and its localisation info along with new element address
		//
		// Done by only one thread per block (i.e. per node)
		uint nodeTileIndex;
		uint nodeTileAddress;
		uint nodeTileOffset;
		uint3 nodeOffset;
		GvCore::GvLocalizationInfo::CodeType parentLocCode;
		GvCore::GvLocalizationInfo::DepthType parentLocDepth;
		if ( nodeTileChildID == 0 )
		{
			// Retrieve node address (the one to refine/subdivide)
			smNodeAddress = pNodeAddresses[ elemNum ] & 0x3FFFFFFF;

			// Compute nodetile address (the start address where node children will be produced)
			const /*uint*/uint3 newNodeTileIndex = GvStructure::VolTreeNodeAddress::unpackAddress( pNewNodeTileIndexes[ elemNum ] );//pNewNodeTileIndexes[ elemNum ] & 0x3FFFFFFF;
			smNewNodeTileAddress = newNodeTileIndex * NodeTileResLinear::get(); // convert into node address

			// Get the localization of the current element
			//
			// Compute the address of the current node tile (and its offset in the node tile)
			// TO DO : check for modulo "%" speed
			nodeTileIndex = smNodeAddress / NodeTileRes::getNumElements();
			nodeTileAddress = nodeTileIndex * NodeTileRes::getNumElements();
			nodeTileOffset = smNodeAddress - nodeTileAddress;
			// Compute the node offset (in 3D, in the node tile)
			nodeOffset = NodeTileRes::toFloat3( nodeTileOffset );
			// Fetch associated localization infos
			parentLocCode = pPageTable.locCodeArray.get( nodeTileIndex ); // replace class with direct array access
			parentLocDepth = pPageTable.locDepthArray.get( nodeTileIndex ); // replace class with direct array access
			// Localization info initialization
			smNodeLocalizationInfo.locCode = parentLocCode.addLevel< NodeTileRes >( nodeOffset );
			smNodeLocalizationInfo.locDepth = parentLocDepth;
		}

		// Thread Synchronization
		//__syncthreads(); // => check for int cudaDeviceProp::warpSize; /**< Warp size in threads */

		// Produce data
		// - call user-defined producer
		uint producerFeedback = pProvider.produceData( pPool, elemNum, nodeTileChildID, smNewNodeTileAddress, smNodeLocalizationInfo );

		// TO DO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		// - check if it is really safe to remove the "Thread Synchronization" __syncthreads();
		// - if yes, it should not be a problem for setPointer() code below
		// - it shoud be OK cause childArray (i.e. octree) is accessed trhough nodeAdress and not newNodeTileAdress :)

		// Update page table (who manages localization information)
		//
		// - write the new "children nodetile address" to the current refined node address
		//
		// Done by only one thread per block (i.e. per node)
		if ( nodeTileChildID == 0 )
		{
			uint packedChildAddress/*wrong name*/ = pPageTable.childArray.get( smNodeAddress );
			uint packedAddress = GvStructure::VolTreeNodeAddress::packAddress( smNewNodeTileAddress ); // for node, does nothing...

			// Update node tile's pointer
			pPageTable.childArray.set( smNodeAddress/*elemAddress*/,
				( packedChildAddress & 0x40000000 /*is that used ? CAUSE in production, only 0x3fffffff is used*/) | ( packedAddress & 0x3FFFFFFF )/*nodetile address of children*/ );

			// Compute the address of the new node tile
			// - already done by pPageTable.getLocalizationInfo()
			//uint newNodeTileIndex = smNewNodeTileAddress/*elemPointer*/.x / ElementRes::getNumElements();
			// => ressemble à "newNodeTileIndex" déjà calculé ?
			uint newNodeTileIndex = smNewNodeTileAddress.x/*elemPointer.x*/ / NodeTileResLinear::getNumElements();
			
			// Update associated localization infos
			GvCore::GvLocalizationInfo::CodeType newLocCode = parentLocCode.addLevel< NodeTileRes >( nodeOffset );
			// => ressemble à "smNodeLocalizationInfo.locCode" déjà calculé ?
			GvCore::GvLocalizationInfo::DepthType newLocDepth = parentLocDepth.addLevel();

			pPageTable.locCodeArray.set( newNodeTileIndex, newLocCode );
			pPageTable.locDepthArray.set( newNodeTileIndex, newLocDepth );
		}
	}
}

/******************************************************************************
 * KERNEL GvKernel_Produce_01
 *
 * This method is a helper for writing into the cache.
 *
 * @param pNbNodes The number of elements we need to produce and write.
 * @param pNodeAddresses buffer of element addresses that producer(s) has to process (subdivide or produce/load)
 * @param pNewNodeTileIndexes buffer of available element addresses in cache where producer(s) can write
 * @param pPool The pool where we will write the produced elements.
 * @param pProvider The provider called for the production.
 * @param pPageTable page table used to retrieve localization information from element addresses
 ******************************************************************************/
template< /*typename TNodeTileRes,*/ typename TElementRes, typename TPoolType, typename TProviderType, typename TPageTableType >
__global__
// TO DO : Prolifing / Optimization
// - use "Launch Bounds" feature to profile / optimize code
// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
void GvKernel_Produce_03( const uint pNbNodes, uint* pNodeAddresses, uint* pNewNodeTileIndexes,
						    TPoolType pPool, TProviderType pProvider, TPageTableType pPageTable )
{
	// TO DO
	// - optimization : actually we are launching 1D block of 32 threads, so no need to synchronize

	typedef GvCore::StaticRes1D< 2 > NodeTileRes;
	typedef GvCore::StaticRes3D< NodeTileRes::numElements, 1, 1 > NodeTileResLinear;

	// NOTE :
	// - 1 block of threads is responsible of the production/refinement/subdiviion of 1 node
	// - 1 node is refined and resulting children are stored in 1 nodetile (contiguously in memory)
	// - 1 thread per 1 child node
	// - ex: for an octree, we use blocks of 32 threads, but only 8 threads are used (TO DO : try to enhance that)

	//typedef GvCore::StaticRes3D< TElementRes::numElements, 1, 1 > NodeTileResLinear;

	// Retrieve global indexes
	//
	// - node ID to refine
	const uint elemNum = blockIdx.x; // TO DO : ERROR !!!!!? => cause kernel is 2D
	// - index of child node (varies from [ 0 ] to [ "nb nodes in a nodetile" - 1 ])
	const uint nodeTileChildID = threadIdx.x + __uimul( threadIdx.y, blockDim.x ) + __uimul( threadIdx.z, __uimul( blockDim.x, blockDim.y ) );

	// Check bound
	if ( elemNum < pNbNodes )
	{
		// TO DO
		// Maybe, if no synchronization, no need to use Shared Memory also ? => local registers are faster that Shared Memory ?

		// Shared Memory declaration
		__shared__ uint smNodeAddress;
		__shared__ /*uint*/uint3 smNewNodeTileAddress;
		__shared__ GvCore::GvLocalizationInfo smNodeLocalizationInfo;

		// Retrieve node address and its localisation info along with new element address
		//
		// Done by only one thread per block (i.e. per node)
		uint nodeTileIndex;
		uint nodeTileAddress;
		uint nodeTileOffset;
		uint3 nodeOffset;
		GvCore::GvLocalizationInfo::CodeType parentLocCode;
		GvCore::GvLocalizationInfo::DepthType parentLocDepth;
		if ( nodeTileChildID == 0 )
		{
			// Retrieve node address (the one to refine/subdivide)
			smNodeAddress = pNodeAddresses[ elemNum ] & 0x3FFFFFFF;

			// Compute nodetile address (the start address where node children will be produced)
			const /*uint*/uint3 newNodeTileIndex = GvStructure::VolTreeNodeAddress::unpackAddress( pNewNodeTileIndexes[ elemNum ] );//pNewNodeTileIndexes[ elemNum ] & 0x3FFFFFFF;
			smNewNodeTileAddress = newNodeTileIndex * NodeTileResLinear::get(); // convert into node address

			// Get the localization of the current element
			//
			// Compute the address of the current node tile (and its offset in the node tile)
			// TO DO : check for modulo "%" speed
			nodeTileIndex = smNodeAddress / NodeTileRes::getNumElements();
			nodeTileAddress = nodeTileIndex * NodeTileRes::getNumElements();
			nodeTileOffset = smNodeAddress - nodeTileAddress;
			// Compute the node offset (in 3D, in the node tile)
			nodeOffset = NodeTileRes::toFloat3( nodeTileOffset );
			// Fetch associated localization infos
			parentLocCode = pPageTable.locCodeArray.get( nodeTileIndex ); // replace class with direct array access
			parentLocDepth = pPageTable.locDepthArray.get( nodeTileIndex ); // replace class with direct array access
			// Localization info initialization
			smNodeLocalizationInfo.locCode = parentLocCode.addLevel< NodeTileRes >( nodeOffset );
			smNodeLocalizationInfo.locDepth = parentLocDepth;
		}

		// Thread Synchronization
		//__syncthreads(); // => check for int cudaDeviceProp::warpSize; /**< Warp size in threads */

		// Produce data
		// - call user-defined producer
		uint producerFeedback = pProvider.produceData( pPool, elemNum, nodeTileChildID, smNewNodeTileAddress, smNodeLocalizationInfo );

		// TO DO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		// - check if it is really safe to remove the "Thread Synchronization" __syncthreads();
		// - if yes, it should not be a problem for setPointer() code below
		// - it shoud be OK cause childArray (i.e. octree) is accessed trhough nodeAdress and not newNodeTileAdress :)

		// Update page table (who manages localization information)
		//
		// - write the new "children nodetile address" to the current refined node address
		//
		// Done by only one thread per block (i.e. per node)
		if ( nodeTileChildID == 0 )
		{
			uint packedChildAddress/*wrong name*/ = pPageTable.childArray.get( smNodeAddress );
			uint packedAddress = GvStructure::VolTreeNodeAddress::packAddress( smNewNodeTileAddress ); // for node, does nothing...

			// Update node tile's pointer
			pPageTable.childArray.set( smNodeAddress/*elemAddress*/,
				( packedChildAddress & 0x40000000 /*is that used ? CAUSE in production, only 0x3fffffff is used*/) | ( packedAddress & 0x3FFFFFFF )/*nodetile address of children*/ );

			// Compute the address of the new node tile
			// - already done by pPageTable.getLocalizationInfo()
			//uint newNodeTileIndex = smNewNodeTileAddress/*elemPointer*/.x / ElementRes::getNumElements();
			// => ressemble à "newNodeTileIndex" déjà calculé ?
			uint newNodeTileIndex = smNewNodeTileAddress.x/*elemPointer.x*/ / NodeTileResLinear::getNumElements();
			
			// Update associated localization infos
			GvCore::GvLocalizationInfo::CodeType newLocCode = parentLocCode.addLevel< NodeTileRes >( nodeOffset );
			// => ressemble à "smNodeLocalizationInfo.locCode" déjà calculé ?
			GvCore::GvLocalizationInfo::DepthType newLocDepth = parentLocDepth.addLevel();

			pPageTable.locCodeArray.set( newNodeTileIndex, newLocCode );
			pPageTable.locDepthArray.set( newNodeTileIndex, newLocDepth );
		}
	}
}

/******************************************************************************
 * KERNEL GvKernel_genericWriteIntoCache
 *
 * This method is a helper for writing into the cache.
 *
 * @param pNumElems The number of elements we need to produce and write.
 * @param pNodesAddressList buffer of element addresses that producer(s) has to process (subdivide or produce/load)
 * @param pElemAddressList buffer of available element addresses in cache where producer(s) can write
 * @param pGpuPool The pool where we will write the produced elements.
 * @param pGpuProvider The provider called for the production.
 * @param pPageTable page table used to retrieve localization information from element addresses
 ******************************************************************************/
template< typename TElementRes, typename TGPUPoolType, typename TGPUProviderType, typename TPageTableType >
__global__
// TO DO : Prolifing / Optimization
// - use "Launch Bounds" feature to profile / optimize code
// __launch_bounds__( maxThreadsPerBlock, minBlocksPerMultiprocessor )
void Gv_enhancedGenericWriteIntoCache( const uint pNumElems, uint* pNodesAddressList, uint* pElemAddressList,
									  TGPUPoolType pGpuPool, TGPUProviderType pGpuProvider, TPageTableType pPageTable )
{
	//--------------------------------------------------------------------------
	typedef GvCore::StaticRes1D< 2 > NodeTileRes;
	typedef GvCore::StaticRes3D< NodeTileRes::numElements, 1, 1 > ElementRes;

	typedef GvStructure::VolTreeNodeAddress AddressType;
	typedef typename AddressType::AddressType ElemAddressType;
	typedef typename AddressType::PackedAddressType	ElemPackedAddressType;
	//--------------------------------------------------------------------------

	// Retrieve global indexes
	//
	// Node index
	const uint elemNum = blockIdx.x;
	// Node child index (3D linearization)
	const uint processID = threadIdx.x + __uimul( threadIdx.y, blockDim.x ) + __uimul( threadIdx.z, __uimul( blockDim.x, blockDim.y ) );

	// Check bound
	if ( elemNum < pNumElems )
	{
		// Clean the syntax a bit
		//typedef typename TPageTableType::ElemAddressType ElemAddressType;

		// Shared Memory declaration
		__shared__ uint nodeAddress;
		__shared__ ElemAddressType/*uint3*/ elemAddress;
		__shared__ GvCore::GvLocalizationInfo parentLocInfo;
		//--------------------------------------------------------------------------
		//__shared__ ElemAddressType newNodeTileIndex;
		//--------------------------------------------------------------------------

		// Retrieve node address and its localisation info along with new element address
		//
		// Done by only one thread of the kernel
		if ( processID == 0 )
		{
			// Compute node address
			//const uint nodeAddressEnc = pNodesAddressList[ elemNum ];
			//nodeAddress = GvStructure::VolTreeNodeAddress::unpackAddress( nodeAddressEnc ).x;
			nodeAddress = GvStructure::VolTreeNodeAddress::unpackAddress( pNodesAddressList[ elemNum ] ).x;
			//nodeAddress = pNodesAddressList[ elemNum ] & 0x3FFFFFFF;

			// Compute element address
			//const uint elemIndexEnc = pElemAddressList[ elemNum ];
			//const ElemAddressType elemIndex = TPageTableType::ElemType::unpackAddress( pElemAddressList[ elemNum ] );
			//elemAddress = elemIndex * TElementRes::get(); // convert into node address             ===> NOTE : for bricks, the resolution holds the border !!!
			//--------------------------------------------------------------------------
			elemAddress = TPageTableType::ElemType::unpackAddress( pElemAddressList[ elemNum ] ) * TElementRes::get();
			//newNodeTileIndex = TPageTableType::ElemType::unpackAddress( pElemAddressList[ elemNum ] );
			//elemAddress = newNodeTileIndex * TElementRes::get();
			//--------------------------------------------------------------------------

			// Get the localization of the current element
			//parentLocInfo = pPageTable.getLocalizationInfo( elemNum );
			//parentLocInfo = pPageTable.getLocalizationInfo( nodeAddress );
			//--------------------------------------------------------------------------
			// Compute the address of the current node tile (and its offset in the node tile)
			uint nodeTileIndex = nodeAddress / NodeTileRes::getNumElements();
			uint nodeTileAddress = nodeTileIndex * NodeTileRes::getNumElements();
			uint nodeTileOffset = nodeAddress - nodeTileAddress;
			// Compute the node offset (in 3D, in the node tile)
			uint3 nodeOffset = NodeTileRes::toFloat3( nodeTileOffset );
			// Fetch associated localization infos
			GvCore::GvLocalizationInfo::CodeType parentLocCode = pPageTable.locCodeArray.get( nodeTileIndex );
			GvCore::GvLocalizationInfo::DepthType parentLocDepth = pPageTable.locDepthArray.get( nodeTileIndex );
			// Localization info initialization
			parentLocInfo.locCode = parentLocCode.addLevel< NodeTileRes >( nodeOffset );
			parentLocInfo.locDepth = parentLocDepth;
			//--------------------------------------------------------------------------
		}

		// Thread Synchronization
		//__syncthreads();

		// Produce data
		/*__shared__*/ uint producerFeedback;	// Shared Memory declaration
		producerFeedback = pGpuProvider.produceData( pGpuPool, elemNum, processID, elemAddress, parentLocInfo );

		// TO DO : optimization
		// Remove this synchonization for brick production
		// DO the stuff in the producer directly

		// Thread Synchronization
		//__syncthreads();

		// Update page table (who manages localisation information)
		//
		// Done by only one thread of the kernel
		if ( processID == 0 )
		{
			//pPageTable.setPointer( nodeAddress, elemAddress, producerFeedback );
			//--------------------------------------------------------------------------
		////	setPointerImpl( uint /*elemAddress*/nodeAddress, ElemAddressType /*elemPointer*/elemAddress, /*un-used*/uint flags )
		//	//{
				ElemPackedAddressType/*uint*/ packedChildAddress/*wrong name*/ = pPageTable.childArray.get( /*elemAddress*/nodeAddress );
				ElemPackedAddressType/*uint*/ packedAddress = AddressType::packAddress( /*elemPointer*/elemAddress ); // for node, does nothing...

		//		// Update node tile's pointer
				pPageTable.childArray.set( /*elemAddress*/nodeAddress,
					( packedChildAddress & 0x40000000 /*is that used ? CAUSE in production, only 0x3fffffff is used*/) | ( packedAddress & 0x3FFFFFFF ) );

		//		// Compute the address of the current node tile
		//		uint nodeTileIndex = /*elemAddress*/nodeAddress / NodeTileRes::getNumElements();
		//		uint nodeTileAddress = nodeTileIndex * NodeTileRes::getNumElements();
		//		uint nodeTileOffset = /*elemAddress*/nodeAddress - nodeTileAddress;

		//		// Compute the node offset
		//		uint3 nodeOffset = NodeTileRes::toFloat3( nodeTileOffset );

		//		// Fetch associated localization infos
		//		//
		//		// TO DO : already done in the production main kernel ?
		//		GvCore::GvLocalizationInfo::CodeType parentLocCode = pPageTable.locCodeArray.get( nodeTileIndex );
		//		GvCore::GvLocalizationInfo::DepthType parentLocDepth = pPageTable.locDepthArray.get( nodeTileIndex );

		//		// Compute the address of the new node tile
				uint newNodeTileIndex = /*elemPointer*/elemAddress.x / ElementRes::getNumElements();
		
		//		// Update associated localization infos
				//GvCore::GvLocalizationInfo::CodeType newLocCode = parentLocCode.addLevel< NodeTileRes >( nodeOffset );
				//GvCore::GvLocalizationInfo::DepthType newLocDepth = parentLocDepth.addLevel();
			//	GvCore::GvLocalizationInfo::CodeType newLocCode = parentLocInfo.locCode;
			//	GvCore::GvLocalizationInfo::DepthType newLocDepth = parentLocInfo.locDepth.addLevel();

				pPageTable.locCodeArray.set( newNodeTileIndex, parentLocInfo.locCode );
				pPageTable.locDepthArray.set( newNodeTileIndex, parentLocInfo.locDepth.addLevel() );
		////	}
			//--------------------------------------------------------------------------
		}
	}
}


} // namespace GvCache

#endif
