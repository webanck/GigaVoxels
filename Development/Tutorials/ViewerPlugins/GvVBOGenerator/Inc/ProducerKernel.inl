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

// GigaVoxels
#include <GvStructure/GvVolumeTreeKernel.h>
#include <GvRendering/GvNodeVisitorKernel.h>
#include <GvCore/vector_types_ext.h>

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Initialize the producer
 * 
 * @param volumeTreeKernel Reference on a volume tree data structure
 ******************************************************************************/
template< typename TDataStructureType >
inline void ProducerKernel< TDataStructureType >
::initialize( DataStructureKernel& pDataStructure )
{
	_volumeTreeKernel = pDataStructure;
}

/******************************************************************************
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
 ******************************************************************************/
template< typename TDataStructureType >
template< typename GPUPoolKernelType >
__device__
inline uint ProducerKernel< TDataStructureType >
::produceData( GPUPoolKernelType& nodePool, uint pRequestID, uint pProcessID, uint3 pNewElemAddress,
			  const GvCore::GvLocalizationInfo& pParentLocInfo, Loki::Int2Type< 0 > )
{
	// NOTE :
	// In this method, you are inside a node tile.
	// The goal is to determine, for each node of the node tile, which type of data it holds.
	// Data type can be :
	// - a constant region,
	// - a region with data,
	// - a region where max resolution is reached.
	// So, one thread is responsible of the production of one node of a node tile.
	
	// Retrieve current node tile localization information code and depth
	const GvCore::GvLocalizationInfo::CodeType *parentLocCode = &pParentLocInfo.locCode;
	const GvCore::GvLocalizationInfo::DepthType *parentLocDepth = &pParentLocInfo.locDepth;

	// Process ID gives the 1D index of a node in the current node tile
	if ( pProcessID < NodeRes::getNumElements() )
	{
		// First, compute the 3D offset of the node in the node tile
		uint3 subOffset = NodeRes::toFloat3( pProcessID );

		// Node production corresponds to subdivide a node tile.
		// So, based on the index of the node, find the node child.
		// As we want to sudbivide the curent node, we retrieve localization information
		// at the next level
		uint3 regionCoords = parentLocCode->addLevel< NodeRes >( subOffset ).get();
		uint regionDepth = parentLocDepth->addLevel().get();

		// Create a new node for which you will have to fill its information.
		GvStructure::GvNode newnode;
		newnode.childAddress = 0;
		newnode.brickAddress = 0;

		// Call what we call an oracle that will determine the type of the region of the node accordingly
		GPUVoxelProducer::GPUVPRegionInfo nodeinfo = getRegionInfo( regionCoords, regionDepth );

		// Now that the type of the region is found, fill the new node information
		if ( nodeinfo == GPUVoxelProducer::GPUVP_CONSTANT )
		{
			newnode.setTerminal( true );
		}
		else if ( nodeinfo == GPUVoxelProducer::GPUVP_DATA )
		{
			newnode.setStoreBrick();
			newnode.setTerminal( false );
		}
		else if ( nodeinfo == GPUVoxelProducer::GPUVP_DATA_MAXRES )
		{
			newnode.setStoreBrick();
			newnode.setTerminal( true );
		}
	
		// Finally, write the new node information into the node pool by selecting channels :
		// - Loki::Int2Type< 0 >() points to node information
		// - Loki::Int2Type< 1 >() points to brick information
		//
		// pNewElemAddress.x + pProcessID : is the adress of the new node in the node pool
		nodePool.getChannel( Loki::Int2Type< 0 >() ).set( pNewElemAddress.x + pProcessID, newnode.childAddress );
		nodePool.getChannel( Loki::Int2Type< 1 >() ).set( pNewElemAddress.x + pProcessID, newnode.brickAddress );
	}

	return 0;
}

/******************************************************************************
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
 ******************************************************************************/
template< typename TDataStructureType >
template< typename GPUPoolKernelType >
__device__
inline uint ProducerKernel< TDataStructureType >
::produceData( GPUPoolKernelType& pDataPool, uint pRequestID, uint pProcessID, uint3 pNewElemAddress,
			  const GvCore::GvLocalizationInfo& pParentLocInfo, Loki::Int2Type< 1 > )
{
	// Pas bon comme ça, car le rendu ne fera pas le ray-tracing pour les sphères
	// Il ne faut pas faire le return 2 comme ça...
	// Il faut d'abord produire les sphères
	//
	//// Test geometric criteria
	////
	//// Node subdivision process is stopped if there is no more than a given number of spheres inside
	//if ( ! isGeometricCriteriaValid( cNbPoints ) )
	//{
	//	return 2;
	//}

	const GvCore::GvLocalizationInfo::CodeType parentLocCode = pParentLocInfo.locCode;
	const GvCore::GvLocalizationInfo::DepthType parentLocDepth = pParentLocInfo.locDepth;

	// For the first levels that we don't take into account
	if ( parentLocDepth.get() < cMinLevelOfResolutionToHandle )
	{
		// Write w component to 0, i.e. number of spheres is 0
		//
		// - brickInfo contient la dimension de la brique dans les 3 premiers parametres puis le nombre de spheres dans la brique en 4eme parametre
		pDataPool.template setValue< 0 >( pNewElemAddress + make_uint3( threadIdx.x, threadIdx.y, threadIdx.z ), make_float4( 0.0f, 0.0f, 0.0f, /*no point*/0.0f ) );
	}
	else
	{
		// Shared Memory declaration
		//
		// - number of points in the brick
		__shared__ uint sPointCounter;
		// - brick info
		__shared__ float sBrickWidth;
		__shared__ float3 sBrickPosition;
		__shared__ float3 sBrickCenter;
		
		// Done by only one thread of the kernel
		if ( pProcessID == 0 )
		{
			// Initialize brick info
			sPointCounter = 0;

			// Initialize brick info
			sBrickWidth = 1.f / static_cast< float >( 1 << parentLocDepth.get() );
			sBrickPosition = make_float3( parentLocCode.get() ) * sBrickWidth;
			sBrickCenter = sBrickPosition + make_float3( 0.5f * sBrickWidth );
		}

		// Thread Synchronization
		__syncthreads();

		if ( parentLocDepth.get() == cMinLevelOfResolutionToHandle )
		{
			/*
			// les thread (0,0,0) et (1,0,0) ne vont pas ecrire de position dans le cache
			// mais renseigner le nombre de spheres et la position de la brique.
			if(!(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) &&
			!(threadIdx.x == 1 && threadIdx.y == 0 && threadIdx.z == 0)
			){
			*/
			// recupere un indice 1 dimension pour recuperer une position dans le buffer de positions
			// on enleve 2 car les deux premieres thread ne vont pas lire dans le buffer de positions
			const uint indice = pProcessID;
			if ( indice < cNbPoints )
			{
				// Retrieve current point
				float4 sphere = _posBuf[ indice ];

				// Scale
				sphere.x *= sBrickWidth;
				sphere.y *= sBrickWidth;
				sphere.z *= sBrickWidth;

				// Bias
				sphere.x += sBrickPosition.x;
				sphere.y += sBrickPosition.y;
				sphere.z += sBrickPosition.z;

				// la sphere et la brique s'intersecent
				//if ( intersectBrick( sphere, sBrickCenter, sBrickWidth ) )
				//{
				// on ajoute 2 car les deux premiers elements sont le nombre de spheres et la position de la brique
				const unsigned int index = atomicAdd( &sPointCounter, 1 ) + 2;

				// on remet index sur 3 dimensions
				uint3 index3D = make_uint3( 0, 0, 0 );
				index3D.x = index % blockDim.x;
				index3D.y = ( index / blockDim.x ) % blockDim.y;
				index3D.z = index / ( blockDim.x * blockDim.y );

				const uint3 destinationAddress = pNewElemAddress + make_uint3( index3D.x, index3D.y, index3D.z );

				// calcul de la position de la sphere dans un repere local a la brique
				const float4 sphereData = make_float4( sphere.x, sphere.y, sphere.z, sphere.w );
				//const float4 sphereData = make_float4( sphere.x - sBrickPosition.x, sphere.y - sBrickPosition.y, sphere.z - sBrickPosition.z, sphere.w );

				// ecriture des donnees dans la cache
				pDataPool.template setValue< 0 >( destinationAddress, sphereData );
				//}
			}

			// Thread Synchronization
			__syncthreads();

			// Write data in the first two memory addresses
			//
			// Done by only one thread of the kernel
			if ( pProcessID == 0 )
			{
				const float4 brickDataInfo = make_float4( blockDim.x, blockDim.y, blockDim.z, sPointCounter );
				pDataPool.template setValue< 0 >( pNewElemAddress + make_uint3( 0, 0, 0 ), brickDataInfo );

				const float4 brickData = make_float4( sBrickPosition.x, sBrickPosition.y, sBrickPosition.z, sBrickWidth );
				pDataPool.template setValue< 0 >( pNewElemAddress + make_uint3( 1, 0, 0 ), brickData );
			}
		}
		else
		{
			// Shared Memory declaration
			__shared__ GvStructure::GvNode sParentNode;
			__shared__ uint3 sParentNodeBrickAdress;
			__shared__ float4 sBrickDataInfo;
			//__shared__ float4 sBrickInfo;

			// Retrieve common values from parent node
			//
			// Done by only one thread of the kernel
			if ( pProcessID == 0 )
			{
				sParentNode.childAddress = 0;
				sParentNode.brickAddress = 0;

				// Retrieve parent node
				GvRendering::GvNodeVisitorKernel::getNodeFather( _volumeTreeKernel, sParentNode, sBrickCenter, parentLocDepth.get() - 1 );

				sParentNodeBrickAdress = make_uint3( 0, 0, 0 );

				sBrickDataInfo = make_float4( 0.f, 0.f, 0.f, 0.f );
				//sBrickInfo = make_float4( 0.f, 0.f, 0.f, 0.f );

				sParentNodeBrickAdress = sParentNode.getBrickAddress();

				// Retrieve brick resolution (nb of elements in each dimension) and number of spheres
				sBrickDataInfo = _volumeTreeKernel.template getSampleValueTriLinear< 0 >( make_float3( sParentNodeBrickAdress.x, sParentNodeBrickAdress.y, sParentNodeBrickAdress.z ) * _volumeTreeKernel.brickCacheResINV - _volumeTreeKernel.brickCacheResINV, 0.5f * _volumeTreeKernel.brickCacheResINV );

				// Retrieve brick position and its size
				_volumeTreeKernel.template getSampleValueTriLinear< 0 >( make_float3( sParentNodeBrickAdress.x, sParentNodeBrickAdress.y, sParentNodeBrickAdress.z ) * _volumeTreeKernel.brickCacheResINV - _volumeTreeKernel.brickCacheResINV, 0.5f * _volumeTreeKernel.brickCacheResINV + make_float3( 1.f, 0.f, 0.f ) * _volumeTreeKernel.brickCacheResINV );
			}

			// Thread Synchronization
			__syncthreads();

			// recupere un indice 1 dimension pour recuperer une position dans le buffer de positions
			// on enleve 2 car les deux premieres thread ne vont pas lire dans le buffer de positions
			//uint parentNodeSphereIndex = threadIdx.x + ( threadIdx.y * blockDim.x ) + ( threadIdx.z * blockDim.x * blockDim.y );
			uint parentNodeSphereIndex = pProcessID;

			if ( parentNodeSphereIndex < sBrickDataInfo.w/*nb particles in parent node*/ )
			{
				// Add brick data offset
				parentNodeSphereIndex += 2;

				uint3 parentNodeSpherePosition = make_uint3( 0, 0, 0 );
				parentNodeSpherePosition.x = parentNodeSphereIndex % blockDim.x;
				parentNodeSpherePosition.y = ( parentNodeSphereIndex / blockDim.x ) % blockDim.y;
				parentNodeSpherePosition.z = parentNodeSphereIndex / ( blockDim.x * blockDim.y );

				// on recupere la position de la sphere dans le parent
				// Sample data structrure to retrieve sphere data (position and radius)
				float4 sphere = _volumeTreeKernel.template getSampleValueTriLinear< 0 >( make_float3( sParentNodeBrickAdress.x, sParentNodeBrickAdress.y, sParentNodeBrickAdress.z ) * _volumeTreeKernel.brickCacheResINV - _volumeTreeKernel.brickCacheResINV,
					0.5f * _volumeTreeKernel.brickCacheResINV +  make_float3( parentNodeSpherePosition.x * _volumeTreeKernel.brickCacheResINV.x,
					parentNodeSpherePosition.y * _volumeTreeKernel.brickCacheResINV.y,
					parentNodeSpherePosition.z * _volumeTreeKernel.brickCacheResINV.z ) );

				// la sphere et la brique s'intersecent
				if ( intersectBrick( sphere, sBrickCenter, sBrickWidth ) )
				{
					// on ajoute 2 car les deux premiers elements sont le nombre de spheres et la position de la brique
					const unsigned int sphereIndex = atomicAdd( &sPointCounter, 1 ) + 2;

					// on remet index sur 3 dimensions
					uint3 spherePosition = make_uint3( 0, 0, 0 );
					spherePosition.x = sphereIndex % blockDim.x;
					spherePosition.y = ( sphereIndex / blockDim.x ) % blockDim.y;
					spherePosition.z = sphereIndex / ( blockDim.x * blockDim.y );

					uint3 destinationAddress = pNewElemAddress + make_uint3( spherePosition.x, spherePosition.y, spherePosition.z );

					// calcul de la position de la sphere dans un repere local a la brique
					float4 sphereData = make_float4( sphere.x, sphere.y, sphere.z, sphere.w );

					// ecriture des donnees dans la cache
					pDataPool.template setValue< 0 >( destinationAddress, sphereData );
				}
			}

			// Thread Synchronization
			__syncthreads();

			// Write data in the first two memory addresses
			//
			// Done by only one thread of the kernel
			if ( pProcessID == 0 )
			{
				const float4 brickDataInfo = make_float4( blockDim.x, blockDim.y, blockDim.z, sPointCounter );
				pDataPool.template setValue< 0 >( pNewElemAddress + make_uint3( 0, 0, 0 ), brickDataInfo );

				const float4 brickData = make_float4( sBrickPosition.x, sBrickPosition.y, sBrickPosition.z, sBrickWidth );
				pDataPool.template setValue< 0 >( pNewElemAddress + make_uint3( 1, 0, 0 ), brickData );
			}
		}
	}

	// Return normal state (this can be customized if needed)
	return 0;
}

/******************************************************************************
 * Set the buffer of spheres
 *
 * @param pSpheresBuffer the buffer of spheres (position and radius)
 ******************************************************************************/
template< typename TDataStructureType >
__host__
inline void ProducerKernel< TDataStructureType >
::setPositionBuffer( float4* buf )
{
    _posBuf = buf;
}

///******************************************************************************
// * Set the data structure (to be able to sample data inside)
// *
// * @param pDataStructure the data structure
// ******************************************************************************/
//template< typename TDataStructureType >
//__host__
//inline void ProducerKernel< TDataStructureType >
//::setVolumeTree( DataStructureKernel pDataStructure )
//{
//    _volumeTreeKernel = pDataStructure;
//}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TDataStructureType >
__device__
inline GPUVoxelProducer::GPUVPRegionInfo ProducerKernel< TDataStructureType >
::getRegionInfo( uint3 regionCoords, uint regionDepth )
{
	// Limit the depth.
	// Currently, 32 is the max depth of the GigaVoxels engine.
	if ( regionDepth >= 32 )
	{
		return GPUVoxelProducer::GPUVP_DATA_MAXRES;
	}

	if ( regionDepth < cMinLevelOfResolutionToHandle )
	{
		return GPUVoxelProducer::GPUVP_DATA;
	}
	else
	{
		//// Test geometric criteria
		////
		//// Node subdivision process is stopped if there is no more than a given number of spheres inside
		//if ( ! isGeometricCriteriaValid( cNbPoints ) )
		//{
		//	return GPUVoxelProducer::GPUVP_DATA_MAXRES;
		//}

		// Brick info
		const float brickWidth = 1.f / static_cast< float >( 1 << regionDepth );
		const float3 brickPosition = make_float3( regionCoords ) * brickWidth;
		const float3 brickCenter = brickPosition + make_float3( 0.5f * brickWidth );
	
		// Au niveau 0, on parcours le tableau de positions	// normalement, ici, on ne rentre jamais pour le produceNode() !!!!!!!
		//if ( regionDepth == 0 )
		if ( regionDepth == cMinLevelOfResolutionToHandle )
		{
			for ( int i = 0 ; i < cNbPoints ; ++i )
			{
				float4 sphere = _posBuf[ i ];

				// Scale
				sphere.x *= brickWidth;
				sphere.y *= brickWidth;
				sphere.z *= brickWidth;

				// Bias
				sphere.x += brickPosition.x;
				sphere.y += brickPosition.y;
				sphere.z += brickPosition.z;

				if ( intersectBrick( sphere, brickCenter, brickWidth ) )
				{
					return GPUVoxelProducer::GPUVP_DATA;
				}
			}
		}
		else
		{ // sinon, on parcours dans le cache les spheres presentes dans la brique du noeud pere

			// TO DO
			//
			// - il n'y pas de parallélisme ici, on peut enlever la shared memory, on la mettre un niveau plus haut dans la methode produceNode()

			// Shared Memory declaration
			__shared__ GvStructure::GvNode sParentNode;

			// Done by only one thread of the kernel
			if ( threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 )
			{
				sParentNode.childAddress = 0;
				sParentNode.brickAddress = 0;

				GvRendering::GvNodeVisitorKernel::getNodeFather( _volumeTreeKernel, sParentNode, brickCenter, regionDepth - 1 );
			}

			// Thread Synchronization
			__syncthreads();

			if ( ! sParentNode.hasBrick() )
			{
				printf( "Pas de brick PARENT - REGION (%d %d %d) - parentNode %d %d\n", regionCoords.x, regionCoords.y, regionCoords.z, sParentNode.childAddress, sParentNode.brickAddress ); // ne devrait pas arriver !!!!!!!!!!!!!!!!

				return GPUVoxelProducer::GPUVP_CONSTANT;
			}
		
			const uint3 parentNodeBrickAdress = sParentNode.getBrickAddress();
			const uint parentNodeBrickAdressEncoded = sParentNode.getBrickAddressEncoded();

			// brickData contient les dimension de la brique puis le nombre de spheres dans la brique
			float4 brickDataInfo = _volumeTreeKernel.template getSampleValueTriLinear< 0 >( make_float3( parentNodeBrickAdress.x, parentNodeBrickAdress.y, parentNodeBrickAdress.z ) * _volumeTreeKernel.brickCacheResINV - _volumeTreeKernel.brickCacheResINV, 0.5f * _volumeTreeKernel.brickCacheResINV );

			// recupere la position de la brick dans la brick GigaVoxel ainsi que la taille de la brick
			float4 brickInfo = _volumeTreeKernel.template getSampleValueTriLinear< 0 >( make_float3( parentNodeBrickAdress.x, parentNodeBrickAdress.y, parentNodeBrickAdress.z ) * _volumeTreeKernel.brickCacheResINV - _volumeTreeKernel.brickCacheResINV, 0.5f * _volumeTreeKernel.brickCacheResINV + make_float3( 1.f, 0.f, 0.f ) * _volumeTreeKernel.brickCacheResINV );
		
			const uint nbSphere = static_cast< uint >( brickDataInfo.w );      // nombre de sheres dans la brick
			const uint dimBrickX = static_cast< uint >( brickDataInfo.x );     // dimension sur X de la brick
			const uint dimBrickY = static_cast< uint >( brickDataInfo.y );     // dimension sur Y de la brick
			//const uint dimBrickZ = static_cast< uint >( brickDataInfo.z );     // dimension sur Z de la brick

			//// Test geometric criteria
			////
			//// Node subdivision process is stopped if there is no more than a given number of spheres inside
			//if ( ! isGeometricCriteriaValid( nbSphere ) )
			//{
			//	return GPUVoxelProducer::GPUVP_DATA_MAXRES;
			//}

			// Check criteria only if enabled
			if ( ! cGeometricCriteria )
			{
				// on parcours les positions dans le cache
				for ( unsigned int i = 2; i < nbSphere + 2; ++i )
				{
					// Retrieve sphere index
					uint3 index3D = make_uint3( 0, 0, 0 );
					index3D.x = i % dimBrickX;
					index3D.y = ( i / dimBrickX ) % dimBrickY;
					index3D.z = i / ( dimBrickX * dimBrickY );

					// Sample data structrure to retrieve sphere data (position and radius)
					float4 sphere = _volumeTreeKernel.template getSampleValueTriLinear< 0 >( make_float3( parentNodeBrickAdress.x, parentNodeBrickAdress.y, parentNodeBrickAdress.z ) * _volumeTreeKernel.brickCacheResINV - _volumeTreeKernel.brickCacheResINV,
																							0.5f * _volumeTreeKernel.brickCacheResINV +  make_float3( index3D.x * _volumeTreeKernel.brickCacheResINV.x,
																							index3D.y * _volumeTreeKernel.brickCacheResINV.y,
																							index3D.z * _volumeTreeKernel.brickCacheResINV.z ) );

					// Need to add parent "brick position"
					//sphere += make_float4( brickInfo.x, brickInfo.y, brickInfo.z, 0.0f /*radius is not mofified !??*/ );

					if ( intersectBrick( sphere, brickCenter, brickWidth ) )
					{
					//	printf( "- REGION (%d %d %d) - sphere %d/%d : (%f %f %f %f) - brick : (%f %f %f %f)\n", regionCoords.x, regionCoords.y, regionCoords.z, i - 1, nbSphere, sphere.x, sphere.y, sphere.z, sphere.w, brickCenter.x, brickCenter.y, brickCenter.z, brickWidth );

						return GPUVoxelProducer::GPUVP_DATA;
					}
				}
			}
			else
			{
				// Used to count how many spheres are in this brick
				unsigned int nbPointsInBrick = 0;

				// on parcours les positions dans le cache
				for ( unsigned int i = 2; i < nbSphere + 2; ++i )
				{
					// Retrieve sphere index
					uint3 index3D = make_uint3( 0, 0, 0 );
					index3D.x = i % dimBrickX;
					index3D.y = ( i / dimBrickX ) % dimBrickY;
					index3D.z = i / ( dimBrickX * dimBrickY );

					// Sample data structrure to retrieve sphere data (position and radius)
					float4 sphere = _volumeTreeKernel.template getSampleValueTriLinear< 0 >( make_float3( parentNodeBrickAdress.x, parentNodeBrickAdress.y, parentNodeBrickAdress.z ) * _volumeTreeKernel.brickCacheResINV - _volumeTreeKernel.brickCacheResINV,
																							0.5f * _volumeTreeKernel.brickCacheResINV +  make_float3( index3D.x * _volumeTreeKernel.brickCacheResINV.x,
																							index3D.y * _volumeTreeKernel.brickCacheResINV.y,
																							index3D.z * _volumeTreeKernel.brickCacheResINV.z ) );

					// Need to add parent "brick position"
					//sphere += make_float4( brickInfo.x, brickInfo.y, brickInfo.z, 0.0f /*radius is not mofified !??*/ );

					if ( intersectBrick( sphere, brickCenter, brickWidth ) )
					{
					//	printf( "- REGION (%d %d %d) - sphere %d/%d : (%f %f %f %f) - brick : (%f %f %f %f)\n", regionCoords.x, regionCoords.y, regionCoords.z, i - 1, nbSphere, sphere.x, sphere.y, sphere.z, sphere.w, brickCenter.x, brickCenter.y, brickCenter.z, brickWidth );

						// Increment sphere counter in current brick
						nbPointsInBrick++;
					}
				}

				if ( nbPointsInBrick == 0 )
				{
					return GPUVoxelProducer::GPUVP_CONSTANT;
				}
				else
				{
					// Test geometric criteria
					//
					// Node subdivision process is stopped if there is no more than a given number of spheres inside
					if ( ! isGeometricCriteriaValid( nbPointsInBrick ) )
					{
						return GPUVoxelProducer::GPUVP_DATA_MAXRES;
					}
					else
					{
						return GPUVoxelProducer::GPUVP_DATA;
					}
				}
			}
		}
	}

    return GPUVoxelProducer::GPUVP_CONSTANT;
}

/******************************************************************************
 * Test the intersection between a sphere and a brick
 *
 * @param pSphere sphere (position and and radius)
 * @param pBrickCenter brick center
 * @param pBoxExtent pBrickWidth brick width
 *
 * @return a flag to tell wheter or not intersection occurs
 ******************************************************************************/
template< typename TDataStructureType >
__device__
inline bool ProducerKernel< TDataStructureType >
::intersectBrick( const float4 pSphere, const float3 pBrickCenter, const float pBrickWidth )
{
	bool result = false;

	switch ( cSphereBrickIntersectionType )
	{
		case 0:
			// Bricks are approximated by spheres (faster)
			result = intersectSphereSphere( pBrickCenter, pBrickWidth * cBrickWidth2PointSize, pSphere );
			break;

		case 1:
			// Bricks are not approximated (use real box-sphere intersection test)
			result = intersectSphereBox( pBrickCenter, pBrickWidth * 0.5f, pSphere );
			break;

		default:
			assert( false );
			break;
	}

	return result;
}

/******************************************************************************
 * Sphere-Sphere intersection test
 *
 * @param pSphereCenter 1st sphere center
 * @param pPointSize 1stsphere radius
 * @param pSphere 2nd sphere (position and and radius)
 *
 * @return a flag to tell wheter or not intersection occurs
 ******************************************************************************/
template< typename TDataStructureType >
__device__
inline bool ProducerKernel< TDataStructureType >
::intersectSphereSphere( const float3 pSphereCenter, const float pPointSize, const float4 pSphere )
{
	// Code is based on the Wild Magic library
	// http://www.geometrictools.com/LibMathematics/Intersection/Wm5IntrSphere3Sphere3.cpp

	const float3 C1mC0 = pSphereCenter - make_float3( pSphere.x, pSphere.y, pSphere.z );
	const float sqrLen = squaredLength( C1mC0 );
	const float r0 = pSphere.w;
	const float r1 = pPointSize;

	const float rSum = r0 + r1;
	const float rSumSqr = rSum * rSum;
	if ( sqrLen > rSumSqr )
	{
		return false;
	}
	
	return true;
}

/******************************************************************************
 * Sphere-Box intersection test
 *
 * @param pBoxCenter box center
 * @param pBoxExtent box extent (distance from center to one side)
 * @param pSphere sphere (position and and radius)
 *
 * @return a flag to tell wheter or not intersection occurs
 ******************************************************************************/
template< typename TDataStructureType >
__device__
inline bool ProducerKernel< TDataStructureType >
::intersectSphereBox( const float3 pBoxCenter, const float pBoxExtent, const float4 pSphere )
{
	// Code is based on the Wild Magic library
	// http://www.geometrictools.com/LibMathematics/Intersection/Wm5IntrBox3Sphere3.cpp

	// Test for intersection in the coordinate system of the box by
    // transforming the sphere into that coordinate system.
    float3 cdiff = make_float3( pSphere.x, pSphere.y, pSphere.z ) - pBoxCenter;

	float ax = fabsf( dot( cdiff, make_float3( 1.f, 0.f, 0.f ) ) );
    float ay = fabsf( dot( cdiff, make_float3( 0.f, 1.f, 0.f ) ) );
    float az = fabsf( dot( cdiff, make_float3( 0.f, 0.f, 1.f ) ) );
    float dx = ax - pBoxExtent;
    float dy = ay - pBoxExtent;
    float dz = az - pBoxExtent;

    if ( ax <= pBoxExtent )
    {
        if ( ay <= pBoxExtent )
        {
            if ( az <= pBoxExtent )
            {
                // Sphere center inside box.
                return true;
            }
            else
            {
                // Potential sphere-face intersection with face z.
                return dz <= pSphere.w;
            }
        }
        else
        {
            if ( az <= pBoxExtent )
            {
                // Potential sphere-face intersection with face y.
                return dy <= pSphere.w;
            }
            else
            {
                // Potential sphere-edge intersection with edge formed
                // by faces y and z.
                float rsqr = pSphere.w * pSphere.w;

                return dy*dy + dz*dz <= rsqr;
            }
        }
    }
    else
    {
        if ( ay <= pBoxExtent )
        {
            if ( az <= pBoxExtent )
            {
                // Potential sphere-face intersection with face x.
                return dx <= pSphere.w;
            }
            else
            {
                // Potential sphere-edge intersection with edge formed
                // by faces x and z.
                float rsqr = pSphere.w * pSphere.w;

                return dx*dx + dz*dz <= rsqr;
            }
        }
        else
        {
            if ( az <= pBoxExtent )
            {
                // Potential sphere-edge intersection with edge formed
                // by faces x and y.
                float rsqr = pSphere.w * pSphere.w;

                return dx*dx + dy*dy <= rsqr;
            }
            else
            {
                // Potential sphere-vertex intersection at corner formed
                // by faces x,y,z.
                float rsqr = pSphere.w * pSphere.w;

                return dx*dx + dy*dy + dz*dz <= rsqr;
            }
        }
    }
}

/******************************************************************************
 * Test wheter or not geometric criteria passes
 *
 * Note : the node subdivision process is stopped if there is no more than a given number of spheres inside
 *
 * @param pNbPointsInBrick number of spheres in a given brick
 *
 * @return a flag to tell wheter or not the criteria passes
 ******************************************************************************/
template< typename TDataStructureType >
__device__
inline bool ProducerKernel< TDataStructureType >
::isGeometricCriteriaValid( const unsigned int pNbPointsInBrick )
{
	return ( pNbPointsInBrick > cMinNbPointsPerBrick );
}

/******************************************************************************
 * Test wheter or not screen based criteria passes
 *
 * Note : the node subdivision process is stopped if ...
 *
 * @return a flag to tell wheter or not the criteria passes
 ******************************************************************************/
template< typename TDataStructureType >
__device__
inline bool ProducerKernel< TDataStructureType >
::isScreenSpaceCriteriaValid()
{
	// TO DO
	// ...
	return true;
}

/******************************************************************************
 * Test wheter or not absolute size criteria passes
 *
 * Note : the node subdivision process is stopped if ...
 *
 * @return a flag to tell wheter or not the criteria passes
 ******************************************************************************/
template< typename TDataStructureType >
__device__
inline bool ProducerKernel< TDataStructureType >
::isApparentMaxSizeCriteriaValid()
{
	// TO DO
	// ...
	return true;
}
