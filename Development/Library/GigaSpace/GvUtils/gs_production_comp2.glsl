////////////////////////////////////////////////////////////////////////////////
//
// COMPUTE SHADER
//
// Production Management
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// VERSION
////////////////////////////////////////////////////////////////////////////////

#version 430

// TO DO
#define BorderSize 1

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

// Number of shader invocations per work group
// - for nodes
//layout ( local_size_x = 32 ) in;
// - for bricks
//layout ( local_size_x = 128 ) in;
layout ( local_size_x = 16, local_size_y = 8 ) in;

////////////////////////////////////////////////////////////////////////////////
// UNIFORM
////////////////////////////////////////////////////////////////////////////////

// ...
uniform uint uNbElements;
uniform uint u_gs_nodeTileNbElements;

uniform uint u_gs_nodeTileResolution_xLog2;
uniform uint u_gs_nodeTileResolution_yLog2;
uniform uint u_gs_nodeTileResolution_zLog2;

// Node Pool parameters
//
// - Child Array
// ---- hierarchical data structure (generalized N-Tree, for instance octree)
// ---- organized in node tiles (each node contains the address of its child)
uniform layout(size1x32) restrict uimageBuffer gs_u_nodePoolChildArray;	// 1 component of 32 bits: unsigned int
// - Data Array
// ---- address in Data Pool where brick (bottom left corner)
uniform layout(size1x32) restrict uimageBuffer gs_u_nodePoolDataArray;	// 1 component of 32 bits: unsigned int

// Node Page Table parameters
//
// - Localization code array
// ---- hierarchical data structure (generalized N-Tree, for instance octree)
// ---- organized in node tiles (each node contains the address of its child)
//uniform layout(size1x32) uimageBuffer u_gs_nodePageTable_localizationCodes;	// 1 component of 32 bits: unsigned int
uniform layout(size1x32) restrict uimageBuffer u_gs_nodePageTable_localizationCodes;	// 1 component of 32 bits: unsigned int
// - Localization depth array
// ---- address in Data Pool where brick (bottom left corner)
uniform layout(size1x32) restrict uimageBuffer uNodePageTableLocalizationDepths;	// 1 component of 32 bits: unsigned int

// Node Pool parameters
//
// - Child Array
// ---- hierarchical data structure (generalized N-Tree, for instance octree)
// ---- organized in node tiles (each node contains the address of its child)
uniform layout(size1x32) restrict uimageBuffer u_gs_nodeAddresses;	// 1 component of 32 bits: unsigned int
// - Data Array
// ---- address in Data Pool where brick (bottom left corner)
uniform layout(size1x32) restrict uimageBuffer u_gs_elementAddresses;	// 1 component of 32 bits: unsigned int

// Data Pool
uniform layout(rgba8) restrict image3D gs_u_dataPoolChannel0;	// 1 component of 32 bits: unsigned int

////////////////////////////////////////////////////////////////////////////////
// SHARED MEMORY
////////////////////////////////////////////////////////////////////////////////

// ...
shared uint smNodeAddress;
shared uvec3 smElemAddress;
shared uvec3 smParentLocInfoCode;
shared uint smParentLocInfoDepth;

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

layout ( binding  = 0, rgba8 ) uniform image2D uColorImage;
//layout ( binding = 1, rgba8 ) uniform image2D uColorImage;

////////////////////////////////////////////////////////////////////////////////
// Functions Declaration
////////////////////////////////////////////////////////////////////////////////

// Data Structure Node
struct gs_Node
{
	// Child address
	uint childAddress;

	// Brick address
	uint brickAddress;
};

// Node region info
#define gs_nodeRegionInfo_CONSTANT 0
#define gs_nodeRegionInfo_DATA 1
#define gs_nodeRegionInfo_MAXRESOLUTION 2
		
// Data Structure Management
uint gs_unpackNodeAddress( uint pAddress );
uint gs_packNodeAddress( uint pAddress );
uvec3 gs_unpackBrickAddress( uint pAddress );
uint gs_packBrickAddress( uvec3 pAddress );

void gs_nodePageTable_getLocalizationInfo( uint pNodeAddress, out uvec3 pLocalizationCode, out uint pLocalizationDepth );
void gs_produceData( uint pRequestID, uint pProcessID, uvec3 pNewElemAddress, uvec3 pLocalizationCode, uint pLocalizationDepth );

void gs_nodeSetPointer( uint pElemAddress, uint pElemPointer );
void gs_nodeSetPointer2( uint pElemAddress, uvec3 pElemPointer );

uvec3 gs_localizationCode_addLevel( uvec3 pLocalizationCode, uvec3 pOffset );
//uint gs_localizationDepth_addLevel( uint pLocalizationDepth );

// Data Structure Management
//void gs_fetchNode( out Node pNode, uint pNodeTileAddress, uint pNodeTileOffset );
//bool gs_nodeIsInitialized( Node pNode );
bool gs_nodeHasSubNodes( gs_Node pNode );
bool gs_nodeIsBrick( gs_Node pNode );
bool gs_nodeHasBrick( gs_Node pNode );
void gs_nodeSetStoreBrick( out gs_Node pNode );
bool gs_nodeIsTerminal( gs_Node pNode );
void gs_nodeSetTerminal( out gs_Node pNode, bool pFlag );
//uint gs_nodeGetChildAddress( Node pNode );
//uvec3 gs_nodeGetBrickAddress( Node pNode );

uint getRegionInfo( uvec3 pRegionCoords, uint pRegionDepth );

uvec3 gs_vector_toVec3( uint pValue );

// Oracle
bool isInSphere( vec3 pPoint );

////////////////////////////////////////////////////////////////////////////////
// Unpack a node address
//
// @param pAddress node address
//
// @return the packed node address
////////////////////////////////////////////////////////////////////////////////
uint gs_unpackNodeAddress( uint pAddress )
{
	return ( pAddress & 0x3FFFFFFFU );
}

////////////////////////////////////////////////////////////////////////////////
// Pack a node address
//
// @param pAddress ...
//
// @return ...
////////////////////////////////////////////////////////////////////////////////
uint gs_packNodeAddress( uint pAddress )
{
	return pAddress;
}

////////////////////////////////////////////////////////////////////////////////
// Unpack a brick address
//
// @param pAddress brick address
//
// @return the packed brick address
////////////////////////////////////////////////////////////////////////////////
uvec3 gs_unpackBrickAddress( uint pAddress )
{
	uvec3 res;
	res.x = ( pAddress & 0x3FF00000U ) >> 20U;
	res.y = ( pAddress & 0x000FFC00U ) >> 10U;
	res.z = pAddress & 0x000003FFU;

	return res;
}

////////////////////////////////////////////////////////////////////////////////
// Pack a brick address
//
// @param pAddress ...
//
// @return ...
////////////////////////////////////////////////////////////////////////////////
uint gs_packBrickAddress( uvec3 pAddress )
{
	return ( pAddress.x << 20 | pAddress.y << 10 | pAddress.z );
}

////////////////////////////////////////////////////////////////////////////////
// Convert a linear value to a three-dimensionnal value.
//
// @param pValue The 1D value to convert
//
// @return the 3D converted value
////////////////////////////////////////////////////////////////////////////////
uvec3 gs_vector_toVec3( uint pValue )
{
	// TO DO
	// - check if data is a "power of 2"
	// ...

	//if ( xIsPOT && yIsPOT && zIsPOT )
	//{
	//	/*r.x = n & xLog2;
	//	r.y = (n >> xLog2) & yLog2;
	//	r.z = (n >> (xLog2 + yLog2)) & zLog2;*/
	//	return uvec3( pValue & xLog2, ( pValue >> xLog2 ) & yLog2, ( pValue >> ( xLog2 + yLog2 ) ) & zLog2 );
	return uvec3( pValue & 1, ( pValue >> 1 ) & 1, ( pValue >> ( 1 + 1 ) ) & 1 );
//	}
	//else
	//{
	//	/*r.x = n % x;
	//	r.y = (n / x) % y;
	//	r.z = (n / (x * y)) % z;*/
	//	return uvec3( pValue % x, ( pValue / x ) % y, ( pValue / ( x * y ) ) % z );
	//}
}

////////////////////////////////////////////////////////////////////////////////
// Return the localization info of a node in the node pool
//
// @param nodeAddress Address of the node in the node pool
//
// @return The localization info of the node
////////////////////////////////////////////////////////////////////////////////
void gs_nodePageTable_getLocalizationInfo( uint pNodeAddress, out uvec3 pLocalizationCode, out uint pLocalizationDepth )
{
	// Compute node tile index
	uint nodeTileIndex = pNodeAddress / u_gs_nodeTileNbElements;

	// Retrieve associated localization depth
	pLocalizationDepth = imageLoad( uNodePageTableLocalizationDepths, int( nodeTileIndex ) ).x;

	// Retrieve associated localization code
	// - Writing to 3-components buffers using the image API in OpenGL
	// - http://rauwendaal.net/category/glsl/
	int texelCoordinate = 3 * int( nodeTileIndex );
	uvec3 parentLocCode;
	parentLocCode.x = imageLoad( u_gs_nodePageTable_localizationCodes, texelCoordinate ).x;
	parentLocCode.y = imageLoad( u_gs_nodePageTable_localizationCodes, texelCoordinate + 1 ).x;
	parentLocCode.z = imageLoad( u_gs_nodePageTable_localizationCodes, texelCoordinate + 2 ).x;

	// Modify localization code
	//
	// Compute the address of the current node tile (and its offset in the node tile)
	uint nodeTileAddress = nodeTileIndex * u_gs_nodeTileNbElements;
	uint nodeTileOffset = pNodeAddress - nodeTileAddress;
	// Compute the node offset (in 3D, in the node tile)
	uvec3 nodeOffset = gs_vector_toVec3( nodeTileOffset );
	// Localization info initialization
	pLocalizationCode = gs_localizationCode_addLevel( parentLocCode, nodeOffset );
}

////////////////////////////////////////////////////////////////////////////////
// Given the current localization code and an offset in a node tile
//
// @param pOffset The offset in a node tile
//
// @return ...
////////////////////////////////////////////////////////////////////////////////
uvec3 gs_localizationCode_addLevel( uvec3 pLocalizationCode, uvec3 pOffset )
{
	uvec3 localizationCode;
	localizationCode.x = pLocalizationCode.x << u_gs_nodeTileResolution_xLog2 | pOffset.x;
	localizationCode.y = pLocalizationCode.y << u_gs_nodeTileResolution_yLog2 | pOffset.y;
	localizationCode.z = pLocalizationCode.z << u_gs_nodeTileResolution_zLog2 | pOffset.z;

	return localizationCode;
}

////////////////////////////////////////////////////////////////////////////////
// Produce data
//
// Optimization
// - remove this synchonization for brick production
// - let user-defined synchronization barriers in the producer directly
////////////////////////////////////////////////////////////////////////////////
void gs_produceData( uint pRequestID, uint pProcessID, uvec3 pNewElemAddress, uvec3 pLocalizationCode, uint pLocalizationDepth )
{
	// NOTE :
	// In this method, you are inside a brick of voxels.
	// The goal is to determine, for each voxel of the brick, the value of each of its channels.
	//
	// Data type has previously been defined by the user, as a
	// typedef Loki::TL::MakeTypelist< uchar4, half4 >::Result DataType;
	//
	// In this tutorial, we have choosen two channels containing color at channel 0 and normal at channel 1.
	
	// Compute useful variables used for retrieving positions in 3D space
	//uvec3 brickRes = BrickRes::get();
	uvec3 brickRes = uvec3( 8, 8, 8 ); // TODO
	uvec3 levelRes = uvec3( 1 << pLocalizationDepth ) * brickRes;	// number of voxels (in each dimension)
	vec3 levelResInv = vec3( 1.0f ) / vec3( levelRes );	// size of a voxel (in each dimension)

	ivec3 brickPos = ivec3( pLocalizationCode * brickRes ) - BorderSize; // TODO BorderSize
	vec3 brickPosF = vec3( brickPos ) * levelResInv;

	// Real brick size (with borders)
	//uint3 elemSize = BrickRes::get() + make_uint3( 2 * BorderSize );
	uvec3 elemSize = uvec3( 8, 8, 8 ) + uvec3( 2 * BorderSize ); // TODO BorderSize
	
	// The original KERNEL execution configuration on the HOST has a 2D block size :
	// dim3 blockSize( 16, 8, 1 );
	//
	// Each block process one brick of voxels.
	//
	// One thread iterate in 3D space given a pattern defined by the 2D block size
	// and the following "for" loops. Loops take into account borders.
	// In fact, each thread of the current 2D block compute elements layer by layer
	// on the z axis.
	//
	// One thread process only a subset of the voxels of the brick.
	//
	// Iterate through z axis step by step as gl_WorkGroupSize.z is equal to 1
	uvec3 elemOffset;
	for ( elemOffset.z = 0; elemOffset.z < elemSize.z; elemOffset.z += gl_WorkGroupSize.z )
	{
		for ( elemOffset.y = 0; elemOffset.y < elemSize.y; elemOffset.y += gl_WorkGroupSize.y )
		{
			for ( elemOffset.x = 0; elemOffset.x < elemSize.x; elemOffset.x += gl_WorkGroupSize.x )
			{
				// Compute position index
				uvec3 locOffset = elemOffset + uvec3( gl_LocalInvocationID.x, gl_LocalInvocationID.y, gl_LocalInvocationID.z );

				// Test if the computed position index is inside the brick (with borders)
				if ( locOffset.x < elemSize.x && locOffset.y < elemSize.y && locOffset.z < elemSize.z )
				{
					// Position of the current voxel's center (relative to the brick)
					//
					// In order to make the mip-mapping mecanism OK,
					// data values must be set at the center of voxels.
					vec3 voxelPosInBrickF = ( vec3( locOffset ) + 0.5f ) * levelResInv;
					// Position of the current voxel's center (absolute, in [0.0;1.0] range)
					vec3 voxelPosF = brickPosF + voxelPosInBrickF;
					// Position of the current voxel's center (scaled to the range [-1.0;1.0])
					vec3 posF = voxelPosF * 2.0f - 1.0f;

					vec4 voxelColor = vec4( 1.0f, 0.0f, 0.0f, 0.0f );
				//	vec4 voxelNormal = vec4( normalize( posF ), 1.0f );

					// Test if the voxel is located inside the unit sphere
					if ( isInSphere( posF ) )
					{
						voxelColor.a = 1.0f;
					}

					// Alpha pre-multiplication used to avoid the "color bleeding" effect
					voxelColor.r *= voxelColor.a;
					voxelColor.g *= voxelColor.a;
					voxelColor.b *= voxelColor.a;

					// Compute the new element's address
					uvec3 destAddress = pNewElemAddress + locOffset;
					// Write the voxel's color in the first field
					imageStore( gs_u_dataPoolChannel0, ivec3( destAddress ), voxelColor );
					// Write the voxel's normal in the second field
				//	dataPool.template setValue< 1 >( destAddress, voxelNormal );
				}
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// Helper function used to determine the type of zones in the data structure.
//
// The data structure is made of regions containing data, empty or constant regions.
// Besides, this function can tell if the maximum resolution is reached in a region.
//
// @param pRegionCoords region coordinates
// @param pRegionDepth region depth
//
// @return the type of the region
////////////////////////////////////////////////////////////////////////////////
uint getRegionInfo( uvec3 pRegionCoords, uint pRegionDepth )
{
	// Limit the depth.
	// Currently, 32 is the max depth of the GigaSpace engine.
	if ( pRegionDepth >= 32 )
	{
		return gs_nodeRegionInfo_MAXRESOLUTION;
	}

	//uvec3 brickRes = BrickRes::get();
	uvec3 brickRes = uvec3( 8, 8, 8 );

	uvec3 levelRes = uvec3( 1 << pRegionDepth ) * brickRes;
	vec3 levelResInv = vec3( 1.f ) / vec3( levelRes );

	// Since we work in the range [-1;1] below, the brick size is two time bigger
	vec3 brickSize = vec3( 1.f ) / vec3( 1 << pRegionDepth ) * 2.f;

	// Build the eight brick corners of a sphere centered in [0;0;0]
	vec3 q000 = vec3( pRegionCoords * brickRes ) * levelResInv * 2.f - 1.f;
	vec3 q001 = vec3( q000.x + brickSize.x, q000.y,			   q000.z);
	vec3 q010 = vec3( q000.x,				 q000.y + brickSize.y, q000.z);
	vec3 q011 = vec3( q000.x + brickSize.x, q000.y + brickSize.y, q000.z);
	vec3 q100 = vec3( q000.x,				 q000.y,			   q000.z + brickSize.z);
	vec3 q101 = vec3( q000.x + brickSize.x, q000.y,			   q000.z + brickSize.z);
	vec3 q110 = vec3( q000.x,				 q000.y + brickSize.y, q000.z + brickSize.z);
	vec3 q111 = vec3( q000.x + brickSize.x, q000.y + brickSize.y, q000.z + brickSize.z);

	// Test if any of the eight brick corner lies in the sphere
	if ( isInSphere( q000 ) || isInSphere( q001 ) || isInSphere( q010 ) || isInSphere( q011 ) ||
		isInSphere( q100 ) || isInSphere( q101 ) || isInSphere( q110 ) || isInSphere( q111 ) )
	{
		return gs_nodeRegionInfo_DATA;
	}

	return gs_nodeRegionInfo_CONSTANT;
}

////////////////////////////////////////////////////////////////////////////////
// ...
////////////////////////////////////////////////////////////////////////////////
bool isInSphere( vec3 pPoint )
{
	return ( length( pPoint ) < 1.0 );
}

////////////////////////////////////////////////////////////////////////////////
// Tell wheter or not a node is initialized
//
// @param pNode the node to test
//
// @return a flag telling wheter or not a node is initialized
////////////////////////////////////////////////////////////////////////////////
bool gs_nodeIsInitialized( gs_Node pNode )
{
	return ( pNode.childAddress != 0 ) || ( pNode.brickAddress != 0 );
}

////////////////////////////////////////////////////////////////////////////////
// Tell wheter or not a node has sub nodes
//
// @param pNode the node to test
//
// @return a flag telling wheter or not a node has sub nodes
////////////////////////////////////////////////////////////////////////////////
bool gs_nodeHasSubNodes( gs_Node pNode )
{
	return ( pNode.childAddress & 0x3FFFFFFFU ) != 0;
}

////////////////////////////////////////////////////////////////////////////////
// Tell wheter or not a node is a brick
//
// @param pNode the node to test
//
// @return a flag telling wheter or not a node is a brick
////////////////////////////////////////////////////////////////////////////////
bool gs_nodeIsBrick( gs_Node pNode )
{
	return ( pNode.childAddress & 0x40000000U ) != 0;
}

////////////////////////////////////////////////////////////////////////////////
// Tell wheter or not a node has a brick
//
// @param pNode the node to test
//
// @return a flag telling wheter or not a node has a brick
////////////////////////////////////////////////////////////////////////////////
bool gs_nodeHasBrick( gs_Node pNode )
{
	return gs_nodeIsBrick( pNode ) && pNode.brickAddress != 0;
}

////////////////////////////////////////////////////////////////////////////////
// Flag the node as containg data or not
//
// @param pFlag a flag telling wheter or not the node contains data
////////////////////////////////////////////////////////////////////////////////
void gs_nodeSetStoreBrick( out gs_Node pNode )
{
	pNode.childAddress = pNode.childAddress | 0x40000000;
}

////////////////////////////////////////////////////////////////////////////////
// Tell wheter or not a node is terminal
//
// @param pNode the node to test
//
// @return a flag telling wheter or not a node is terminal
////////////////////////////////////////////////////////////////////////////////
bool gs_nodeIsTerminal( gs_Node pNode )
{
	return ( pNode.childAddress & 0x80000000U ) != 0;
}

////////////////////////////////////////////////////////////////////////////////
// Flag the node as beeing terminal or not
//
// @param pNode the node to test
// @param pFlag a flag telling wheter or not the node is terminal
////////////////////////////////////////////////////////////////////////////////
void gs_nodeSetTerminal( out gs_Node pNode, bool pFlag )
{
	if ( pFlag )
	{
		pNode.childAddress = ( pNode.childAddress | 0x80000000 );
	}
	else
	{
		pNode.childAddress = ( pNode.childAddress & 0x7FFFFFFF );
	}
}

////////////////////////////////////////////////////////////////////////////////
// @param pElemAddress ...
// @param pElemPointer ...
// @param pFlags ...
////////////////////////////////////////////////////////////////////////////////
void gs_nodeSetPointer( uint pElemAddress, uint pElemPointer )
{
	uint packedChildAddress = imageLoad( gs_u_nodePoolChildArray, int( pElemAddress ) ).x;
	uint packedAddress = /*gs_packNodeAddress(*/ pElemPointer /*)*/;

	// Update node tile's pointer
	imageStore( gs_u_nodePoolChildArray, int( pElemAddress ), uvec4( ( packedChildAddress & 0x40000000 ) | ( packedAddress & 0x3FFFFFFF ) ) );

	// Compute the address of the current node tile
	uint nodeTileIndex = pElemAddress / u_gs_nodeTileNbElements;
	uint nodeTileAddress = nodeTileIndex * u_gs_nodeTileNbElements;
	uint nodeTileOffset = pElemAddress - nodeTileAddress;

	// Compute the node offset
	uvec3 nodeOffset = gs_vector_toVec3( nodeTileOffset );

	// Fetch associated localization infos
	//
	// Retrieve associated localization depth
	uint parentLocDepth = imageLoad( uNodePageTableLocalizationDepths, int( nodeTileIndex ) ).x;
	//
	// Retrieve associated localization code
	// - Writing to 3-components buffers using the image API in OpenGL
	// - http://rauwendaal.net/category/glsl/
	int texelCoordinate = 3 * int( nodeTileIndex );
	uvec3 parentLocCode;
	parentLocCode.x = imageLoad( u_gs_nodePageTable_localizationCodes, texelCoordinate ).x;
	parentLocCode.y = imageLoad( u_gs_nodePageTable_localizationCodes, texelCoordinate + 1 ).x;
	parentLocCode.z = imageLoad( u_gs_nodePageTable_localizationCodes, texelCoordinate + 2 ).x;
	
	// Compute the address of the new node tile
	uint newNodeTileIndex = pElemPointer / u_gs_nodeTileNbElements;
	
	// Update associated localization infos
	uvec3 newLocCode = gs_localizationCode_addLevel( parentLocCode, nodeOffset );
	uint newLocDepth = parentLocDepth + 1;//parentLocDepth.addLevel();

	int newTexelCoordinate = 3 * int( newNodeTileIndex );
	imageStore( u_gs_nodePageTable_localizationCodes, int( newTexelCoordinate ), uvec4( newLocCode.x ) );
	imageStore( u_gs_nodePageTable_localizationCodes, int( newTexelCoordinate + 1 ), uvec4( newLocCode.y ) );
	imageStore( u_gs_nodePageTable_localizationCodes, int( newTexelCoordinate + 2 ), uvec4( newLocCode.z ) );

	imageStore( uNodePageTableLocalizationDepths, int( newNodeTileIndex ), uvec4( newLocDepth ) );
}

////////////////////////////////////////////////////////////////////////////////
// @param pElemAddress ...
// @param pElemPointer ...
// @param pFlags ...
////////////////////////////////////////////////////////////////////////////////
void gs_nodeSetPointer2( uint pElemAddress, uvec3 pElemPointer )
{
	// XXX: Should be removed
	uvec3 brickPointer = pElemPointer + uvec3( 1 ); // Warning: fixed border size !	=> QUESTION ??

	uint packedChildAddress = imageLoad( gs_u_nodePoolChildArray, int( pElemAddress ) ).x;
	uint packedBrickAddress = gs_packBrickAddress( brickPointer );

	// We store brick
	packedChildAddress |= 0x40000000;

	//// Check flags value and modify address accordingly.
	//// If flags is greater than 0, it means that the node containing the brick is terminal
	//if ( flags > 0 )
	//{
	//	// If flags equals 2, it means that the brick is empty
	//	if ( flags == 2 )
	//	{
	//		// Empty brick flag
	//		packedBrickAddress = 0;
	//		packedChildAddress &= 0xBFFFFFFF;
	//	}

	//	// Terminal flag
	//	packedChildAddress |= 0x80000000;
	//}

	imageStore( gs_u_nodePoolChildArray, int( pElemAddress ), uvec4( packedChildAddress ) );
	imageStore( gs_u_nodePoolDataArray, int( pElemAddress ), uvec4( packedBrickAddress ) );
}

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
////////////////////////////////////////////////////////////////////////////////
void main()
{
	// Retrieve global indexes
	uint elemNum = gl_WorkGroupID.x;	// NOTE : size limit !!!!!!!!
	//uint processID = gl_LocalInvocationIndex ;
	uint processID = gl_LocalInvocationIndex;

	// Check bound
	if ( elemNum < uNbElements )
	{
		// Retrieve node address and its localisation info along with new element address
		//
		// Done by only one thread of the kernel
		if ( processID == 0 )
		{
			// Compute node address
			uint nodeAddressEnc = imageLoad( u_gs_nodeAddresses, int( elemNum ) ).x;
			smNodeAddress = gs_unpackNodeAddress( nodeAddressEnc ).x;

			// Compute element address
			uint elemIndexEnc = imageLoad( u_gs_elementAddresses, int( elemNum ) ).x;
			uvec3 elemIndex = gs_unpackBrickAddress( elemIndexEnc );
			//smElemAddress = elemIndex * u_gs_nodeTileNbElements; // convert into node address
			smElemAddress = elemIndex * uvec3( 10, 10, 10 ); // TODO...

			// Get the localization of the current element
			gs_nodePageTable_getLocalizationInfo( smNodeAddress, smParentLocInfoCode, smParentLocInfoDepth );
		}

		// Thread Synchronization
		barrier();

		// Produce data
		//
		// Optimization
		// - remove this synchonization for brick production
		// - let user-defined synchronization barriers in the producer directly
		gs_produceData( elemNum, processID, smElemAddress, smParentLocInfoCode, smParentLocInfoDepth );

		// Note : for "nodes", producerFeedback is un-un-used for the moment

		// Thread Synchronization
		barrier();

		// Update page table (who manages localisation information)
		//
		// Done by only one thread of the kernel
		if ( processID == 0 )
		{
			gs_nodeSetPointer2( smNodeAddress, smElemAddress );
		}
	}
}
