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

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

// Number of shader invocations per work group
layout ( local_size_x = 128 ) in;

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
uniform layout(size1x32) uimageBuffer gs_u_nodePoolChildArray;	// 1 component of 32 bits: unsigned int
// - Data Array
// ---- address in Data Pool where brick (bottom left corner)
uniform layout(size1x32) uimageBuffer gs_u_nodePoolDataArray;	// 1 component of 32 bits: unsigned int

// Node Page Table parameters
//
// - Localization code array
// ---- hierarchical data structure (generalized N-Tree, for instance octree)
// ---- organized in node tiles (each node contains the address of its child)
uniform layout(size1x32) uimageBuffer u_gs_nodePageTable_localizationCodes;	// 1 component of 32 bits: unsigned int
// - Localization depth array
// ---- address in Data Pool where brick (bottom left corner)
uniform layout(size1x32) uimageBuffer uNodePageTableLocalizationDepths;	// 1 component of 32 bits: unsigned int

// Node Pool parameters
//
// - Child Array
// ---- hierarchical data structure (generalized N-Tree, for instance octree)
// ---- organized in node tiles (each node contains the address of its child)
uniform layout(size1x32) uimageBuffer u_gs_nodeAddresses;	// 1 component of 32 bits: unsigned int
// - Data Array
// ---- address in Data Pool where brick (bottom left corner)
uniform layout(size1x32) uimageBuffer u_gs_elementAddresses;	// 1 component of 32 bits: unsigned int

////////////////////////////////////////////////////////////////////////////////
// SHARED MEMORY
////////////////////////////////////////////////////////////////////////////////

// ...
shared uint smNodeAddress;
shared uint smElemAddress;
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
uvec3 gs_unpackBrickAddress( uint pAddress );
void gs_nodePageTable_getLocalizationInfo( uint pNodeAddress, out uvec3 pLocalizationCode, out uint pLocalizationDepth );
void gs_produceData( uint pRequestID, uint pProcessID, uint pNewElemAddress, uvec3 pLocalizationCode, uint pLocalizationDepth );
void gs_nodeSetPointer( uint pElemAddress, uint pElemPointer );
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
void gs_produceData( uint pRequestID, uint pProcessID, uint pNewElemAddress, uvec3 pLocalizationCode, uint pLocalizationDepth )
{
	// Process ID gives the 1D index of a node in the current node tile
	if ( pProcessID < u_gs_nodeTileNbElements )
	{
		// First, compute the 3D offset of the node in the node tile
		uvec3 subOffset = gs_vector_toVec3( pProcessID );

		// Node production corresponds to subdivide a node tile.
		// So, based on the index of the node, find the node child.
		// As we want to sudbivide the curent node, we retrieve localization information
		// at the next level
		uvec3 regionCoords = gs_localizationCode_addLevel( pLocalizationCode, subOffset );
		// uint regionDepth = parentLocDepth.addLevel().get();
		uint regionDepth = pLocalizationDepth + 1;

		// Create a new node for which you will have to fill its information.
		gs_Node newNode;
		newNode.childAddress = 0;
		newNode.brickAddress = 0;

		// Call what we call an oracle that will determine the type of the region of the node accordingly
		uint nodeRegionInfo = getRegionInfo( regionCoords, regionDepth );

		// Now that the type of the region is found, fill the new node information
		if ( nodeRegionInfo == gs_nodeRegionInfo_CONSTANT )
		{
			gs_nodeSetTerminal( newNode, true );
		}
		else if ( nodeRegionInfo == gs_nodeRegionInfo_DATA )
		{
			gs_nodeSetStoreBrick( newNode );
			gs_nodeSetTerminal( newNode, false );
		}
		else if ( nodeRegionInfo == gs_nodeRegionInfo_MAXRESOLUTION )
		{
			gs_nodeSetStoreBrick( newNode );
			gs_nodeSetTerminal( newNode, true );
		}

		// Finally, write the new node information into the node pool by selecting channels :
		// - Loki::Int2Type< 0 >() points to node information
		// - Loki::Int2Type< 1 >() points to brick information
		//
		// newElemAddress.x + pProcessID : is the adress of the new node in the node pool
		//
		// - write a single texel into an image
		imageStore( gs_u_nodePoolChildArray, int( pNewElemAddress.x + pProcessID ), uvec4( newNode.childAddress ) );
		imageStore( gs_u_nodePoolDataArray, int( pNewElemAddress.x + pProcessID ), uvec4( newNode.brickAddress ) );
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

	return gs_nodeRegionInfo_DATA;
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
// PROGRAM
////////////////////////////////////////////////////////////////////////////////
void main()
{
	// Retrieve global indexes
	uint elemNum = gl_WorkGroupID.x;
	uint processID = gl_LocalInvocationIndex ;

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
			uint elemIndex = gs_unpackNodeAddress( elemIndexEnc );
			smElemAddress = elemIndex * u_gs_nodeTileNbElements; // convert into node address

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

		// Update page table (who manages localisation information)
		//
		// Done by only one thread of the kernel
		if ( processID == 0 )
		{
			gs_nodeSetPointer( smNodeAddress, smElemAddress );
		}
	}
}
