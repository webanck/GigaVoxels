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

#define MAX_ITERATIONS 100

uniform vec4 uComputeSpaceWindow/* = vec4( 0, 0, 256, 256 )*/;
uniform uint uWidth/* = 256*/;
uniform uint uHeight/* = 256*/;
// ...
uniform uint uNbElements;
uniform uint uNodeTileNbElements;

// Node Pool parameters
//
// - Child Array
// ---- hierarchical data structure (generalized N-Tree, for instance octree)
// ---- organized in node tiles (each node contains the address of its child)
uniform layout(size1x32) uimageBuffer uNodePoolChildArray;	// 1 component of 32 bits: unsigned int
// - Data Array
// ---- address in Data Pool where brick (bottom left corner)
uniform layout(size1x32) uimageBuffer uNodePoolDataArray;	// 1 component of 32 bits: unsigned int
SSS
// Node Page Table parameters
//
// - Localization code array
// ---- hierarchical data structure (generalized N-Tree, for instance octree)
// ---- organized in node tiles (each node contains the address of its child)
uniform layout(size1x32) uimageBuffer u_gs_nodePageTable_localizationCodes;	// 1 component of 32 bits: unsigned int
// - Localization depth array
// ---- address in Data Pool where brick (bottom left corner)
uniform layout(size1x32) uimageBuffer uNodePageTableLocalizationDepths;	// 1 component of 32 bits: unsigned int

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

layout ( binding  = 0, rgba8 ) uniform image2D uColorImage;
//layout ( binding = 1, rgba8 ) uniform image2D uColorImage;

////////////////////////////////////////////////////////////////////////////////
// Functions Declaration
////////////////////////////////////////////////////////////////////////////////

// Data Structure Node
struct Node
{
	// Child address
	uint childAddress;

	// Brick address
	uint brickAddress;

};

// Data Structure Management
uint gv_unpackNodeAddress( uint pAddress );
uvec3 gv_unpackBrickAddress( uint pAddress );
void gv_nodePageTable_getLocalizationInfo( uint pNodeAddress, out pLocalizationCode, out pLocalizationDepth );
void gv_produceData( uint pRequestID, uint pProcessID, uvec3 pNewElemAddress, pLocalizationCode, pLocalizationDepth );
void gv_setPointer( uint pElemAddress, uint pElemPointer );

////////////////////////////////////////////////////////////////////////////////
// Return the localization info of a node in the node pool
//
// @param nodeAddress Address of the node in the node pool
//
// @return The localization info of the node
////////////////////////////////////////////////////////////////////////////////
void gv_nodePageTable_getLocalizationInfo( uint pNodeAddress, out pLocalizationCode, out pLocalizationDepth )
{
	// Compute node tile index
	uint nodeTileIndex = pNodeAddress / uNodeTileNbElements;

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

	// Compute the address of the current node tile (and its offset in the node tile)
	uint nodeTileAddress = nodeTileIndex * uNodeTileNbElements;
	uint nodeTileOffset = pNodeAddress - nodeTileAddress;
	// Compute the node offset (in 3D, in the node tile)
	uvec3 nodeOffset = NodeTileRes::toVec3( nodeTileOffset );
	// Localization info initialization
	pLocalizationCode = parentLocCode.addLevel< NodeTileRes >( nodeOffset );
}

////////////////////////////////////////////////////////////////////////////////
// Retrieve node information at given address in cache (i.e. node pool)
//
// @param pNode the node information at given address in cache
// @param pNodeTileAddress node tile address in cache
// @param pNodeTileOffset child offset address from node tile
////////////////////////////////////////////////////////////////////////////////
void gv_fetchNode( out Node pNode, uint pNodeTileAddress, uint pNodeTileOffset )
{
	// Load a single texel from an image
	pNode.childAddress = imageLoad( uNodePoolChildArray, int( pNodeTileAddress + pNodeTileOffset ) ).x;

	// Load a single texel from an image
	pNode.brickAddress = imageLoad( uNodePoolDataArray, int( pNodeTileAddress + pNodeTileOffset ) ).x;

	// TO DO
	// - optimization ? => only if gv_nodeHisBrick( pNode ), ask for pNode.brickAddress = imageLoad()..., if not do simply "pNode.brickAddress = 0"
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
		// Shared Memory declaration
		shared uint nodeAddress;
		shared uint elemAddress;
		shared uvec3 parentLocInfoCode;
		shared uint parentLocInfoDepth;

		// Retrieve node address and its localisation info along with new element address
		//
		// Done by only one thread of the kernel
		if ( processID == 0 )
		{
			// Compute node address
			uint nodeAddressEnc = pNodesAddressList[ elemNum ];
			nodeAddress = gv_unpackNodeAddress( nodeAddressEnc ).x;

			// Compute element address
			uint elemIndexEnc = pElemAddressList[ elemNum ];
			uint elemIndex = gv_unpackNodeAddress( elemIndexEnc );
			elemAddress = elemIndex * 8; // convert into node address

			// Get the localization of the current element
			gl_nodePageTable_getLocalizationInfo( nodeAddress, parentLocInfoCode, parentLocInfoDepth );
		}

		// Thread Synchronization
		barrier();

		// Produce data
		//
		// Optimization
		// - remove this synchonization for brick production
		// - let user-defined synchronization barriers in the producer directly
		//uint producerFeedback = pGpuProvider.produceData( pGpuPool, elemNum, processID, elemAddress, parentLocInfoCode, parentLocInfoDepth );
		gv_produceData( elemNum, processID, elemAddress, parentLocInfoCode, parentLocInfoDepth );

		// Note : for "nodes", producerFeedback is un-un-used for the moment

		// Update page table (who manages localisation information)
		//
		// Done by only one thread of the kernel
		if ( processID == 0 )
		{
			gv_setPointer( nodeAddress, elemAddress );
		}
	}
}
