#ifndef _GV_GLSL_DATA_STRUCTURE_MANAGEMENT_H_
#define _GV_GLSL_DATA_STRUCTURE_MANAGEMENT_H_

////////////////////////////////////////////////////////////////////////////////
//
// GigaSpace SHADER library
//
// Data Structure Management 
//
// - ...
//
////////////////////////////////////////////////////////////////////////////////

// Node Pool parameters
//
// - Child Array
// ---- hierarchical data structure (generalized N-Tree, for instance octree)
// ---- organized in node tiles (each node contains the address of its child)
//
// - Data Array
// ---- address in Data Pool where brick (bottom left corner)
uniform layout(size1x32) uimageBuffer uNodePoolChildArray;
uniform layout(size1x32) uimageBuffer uNodePoolDataArray;

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
void gv_fetchNode( out Node pNode, uint pNodeTileAddress, uint pNodeTileOffset );
bool gv_nodeIsInitialized( Node pNode );
bool gv_nodeHasSubNodes( Node pNode );
bool gv_nodeIsBrick( Node pNode );
bool gv_nodeHasBrick( Node pNode );
bool gv_nodeIsTerminal( Node pNode );
uint gv_nodeGetChildAddress( Node pNode );
uvec3 gv_nodeGetBrickAddress( Node pNode );

////////////////////////////////////////////////////////////////////////////////
// Unpack a node address
//
// @param pAddress node address
//
// @return the packed node address
////////////////////////////////////////////////////////////////////////////////
uint gv_unpackNodeAddress( uint pAddress )
{
	return ( pAddress & 0x3FFFFFFFU );
}

////////////////////////////////////////////////////////////////////////////////
// Unpack a brick address
//
// @param pAddress brick address
//
// @return the packed brick address
////////////////////////////////////////////////////////////////////////////////
uvec3 gv_unpackBrickAddress( uint pAddress )
{
	uvec3 res;
	res.x = ( pAddress & 0x3FF00000U ) >> 20U;
	res.y = ( pAddress & 0x000FFC00U ) >> 10U;
	res.z = pAddress & 0x000003FFU;

	return res;
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
}

////////////////////////////////////////////////////////////////////////////////
// Tell wheter or not a node is initialized
//
// @param pNode the node to test
//
// @return a flag telling wheter or not a node is initialized
////////////////////////////////////////////////////////////////////////////////
bool gv_nodeIsInitialized( Node pNode )
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
bool gv_nodeHasSubNodes( Node pNode )
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
bool gv_nodeIsBrick( Node pNode )
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
bool gv_nodeHasBrick( Node pNode )
{
	return gv_nodeIsBrick( pNode ) && pNode.brickAddress != 0;
}

////////////////////////////////////////////////////////////////////////////////
// Tell wheter or not a node is terminal
//
// @param pNode the node to test
//
// @return a flag telling wheter or not a node is terminal
////////////////////////////////////////////////////////////////////////////////
bool gv_nodeIsTerminal( Node pNode )
{
	return ( pNode.childAddress & 0x80000000U ) != 0;
}

////////////////////////////////////////////////////////////////////////////////
// Retrieve the node's child address
//
// @param pNode the node to test
//
// @return the node's child address
////////////////////////////////////////////////////////////////////////////////
uint gv_nodeGetChildAddress( Node pNode )
{
	return gv_unpackNodeAddress( pNode.childAddress );
}

////////////////////////////////////////////////////////////////////////////////
// Retrieve the node's data address
//
// @param pNode the node to test
//
// @return the node's data address
////////////////////////////////////////////////////////////////////////////////
uvec3 gv_nodeGetBrickAddress( Node pNode )
{
	return gv_unpackBrickAddress( pNode.brickAddress );
}

#endif // _GV_GLSL_DATA_STRUCTURE_MANAGEMENT_H_
