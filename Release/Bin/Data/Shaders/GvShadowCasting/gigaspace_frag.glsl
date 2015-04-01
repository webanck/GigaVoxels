////////////////////////////////////////////////////////////////////////////////
//
// FRAGMENT SHADER
//
// GigaSpace Pass 
//
// - Hierarchical data structure traversal
// - Requests of production are emitted when no data is encountered
// - Multi-resolution voxel-based volume rendering pass with cone-tracing
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// VERSION
////////////////////////////////////////////////////////////////////////////////

// TO DO
//
// - check that (imageLoad, imageStore)
#version 410

////////////////////////////////////////////////////////////////////////////////
// EXTENSION
////////////////////////////////////////////////////////////////////////////////

// TO DO
//
// - check that (imageLoad, imageStore)
#extension GL_NV_gpu_shader5 : enable
#extension GL_EXT_shader_image_load_store : enable

////////////////////////////////////////////////////////////////////////////////
// INCLUDE
////////////////////////////////////////////////////////////////////////////////

// TO DO
//
// - use #include to include a GLSL GigaSpace library
// - separate features in different headers
// Exemple :
//    #include "gv_GigaSpace_GLSL.h"
// that may include :
//    #include "gv_DataStructure_glsl.h"
//    #include "gv_DataProductionManager_glsl.h"
//    #include "gv_Renderer_glsl.h"

////////////////////////////////////////////////////////////////////////////////
// DEFINE AND CONSTANT SECTION
////////////////////////////////////////////////////////////////////////////////

// TO DO
//
// - add #define to be able to customize code à la Ubershader
#define GV_USE_LIGHTING
//#define GV_USE_MIPMAPPING
#define GV_USE_OPTIMIZED_NORMAL
//#define GV_USE_DEBUG_COUNTER

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// UNIFORM
////////////////////////////////////////////////////////////////////////////////

// Viewing System parameters
//
// - Camera
uniform vec3 uViewPos;
uniform vec3 uViewPlane;
uniform vec3 uViewAxisX;
uniform vec3 uViewAxisY;
uniform vec2 uPixelSize;
uniform float uFrustumNearInv;

// Data Traversal parameters
uniform uint uMaxDepth;
uniform float uConeApertureScale;
uniform uint uMaxNbLoops;

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

// Data Pool parameters
//
// - Cache of bricks of data associated to nodes (i.e. voxels)
//
// - uDataPool : sampler 3D to access user defined voxels data
// - uBrickPoolResInv : size of one piece of data in cache space (i.e one voxel)
//
// Note : Here only one channel is defined (users need one sampler per channel)
uniform sampler3D uDataPool;
uniform vec3 uBrickPoolResInv;

// Data Production Management parameters
//
// - uUpdateBufferArray : buffer of requests
// - uNodeTimeStampArray : buffer of nodes time stamps
// - uBrickTimeStampArray : buffer of bricks time stamps
// - uCurrentTime : current time (number of frames)
// - uBrickCacheSize : size of bricks in cache (i.e in Data Pool)
uniform layout(size1x32) uimageBuffer uUpdateBufferArray;
uniform layout(size1x32) uimageBuffer uNodeTimeStampArray;
uniform layout(size1x32) uimageBuffer uBrickTimeStampArray;
uniform uint uCurrentTime;
uniform uvec3 uBrickCacheSize;

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

// The output of the shader program
out vec4 oColor;

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

// Data Production Management
void gv_setNodeUsage( uint pAddress );
void gv_setBrickUsage( uvec3 pAddress );
void gv_cacheLoadRequest( uint pNodeAddressEnc );
void gv_cacheSubdivRequest( uint pNodeAddressEnc );

// Rendering Management
bool gv_intersectBox( vec3 pRayStart, vec3 pRayDir, vec3 boxmin, vec3 boxmax, out float tnear, out float tfar );
float gv_getRayNodeLength( vec3 posInNode, float nsize, vec3 pRayDir );
float gv_getConeAperture( float pDistance );
void gv_NodeVisitor_visit( vec3 pos,
					out uint nodeIdx, out uint nodeIdxPrev,  //! numData: number of point samples linked/ or address of the brick
					out vec3 nodePos, out vec3 nodePosPrev, //Prev== parent node
					out float nodeSize, float pConeAperture );
vec4 gv_sampleBrick( uint brickIdxEnc, vec3 samplePos, vec3 nodePos, float nodeSize );
vec4 gv_sampleMipMapInterp( float maxDepthF, uint numData, uint numDataPrev, vec3 samplePos, vec3 nodePos, float nodeSize, vec3 nodePosPrev );
vec4 gv_traceVoxelConeRayCast( vec3 pRayStart, vec3 pRayDir, float t, float tMax );

// Shader Management
void gv_Shader_postShade( inout vec4 pColor, uint pCounter );

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

////////////////////////////////////////////////////////////////////////////////
// Update timestamp usage information of a node tile
// with current time (i.e. current rendering pass)
// given its address in the node pool.
//
// @param pAddress The address of the node for which we want to update usage information
////////////////////////////////////////////////////////////////////////////////
void gv_setNodeUsage( uint pAddress )
{
	// Specify the coordinate at which to store the texel.
	//
	// FIXME : divide by elemRes or >> if POT
	uint elemOffset = pAddress / 8;

	// Write a single texel into an image
	imageStore( uNodeTimeStampArray, int( elemOffset ), uvec4( uCurrentTime ) );
}

////////////////////////////////////////////////////////////////////////////////
// Update timestamp usage information of a brick
// with current time (i.e. current rendering pass)
// given its address in brick pool.
//
// @param pAddress The address of the node for which we want to update usage information
////////////////////////////////////////////////////////////////////////////////
void gv_setBrickUsage( uvec3 pAddress )
{
	// Specify the coordinate at which to store the texel.
	//
	// FIXME : divide by elemRes or >> if POT
	uvec3 elemOffset = pAddress / 10;

	uint elemOffsetLinear =
		elemOffset.x + elemOffset.y * uBrickCacheSize.x +
		elemOffset.z * uBrickCacheSize.x * uBrickCacheSize.y;

	// Write a single texel into an image
	imageStore( uBrickTimeStampArray, int( elemOffsetLinear ), uvec4( uCurrentTime ) );
}

////////////////////////////////////////////////////////////////////////////////
// Update buffer with a load request for a given node.
//
// @param pNodeAddressEnc the encoded node address
////////////////////////////////////////////////////////////////////////////////
void gv_cacheLoadRequest( uint pNodeAddressEnc )
{
	// Write a single texel into an image.
	//
	// GigaVoxels bit mask for load request is 0x80000000U (31th bit)
	imageStore( uUpdateBufferArray, int( pNodeAddressEnc ), uvec4( ( pNodeAddressEnc & 0x3FFFFFFFU ) | 0x80000000U ) );
}

////////////////////////////////////////////////////////////////////////////////
// Update buffer with a subdivision request for a given node.
//
// @param pNodeAddressEnc the encoded node address
////////////////////////////////////////////////////////////////////////////////
void gv_cacheSubdivRequest( uint pNodeAddressEnc )
{
	// Write a single texel into an image.
	//
	// GigaVoxels bit mask for subdivision request is 0x40000000U (30th bit)
	imageStore( uUpdateBufferArray, int( pNodeAddressEnc ), uvec4( ( pNodeAddressEnc & 0x3FFFFFFFU) | 0x40000000U ) );
}

////////////////////////////////////////////////////////////////////////////////
// Intersect ray with a box.
//
// Algorithm comes from :
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm*
//
// @param pRayStart ray start
// @param pRayDir ray direction
// @param boxmin box min position
// @param boxmax box max position
// @param tnear (output) the resulting near distance of intersection
// @param tfar (output) the resulting far distance of intersection
//
// @return a flag telling wheter or not an intersection occurs
////////////////////////////////////////////////////////////////////////////////
bool gv_intersectBox( vec3 pRayStart, vec3 pRayDir, vec3 boxmin, vec3 boxmax, out float tnear, out float tfar )
{
	// Compute intersection of ray with all six bbox planes
	vec3 invR = vec3( 1.0f ) / pRayDir;
	vec3 tbot = invR * ( boxmin - pRayStart );
	vec3 ttop = invR * ( boxmax - pRayStart );

	// Re-order intersections to find smallest and largest on each axis
	vec3 tmin = min( ttop, tbot );
	vec3 tmax = max( ttop, tbot );

	// Find the largest tmin and the smallest tmax
	float largest_tmin = max( max( tmin.x, tmin.y), max( tmin.x, tmin.z ) );
	float smallest_tmax = min( min( tmax.x, tmax.y), min( tmax.x, tmax.z ) );

	tnear = largest_tmin;
	tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

////////////////////////////////////////////////////////////////////////////////
// gv_getRayNodeLength() function
//
// @param posInNode ...
// @param nsize node size
// @param pRayDir ray direction
//
// @return ...
////////////////////////////////////////////////////////////////////////////////
float gv_getRayNodeLength( vec3 posInNode, float nsize, vec3 pRayDir )
{
#if 1

    vec3 directions = step( 0.0, pRayDir );	// To precompute somewhere

    vec3 miniT;
    vec3 miniraypos = posInNode;
    vec3 planes = directions * nsize;
    miniT = ( planes - miniraypos ) / pRayDir;

    return min( miniT.x, min( miniT.y, miniT.z ) );	// * length( ray );

#else

	// Intersect ray with box
    float boxInterMin = 0.0f;
	float boxInterMax = 0.0f;
    bool hit = gv_intersectBox( posInNode, pRayDir, vec3( 0.0f ), vec3( nsize ), boxInterMin, boxInterMax );

	// Check for intersection
	if ( hit )
	{
		return boxInterMax - boxInterMin;
	}
	else
	{
		return 1.0f / 2048.0f;
	}

#endif
}

////////////////////////////////////////////////////////////////////////////////
// This method returns the cone aperture for a given distance.
//
// @param pDistance the current distance along the ray's direction.
//
// @return the cone aperture
////////////////////////////////////////////////////////////////////////////////
float gv_getConeAperture( float pDistance )
{
	// Note :
	// - uConeApertureScale is used to over-estimate to avoid aliasing
	
	// It is an estimation of the size of a voxel at given distance from the camera.
	// It is based on THALES theorem. Its computation is rotation invariant.
	return uPixelSize.x * pDistance * uFrustumNearInv * uConeApertureScale;
}

////////////////////////////////////////////////////////////////////////////////
// Node visitor function
// 
// - Hierarchical data structure traversal
// - Requests of production are emitted when no data is encountered
//
// @param pPosition current position along the ray
// @param pNodeIdx (output) address in cache of visited node
// @param pNodeIdxPrev (output) address in cache of father's visited node
// @param pNodePosition (output) bottom left 3D position of visited node
// @param pNodePositionPrev (output) bottom left 3D position of father's visited node
// @param pNodeSize (output) 3D size of visited node
// @param pConeAperture current cone aperture (according to current position along the ray)
////////////////////////////////////////////////////////////////////////////////
void gv_NodeVisitor_visit( vec3 pPosition,
					out uint pNodeIdx, out uint pNodeIdxPrev,  //! numData: number of point samples linked/ or address of the brick
					out vec3 pNodePosition, out vec3 pNodePositionPrev, //Prev== parent node
					out float pNodeSize, float pConeAperture )
{
	// Start from root node (first nodetile is unused to be able to use "NULL" address)
	uint rootIdx = 8;	// getNodeIdxInit();	// (hard-coded...)

	pNodePosition = vec3( 0.0 );
	pNodeSize = 2.0;
	float nodeSizeInv = 1.0 / pNodeSize;
	float voxelSize = pNodeSize / 8.0;

	uint curIdx = 0;
	uint prevIdx = 0;
	uint prevPrevIdx = 0;

	vec3 prevNodePos = vec3( 0.0f );
	vec3 prevPrevNodePos = vec3( 0.0f );

	uint nodeChildIdx = rootIdx;

	uint depth = 0;	// getNodeDepthInit();

	Node node;

	// Visit the data structure
	bool descentSizeCriteria;
	do
	{
		// Each time, modify size to children ones
		//
		// - here, for octree, divide by 2
		pNodeSize = pNodeSize * 0.5;
		nodeSizeInv = nodeSizeInv * 2.0;
		voxelSize *= 0.5;

		// For a given node, find in which child we are
		uvec3 curOffsetI = uvec3( ( pPosition - pNodePosition ) * nodeSizeInv );
		// Linearise offset (3D to 1D)
		// - written in hardware FMAD compliant way
		uint curOffset = curOffsetI.z * 4 + ( curOffsetI.y * 2 + curOffsetI.x );

		// Fetch the node
		gv_fetchNode( node, nodeChildIdx, curOffset );

		prevPrevIdx = prevIdx;
		prevIdx = curIdx;
		curIdx = nodeChildIdx + curOffset;

		// Flag current node tile as used
		gv_setNodeUsage( nodeChildIdx );

		// Flag current brick as used
		if ( gv_nodeHasBrick( node ) )
		{
			gv_setBrickUsage( gv_nodeGetBrickAddress( node ) );
		}

		prevPrevNodePos = prevNodePos;
		prevNodePos = pNodePosition;

		// Update bottom left corner of node
		pNodePosition = pNodePosition + pNodeSize * vec3( curOffsetI );

		// Update descent condition
		descentSizeCriteria = ( voxelSize > pConeAperture ) && ( depth < uMaxDepth );

		// Update current depth
		// - next LOD (level of detail)
		depth++;

		// TO DO
		// - sue a #define or uniform to handle the following policy
#if 1	// Low resolution first

		if ( ! gv_nodeIsInitialized( node ) || ( gv_nodeIsBrick( node ) && !gv_nodeHasBrick( node ) ) )
		{
			// Flag node to request a data production (i.e load)
			gv_cacheLoadRequest( curIdx );
		}
		else if ( !gv_nodeHasSubNodes( node ) && descentSizeCriteria && !gv_nodeIsTerminal( node ) )
		{
			// Flag node to request a node subdivision
			gv_cacheSubdivRequest( curIdx );
		}

#else	// High resolution immediatly

#endif

		// Update base nodetile address
		nodeChildIdx = gv_nodeGetChildAddress( node );
	}
	while ( descentSizeCriteria && gv_nodeHasSubNodes( node ) );	// END of the data structure traversal

	pNodeIdx = curIdx;
	pNodeIdxPrev = prevIdx;

	pNodePositionPrev = prevNodePos;
}

////////////////////////////////////////////////////////////////////////////////
// Sample the data pool (i.e bricks of voxels) at a given position
// to retrieve the underlying value.
//
// @param brickIdxEnc the current brick's encoded address in the data pool
// @param samplePos the current sample's position
// @param nodePos the current node's position
// @param pNodeSize the current node's size
//
// @return the sampled value at given position
////////////////////////////////////////////////////////////////////////////////
vec4 gv_sampleBrick( uint brickIdxEnc, vec3 samplePos, vec3 nodePos, float pNodeSize )
{
	// Unpack the brick address to obtain localization code
	uvec3 brickIdx;
	brickIdx.x = ( brickIdxEnc & 0x3FF00000U ) >> 20U;
	brickIdx.y = ( brickIdxEnc & 0x000FFC00U ) >> 10U;
	brickIdx.z = brickIdxEnc & 0x000003FFU;

	// FIXME : why is it working with 0.0 and not 1.0 ?
	const float cUsedBrickSize = 8.0;	// float( VOXEL_POOL_BRICK_RES - VOXEL_POOL_BRICK_BORDER );

	// Compute texture coordinates to sample data
	vec3 brickPos = vec3( brickIdx );
	vec3 posInNode = ( samplePos - nodePos ) / pNodeSize;
	vec3 samplePosBrick = posInNode * cUsedBrickSize;

	// Sample data
	return texture( uDataPool, ( brickPos + samplePosBrick ) * uBrickPoolResInv );
}

////////////////////////////////////////////////////////////////////////////////
// Compute normal (with gradient)
//
// @param brickIdxEnc the current brick's encoded address in the data pool
// @param samplePos the current sample's position
// @param nodePos the current node's position
// @param pNodeSize the current node's size
// @param pRayStep the current ray step
//
// @return the normal at given position
////////////////////////////////////////////////////////////////////////////////
vec3 gv_computeGradient( uint brickIdxEnc, vec3 pSamplePos, vec3 pNodePos, float pNodeSize, float pRayStep )
{
	// Note : alpha component stores the density of matter (i.e. w)
	// - so, gradient computation on "w" component is quivalent to normal

	vec3 grad = vec3( 0.f );

	// Unpack the brick address to obtain localization code
	uvec3 brickIdx;
	brickIdx.x = ( brickIdxEnc & 0x3FF00000U ) >> 20U;
	brickIdx.y = ( brickIdxEnc & 0x000FFC00U ) >> 10U;
	brickIdx.z = brickIdxEnc & 0x000003FFU;

	// TO DO
	// - remove this hard-coded value
	const float cUsedBrickSize = 8.0;	// float( VOXEL_POOL_BRICK_RES - VOXEL_POOL_BRICK_BORDER );
	float cTreeToCacheSpaceFactor = cUsedBrickSize * uBrickPoolResInv / pNodeSize;

	// Compute texture coordinates to sample data
	vec3 brickPosInCache = /*BrickPos*/vec3( brickIdx ) * uBrickPoolResInv;
			
	float gradStepInCache = /*gradStep*/( pRayStep * 0.25f ) * cTreeToCacheSpaceFactor;

	// Written in hardware FMAD compliant way
	vec3 samplePosInCache = /*offsetInNode*/( pSamplePos - pNodePos ) * cTreeToCacheSpaceFactor + brickPosInCache;

	// x component
	float nX_1 = texture( uDataPool, samplePosInCache + vec3( gradStepInCache, 0.0f, 0.0f ) ).w;
	float nX_2 = texture( uDataPool, samplePosInCache - vec3( gradStepInCache, 0.0f, 0.0f ) ).w;
	// y component
	float nY_1 = texture( uDataPool, samplePosInCache + vec3( 0.0f, gradStepInCache, 0.0f ) ).w;
	float nY_2 = texture( uDataPool, samplePosInCache - vec3( 0.0f, gradStepInCache, 0.0f ) ).w;
	// z component
	float nZ_1 = texture( uDataPool, samplePosInCache + vec3( 0.0f, 0.0f, gradStepInCache ) ).w;
	float nZ_2 = texture( uDataPool, samplePosInCache - vec3( 0.0f, 0.0f, gradStepInCache ) ).w;
	
	grad.x = nX_2 - nX_1;
	grad.y = nY_2 - nY_1;
	grad.z = nZ_2 - nZ_1;

	return grad;
}

////////////////////////////////////////////////////////////////////////////////
// Sample the data pool (i.e bricks of voxels) at a given position
// to retrieve the underlying value by using mip-mapping with parent's coarser level of resolution
//
// @param maxDepthF ...
// @param maxDepthNew ...
// @param depth ...
// @param numData ...
// @param numDataPrev ...
// @param samplePos ...
// @param nodePos ...
// @param pNodeSize ...
// @param nodePosPrev ...
//
// @return the sampled value at given position
////////////////////////////////////////////////////////////////////////////////
vec4 gv_sampleMipMapInterp( float maxDepthF,
						uint numData, uint numDataPrev, vec3 samplePos,
						vec3 nodePos, float pNodeSize, vec3 nodePosPrev )
{
	// TO DO
	// - modify this function to be operational
	// ...

	float quadInterp = fract( maxDepthF );

	// Sample data pool (i.e bricks of voxels) at given position
	vec4 sampleVal = gv_sampleBrick( numData, samplePos, nodePos, pNodeSize );
	vec4 sampleValParent = vec4( 0. );

	// Check if there is a parent with data
	if ( numDataPrev != 0 )
	{
		// Sample data pool at same position but at the previous level of resolution (i.e. coarser one)
		sampleValParent = gv_sampleBrick( numDataPrev, samplePos, nodePosPrev, pNodeSize * 2.0f );
	}

	//return mix( sampleValParent, sampleVal, quadInterp );
	return sampleVal;
	//return sampleValParent;
}

////////////////////////////////////////////////////////////////////////////////
// Main GigaSpace pass
// 
// - Hierarchical data structure traversal
// - Requests of production are emitted when no data is encountered
// - Requests of production are emitted when no data is encountered
// - Multi-resolution voxel-based volume rendering pass with cone-tracing
//
// @param pPosition current position along the ray
// @param pNodeIdx (output) address in cache of visited node
// @param pNodeIdxPrev (output) address in cache of father's visited node
// @param pNodePosition (output) bottom left 3D position of visited node
// @param pNodePositionPrev (output) bottom left 3D position of father's visited node
// @param pNodeSize (output) 3D size of visited node
// @param pConeAperture current cone aperture (according to current position along the ray)
//
// @param pRayStart ray start position
// @param pRayDir ray direction
// @param t distance to camera from which to start ray tracing (given by box intersection and near far clipping plane)
// @param tMax distance to camera at which to end ray tracing (given by box intersection and far far clipping plane)
//
// @return The color accumulated along the ray
////////////////////////////////////////////////////////////////////////////////
vec4 gv_traceVoxelConeRayCast( vec3 pRayStart, vec3 pRayDir, float t, float tMax )
{
    float tTree = t;

    // XXX: really needed ? We don't put the address (0,0,0) in the cache's list anyway.
    gv_setNodeUsage( 0 );

	// Update current position along the ray
	// - i.e. start position
    vec3 samplePosTree = tTree * pRayDir + pRayStart; // written in hardware FMAD compliant way

	// TO DO
	//
	// - add user customizable function
	//
	// - Shader pre-shade process
	// - pShader.preShade( pRayStartTree, pRayDirTree, ptTree );
	// ...
	// Pre-shade
    vec4 accColor = vec4( 0.0 );
	
    // Advance along the ray, node by node
	uint numLoop = 0;
    while ( tTree < tMax && /*stop criteria*/accColor.a < 0.99 && numLoop < uMaxNbLoops )
    {
   		uint nodeIdx;
		uint nodeIdxPrev;
		float nodeSize;
		vec3 nodePos;
		vec3 nodePosPrev;
		uint depth;

		float coneAperture = gv_getConeAperture( tTree );
		
        // Node visitor
		//
		// - traverse the data structure
		// - emits requests if required
        gv_NodeVisitor_visit( samplePosTree, nodeIdx, nodeIdxPrev, nodePos, nodePosPrev, nodeSize, coneAperture );

        vec3 posInNode = samplePosTree - nodePos;

        float nodeLength = gv_getRayNodeLength( posInNode, nodeSize, pRayDir );

		// Retrieve node's brick address
        uint brickAddress = 0;
		uint brickAddressPrev = 0;
		if ( nodeIdx != 0 )
		{
			brickAddress = imageLoad( uNodePoolDataArray, int( nodeIdx ) ).x;
		}
		if ( nodeIdxPrev != 0 )
		{
			brickAddressPrev = imageLoad( uNodePoolDataArray, int( nodeIdxPrev ) ).x;
		}

        // Brick visitor
		//
		// - traverse the brick
		// - apply shader
        if ( brickAddress != 0 )
        {
			// Local distance
			float dt = 0.0f;

			// Step
			float rayStep = 0.0f;
			
			// Visit the brick
		    while ( dt <= nodeLength && /*stop criteria*/accColor.a < 0.99 )
            {
				// Update global distance
				float fullT = tTree + dt;
				
				// Get the cone aperture at the given distance
				float coneAperture = gv_getConeAperture( fullT );
				
				// Update current position along ray
                samplePosTree = fullT * pRayDir + pRayStart; // written in hardware FMAD compliant way
				
				// Compute next step
				rayStep = max( coneAperture, ( nodeSize / 8.0f )/*voxelSize*/ * 0.66f );	// 2/3 size of one voxel (hard-coded)
				
                // Sample data (i.e. voxels brick)
#ifdef GV_USE_MIPMAPPING
				// TO DO
				// - add mipmapping parameter computation
				// ...
				// vec4 color = gv_sampleMipMapInterp( maxDepthF, brickAddress, brickAddressPrev, samplePosTree, nodePos, nodeSize, nodePosPrev );
#else
                vec4 color = gv_sampleBrick( brickAddress, samplePosTree, nodePos, nodeSize );
#endif				

				// TO DO
				//
				// - add user customizable function
				//
                // Shading / Lighting
#ifdef GV_USE_LIGHTING
				// Compute normal (with gradient)
	#ifdef GV_USE_OPTIMIZED_NORMAL
				vec3 normal = normalize( gv_computeGradient( brickAddress, samplePosTree, nodePos, nodeSize, rayStep ) );
	#else
                vec3 normal = vec3( 0.f );
				float gradStep = rayStep * 0.25f;
				vec4 v0;
                vec4 v1;
				// x component
				v0 = gv_sampleBrick( brickAddress, samplePosTree + vec3( gradStep, 0.0f, 0.0f ), nodePos, nodeSize );
				v1 = gv_sampleBrick( brickAddress, samplePosTree - vec3( gradStep, 0.0f, 0.0f ), nodePos, nodeSize );
				normal.x = v0.w - v1.w;
				// y component
				v0 = gv_sampleBrick( brickAddress, samplePosTree + vec3( 0.0f, gradStep, 0.0f ), nodePos, nodeSize );
				v1 = gv_sampleBrick( brickAddress, samplePosTree - vec3( 0.0f, gradStep, 0.0f ), nodePos, nodeSize );
				normal.y = v0.w - v1.w;
				// z component
				v0 = gv_sampleBrick( brickAddress, samplePosTree + vec3( 0.0f, 0.0f, gradStep ), nodePos, nodeSize );
				v1 = gv_sampleBrick( brickAddress, samplePosTree - vec3( 0.0f, 0.0f, gradStep ), nodePos, nodeSize );
				normal.z = v0.w - v1.w;

				normal = -normal;
				normal = normalize( normal );
	#endif
				// Light vector
				vec3 lightVec = normalize( vec3( 1.0 ) - samplePosTree );
				
				// View vector (no used for the moment)
				//vec3 viewVec = -pRayDir;

				// Diffuse lighting model
				color.rgb = color.rgb * max( 0.0, dot( normal, lightVec ) );
#endif

				// Accumulate color
				//
				// - TO DO : modify this and use real volume equation (not the usual compositing one)
				accColor = accColor + ( 1.0 - accColor.a ) * color;

				// Update local distance
				dt += rayStep;
			}

			// Update current distance
			tTree = tTree + dt;
		}
		else
        {
			// Here node is either empty either its associated brick is not yet in cache

			// Update current distance at end of the node
            tTree = tTree + nodeLength;

			// Epsilon shifting
            //tTree = tTree + 1. / 512.;
			tTree += gv_getConeAperture( tTree );
        }
        
		// Update current position along the ray
        samplePosTree = pRayStart + tTree * pRayDir;

		// Update loop counter
        numLoop++;
    }

	// TO DO
	//
	// - add user customizable function
	//
	// - Shader post-shade process
	gv_Shader_postShade( accColor, numLoop );

	return accColor;
}

////////////////////////////////////////////////////////////////////////////////
// This method is called after the ray stopped or left the bounding
// volume. You may want to do some post-treatment of the color.
////////////////////////////////////////////////////////////////////////////////
void gv_Shader_postShade( inout vec4 pColor, uint pCounter )
{
	/*if ( pColor.w >= 0.99 )
	{
		pColor.w = 1.f;
	}*/

	// Debug output
	//
	// TO DO : set min/max adaptative scale
	// ...
#ifdef GV_USE_DEBUG_COUNTER
	if ( pCounter < 10 ) { pColor = vec4( 0.0, 0.0, 0.0, 1.0 ); return; }	// black
	if ( pCounter < 20 ) { pColor = vec4( 0.0, 0.0, 1.0, 1.0 ); return; }	// blue
	if ( pCounter < 30 ) { pColor = vec4( 0.0, 1.0, 0.0, 1.0 ); return; }	// green
	if ( pCounter < 40 ) { pColor = vec4( 0.0, 1.0, 1.0, 1.0 ); return; }	// cyan
	if ( pCounter < 50 ) { pColor = vec4( 1.0, 0.0, 0.0, 1.0 ); return; }	// red
	if ( pCounter < 60 ) { pColor = vec4( 1.0, 0.0, 1.0, 1.0 ); return; }	// pink
	if ( pCounter < 70 ) { pColor = vec4( 1.0, 1.0, 0.0, 1.0 ); return; }	// yellow
	if ( pCounter < 80 ) { pColor = vec4( 1.0, 1.0, 1.0, 1.0 ); return; }	// white
#endif
}

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
//
// Main fragment program.
//
// In GigaVoxels, ray casting algorithm is replaced by a custom
// voxel cone tracing algorithm.
//
// The output is the color accumulated during ray traversal.
////////////////////////////////////////////////////////////////////////////////
void main()
{
	// Ray generation
	//
	// - compute ray direction from view point through current pixel
    vec3 viewDir = normalize( uViewPlane + uViewAxisX * gl_FragCoord.x + uViewAxisY * gl_FragCoord.y );

	// Ray intersection
	//
	// - intersect current ray with box
	// - initialize the GigaVoxels bounding box to test ray casting intersection
    const vec3 boxMin = vec3( 0.001, 0.001, 0.001 );
    const vec3 boxMax = vec3( 0.999, 0.999, 0.999 );
	float tnear, tfar;
	bool hit = gv_intersectBox( uViewPos, viewDir, boxMin, boxMax, tnear, tfar );

	// Stop if no intersection
	if ( ! hit )
	{
		return;
	}

	// Clamp "tnear" value to near plane if needed
	if ( tnear < 0.0 )
	{
		tnear = 0.0;
	}

	//float initT = tnear + /*espilon*/1.0 / 512.0;
	float initT = tnear + gv_getConeAperture( tnear );
	
	// Launch voxel cone tracing algorithm
    oColor = gv_traceVoxelConeRayCast( uViewPos, viewDir, initT, tfar );
}
