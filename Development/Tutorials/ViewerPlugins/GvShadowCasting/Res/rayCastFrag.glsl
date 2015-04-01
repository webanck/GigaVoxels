////////////////////////////////////////////////////////////////////////////////
//
// FRAGMENT SHADER
//
// GigaVoxels Renderer
// - ...
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// VERSION
////////////////////////////////////////////////////////////////////////////////

#version 410

////////////////////////////////////////////////////////////////////////////////
// EXTENSION
////////////////////////////////////////////////////////////////////////////////

#extension GL_NV_gpu_shader5 : enable
#extension GL_EXT_shader_image_load_store : enable

////////////////////////////////////////////////////////////////////////////////
// INCLUDE
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// UNIFORM
////////////////////////////////////////////////////////////////////////////////

// Uniform parameters of the view
uniform vec3 uViewPos;
uniform vec3 uViewPlane;
uniform vec3 uViewAxisX;
uniform vec3 uViewAxisY;

// Uniform parameters of the caches
uniform uvec3 uBrickCacheSize;

// Uniform parameters of the pools
uniform vec3 uBrickPoolResInv;

uniform layout(size1x32) uimageBuffer uUpdateBufferArray;
uniform layout(size1x32) uimageBuffer uNodeTimeStampArray;
uniform layout(size1x32) uimageBuffer uBrickTimeStampArray;
uniform uint uCurrentTime;

uniform uint uMaxDepth;

// Sampler 3D to access user defined voxels data
uniform sampler3D uDataPool;

// ...
uniform layout(size1x32) uimageBuffer uNodePoolChildArray;
uniform layout(size1x32) uimageBuffer uNodePoolDataArray;

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
uint unpackNodeAddress( uint pAddress );
uvec3 unpackBrickAddress( uint pAddress );
void fetchNode( out Node pNode, uint pNodeTileAddress, uint pNodeTileOffset );
bool nodeIsInitialized( Node pNode );
bool nodeHasSubNodes( Node pNode );
bool nodeIsBrick( Node pNode );
bool nodeHasBrick( Node pNode );
bool nodeIsTerminal( Node pNode );
uint nodeGetChildAddress( Node pNode );
uvec3 nodeGetBrickAddress( Node pNode );

// Data Production Management
void setNodeUsage( uint pAddress );
void setBrickUsage( uvec3 pAddress );
void cacheLoadRequest( uint pNodeAddressEnc );
void cacheSubdivRequest( uint pNodeAddressEnc );

// Rendering Management
bool intersectBox( vec3 pRayStart, vec3 pRayDir, vec3 boxmin, vec3 boxmax, out float tnear, out float tfar );
float getRayNodeLength( vec3 posInNode, float nsize, vec3 pRayDir );
uint nodeVisitor_visit( vec3 pos, uint maxDepth, 
					out uint nodeIdx, out uint nodeIdxPrev,  //! numData: number of point samples linked/ or address of the brick
					out vec3 nodePos, out vec3 nodePosPrev, //Prev== parent node
					out float nodeSize );
vec4 sampleBrick( uint brickIdxEnc, vec3 samplePos, vec3 nodePos, float nodeSize );
vec4 sampleMipMapInterp( float maxDepthF, float maxDepthNew/*, float depth*/,
						uint numData, uint numDataPrev, vec3 samplePos,
						vec3 nodePos, float nodeSize, vec3 nodePosPrev );
vec4 traceVoxelConeRayCast( vec3 pRayStart, vec3 pRayDir, float t, float pConeFactor, float tMax );

////////////////////////////////////////////////////////////////////////////////
// Unpack a node address
//
// @param pAddress node address
//
// @return the packed node address
////////////////////////////////////////////////////////////////////////////////
uint unpackNodeAddress( uint pAddress )
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
uvec3 unpackBrickAddress( uint pAddress )
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
void fetchNode( out Node pNode, uint pNodeTileAddress, uint pNodeTileOffset )
{
	// Load a single texel from an image
	pNode.childAddress = imageLoad( uNodePoolChildArray, int( pNodeTileAddress + pNodeTileOffset ) ).x;

	// Load a single texel from an image
	pNode.brickAddress = imageLoad( uNodePoolDataArray, int( pNodeTileAddress + pNodeTileOffset ) ).x;
}

////////////////////////////////////////////////////////////////////////////////
// ...
//
// @param pNode ...
//
// @return ...
////////////////////////////////////////////////////////////////////////////////
bool nodeIsInitialized( Node pNode )
{
	return ( pNode.childAddress != 0 ) || ( pNode.brickAddress != 0 );
}

////////////////////////////////////////////////////////////////////////////////
// ...
//
// @param pNode ...
//
// @return ...
////////////////////////////////////////////////////////////////////////////////
bool nodeHasSubNodes( Node pNode )
{
	return ( pNode.childAddress & 0x3FFFFFFFU ) != 0;
}

////////////////////////////////////////////////////////////////////////////////
// ...
//
// @param pNode ...
//
// @return ...
////////////////////////////////////////////////////////////////////////////////
bool nodeIsBrick( Node pNode )
{
	return ( pNode.childAddress & 0x40000000U ) != 0;
}

////////////////////////////////////////////////////////////////////////////////
// ...
//
// @param pNode ...
//
// @return ...
////////////////////////////////////////////////////////////////////////////////
bool nodeHasBrick( Node pNode )
{
	return nodeIsBrick( pNode ) && pNode.brickAddress != 0;
}

////////////////////////////////////////////////////////////////////////////////
// ...
//
// @param pNode ...
//
// @return ...
////////////////////////////////////////////////////////////////////////////////
bool nodeIsTerminal( Node pNode )
{
	return ( pNode.childAddress & 0x80000000U ) != 0;
}

////////////////////////////////////////////////////////////////////////////////
// ...
//
// @param pNode ...
//
// @return ...
////////////////////////////////////////////////////////////////////////////////
uint nodeGetChildAddress( Node pNode )
{
	return unpackNodeAddress( pNode.childAddress );
}

////////////////////////////////////////////////////////////////////////////////
// ...
//
// @param pNode ...
//
// @return ...
////////////////////////////////////////////////////////////////////////////////
uvec3 nodeGetBrickAddress( Node pNode )
{
	return unpackBrickAddress( pNode.brickAddress );
}

////////////////////////////////////////////////////////////////////////////////
// Update timestamp usage information of a node tile
// with current time (i.e. current rendering pass)
// given its address in the node pool.
//
// @param pAddress The address of the node for which we want to update usage information
////////////////////////////////////////////////////////////////////////////////
void setNodeUsage( uint pAddress )
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
void setBrickUsage( uvec3 pAddress )
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
void cacheLoadRequest( uint pNodeAddressEnc )
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
void cacheSubdivRequest( uint pNodeAddressEnc )
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
// @param tnear ...
// @param tfar ...
//
// @return a flag telling wheter or not an intersection occurs
////////////////////////////////////////////////////////////////////////////////
bool intersectBox( vec3 pRayStart, vec3 pRayDir, vec3 boxmin, vec3 boxmax, out float tnear, out float tfar )
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
// getRayNodeLength() function
//
// @param posInNode ...
// @param nsize node size
// @param pRayDir ray direction
//
// @return ...
////////////////////////////////////////////////////////////////////////////////
float getRayNodeLength( vec3 posInNode, float nsize, vec3 pRayDir )
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
    bool hit = intersectBox( posInNode, pRayDir, vec3( 0.0f ), vec3( nsize ), boxInterMin, boxInterMax );

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
// This function is equivalent to the GigaVoxels rendererDescentOctree() function.
//
// @param pPosition ...
// @param pMaxDepth ...
// @param pNodeIdx ...
// @param pNodeIdxPrev ...
// @param pNodePosition ...
// @param pNodePositionPrev ...
// @param pNodeSize ...
//
// @return ...
////////////////////////////////////////////////////////////////////////////////
uint nodeVisitor_visit( vec3 pPosition, uint pMaxDepth, 
					out uint pNodeIdx, out uint pNodeIdxPrev,  //! numData: number of point samples linked/ or address of the brick
					out vec3 pNodePosition, out vec3 pNodePositionPrev, //Prev== parent node
					out float pNodeSize )
{
	// Start from root node (first nodetile is unused to be able to use "NULL" address)
	uint rootIdx = 8;	// getNodeIdxInit();	// (hard-coded...)

    pNodePosition = vec3( 0.0 );
    pNodeSize = 2.0;
    float nodeSizeInv = 1.0 / pNodeSize;

	uint curIdx = 0;
	uint prevIdx = 0;
	uint prevPrevIdx = 0;

	vec3 prevNodePos = vec3( 0.0f );
	vec3 prevPrevNodePos = vec3( 0.0f );

	uint nodeChildIdx = rootIdx;

	uint depth = 0;	// getNodeDepthInit();

	Node node;

	// Visit the data structure
	while ( nodeChildIdx != 0 && depth <= pMaxDepth && depth <= uMaxDepth )
    //do
	{
		// Each time, modify size to children ones
		//
		// - here, for octree, divide by 2
		pNodeSize = pNodeSize * 0.5;
		nodeSizeInv = nodeSizeInv * 2.0;

		// For a given node, find in which child we are
		uvec3 curOffsetI = uvec3( ( pPosition - pNodePosition ) * nodeSizeInv );
		// Linearise offset (3D to 1D)
		uint curOffset = curOffsetI.x + curOffsetI.y * 2 + curOffsetI.z * 4;

		prevPrevIdx = prevIdx;
		prevIdx = curIdx;
		curIdx = nodeChildIdx + curOffset;

		// Fetch the node
		fetchNode( node, nodeChildIdx, curOffset );

		// Flag current node tile as used
		setNodeUsage( nodeChildIdx );
		
		// Flag current brick as used
		if ( nodeHasBrick( node ) )
		{
			setBrickUsage( nodeGetBrickAddress( node ) );
		}

        prevPrevNodePos = prevNodePos;
		prevNodePos = pNodePosition;

		// Update bottom left corner of node
        pNodePosition = pNodePosition + pNodeSize * vec3( curOffsetI );

		// Update current depth
		// - next LOD (level of detail)
		depth++;

#if 1	// Low resolution first

		if ( ! nodeIsInitialized( node ) || ( nodeIsBrick( node ) && !nodeHasBrick( node ) ) )
		{
			// Flag node to request a data production (i.e load)
			cacheLoadRequest( curIdx );
		}
		else if ( !nodeHasSubNodes( node ) && !nodeIsTerminal( node ) /*&& depth <= pMaxDepth*/ && depth <= uMaxDepth )
		{
			// Flag node to request a node subdivision
			cacheSubdivRequest( curIdx );
		}

#else	// High resolution immediatly

#endif

		// Update base nodetile address
		nodeChildIdx = nodeGetChildAddress( node );
	}
    //while ( nodeChildIdx != 0 && depth < pMaxDepth && depth < uMaxDepth );

    pNodeIdx = curIdx;
    pNodeIdxPrev = prevIdx;

    pNodePositionPrev = prevNodePos;

	return curIdx;
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
 vec4 sampleBrick( uint brickIdxEnc, vec3 samplePos, vec3 nodePos, float pNodeSize )
 {
	// Unpack the brick address to obtain localization code
	uvec3 brickIdx;
	brickIdx.x = ( brickIdxEnc & 0x3FF00000U ) >> 20U;
	brickIdx.y = ( brickIdxEnc & 0x000FFC00U ) >> 10U;
	brickIdx.z = brickIdxEnc & 0x000003FFU;

	// FIXME : why is it working with 0.0 and not 1.0 ?
	float usedBrickSize = 8.0;	// float( VOXEL_POOL_BRICK_RES - VOXEL_POOL_BRICK_BORDER );

	// Compute texture coordinates to sample data
	vec3 brickPos = vec3( brickIdx );
	vec3 posInNode = ( samplePos - nodePos ) / pNodeSize;
	vec3 samplePosBrick = posInNode * usedBrickSize;

	// Sample data
	return texture( uDataPool, ( brickPos + samplePosBrick ) * uBrickPoolResInv );
}

////////////////////////////////////////////////////////////////////////////////
// ...
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
// @return ...
////////////////////////////////////////////////////////////////////////////////
vec4 sampleMipMapInterp( float maxDepthF, float maxDepthNew/*, float depth*/,
						uint numData, uint numDataPrev, vec3 samplePos,
						vec3 nodePos, float pNodeSize, vec3 nodePosPrev )
{

 	float quadInterp;
	quadInterp = fract( maxDepthF );

	// Sample data pool (i.e bricks of voxels) at given position
	vec4 sampleVal = sampleBrick( numData, samplePos, nodePos, pNodeSize );
	vec4 sampleValParent = vec4( 0. );

	// Check if there is a parent with data
	if ( numDataPrev != 0 )
	{
		// Sample data pool at same position but at the previous level of resolution (i.e. coarser one)
		sampleValParent = sampleBrick( numDataPrev, samplePos, nodePosPrev, pNodeSize * 2.0f );
	}

	//return mix( sampleValParent, sampleVal, quadInterp );
	return sampleVal;
	//return sampleValParent;
}

////////////////////////////////////////////////////////////////////////////////
// renderVolTree_Std
//
// @param pRayStart ray start
// @param pRayDir ray direction
// @param t ...
// @param pConeFactor ...
// @param tMax ...
//
// @return ...
////////////////////////////////////////////////////////////////////////////////
vec4 traceVoxelConeRayCast( vec3 pRayStart, vec3 pRayDir, float t, float pConeFactor, float tMax )
{
    float tTree = t;

    // XXX: really needed ? We don't put the address (0,0,0) in the cache's list anyway.
    setNodeUsage( 0 );

	// Update current position along the ray
	// - i.e. start position
    vec3 samplePosTree = pRayStart + tTree * pRayDir;

	// Pre-shade
    vec4 accColor = vec4( 0. );

    uint numLoop = 0;

    float voxelSize = 0.0;

	// Advance along the ray, node by node
    while ( tTree < tMax && /*stop criteria*/accColor.a < 0.99 && numLoop < 200 )
    {
   		uint nodeIdx;
		uint nodeIdxPrev;
		float nodeSize;
		vec3 nodePos;
		vec3 nodePosPrev;
		uint depth;

        // Update constants
        voxelSize = tTree * pConeFactor;

        // log( 1.0 / x ) = -log( x )
		float maxDepthF = -log2( voxelSize );
		uint maxDepth = (uint)ceil( maxDepthF );

        // Node visitor
		//
		// - traverse the data structure
		// - emits requests if required
        nodeVisitor_visit( samplePosTree, maxDepth, nodeIdx, nodeIdxPrev, nodePos, nodePosPrev, nodeSize );

        vec3 posInNode = samplePosTree - nodePos;

        float nodeLength = getRayNodeLength( posInNode, nodeSize, pRayDir );

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
			// Ray step
            float tStep = ( nodeSize / 8.0f ) * 0.66f;	// 2/3 size of one voxel (hard-coded)
            float tEnd = tTree + nodeLength;

		    while ( tTree < tEnd && /*stop criteria*/accColor.a < 0.99 )
            {
				// Update current position along ray
                samplePosTree = pRayStart + tTree * pRayDir;

                // Update constants
                voxelSize = tTree * pConeFactor;
                maxDepthF = -log2( voxelSize );
				uint maxDepthNew = (uint)ceil( maxDepthF );

                // Stop the raymarching if the two depth does not match.
       			//if ( maxDepthNew != maxDepth )
				//{
                //   break;
				//}

                // Sample the brick
                vec4 color = sampleBrick( brickAddress, samplePosTree, nodePos, nodeSize );
				// vec4 color = sampleMipMapInterp( maxDepthF, maxDepthNew/*, depth*/,
                //   brickAddress, brickAddressPrev, samplePosTree, nodePos, nodeSize, nodePosPrev );

                // Lighting
#if 1
                vec3 grad = vec3( 0.f );

                float gradStep = tStep * 0.25f;

                vec4 v0;
                vec4 v1;

                v0 = sampleBrick( brickAddress, samplePosTree + vec3( gradStep, 0.0f, 0.0f ), nodePos, nodeSize );
                v1 = sampleBrick( brickAddress, samplePosTree - vec3( gradStep, 0.0f, 0.0f ), nodePos, nodeSize );
                grad.x = v0.w - v1.w;

                v0 = sampleBrick( brickAddress, samplePosTree + vec3( 0.0f, gradStep, 0.0f ), nodePos, nodeSize );
                v1 = sampleBrick( brickAddress, samplePosTree - vec3( 0.0f, gradStep, 0.0f ), nodePos, nodeSize );
                grad.y = v0.w - v1.w;

                v0 = sampleBrick( brickAddress, samplePosTree + vec3( 0.0f, 0.0f, gradStep ), nodePos, nodeSize );
                v1 = sampleBrick( brickAddress, samplePosTree - vec3( 0.0f, 0.0f, gradStep ), nodePos, nodeSize );
                grad.z = v0.w - v1.w;

                grad = -grad;
                grad = normalize( grad );

                vec3 lightVec = normalize( vec3( 1. ) - samplePosTree );
                vec3 viewVec = -pRayDir;

                color.rgb = color.rgb * max( 0., dot( grad, lightVec ) );
#endif

				// Accumulate color
                accColor = accColor + ( 1.0 - accColor.a ) * color;

				// Update current distance
                tTree = tTree + tStep;
            }
        }
        else
        {
			// Here node is either empty either its associated brick is not yet in cache

			// Update current distance at end of the node
            tTree = tTree + nodeLength;

			// Epsilon shifting...
            tTree = tTree + 1. / 512.;
        }
        
		// Update current position along the ray
        samplePosTree = pRayStart + tTree * pRayDir;

		// Update loop counter
        numLoop++;
    }

	return accColor;
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
	bool hit = intersectBox( uViewPos, viewDir, boxMin, boxMax, tnear, tfar );

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

	// Hardcode things for testing purposes
	//
	// radians() convert a quantity in degrees to radians
	float lodLevelCone = tan( radians( 1.0 * 10.0 ) ) * 2.0 * 0.01;

	float initT = tnear + /*espilon*/1.0 / 512.0;
	float coneFactor = max( 1.0 / 2048.0, lodLevelCone );

	// Launch voxel cone tracing algorithm
    oColor = traceVoxelConeRayCast( uViewPos, viewDir, initT, coneFactor, tfar );
}
