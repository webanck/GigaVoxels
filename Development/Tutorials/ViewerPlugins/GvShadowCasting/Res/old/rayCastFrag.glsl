#version 410
#extension GL_NV_gpu_shader5 : enable
#extension GL_EXT_shader_image_load_store : enable

#include "printf.hglsl"
#include "volumeTree.hglsl"
//#include "volumeTreeCache.hglsl"

layout(location = 0) in vec3 fragViewDir;

/**
 * The output of the shader program
 */
out vec4 fragOutColor;

/**
 * Uniform parameters of the view
 */
uniform vec3 viewPos;
uniform vec3 viewPlane;
uniform vec3 viewAxisX;
uniform vec3 viewAxisY;

/**
 * Uniform parameters of the caches
 */
uniform uvec3 nodeCacheSize;
uniform uvec3 brickCacheSize;

/**
 * Uniform parameters of the pools
 */
uniform vec3 nodePoolResInv;
uniform vec3 brickPoolResInv;

uniform layout(size1x32) uimageBuffer d_updateBufferArray;
uniform layout(size1x32) uimageBuffer d_nodeTimeStampArray;
uniform layout(size1x32) uimageBuffer d_brickTimeStampArray;
uniform uint k_currentTime;

uniform uint maxVolTreeDepth;
//#define maxVolTreeDepth 1

/**
 * Sampler 3D to access user defined voxels data.
 */
uniform sampler3D dataPool;

/******************************************************************************
 * Update timestamp usage information of a node tile
 * with current time (i.e. current rendering pass)
 * given its address in the node pool.
 *
 * @param pAddress The address of the node for which we want to update usage information
 ******************************************************************************/
void setNodeUsage( uint pAddress )
{
	// Specify the coordinate at which to store the texel.
	//
	// FIXME : divide by elemRes or >> if POT
	uint elemOffset = pAddress / 8;

	// Write a single texel into an image
	imageStore( d_nodeTimeStampArray, int( elemOffset ), uvec4( k_currentTime ) );
}

/******************************************************************************
 * Update timestamp usage information of a brick
 * with current time (i.e. current rendering pass)
 * given its address in brick pool.
 *
 * @param pAddress The address of the node for which we want to update usage information
 ******************************************************************************/
void setBrickUsage( uvec3 pAddress )
{
	// Specify the coordinate at which to store the texel.
	//
	// FIXME : divide by elemRes or >> if POT
	uvec3 elemOffset = pAddress / 10;

	uint elemOffsetLinear =
		elemOffset.x + elemOffset.y * brickCacheSize.x +
		elemOffset.z * brickCacheSize.x * brickCacheSize.y;

	// Write a single texel into an image
	imageStore( d_brickTimeStampArray, int( elemOffsetLinear ), uvec4( k_currentTime ) );
}

/******************************************************************************
 * Update buffer with a load request for a given node.
 *
 * @param pNodeAddressEnc the encoded node address
 ******************************************************************************/
void cacheLoadRequest( uint pNodeAddressEnc )
{
	// Write a single texel into an image.
	//
	// GigaVoxels bit mask for load request is 0x80000000U (31th bit)
	imageStore( d_updateBufferArray, int( pNodeAddressEnc ), uvec4( ( pNodeAddressEnc & 0x3FFFFFFFU ) | 0x80000000U ) );
}

/******************************************************************************
 * Update buffer with a subdivision request for a given node.
 *
 * @param pNodeAddressEnc the encoded node address
 ******************************************************************************/
void cacheSubdivRequest( uint pNodeAddressEnc )
{
	// Write a single texel into an image.
	//
	// GigaVoxels bit mask for subdivision request is 0x40000000U (30th bit)
	imageStore( d_updateBufferArray, int( pNodeAddressEnc ), uvec4( ( pNodeAddressEnc & 0x3FFFFFFFU) | 0x40000000U ) );
}

/******************************************************************************
 * Intersect ray with a box.
 *
 * Algorithm comes from :
 * http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm*
 *
 * @param rayStart ...
 * @param rayDir ...
 * @param boxmin ...
 * @param boxmax ...
 * @param tnear ...
 * @param tfar ...
 *
 * @return ...
 ******************************************************************************/
bool intersectBox( vec3 rayStart, vec3 rayDir, vec3 boxmin, vec3 boxmax, out float tnear, out float tfar )
{
	// Compute intersection of ray with all six bbox planes
	vec3 invR = vec3( 1.0f ) / rayDir;
	vec3 tbot = invR * ( boxmin - rayStart );
	vec3 ttop = invR * ( boxmax - rayStart );

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

/******************************************************************************
 * getRayNodeLength() function
 *
 * @param posInNode ...
 * @param nsize ...
 * @param rayDir ...
 *
 * @return ...
 ******************************************************************************/
float getRayNodeLength( vec3 posInNode, float nsize, vec3 rayDir )
{
#if 1

    vec3 directions = step( 0.0, rayDir );	// To precompute somewhere

    vec3 miniT;
    vec3 miniraypos = posInNode;
    vec3 planes = directions * nsize;
    miniT = ( planes - miniraypos ) / rayDir;

    return min( miniT.x, min( miniT.y, miniT.z ) );	// * length( ray );

#else

	// Intersect ray with box
    float boxInterMin = 0.0f;
	float boxInterMax = 0.0f;
    bool hit = intersectBox( posInNode, rayDir, vec3( 0.0f ), vec3( nsize ), boxInterMin, boxInterMax );

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

/******************************************************************************
 * This function is equivalent to the GigaVoxels rendererDescentOctree() function.
 *
 * @param pos ...
 * @param maxDepth ...
 * @param nodeIdx ...
 * @param nodeIdxPrev ...
 * @param nodePos ...
 * @param nodePosPrev ...
 * @param nodeSize ...
 *
 * @return ...
 ******************************************************************************/
uint octreeTextureTraversal2( vec3 pos, uint maxDepth, 
					out uint nodeIdx, out uint nodeIdxPrev,  //! numData: number of point samples linked/ or address of the brick
					out vec3 nodePos, out vec3 nodePosPrev, //Prev== parent node
					out float nodeSize )
{
	uint rootIdx = 8;	// getNodeIdxInit();

    nodePos = vec3( 0.0 );
    nodeSize = 2.0;
    float nodeSizeInv = 1.0 / nodeSize;

	uint curIdx = 0;
	uint prevIdx = 0;
	uint prevPrevIdx = 0;

	vec3 prevNodePos = vec3( 0.0f );
	vec3 prevPrevNodePos = vec3( 0.0f );

	uint nodeChildIdx = rootIdx;

	uint depth = 0;	// getNodeDepthInit();

	OctreeNode node;

	while ( nodeChildIdx != 0 && depth <= maxDepth && depth <= maxVolTreeDepth )
    //do
	{
		nodeSize = nodeSize * 0.5;
		nodeSizeInv = nodeSizeInv * 2.0;

		uvec3 curOffsetI = uvec3( ( pos - nodePos ) * nodeSizeInv );
		uint curOffset = curOffsetI.x + curOffsetI.y * 2 + curOffsetI.z * 4;

		prevPrevIdx = prevIdx;
		prevIdx = curIdx;
		curIdx = nodeChildIdx + curOffset;

		// Fetch the node
		fetchOctreeNode( node, nodeChildIdx, curOffset );

		// Mark the current node tile as used
		setNodeUsage( nodeChildIdx );
		
		// Mark the current brick as used
		if ( nodeHasBrick( node ) )
		{
			setBrickUsage( nodeGetBrickAddress( node ) );
		}

        prevPrevNodePos = prevNodePos;
		prevNodePos = nodePos;

        nodePos = nodePos + nodeSize * vec3( curOffsetI );

		// Next LOD (level of detail)
		depth++;

#if 1	// Low res first

		if ( ! nodeIsInitialized( node ) || ( nodeIsBrick( node ) && !nodeHasBrick( node ) ) )
		{
			// Flag node to request a data production (i.e load)
			cacheLoadRequest( curIdx );
		}
		else if ( !nodeHasSubNodes( node ) && !nodeIsTerminal( node ) /*&& depth <= maxDepth*/ && depth <= maxVolTreeDepth )
		{
			// Flag node to request a node subdivision
			cacheSubdivRequest( curIdx );
		}

#else	// High res immediatly

#endif

		nodeChildIdx = nodeGetChildAddress( node );
	}
    //while ( nodeChildIdx != 0 && depth < maxDepth && depth < maxVolTreeDepth );

    nodeIdx = curIdx;
    nodeIdxPrev = prevIdx;

    nodePosPrev = prevNodePos;

	return curIdx;
 }

/******************************************************************************
 * Sample the data pool (i.e bricks of voxels) at a given position
 * to retrieve the underlying value.
 *
 * @param brickIdxEnc the current brick's encoded address in the data pool
 * @param samplePos the current sample's position
 * @param nodePos the current node's position
 * @param nodeSize the current node's size
 *
 * @return the sampled value at given position
 ******************************************************************************/
 vec4 sampleBrick( uint brickIdxEnc, vec3 samplePos, vec3 nodePos, float nodeSize )
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
	vec3 posInNode = ( samplePos - nodePos ) / nodeSize;
	vec3 samplePosBrick = posInNode * usedBrickSize;

	// Sample data
	return texture( dataPool, ( brickPos + samplePosBrick ) * brickPoolResInv );
}

/******************************************************************************
 * ...
 *
 * @param maxDepthF ...
 * @param maxDepthNew ...
 * @param depth ...
 * @param numData ...
 * @param numDataPrev ...
 * @param samplePos ...
 * @param nodePos ...
 * @param nodeSize ...
 * @param nodePosPrev ...
 *
 * @return ...
 ******************************************************************************/
vec4 sampleMipMapInterp( float maxDepthF, float maxDepthNew/*, float depth*/,
						uint numData, uint numDataPrev, vec3 samplePos,
						vec3 nodePos, float nodeSize, vec3 nodePosPrev )
{

 	float quadInterp;
	quadInterp = fract( maxDepthF );

	// Sample data pool (i.e bricks of voxels) at given position
	vec4 sampleVal = sampleBrick( numData, samplePos, nodePos, nodeSize );
	vec4 sampleValParent = vec4( 0. );

	// Check if there is a parent with data
	if ( numDataPrev != 0 )
	{
		// Sample data pool at same position but at the previous level of resolution (i.e. coarser one)
		sampleValParent = sampleBrick( numDataPrev, samplePos, nodePosPrev, nodeSize * 2.0f );
	}

	//return mix( sampleValParent, sampleVal, quadInterp );
	return sampleVal;
	//return sampleValParent;
}

/******************************************************************************
 * renderVolTree_Std
 *
 * @param rayStart ...
 * @param rayDir ...
 * @param t ...
 * @param tMax ...
 *
 * @return ...
 ******************************************************************************/
vec4 traceVoxelConeRayCast1( vec3 rayStart, vec3 rayDir, float t, float coneFactor, float tMax )
{
    float tTree = t;

    // XXX: really needed ? We don't put the address (0,0,0) in the cache's list anyway.
    setNodeUsage( 0 );

    vec3 samplePosTree = rayStart + tTree * rayDir;

    vec4 accColor = vec4(0.);

    uint numLoop = 0;

    float voxelSize = 0.0;

    while ( tTree < tMax && accColor.a < 0.99 && numLoop < 200 )
    {
   		uint nodeIdx;
		uint nodeIdxPrev;
		float nodeSize;
		vec3 nodePos;
		vec3 nodePosPrev;
		uint depth;

        // Update constants
        voxelSize = tTree * coneFactor;

        // log( 1.0 / x ) = -log( x )
		float maxDepthF = -log2( voxelSize );
		uint maxDepth = (uint)ceil( maxDepthF );

        // Traverse the tree
        octreeTextureTraversal2( samplePosTree, maxDepth, nodeIdx, nodeIdxPrev, nodePos, nodePosPrev, nodeSize );

        vec3 posInNode = samplePosTree - nodePos;

        float nodeLength = getRayNodeLength( posInNode, nodeSize, rayDir );

        uint brickAddress = 0;
		uint brickAddressPrev = 0;

   		if ( nodeIdx != 0 )
		{
			brickAddress = imageLoad( d_volTreeDataArray, int( nodeIdx ) ).x;
		}
		if ( nodeIdxPrev != 0 )
		{
			brickAddressPrev = imageLoad( d_volTreeDataArray, int( nodeIdxPrev ) ).x;
		}

        // Traverse the brick
        if ( brickAddress != 0 )
        {
            float tStep = ( nodeSize / 8.0f ) * 0.66f;
        	//float tStep = ( nodeSize / 8.0f ) * 0.33f;
            float tEnd = tTree + nodeLength;

		    while ( tTree < tEnd && accColor.a < 0.99 )
            {
                samplePosTree = rayStart + tTree * rayDir;

                // Update constants
                voxelSize = tTree * coneFactor;
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
                vec3 viewVec = -rayDir;

                color.rgb = color.rgb * max( 0., dot( grad, lightVec ) );
#endif

                accColor = accColor + ( 1.0 - accColor.a ) * color;
                tTree = tTree + tStep;
            }
        }
        else
        {
            tTree = tTree + nodeLength;
            tTree = tTree + 1. / 512.;
        }
        
        samplePosTree = rayStart + tTree * rayDir;
        numLoop++;
    }

	return accColor;
}

/******************************************************************************
 * Main fragment program.
 *
 * In GigaVoxels, ray casting algorithm is replaced by a custom
 * voxel cone tracing algorithm.
 *
 * The output is the color accumulated during ray traversal.
 ******************************************************************************/
void main()
{
	// Compute ray direction from view point through current pixel
	//vec3 viewDir = normalize( fragViewDir );
    vec3 viewDir = normalize( viewPlane + viewAxisX * gl_FragCoord.x + viewAxisY * gl_FragCoord.y );

	// Initialize the GigaVoxels bounding box to test ray casting intersection
	//const vec3 boxMin = vec3( 0.0, 0.0, 0.0 );
	//const vec3 boxMax = vec3( 1.0, 1.0, 1.0 );
    const vec3 boxMin = vec3( 0.001, 0.001, 0.001 );
    const vec3 boxMax = vec3( 0.999, 0.999, 0.999 );

	// Intersect current ray with box
	float tnear, tfar;
	bool hit = intersectBox( viewPos, viewDir, boxMin, boxMax, tnear, tfar );

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

	float initT = 1.0 / 512.0 + tnear;
	float coneFactor = max( 1.0 / 2048.0, lodLevelCone );

	// Launch voxel cone tracing algorithm
	//
	//vec3 samplePos = viewPos + tnear * viewDir;
	//fragOutColor = vec4( samplePos.xyz, 1. );
    fragOutColor = traceVoxelConeRayCast1( viewPos, viewDir, initT, coneFactor, tfar );
}
