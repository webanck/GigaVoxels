#version 410
#extension GL_NV_gpu_shader5 : enable
#extension GL_EXT_shader_image_load_store : enable

//#include "printf.hglsl"
#include "volumeTree.hglsl"
//#include "volumeTreeCache.hglsl"

layout(location = 0) in vec3 fragViewDir;

out vec4 fragOutColor;

uniform vec3 viewPos;

uniform uvec3 nodeCacheSize;
uniform uvec3 brickCacheSize;

uniform vec3 nodePoolResInv;
uniform vec3 brickPoolResInv;

uniform layout(size1x32) uimageBuffer d_updateBufferArray;
uniform layout(size1x32) uimageBuffer d_nodeTimeStampArray;
uniform layout(size1x32) uimageBuffer d_brickTimeStampArray;
uniform uint k_currentTime;

uniform uint maxVolTreeDepth;
//#define maxVolTreeDepth 6

uniform sampler3D dataPool;

void setNodeUsage(uint address)
{
	// FIXME : divide by elemRes or >> if POT
	uint elemOffset = address / 8;

	imageStore(d_nodeTimeStampArray, int(elemOffset), uvec4(k_currentTime));
}

void setBrickUsage(uvec3 address)
{
	// FIXME : divide by elemRes or >> if POT
	uvec3 elemOffset = address / 10;

	uint elemOffsetLinear =
		elemOffset.x + elemOffset.y * brickCacheSize.x +
		elemOffset.z * brickCacheSize.x * brickCacheSize.y;

	imageStore(d_brickTimeStampArray, int(elemOffsetLinear), uvec4(k_currentTime));
}

void cacheLoadRequest(uint nodeAddressEnc)
{
	imageStore(d_updateBufferArray, int(nodeAddressEnc), uvec4((nodeAddressEnc & 0x3FFFFFFFU) | 0x80000000U));
}

void cacheSubdivRequest(uint nodeAddressEnc)
{
	imageStore(d_updateBufferArray, int(nodeAddressEnc), uvec4((nodeAddressEnc & 0x3FFFFFFFU) | 0x40000000U));
}

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
bool intersectBox(vec3 rayStart, vec3 rayDir, vec3 boxmin, vec3 boxmax, out float tnear, out float tfar)
{
	// compute intersection of ray with all six bbox planes
	vec3 invR = vec3(1.0f) / rayDir;
	vec3 tbot = invR * (boxmin - rayStart);
	vec3 ttop = invR * (boxmax - rayStart);

	// re-order intersections to find smallest and largest on each axis
	vec3 tmin = min(ttop, tbot);
	vec3 tmax = max(ttop, tbot);

	// find the largest tmin and the smallest tmax
	float largest_tmin = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
	float smallest_tmax = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));

	tnear = largest_tmin;
	tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

float getRayNodeLength(vec3 posInNode, float nsize, vec3 rayDir)
{
#if 1
    vec3 directions = step(0.0, rayDir); //To precompute somewhere

    vec3 miniT;
    vec3 miniraypos=posInNode;
    vec3 planes=directions*nsize;
    miniT=(planes-miniraypos)/rayDir;

    return min(miniT.x, min(miniT.y, miniT.z));//*length(ray);
#else
    float boxInterMin=0.0f; float boxInterMax=0.0f;
    bool hit=intersectBox(posInNode, rayDir, vec3(0.0f), vec3(nsize), boxInterMin, boxInterMax);

	if(hit)
		return boxInterMax-boxInterMin;
	else
		return 1.0f/2048.0f;

#endif
}

// rendererDescentOctree
uint octreeTextureTraversal2(vec3 pos, uint maxDepth, 
					out uint nodeIdx, out uint nodeIdxPrev,  //! numData: number of point samples linked/ or address of the brick
					out vec3 nodePos, out vec3 nodePosPrev, //Prev== parent node
					out float nodeSize)
{
	uint rootIdx = 8;//getNodeIdxInit();

    nodePos = vec3(0.0);
    nodeSize = 2.0;
    float nodeSizeInv = 1.0 / nodeSize;

	uint curIdx = 0;
	uint prevIdx = 0;
	uint prevPrevIdx = 0;

	vec3 prevNodePos = vec3(0.0f);
	vec3 prevPrevNodePos = vec3(0.0f);

	uint nodeChildIdx = rootIdx;

	uint depth = 0;//getNodeDepthInit();

	OctreeNode node;

	while (nodeChildIdx != 0 && depth < maxDepth && depth < maxVolTreeDepth)
	{
		nodeSize = nodeSize * 0.5;
		nodeSizeInv = nodeSizeInv * 2.0;

		uvec3 curOffsetI = uvec3((pos - nodePos) * nodeSizeInv);
		uint curOffset = curOffsetI.x + curOffsetI.y * 2 + curOffsetI.z * 4;

		prevPrevIdx = prevIdx;
		prevIdx = curIdx;
		curIdx = (nodeChildIdx & 0x3FFFFFFFU) + curOffset;

		// fetch the node
		fetchOctreeNode(node, nodeChildIdx, curOffset);

		// mark the node tile used
		setNodeUsage(nodeChildIdx);
		// mark the brick used
		if (nodeHasBrick(node))
			setBrickUsage(nodeGetBrickAddress(node));

        prevPrevNodePos = prevNodePos;
		prevNodePos = nodePos;

        nodePos = nodePos + vec3(curOffsetI) * nodeSize;

		// next depth
		depth++;

#if 1 //Low res first
		if (!nodeIsInitialized(node) || (nodeIsBrick(node) && !nodeHasBrick(node)))
		{
			cacheLoadRequest(curIdx);
		}
		else if (!nodeHasSubNodes(node) && !nodeIsTerminal(node) && depth < maxDepth && depth < maxVolTreeDepth)
		{
			cacheSubdivRequest(curIdx);
		}
#else //High res immediatly
#endif
		nodeChildIdx = nodeGetChildAddress(node);
	}

	if (depth == maxDepth || depth == maxVolTreeDepth)
	{
		nodeIdx = curIdx;
		nodeIdxPrev = prevIdx;

		//nodePos = ;
		nodePosPrev = prevNodePos;
		//nodeSize = ;
	}
	else
	{
		nodeIdx = prevIdx;
		nodeIdxPrev = prevPrevIdx;

		nodePos = prevNodePos;
		nodePosPrev = prevPrevNodePos;

		nodeSize = nodeSize * 2.0f;
	}

	return curIdx;
 }

 vec4 sampleBrick(uint brickIdxEnc, vec3 samplePos, vec3 nodePos, float nodeSize)
 {
	vec4 sampleVal = texture(dataPool, vec3(0.5));//vec4(0.);

	// Unpack the brick address
	uvec3 brickIdx;
	brickIdx.x = (brickIdxEnc & 0x3FF00000U) >> 20U;
	brickIdx.y = (brickIdxEnc & 0x000FFC00U) >> 10U;
	brickIdx.z = brickIdxEnc & 0x000003FFU;

	// FIXME: why is it working with 0.0 and not 1.0 ?
	float texelOffset = 0.0;//1.0;
	float usedBrickSize = 8.0;//float(VOXEL_POOL_BRICK_RES-VOXEL_POOL_BRICK_BORDER);

	vec3 brickPos = vec3(brickIdx);// +vec3(texelOffset);
	vec3 posInNode = (samplePos - nodePos) / nodeSize;
	vec3 samplePosBrick = posInNode * usedBrickSize;

	vec3 voxelPos = brickPos + samplePosBrick;

	if (brickIdxEnc > 0)
	{
        sampleVal = texture(dataPool, voxelPos / vec3(320.));
        //sampleVal = vec4(1., 0., 0., 1.);//texture(dataPool, samplePos);
        /*
		vec3 posF = voxelPos;
		uvec3 posI = uvec3(floor(posF));
		vec2 samplePos0;
		vec2 samplePos1;
		samplePos0.x = posF.x + 406.0f * (posI.z % 21);
		samplePos0.y = posF.y + 406.0f * (posI.z / 21);
		samplePos1.x = posF.x + 406.0f * ((posI.z + 1) % 21);
		samplePos1.y = posF.y + 406.0f * ((posI.z + 1) / 21);
		vec4 p0 = texture(dataPool, samplePos0);
		vec4 p1 = texture(dataPool, samplePos1);
		sampleVal = mix(p0, p1, posF.z - (float)posI.z);
		//sampleVal = p0;
        */
	}

	return sampleVal;
 }

vec4 sampleMipMapInterp(float maxDepthF, float maxDepthNew, float depth,
						uint numData, uint numDataPrev, vec3 samplePos,
						vec3 nodePos, float nodeSize, vec3 nodePosPrev)
{

 	float quadInterp;
	quadInterp = fract(maxDepthF);

	vec4 sampleVal = sampleBrick(numData, samplePos, nodePos, nodeSize);
	vec4 sampleValParent = vec4(0.);

	if (numDataPrev != 0)
		sampleValParent = sampleBrick(numDataPrev, samplePos, nodePosPrev, nodeSize * 2.0f);

	return mix(sampleValParent, sampleVal, quadInterp);
	//return sampleVal;
	//return sampleValParent;
}

// renderVolTree_Std
//vec4 traceVoxelConeRayCast1(vec3 rayStart, vec3 rayDir, float t, float tMax)
vec4 traceVoxelConeRayCast1(vec3 rayStart, vec3 rayDir, float t, float coneFactor, float tMax)
{
    float tTree = t;

    // XXX: really needed ? We don't put the address (0,0,0) in the cache's list anyway.
    setNodeUsage(0);

    vec3 samplePosTree = rayStart + tTree * rayDir;

    vec4 accColor = vec4(0.);

    uint numLoop = 0;

    float voxelSize = 0.0;

    while (tTree < tMax && accColor.a < 0.99 && numLoop < 500)
    {
   		uint nodeIdx;
		uint nodeIdxPrev;
		float nodeSize;
		vec3 nodePos;
		vec3 nodePosPrev;
		uint depth;

        // update constants
        voxelSize = tTree * coneFactor;

        // log(1.0 / x) = -log(x)
		float maxDepthF = -log2(voxelSize);
		uint maxDepth = (uint)ceil(maxDepthF);

        // traverse the tree
        octreeTextureTraversal2(samplePosTree, maxDepth, nodeIdx, nodeIdxPrev, nodePos, nodePosPrev, nodeSize);

        vec3 posInNode = samplePosTree - nodePos;
        
        float nodeLength = getRayNodeLength(posInNode, nodeSize, rayDir);
        
        uint brickAddress = 0;
		uint brickAddressPrev = 0;

   		if (nodeIdx != 0)
            brickAddress = imageLoad(d_volTreeDataArray, int(nodeIdx)).x;
		if (nodeIdxPrev != 0)
            brickAddressPrev = imageLoad(d_volTreeDataArray, int(nodeIdxPrev)).x;

        // traverse the brick
        if (brickAddress != 0)
        {
            //float tStep = (nodeSize / 8.0f) * 0.66f;
        	float tStep = (nodeSize / 8.0f) * 0.33f;
            float tEnd = tTree + nodeLength;

            while (tTree < tEnd && accColor.a < 0.99)
            {
                samplePosTree = rayStart + tTree * rayDir;

                // update constants
                voxelSize = tTree * coneFactor;
                maxDepthF = -log2(voxelSize);

                uint maxDepthNew = (uint)ceil(maxDepthF);

                // stop the raymarching if the two depth does not match.
       			if (maxDepthNew != maxDepth)
                    break;

                // sample the brick
                vec4 color = sampleBrick(brickAddress, samplePosTree, nodePos, nodeSize);
                //vec4 color = sampleMipMapInterp(maxDepthF, maxDepthNew, depth,
                //    brickAddress, brickAddressPrev, samplePosTree, nodePos, nodeSize, nodePosPrev);

                // lighting
#if 1
                vec3 grad = vec3(0.f);

                float gradStep = tStep * 0.25f;

                vec4 v0;
                vec4 v1;

                v0 = sampleBrick(brickAddress, samplePosTree + vec3(gradStep, 0.0f, 0.0f), nodePos, nodeSize);
                v1 = sampleBrick(brickAddress, samplePosTree - vec3(gradStep, 0.0f, 0.0f), nodePos, nodeSize);
                grad.x = v0.w - v1.w;

                v0 = sampleBrick(brickAddress, samplePosTree + vec3(0.0f, gradStep, 0.0f), nodePos, nodeSize);
                v1 = sampleBrick(brickAddress, samplePosTree - vec3(0.0f, gradStep, 0.0f), nodePos, nodeSize);
                grad.y = v0.w - v1.w;

                v0 = sampleBrick(brickAddress, samplePosTree + vec3(0.0f, 0.0f, gradStep), nodePos, nodeSize);
                v1 = sampleBrick(brickAddress, samplePosTree - vec3(0.0f, 0.0f, gradStep), nodePos, nodeSize);
                grad.z = v0.w - v1.w;

                grad = -grad;
                grad = normalize(grad);

                vec3 lightVec = normalize(vec3(1.) - samplePosTree);
                vec3 viewVec = -rayDir;

                color.rgb = color.rgb * max(0., dot(grad, lightVec));
#endif

                accColor = accColor + (1.0 - accColor.a) * color;
                tTree = tTree + tStep;
            }
        }
        else
        {
            tTree = tTree + nodeLength;
        }
        
        samplePosTree = rayStart + tTree * rayDir;
        numLoop++;
    }

    return accColor;
}

// renderVolTree_Std
vec4 traceVoxelConeRayCast2(vec3 rayStart, vec3 rayDir, float initT, float coneFactor, float maxRayLength)
{
	// already done before calling this function
	//coneFactor=max(coneFactor, 1.0f/2048.0f);

	//Keep root node in cache
	setNodeUsage(0);

	float t = initT;

	//samplePos = clamp(samplePos, 0.001, 0.999);
	//vec3 samplePos = clamp(rayStart + t * rayDir, 0.001, 0.999);
    vec3 samplePos = rayStart + t * rayDir;

	vec4 accColor = vec4(0.);
	vec4 col = vec4(0.);

	bool loopcr = true;

	float voxelSize = 0.0;

	int i=0;

	while (loopcr)
	{
		voxelSize = t * coneFactor;

		float maxDepthF = log2(1.0 / voxelSize);
		uint maxDepth = (uint)ceil(maxDepthF);

		uint nodeIdx;
		uint nodeIdxPrev;
		float nodeSize=0.5;
		vec3 nodePos=vec3(0.0);
		vec3 nodePosPrev=vec3(0.0);
		uint depth;

		/*uint curIdx = */octreeTextureTraversal2(samplePos, maxDepth, nodeIdx, nodeIdxPrev, nodePos, nodePosPrev, nodeSize, depth);

		uint numData=0;
		uint numDataPrev=0;

		if(nodeIdx!=0)
			numData = imageLoad(d_volTreeDataArray, int(nodeIdx)).x;
		if(nodeIdxPrev!=0)
			numDataPrev = imageLoad(d_volTreeDataArray, int(nodeIdxPrev)).x;

		vec3 posInNode=samplePos-nodePos;
		float nodeLength=getRayNodeLength(posInNode, nodeSize, rayDir);
		float tStep=(nodeSize / 8.0f) * 0.33f;
		float tEnd=t+nodeLength;

		/*
		if (gl_FragCoord.x == 255.5f && gl_FragCoord.y == 255.5f)
		{
			putFloat(posInNode.x);
			putFloat(posInNode.y);
			putFloat(posInNode.z);
			putFloat(nodeSize);
			putFloat(tStep);
			putFloat(tEnd);
			putFloat(nodeLength);
		}
		*/

        // rendererBrickSampling
		do
		{
			voxelSize= t*coneFactor;
			maxDepthF=log2(1.0f/voxelSize);
			uint maxDepthNew=(uint)ceil(maxDepthF);

			if(maxDepthNew!=maxDepth)
				break;

			vec4 sampleVal;

			sampleVal = sampleBrick(numData, samplePos, nodePos, nodeSize);
			//sampleVal = sampleMipMapInterp(maxDepthF, maxDepthNew, depth,
			//	numData, numDataPrev, samplePos, nodePos, nodeSize, nodePosPrev);
#if 0
			// lighting test
			vec3 grad = vec3(0.f);

			float gradStep = tStep * 0.25f;

			vec4 v0;
			vec4 v1;

			v0 = sampleBrick(numData, samplePos + vec3(gradStep, 0.0f, 0.0f), nodePos, nodeSize);
			v1 = sampleBrick(numData, samplePos - vec3(gradStep, 0.0f, 0.0f), nodePos, nodeSize);
			grad.x = v0.w - v1.w;

			v0 = sampleBrick(numData, samplePos + vec3(0.0f, gradStep, 0.0f), nodePos, nodeSize);
			v1 = sampleBrick(numData, samplePos - vec3(0.0f, gradStep, 0.0f), nodePos, nodeSize);
			grad.y = v0.w - v1.w;

			v0 = sampleBrick(numData, samplePos + vec3(0.0f, 0.0f, gradStep), nodePos, nodeSize);
			v1 = sampleBrick(numData, samplePos - vec3(0.0f, 0.0f, gradStep), nodePos, nodeSize);
			grad.z = v0.w - v1.w;

			grad = -grad;
			grad = normalize(grad);

			vec3 lightVec=normalize(vec3(1.)-samplePos);
			vec3 viewVec=(-rayDir);

			sampleVal.rgb = sampleVal.rgb * max(0., dot(grad, lightVec));
#endif

			vec4 colPrev=col;
			col=sampleVal;

			vec4 slabCol;
			float aNew;

			float crtn = (tStep)/( 1.0f / 512.0f * 1.5f ); //
			aNew = 1.0f - pow(1.0f - col.a, crtn);
			//aNew = 1.0 - pow(1.0 - col.a, tStep * 1024.0);

			if(col.a>0.0f)
			{
				slabCol.rgb=(col.rgb/col.a)*aNew;
			}
			else
				slabCol.rgb=vec3(0.0f);

			slabCol.a=aNew;

			accColor=accColor+(1.0-accColor.a)*slabCol;

			t=t+tStep;
			samplePos=rayStart+rayDir*t;
		}
		while(t<tEnd && accColor.a<0.99);

		loopcr= i<10000 && accColor.a<0.99 && t<maxRayLength;

		i++;
	}

	accColor=clamp(accColor, vec4(0.0f), vec4(1.0f));

	return accColor;
}

void main()
{
	vec3 viewDir = normalize(fragViewDir);

	//const vec3 boxMin = vec3(0.0, 0.0, 0.0);
	//const vec3 boxMax = vec3(1.0, 1.0, 1.0);
    const vec3 boxMin = vec3(0.001, 0.001, 0.001);
    const vec3 boxMax = vec3(0.999, 0.999, 0.999);

	float tnear, tfar;
	bool hit = intersectBox(viewPos, viewDir, boxMin, boxMax, tnear, tfar);

	if (!hit) return;
	if (tnear < 0.0)
		tnear = 0.0;	// clamp to near plane

	// Hardcode things for testing purposes
	float lodLevelCone=tan( radians(1.0 * 10.0) ) * 2.0 * 0.01;

	float initT = 1.0 / 512.0 + tnear;
	float coneFactor = max(1.0 / 2048.0, lodLevelCone);

	fragOutColor = traceVoxelConeRayCast1(viewPos, viewDir, initT, coneFactor, tfar);
}
