#ifndef _GV_GLSL_DATA_PRODUCTION_MANAGEMENT_H_
#define _GV_GLSL_DATA_PRODUCTION_MANAGEMENT_H_

////////////////////////////////////////////////////////////////////////////////
//
// GigaSpace SHADER library
//
// Data Production Management 
//
// - ...
//
////////////////////////////////////////////////////////////////////////////////

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
// Functions Declaration
////////////////////////////////////////////////////////////////////////////////

// Data Production Management
void gv_setNodeUsage( uint pAddress );
void gv_setBrickUsage( uvec3 pAddress );
void gv_cacheLoadRequest( uint pNodeAddressEnc );
void gv_cacheSubdivRequest( uint pNodeAddressEnc );

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

#endif // _GV_GLSL_DATA_PRODUCTION_MANAGEMENT_H_
