////////////////////////////////////////////////////////////////////////////////
//
// FRAGMENT SHADER
//
// On-the-fly voxelization
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// VERSION
////////////////////////////////////////////////////////////////////////////////

#version 420

// imageAtomicAdd()
// - atomically add a value to an existing value in memory and return the original value

////////////////////////////////////////////////////////////////////////////////
// Extensions
////////////////////////////////////////////////////////////////////////////////

#extension GL_ARB_shader_atomic_counters : enable
#extension GL_EXT_shader_image_load_store : enable
#extension GL_NV_shader_atomic_float : enable

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

// ...
in Data
{
	// ...
	vec3 normal;

	// ...
	vec3 worldPosition;

	// ...
	//vec4 aabb;

	// ...
	//vec2 clipSpace;

} iData;

////////////////////////////////////////////////////////////////////////////////
// UNIFORM
////////////////////////////////////////////////////////////////////////////////

// Atomic counter to know if something is in the brick
layout (binding = 0, offset = 0) uniform atomic_uint uCounter;

// Change-of-basis matrix : from world's base to brick's base
uniform mat4 uBrickMatrix;

// To write in the gigaVoxel dataPool
//uniform layout( rgba32f ) image3D uDataPool;
uniform layout( r32f ) coherent image3D uDataPoolx;
uniform layout( r32f ) coherent image3D uDataPooly;
uniform layout( r32f ) coherent image3D uDataPoolz;

// The brick address in the texture
//
// todo : uBrickAddress is not used directly, instead precompute "uBrickAddress * 10" to have the real brick osset in cache
// as cache is indexed by voxels
uniform ivec3 uBrickAddress;

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
////////////////////////////////////////////////////////////////////////////////
//void main()
//{
//	if ( atomicCounter( uCounter ) == 0 )
//	{
//		atomicCounterIncrement( uCounter );
//	}
//
//	// We compute voxel offset
//	ivec3 voxelOffset = ivec3( floor( 8 * ( uBrickMatrix * vec4( iData.worldPosition, 1.0 ) ) ).xyz );
//
//	// rmk : ivec3( 1, 1, 1 ) is the borderOffset
//	imageStore( uDataPool, uBrickAddress * 10 + voxelOffset + ivec3( 1, 1, 1 ), vec4( 1.0, 1.0, 1.0, 1.0 ) );
//}

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
////////////////////////////////////////////////////////////////////////////////
void main()
{
	//discard( iData.clipSpace.x < iData.aabb.x || iData.clipSpace.y < iData.aabb.y || iData.clipSpace.x > iData.aabb.z || iData.clipSpace.y > iData.aabb.w );

	if ( atomicCounter( uCounter ) == 0 )
	{
		atomicCounterIncrement( uCounter );
	}
	
	// We compute voxel offset
	vec3 brickCoord = /*nbElementsInBrick*/10.0 * ( uBrickMatrix * vec4( iData.worldPosition, 1.0 ) ).xyz;

	ivec3 voxelOffset;
	voxelOffset.xy = ivec2( floor( brickCoord.xy ) );
	voxelOffset.z = int( floor( brickCoord.z ) );

	// To know how many voxels we intersect
	//uint nbDepthVoxel = floor( max( dFdx( brickCoord.z ), dFdy( brickCoord.z ) ) + 1;
		
	// imageAtomicAdd()
	// - atomically add a value to an existing value in memory and return the original value
	//
	// "uBrickAddress" is the index of the brick in the cache (i.e. data pool)
	// so "uBrickAddress * 10" is the index of the equivalent voxel in the cache
	imageAtomicAdd( /*image*/uDataPoolx, /*coordinate*/uBrickAddress * 10 + voxelOffset, /*data*/iData.normal.x );
	imageAtomicAdd( /*image*/uDataPooly, /*coordinate*/uBrickAddress * 10 + voxelOffset, /*data*/iData.normal.y );
	imageAtomicAdd( /*image*/uDataPoolz, /*coordinate*/uBrickAddress * 10 + voxelOffset, /*data*/iData.normal.z );
}
