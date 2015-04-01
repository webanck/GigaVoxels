////////////////////////////////////////////////////////////////////////////////
//
// FRAGMENT SHADER
//
// On the fly "voxelization "
// - signed distance field generation
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// VERSION
////////////////////////////////////////////////////////////////////////////////

#version 420
// - due to imageAtomicCompSwap()

////////////////////////////////////////////////////////////////////////////////
// EXTENSIONS
////////////////////////////////////////////////////////////////////////////////

#extension GL_EXT_shader_image_load_store : enable

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

// Input data structure
in Data
{
	// Interpolated distance to camera plan at a fragment
	// - orthographic projection
	float distMin;

	// Interpolated normal at a fragment
	vec3 normal;

} dataIn;

////////////////////////////////////////////////////////////////////////////////
// UNIFORM
////////////////////////////////////////////////////////////////////////////////

// z-slice index
// - values go between [ 0 ; 9 ] => i.e. brick size in cache
uniform int uSlice;

// To know along whitch axis we are working
uniform int uAxe;

uniform layout( r32ui ) coherent volatile uimage3D uDistance[ 3 ];

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Function that put distance in the texture, emulating atomic operation
//
// @param image specify the image unit into which to compare and conditionally store data
// @param coords specify the coordinate at which to compare and conditionally store the data
// @param val specifies the value to store in the image if compare is equal to the existing image content
// @param sign ...
////////////////////////////////////////////////////////////////////////////////
void atomicPutDist( layout( r32ui ) coherent volatile uimage3D image, ivec3 coords, float val, float sign )
{
	// Produce the encoding of a floating point value as an integer
	uint newVal = floatBitsToUint( val );

	uint prevVal = 0;
	uint curVal;
	float aux;

	// Loop as long as destination value gets changed by other threads
	//
	// - imageAtomicCompSwap() : atomically compares supplied data with that in memory and conditionally stores it to memory
	while ( ( curVal = imageAtomicCompSwap( image, coords, prevVal/*compare*/, newVal/*data*/ ) ) != prevVal )
	{
		prevVal = curVal;
		aux = min( abs( uintBitsToFloat( curVal ) ), val );
		if ( aux == val )
		{
			// We change the value only if val is the min
			newVal = floatBitsToUint( sign * val );
		}
		else
		{
			newVal = curVal;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
//
// "view", "projection" and "viewport" matrices have been configured
//  to be aligned with exactly 1 brick. Therefore, gl_FragCoord.x and
//  gl_FragCoord.y values go between [ 0 ; 9 ] => i.e. brick size in cache.
//   
////////////////////////////////////////////////////////////////////////////////
void main()
{
	// Flag to tell wheter or not we are inside the mesh
	//
	// - negative (i.e. -1.0) if we are inside the mesh
	// - the sign is computed thanks to the normal
	float signe;

	if ( uAxe == 0 )
	{
		signe = - sign( dataIn.normal.x * dataIn.distMin );

		atomicPutDist( uDistance[ uAxe ]/*image (data pool where to write)*/,
			           ivec3( uSlice, gl_FragCoord.x, gl_FragCoord.y )/*voxel coordinates (3D index) where to write*/,
					   abs( dataIn.distMin )/*value to write*/,
					   signe );
	}
	
	if ( uAxe == 1 )
	{
		signe = - sign( dataIn.normal.y * dataIn.distMin );

		atomicPutDist( uDistance[ uAxe ],
					   ivec3( gl_FragCoord.y, uSlice, gl_FragCoord.x ),
					   abs( dataIn.distMin ),
					   signe );
	}

	if ( uAxe == 2 )
	{
		signe = - sign( dataIn.normal.z * dataIn.distMin );

		atomicPutDist( uDistance[ uAxe ],
					   ivec3( gl_FragCoord.x, gl_FragCoord.y, uSlice ),
					   abs( dataIn.distMin ),
					   signe );
	}
}
