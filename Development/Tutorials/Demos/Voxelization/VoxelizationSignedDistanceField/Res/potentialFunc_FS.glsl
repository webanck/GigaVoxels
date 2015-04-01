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
// - due to imageAtomicExchange() function

////////////////////////////////////////////////////////////////////////////////
// EXTENSIONS
////////////////////////////////////////////////////////////////////////////////

#extension GL_EXT_shader_image_load_store : enable
#extension GL_NV_shader_atomic_float : enable

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// UNIFORM
////////////////////////////////////////////////////////////////////////////////

// 3D texture of a brick containing min-distance from mesh to each voxel of the brick along its x-axis
uniform layout( r32f ) coherent volatile image3D uDistanceX;

// 3D texture of a brick containing min-distance from mesh to each voxel of the brick along its y-axis
uniform layout( r32f ) coherent volatile image3D uDistanceY;

// 3D texture of a brick containing min-distance from mesh to each voxel of the brick along its z-axis
uniform layout( r32f ) coherent volatile image3D uDistanceZ;

// GigaVoxels data pool (3D texture) corresponding to current selected channel,
// .i.e "channel 0" the one used to store the final closest distance to the mesh at a given voxel
uniform layout( r32f ) coherent image3D uPotential;

// The brick address in the texture (this is the address in cache)
uniform ivec3 uBrickAddress;

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
//
// Implementation of Cyril Crassin algorithm ( Cyril Crassin thesis : Computing distance fields on GPU )
//
//  Note : code make use of imageAtomicExchange() function
//       - which atomically store supplied data into memory and return the original value from memory
////////////////////////////////////////////////////////////////////////////////
void main()
{
	vec4 texelDist;

	// Iterate through 3D texture z-slices
	//
	// - given fragments generated on the x-y plane of the window (viewport)
	// - iteration is used to fill all voxels of current brick
	for ( int slice = 0; slice < 10; slice++ )
	{
		// Retrieve previously generated min (x,y,z) signed distances from mesh to current voxel
		texelDist.x = imageLoad( uDistanceX, ivec3( gl_FragCoord.x, gl_FragCoord.y, slice ) );
		texelDist.y = imageLoad( uDistanceY, ivec3( gl_FragCoord.x, gl_FragCoord.y, slice ) );
		texelDist.z = imageLoad( uDistanceZ, ivec3( gl_FragCoord.x, gl_FragCoord.y, slice ) );

		// Check wheter or not we are inside the mesh
		//
		// - distance computation 
		int sign = +1;
		if ( ( texelDist.x < 0.0 ) && ( texelDist.y < 0.0 ) && ( texelDist.z < 0.0 ) )
		{
			// Here, we know that we are in the mesh
			sign = -1;
		}

		// Handle 3 main differents cases, depending wheter or not 3, 2 or only 1 directions have significative information
		// - indeed the previous "signed distance field" on each axes may have no information where no triangles fragment projects on a givent axis
		// - in such a case, the corresponding distance value has previously been set to "1.0"
		if ( ( texelDist.x < 1.0 ) && ( texelDist.y < 1.0 ) && ( texelDist.z < 1.0 ) )
		{
			// Here, the 3 directions have significative information
			float a = sqrt( texelDist.x * texelDist.x + texelDist.y * texelDist.y );
			float b = sqrt( texelDist.x * texelDist.x + texelDist.z * texelDist.z );
			float c = sqrt( texelDist.y * texelDist.y + texelDist.z * texelDist.z );

			// The distance is the distance to the plane defined as the closest intersection along the 3 axis,
			// we use V = 1/3 * h * B and B = 1/4 * sqrt( ( a + b + c )(-a + b + c )( a - b + c )( a + b - c ) ) where a, b and c are the 3 side of the basis
			imageAtomicExchange( uPotential, uBrickAddress * 10 + ivec3( gl_FragCoord.xy, slice ), 
								 sign * abs( texelDist.x ) * abs( texelDist.y ) * abs( texelDist.z ) 
								 / ( ( 1.0 / 4.0 ) * sqrt ( ( a + b + c ) * ( - a + b + c ) * ( a - b + c ) * ( a + b - c ) ) ) );

		}
		// Cases where there are only information into 2 directions 
		// - compute distance to a line
		else if ( ( texelDist.x < 1.0 ) && ( texelDist.y < 1.0 ) && !( texelDist.z < 1.0 ) )
		{
			imageAtomicExchange( uPotential, uBrickAddress * 10 + ivec3( gl_FragCoord.xy, slice ),
								 abs( texelDist.x ) * abs( texelDist.y ) / sqrt( texelDist.x * texelDist.x + texelDist.y * texelDist.y ) );
		}
		else if ( ( texelDist.x < 1.0 ) && !( texelDist.y < 1.0 ) && ( texelDist.z < 1.0 ) )
		{
			imageAtomicExchange( uPotential, uBrickAddress * 10 + ivec3( gl_FragCoord.xy, slice ), 
								 abs( texelDist.x ) * abs( texelDist.z ) / sqrt( texelDist.x * texelDist.x + texelDist.z * texelDist.z ) );
		}
		else if ( ! ( texelDist.x < 1.0 ) && ( texelDist.y < 1.0 ) && ( texelDist.z < 1.0 ) )
		{
			imageAtomicExchange( uPotential, uBrickAddress * 10 + ivec3( gl_FragCoord.xy, slice ), 
								 abs( texelDist.y ) * abs( texelDist.z ) / sqrt( texelDist.y * texelDist.y + texelDist.z * texelDist.z ) );
		}
		// Cases where only one direction has significative information
		else if ( ( texelDist.x < 1.0 ) )
		{
			imageAtomicExchange( uPotential, uBrickAddress * 10 + ivec3( gl_FragCoord.xy, slice ), abs( texelDist.x ) );
		}
		else if ( ( texelDist.y < 1.0 ) )
		{
			imageAtomicExchange( uPotential, uBrickAddress * 10 + ivec3( gl_FragCoord.xy, slice ), abs( texelDist.y ) );
		}
		else if ( ( texelDist.z < 1.0 ) )
		{
			imageAtomicExchange( uPotential, uBrickAddress * 10 + ivec3( gl_FragCoord.xy, slice ), abs( texelDist.z ) );
		}
		else
		{
			imageAtomicExchange( uPotential, uBrickAddress * 10 + ivec3( gl_FragCoord.xy, slice ), 1.0 );
		}
	}
}
