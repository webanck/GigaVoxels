////////////////////////////////////////////////////////////////////////////////
//
// FRAGMENT SHADER
//
// ...
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

// ...
uniform layout( r32f ) coherent volatile image3D uDistanceX;

// ...
uniform layout( r32f ) coherent volatile image3D uDistanceY;

// ...
uniform layout( r32f ) coherent volatile image3D uDistanceZ;

// ...
uniform layout( r32f ) coherent image3D uPotential;

// The brick adress in the texture
uniform ivec3 uBrickAddress;

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
//
// Implementation of Cyril crassin algorithm ( Cyril Crassin thesis : Computing distance fields on GPU )
//
//  Note : code make use of imageAtomicExchange() function
//       - which atomically store supplied data into memory and return the original value from memory
////////////////////////////////////////////////////////////////////////////////
void main()
{
	vec4 texelDist;

	// Iterate through 3D texture z-slices
	for ( int slice = 0; slice < 10; slice++ )
	{
		// Retrieve Z dist
		texelDist.z = imageLoad( uDistanceZ, ivec3( gl_FragCoord.x, gl_FragCoord.y, slice ) );

		// Retrieve X dist
		texelDist.x = imageLoad( uDistanceX, ivec3( gl_FragCoord.x, gl_FragCoord.y, slice ) );

		// Retrieve Y dist
		texelDist.y = imageLoad( uDistanceY, ivec3( gl_FragCoord.x, gl_FragCoord.y, slice ) );

		// Check wheter or not we are in the mesh
		// - distance computation 
		int sign = +1;
		if ( ( texelDist.x < 0.0 ) && ( texelDist.y < 0.0 ) && ( texelDist.z < 0.0 ) )
		{
			// Here, we know that we are in the mesh
			sign = -1;
		}

		if ( ( texelDist.x < 1.0 ) && ( texelDist.y < 1.0 ) && ( texelDist.z < 1.0 ) )
		{
			// The 3 direction got sinificative information
			float a = sqrt( texelDist.x * texelDist.x + texelDist.y * texelDist.y );
			float b = sqrt( texelDist.x * texelDist.x + texelDist.z * texelDist.z );
			float c = sqrt( texelDist.y * texelDist.y + texelDist.z * texelDist.z );

			// The distance is the distance to the plane defined as the closest intersection along the 3 axis,
			// we use V = 1/3 * h * B and B = 1/4 * sqrt( ( a + b + c )(-a + b +c )(a-b+c)(a+b-c) ) where a, b and c are the 3 side of the basis
			imageAtomicExchange( uPotential, uBrickAddress * 10 + ivec3( gl_FragCoord.xy, slice ), 
				sign * abs( texelDist.x ) * abs( texelDist.y ) * abs( texelDist.z ) 
				/ ( ( 1.0 / 4.0 ) 
				* sqrt ( ( a + b + c ) 
				* ( - a + b + c )
				* ( a - b + c )
				* ( a + b - c ) ) ) );

			// Case where there are only information into 2 dirrections 
			// Computing distance to a line
		}
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
			// Only one dirrections
		}
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
