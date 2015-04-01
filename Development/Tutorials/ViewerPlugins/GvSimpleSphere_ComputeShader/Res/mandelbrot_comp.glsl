////////////////////////////////////////////////////////////////////////////////
//
// COMPUTE SHADER
//
// Mandelbrot
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// VERSION
////////////////////////////////////////////////////////////////////////////////

#version 430

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

// Number of shader invocations per work group
layout ( local_size_x = 32, local_size_y = 32 ) in;

////////////////////////////////////////////////////////////////////////////////
// UNIFORM
////////////////////////////////////////////////////////////////////////////////

#define MAX_ITERATIONS 100

uniform vec4 uComputeSpaceWindow/* = vec4( 0, 0, 256, 256 )*/;
uniform uint uWidth/* = 256*/;
uniform uint uHeight/* = 256*/;

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

layout ( binding  = 0, rgba8 ) uniform image2D uColorImage;
//layout ( binding = 1, rgba8 ) uniform image2D uColorImage;

////////////////////////////////////////////////////////////////////////////////
// FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
uint mandelbrot( vec2 pC )
{
	uint i = 0;
	
	vec2 z = vec2( 0.0, 0.0 );
	while ( i < MAX_ITERATIONS && ( z.x * z.x + z.y * z.y  ) < 4.0 )
	{
		z = vec2( z.x * z.x - z.y * z.y + pC.x, 2.0 * z.x * z.y + pC.y );
		i++;
	}

	return i;
}

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
////////////////////////////////////////////////////////////////////////////////
void main()
{
	// Size of pixels in the complex space
	float dx = ( uComputeSpaceWindow.z - uComputeSpaceWindow.x ) / uWidth;
	float dy = ( uComputeSpaceWindow.w - uComputeSpaceWindow.y ) / uHeight;

	// Value of "c"
	vec2 c = vec2( dx * gl_GlobalInvocationID.x + uComputeSpaceWindow.x, dy * gl_GlobalInvocationID.y + uComputeSpaceWindow.y );

	// Mandelbrot color computation
	uint i = mandelbrot( c );
	vec4 color = vec4( 0.0, 0.5, 0.5, 1.0 );
	if ( i < MAX_ITERATIONS )
	{
		if ( i < 5 )
		{
			color = vec4( float( i ) / 5.0, 0.0, 0.0, 1.0 );
		}
		else if ( i < 10 )
		{
			color = vec4( ( float( i ) - 5.0 ) / 5.0, 1.0, 0.0, 1.0 );
		}
		else if ( i < 15 )
		{
			color = vec4( 1.0, ( float( i ) - 10.0 ) / 5.0, 0.0, 1.0 );
		}
		else
		{
			color = vec4( 0.0, 0.0, 1.0, 1.0 );
		}

	}
	else
	{
		color = vec4( 0.0, 0.0, 0.0, 1.0 );
	}

	// Write color output
	imageStore( uColorImage, ivec2( gl_GlobalInvocationID.xy ), color );
	//imageStore( uColorImage, ivec2( gl_GlobalInvocationID.xy ), vec4( 1.0, 0.0, 0.0, 1.0 ) );
}
