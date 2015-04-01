////////////////////////////////////////////////////////////////////////////////
//
// GEOMETRY SHADER
//
// Spiral arms with wireframe
// - Gardner : cloud opacity simulation
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// VERSION
////////////////////////////////////////////////////////////////////////////////

#version 400

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

layout (triangles) in;

in vec3 tePosition[];
in vec3 teNormal[];

////////////////////////////////////////////////////////////////////////////////
// UNIFORM
////////////////////////////////////////////////////////////////////////////////

uniform mat4 uViewportMatrix;

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

layout (triangle_strip, max_vertices = 3) out;

out vec3 gPosition;
out vec3 gNormal;
noperspective out vec3 EdgeDistance;

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
////////////////////////////////////////////////////////////////////////////////
void main()
{
	// Transform points to window space
	vec3 p0 = vec3( uViewportMatrix * ( gl_in[ 0 ].gl_Position / gl_in[ 0 ].gl_Position.w ) );
	vec3 p1 = vec3( uViewportMatrix * ( gl_in[ 1 ].gl_Position / gl_in[ 1 ].gl_Position.w ) );
	vec3 p2 = vec3( uViewportMatrix * ( gl_in[ 2 ].gl_Position / gl_in[ 2 ].gl_Position.w ) );
	
	// Find triangle altitudes (distance from each vertex to opposite edge)
	float a = length( p1 - p2 );
	float b = length( p2 - p0 );
	float c = length( p1 - p0 );
	float alpha = acos( ( b * b + c * c - a * a ) / ( 2.0 * b * c ) );
	float beta = acos( ( a * a + c * c - b * b ) / ( 2.0 * a * c ) );
	float ha = abs( c * sin( beta ) );
	float hb = abs( c * sin( alpha ) );
	float hc = abs( b * sin( alpha ) );
	
	// Send triangle with edge information
	
	gPosition = tePosition[ 0 ];
	gNormal = teNormal[ 0 ];
	EdgeDistance = vec3( ha, 0.0, 0.0 );
	gl_Position = gl_in[ 0 ].gl_Position;
	EmitVertex();
	
	gPosition = tePosition[ 1 ];
	gNormal = teNormal[ 1 ];
	EdgeDistance = vec3( 0.0, hb, 0.0 );
	gl_Position = gl_in[ 1 ].gl_Position;
	EmitVertex();
	
	gPosition = tePosition[ 2 ];
	gNormal = teNormal[ 2 ];
	EdgeDistance = vec3( 0.0, 0.0, hc );
	gl_Position = gl_in[ 2 ].gl_Position;
	EmitVertex();

	EndPrimitive();
}
