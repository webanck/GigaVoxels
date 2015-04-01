////////////////////////////////////////////////////////////////////////////////
//
// VERTEX SHADER
//
// Points
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// VERSION
////////////////////////////////////////////////////////////////////////////////

#version 400

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

layout (location = 0) in vec3 VertexPosition;

////////////////////////////////////////////////////////////////////////////////
// UNIFORM
////////////////////////////////////////////////////////////////////////////////

uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;
uniform float PointSize;
uniform float uTime;

uniform vec2 uWindowSize;
const float worldPointSize = 0.01;
//const vec2 screenSize = vec2( 1325, 1083 );

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
////////////////////////////////////////////////////////////////////////////////
void main()
{
	// IDEA
	// - we want to compute the footprint of a point-sprite on screen
	// - we will have same world size but different screen-space size
	
	// Transform point to Eye space
	vec4 eyePosition = ModelViewMatrix * vec4( VertexPosition, 1.0 );
	// Modify x and y components
	// - keep same distance from point to camera view-plane
	// - but use "world size" of point for x and y components
	// => the idea is to keep constant size in world space, hence a different screen size
	vec4 clipPosition = ProjectionMatrix * vec4( worldPointSize, worldPointSize, eyePosition.z, eyePosition.w );
	// Check foorptint in screen space
	//vec2 projectedSize = /*viewport transform*/( 0.5 * screenSize ) * /*NDC space*/( ( clipPosition.xy / clipPosition.w ) ) * /*scaling coefficient*/PointSize;
	vec2 projectedSize = /*viewport transform*/( 0.5 * uWindowSize ) * /*NDC space*/( ( clipPosition.xy / clipPosition.w ) ) * /*scaling coefficient*/PointSize;
	float size = 0.5 * ( projectedSize.x + projectedSize.y );
	//gl_PointSize = size - size * /*half variation*/0.5 * /*animation*/( 0.5f * ( 1.0 + cos( 3.141592 * uTime / 100.0 ) ) );
	gl_PointSize = size;
	
	// Send position to clip space
	gl_Position = ProjectionMatrix * eyePosition;
}
