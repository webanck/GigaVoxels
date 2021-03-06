////////////////////////////////////////////////////////////////////////////////
//
// VERTEX SHADER
//
// GigaSpace Pass 
//
// - Hierarchical data structure traversal
// - Requests of production are emitted when no data is encountered
// - Multi-resolution voxel-based volume rendering pass with cone-tracing
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// VERSION
////////////////////////////////////////////////////////////////////////////////

#version 400

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

// Vertex position
// - position is already in Clip space
in vec2 iPosition;

////////////////////////////////////////////////////////////////////////////////
// UNIFORM
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
////////////////////////////////////////////////////////////////////////////////
void main()
{
	// Pass vertex position
	// - position is already in Clip space
	gl_Position = vec4( iPosition, 0.0, 1.0 );
}
