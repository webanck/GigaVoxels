#version 330
#extension GL_EXT_shader_image_load_store : enable

// To write in the gigaVoxel dataPool
uniform layout( rgba32f ) image3D dataPool;

// The brick adress in the texture
uniform ivec3 brickAdress;

void main()
{
	// We clear the dataPool
	imageStore( dataPool, brickAdress + ivec3( gl_FragCoord.x, gl_FragCoord.y, 0 ), vec4( 0.0, 0.0, 0.0, 0.0 ) );
	imageStore( dataPool, brickAdress + ivec3( gl_FragCoord.x, gl_FragCoord.y, 1 ), vec4( 0.0, 0.0, 0.0, 0.0 ) );	
}