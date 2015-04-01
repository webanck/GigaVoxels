/**
 * FRAGMENT SHADER
 *
 * Writing the desired values in a frame buffer 
 * object to be used later in the GigaVoxels pipeline.
**/

#version 400 compatibility

varying vec4 eyeVertexPosition;
varying vec3 newNormal;

void main() {
	gl_FragData[0] = vec4(length(eyeVertexPosition.xyz), 0.0, 0.0, 0.0);//depth, aka distance to camera
	gl_FragData[1] = vec4(newNormal, 0.0);//re-computed normal
}