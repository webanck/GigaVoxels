#version 420

in vec3 position;
in vec3 normal;
in uint index;

out Data {
	uint index;
} dataOut;

void main()
{
	gl_Position =  vec4(position, 1.0);
	
	dataOut.index = index;
}
