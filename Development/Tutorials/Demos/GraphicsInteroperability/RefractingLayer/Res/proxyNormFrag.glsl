#version 400 compatibility

in vec3 Normal;
layout (location = 0) out vec3 fragNormal;

void main() {
	fragNormal = Normal;
	
}