uniform mat4 cubeModelMatrix;

void main() {
	gl_TexCoord[0].xyz = gl_Vertex.xyz;
	//gl_TexCoord[0].xyz = vec3(inverse(cubeModelMatrix)*gl_Vertex);
	gl_Position = gl_ModelViewProjectionMatrix*gl_Vertex;
}
