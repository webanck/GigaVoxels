uniform samplerCube s;

void main() {
	gl_FragColor = textureCube(s, gl_TexCoord[0].xyz);
	//gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0);
}
