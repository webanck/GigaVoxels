/** FRAGMENT SHADER
 * Cube Map.
 **/

uniform samplerCube s;

void main() {
	gl_FragColor = textureCube(s, gl_TexCoord[0].xyz);
}
