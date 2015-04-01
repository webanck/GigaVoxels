/** VERTEX SHADER
 * Cube Map.
 **/

void main() {
	vec4 pos = gl_ModelViewProjectionMatrix*gl_Vertex;
	gl_Position = pos.xyww;//trick used to the cube map stays at zfar
	gl_TexCoord[0].xyz = gl_Vertex.xyz;
}
