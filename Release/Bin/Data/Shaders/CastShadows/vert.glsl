varying vec3 point, normal;
uniform vec3 lightPos;


void main() {
	point = vec3(gl_ModelViewMatrix*gl_Vertex); 
	normal = gl_NormalMatrix*gl_Normal;
	gl_FrontColor = gl_Color;
	gl_Position = gl_ModelViewProjectionMatrix*gl_Vertex;
	gl_TexCoord[0].xy   = gl_MultiTexCoord0.xy;
}