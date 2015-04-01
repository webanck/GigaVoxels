varying float distMin;
varying vec3 normal;

void main()
{

	vec4 positionInView = gl_ModelViewMatrix * gl_Vertex;
	
	distMin = abs(positionInView.z);

	gl_Position = gl_ProjectionMatrix * positionInView;
	
	normal = gl_Normal;
}
