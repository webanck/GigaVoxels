void main()
{

	vec4 positionInView = gl_ModelViewMatrix * gl_Vertex;
	gl_Position = gl_ProjectionMatrix * positionInView;
}
