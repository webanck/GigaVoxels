//varying float depthInWorld;
varying vec4 positionInView;

void main()
{
	positionInView = gl_ModelViewMatrix * gl_Vertex;

	gl_Position = gl_ProjectionMatrix * positionInView;
	//depthInWorld = length(positionInView.xyz);
}
