//varying float depthInWorld;
varying vec4 positionInView;

void main()
{
	//gl_FragColor = vec4(depthInWorld);
	gl_FragColor = length(positionInView.xyz);
}
