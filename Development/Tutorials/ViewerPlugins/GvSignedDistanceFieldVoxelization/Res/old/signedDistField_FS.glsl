varying float distMin;
varying vec3 normal;

uniform int x;
uniform int y;
uniform int z;

uniform sampler3D signedDistField;

uniform int slice;

uniform vec3 direction;

float min ( in float a, in float b ) {
	if ( a < b ) { 
		return a; 
	} else {
		return b;
	}
}

void main()
{
	// retrieve current texel value in the slice of the distance flield
	vec4 currentDist = texelFetch( signedDistField, ivec3( gl_FragCoord.x, gl_FragCoord.y, slice), 0);

	float sign;
	
	if ( dot( direction, normal ) > 0.0 ) {
		sign = + 1.0;
	} else {
		sign = -1.0;
	}
		
	if (x) {
		gl_FragColor.x = sign *  min( distMin, abs( currentDist.x ) );
		gl_FragColor.z = currentDist.z;
		gl_FragColor.y = currentDist.y;
	}
	if (y) {
		gl_FragColor.y = sign * min( distMin, abs ( currentDist.y ) );
		gl_FragColor.x = currentDist.x;
		gl_FragColor.z = currentDist.z;
	}
	if (z) {
		gl_FragColor.z = sign * min( distMin, abs( currentDist.z ) );
		gl_FragColor.x = currentDist.x;
		gl_FragColor.y = currentDist.y;
	}
}
