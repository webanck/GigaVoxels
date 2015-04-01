uniform sampler3D signedDistField;
uniform int slice;

void main()
{
	// retrieve Z dist
	vec4 texelDist = texelFetch( signedDistField, ivec3( gl_FragCoord.x, gl_FragCoord.y, slice), 0);
	// X
	vec4 aux = texelFetch( signedDistField, ivec3( gl_FragCoord.y, slice, gl_FragCoord.x), 0);
	texelDist.x = aux.x;
	// Y
	aux = texelFetch( signedDistField, ivec3( slice, gl_FragCoord.x,  gl_FragCoord.y), 0);
	texelDist.y = aux.y;

	int sign = +1 ;
	// distance computation 
	if ( ( texelDist.x < 0.0 ) && ( texelDist.y < 0.0 ) && ( texelDist.z < 0.0 ) ) {
			// We are in the mesh in this case
			sign = -1;
	}
	
	if ( ( texelDist.x < 1.0 ) && ( texelDist.y < 1.0 ) && ( texelDist.z < 1.0 ) ) {
		gl_FragColor.x = (float)sign * abs( texelDist.x ) * abs( texelDist.y ) * abs( texelDist.z ) 
			/ ( ( 1.0 / 4.0 ) 
			    * sqrt ( ( sqrt( texelDist.x * texelDist.x + texelDist.y * texelDist.y ) + 
				       sqrt( texelDist.x * texelDist.x + texelDist.z * texelDist.z ) +
				       sqrt( texelDist.y * texelDist.y + texelDist.z * texelDist.z ) ) 
				     * ( - sqrt( texelDist.x * texelDist.x + texelDist.y * texelDist.y ) + 
					 sqrt( texelDist.x * texelDist.x + texelDist.z * texelDist.z ) +
					 sqrt( texelDist.y * texelDist.y + texelDist.z * texelDist.z ) )
				     * ( sqrt( texelDist.x * texelDist.x + texelDist.y * texelDist.y ) - 
					 sqrt( texelDist.x * texelDist.x + texelDist.z * texelDist.z ) +
					 sqrt( texelDist.y * texelDist.y + texelDist.z * texelDist.z ) )
				     * ( sqrt( texelDist.x * texelDist.x + texelDist.y * texelDist.y ) +
					 sqrt( texelDist.x * texelDist.x + texelDist.z * texelDist.z ) -
					 sqrt( texelDist.y * texelDist.y + texelDist.z * texelDist.z ) ) ) );
	} else if ( ( texelDist.x < 1.0 ) && ( texelDist.y < 1.0 ) && !( texelDist.z < 1.0 ) ) {
		gl_FragColor.x = abs( texelDist.x ) * abs( texelDist.y ) / sqrt( texelDist.x * texelDist.x + texelDist.y * texelDist.y );
	} else if ( ( texelDist.x < 1.0 ) && !( texelDist.y < 1.0 ) && ( texelDist.z < 1.0 ) ) {
		gl_FragColor.x = abs( texelDist.x ) * abs( texelDist.z ) / sqrt( texelDist.x * texelDist.x + texelDist.z * texelDist.z );
	} else if ( ! ( texelDist.x < 1.0 ) && ( texelDist.y < 1.0 ) && ( texelDist.z < 1.0 ) ) {
		gl_FragColor.x = abs( texelDist.y ) * abs( texelDist.z ) / sqrt( texelDist.y * texelDist.y + texelDist.z * texelDist.z );
	} else if ( ( texelDist.x < 1.0 ) ) {
		gl_FragColor.x = abs( texelDist.x );
	} else if ( ( texelDist.y < 1.0 ) ) {
		gl_FragColor.x = abs( texelDist.y ); 
	} else if ( ( texelDist.z < 1.0 ) ) {
		gl_FragColor.x = abs( texelDist.z ); 
	} else {
		gl_FragColor.x = 1.0;
	}
}
