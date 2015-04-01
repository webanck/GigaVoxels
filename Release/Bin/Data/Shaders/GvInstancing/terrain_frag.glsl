////////////////////////////////////////////////////////////////////////////////
//
// FRAGMENT SHADER
//
// Sky box rendering
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// VERSION
////////////////////////////////////////////////////////////////////////////////

#version 400

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

noperspective in vec3 EdgeDistance;
in vec2 uv;

////////////////////////////////////////////////////////////////////////////////
// UNIFORM
////////////////////////////////////////////////////////////////////////////////

uniform sampler2D heightMapTexture;	// heightmap texture

// Wireframe parameter
uniform float uLineWidth;
uniform vec4 uLineColor;

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

// Ouput color
layout (location = 0) out vec4 oColor;

//const vec3 LUT[8] = { vec3(0.0,0.0,0.0), vec3(0.0,0.0,1.0), vec3(0.0,1.0,0.0), vec3(0.0,1.0,1.0), vec3(1.0,0.0,0.0), vec3(1.0,0.0,1.0), vec3(1.0,1.0,0.0), vec3(1.0,1.0,1.0) };
	
////////////////////////////////////////////////////////////////////////////////
// PROGRAM
////////////////////////////////////////////////////////////////////////////////
void main()
{
	vec3 terrainColor = vec3( 0.0, 0.0, 0.0 );
	float height = texture( heightMapTexture, uv ).r / 255.0 * 250;
	if ( height < 0.0125 ) terrainColor = vec3(0.0,0.0,0.0);
	else if ( height < 0.25 ) terrainColor = vec3(0.0,0.0,1.0);
	else if ( height < 0.375 ) terrainColor = vec3(0.0,1.0,0.0);
	else if ( height < 0.5 ) terrainColor = vec3(0.0,1.0,1.0);
	else if ( height < 0.625 ) terrainColor = vec3(1.0,0.0,0.0);
	else if ( height < 0.75 ) terrainColor = vec3(1.0,0.0,1.0);
	else if ( height < 0.875 ) terrainColor = vec3(1.0,1.0,0.0);
	else /*if ( height < 0.875 )*/ terrainColor = vec3(1.0,1.0,1.0);

	// Wireframe
	float distanceToEdge = min( EdgeDistance.z, min( EdgeDistance.x, EdgeDistance.y ) );
	float mixValue = smoothstep( uLineWidth - 1.0, uLineWidth + 1.0, distanceToEdge );

	//oColor = vec4( 1.0, 0.0, 0.0, 1.0 );

	//ivec2 textureSize = textureSize( heightMapTexture, 0 );
	//vec3 terrainColor = texture( heightMapTexture, vec2( gl_TexCoord.xy / textureSize.xy ) ).xyz;
	//vec3 terrainColor = texture( heightMapTexture, uv ).xyz / 255.0;
	//vec3 terrainColor = vec3( 0.0, 0.0, 0.0 );
	//oColor = vec4( terrainColor, 1.0 );
	//oColor = vec4( terrainColor.x, terrainColor.y, terrainColor.z, 1.0 );
	//vec4 terrainColor = texture( heightMapTexture, uv );

	// Output color
	oColor = mix( uLineColor, vec4( terrainColor, 1.0 ), mixValue );
	//oColor = mix( uLineColor, terrainColor, mixValue );
	//oColor = mix( uLineColor, vec4( terrainColor.x, 0.25, 1 - terrainColor.x, terrainColor.w ), mixValue );
}
