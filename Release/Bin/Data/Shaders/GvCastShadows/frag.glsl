////////////////////////////////////////////////////////////////////////////////
//
// FRAGMENT SHADER
//
// ...
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// VERSION
////////////////////////////////////////////////////////////////////////////////

//#version 400

////////////////////////////////////////////////////////////////////////////////
// INCLUDE SECTION
////////////////////////////////////////////////////////////////////////////////

//#include test2.h
//#include test.h

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

// Position and normal in View coordinates system
varying vec3 point, normal;

// Not used ?
//varying vec2 texCoord;

////////////////////////////////////////////////////////////////////////////////
// UNIFORM
////////////////////////////////////////////////////////////////////////////////

// Light parameter
uniform vec3 lightPos;

// Material parameters
uniform vec4 ambientLight; //between 0 and 1
uniform float shininess;
uniform vec4 specularColor;
uniform bool hasTex;
uniform sampler2D samplera;
uniform sampler2D samplerd;
uniform sampler2D samplers;

// Lighting stuff
// TO DO
// - declare them in main() function
vec4 ambient; 
vec4 diffuse;
vec4 specular;
// float attenuation;

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
////////////////////////////////////////////////////////////////////////////////
void main()
{
	// Default light parameters
	vec4 lightIntensityAmb = vec4( 0.0f, 0.0f, 0.0f, 1.0f );
	vec4 lightIntensityDiff = vec4( 0.8f, 0.8f, 0.8f, 1.0f );
	vec4 lightIntensitySpec = vec4( 0.7f, 0.7f, 0.7f, 1.0f );

	// Material's diffuse color
	vec4 dcolor;
	if ( hasTex )
	{
		dcolor = texture( samplerd, gl_TexCoord[ 0 ].xy );
	} else
	{
		// Default material's diffuse color
		dcolor = vec4( 0.8, 0.8, 0.8, 1.0 );
	}

	vec3 pointToLight = vec3( lightPos - point );
	vec3 N = normalize( normal );
	vec3 L = normalize( pointToLight );

	// ADS Lighting model computations (i.e. ambient, diffuse, specular)

	// Ambient term
	ambient = ambientLight * lightIntensityAmb;
	
	// Front facing faces (i.e. ambient, diffuse, specular)
	//
	// Diffuse term
	diffuse = max( 0.0, dot( N, L ) ) * lightIntensityDiff * dcolor; 
	// Specular term
	vec3 reflVec = reflect( -L, N );
	vec3 EV = normalize( -point );
	float cosAngle = max( 0.0, dot( EV, reflVec ) );
	specular = vec4( 0.0 );
	if ( max( 0.0, dot( N, L ) ) > 0.0 )
	{
		specular = pow( cosAngle, shininess ) * specularColor * lightIntensitySpec;
	}

	// Back facing faces (i.e. ambient, diffuse, specular)
	//
	// Diffuse term
	vec4 backdiffuse = max( 0.0, dot( -N, L ) ) * lightIntensityDiff * dcolor;
	// Specular term
	vec3 backreflVec = reflect( -L, -N );
	float backcosAngle = max( 0.0, dot( EV, backreflVec ) );
	vec4 backspecular = vec4( 0.0 );
	if ( max( 0.0, dot( -N, L ) ) > 0.0 )
	{
		backspecular = pow( backcosAngle, shininess ) * specularColor * lightIntensitySpec;
	}
	
	// ADS Lighting model (i.e. ambient, diffuse, specular)
	if ( gl_FrontFacing )
	{
		gl_FragColor = ambient + diffuse + specular;
	}
	else
	{
		gl_FragColor = ambient + backdiffuse + backspecular;
	}
}
