       //#include test2.h 
         // #include test.h    

//Lighting stuff
vec4 ambient; 
vec4 diffuse;
vec4 specular;
//float attenuation;

//Lighting parameters
uniform vec4 ambientLight; //between 0 and 1
uniform float shininess;
uniform vec4 specularColor;
uniform vec3 lightPos;
uniform bool hasTex;

varying vec3 point, normal;
varying vec2 texCoord;
uniform sampler2D samplera;
uniform sampler2D samplerd;
uniform sampler2D samplers;


void main() {

	vec4 lightIntensityAmb = vec4(0.0f, 0.0f, 0.0f, 1.0f);
	vec4 lightIntensityDiff = vec4(0.8f, 0.8f, 0.8f, 1.0f);
	vec4 lightIntensitySpec = vec4(0.7f, 0.7f, 0.7f, 1.0f);

	vec4 dcolor;
	if (hasTex) {
		dcolor = texture(samplerd, gl_TexCoord[0].xy);
	} else {
		dcolor = vec4(0.8, 0.8, 0.8, 1.0);
	}
	vec3 pointToLight = vec3(lightPos - point);
	vec3 N = normalize(normal);
	vec3 L = normalize(pointToLight);

	//Computations for front facing faces
	ambient = ambientLight*lightIntensityAmb;

	diffuse = max(0.0, dot(N, L))*lightIntensityDiff*dcolor; 

	vec3 reflVec = reflect(-L, N);
	vec3 EV = normalize(-point);
	float cosAngle = max(0.0, dot(EV, reflVec));
	specular = vec4(0.0);
	if (max(0.0, dot (N, L)) > 0.0) {
		specular = pow(cosAngle, shininess)*specularColor*lightIntensitySpec;
	}

	//Computations for back facing faces
	vec4 backdiffuse = max(0.0, dot(-N, L))*lightIntensityDiff*dcolor; 
	vec3 backreflVec = reflect(-L, -N);
	float backcosAngle = max(0.0, dot(EV, backreflVec));
	vec4 backspecular = vec4(0.0);
	if (max(0.0, dot (-N, L)) > 0.0) {
		backspecular = pow(backcosAngle, shininess)*specularColor*lightIntensitySpec;
	}


	if (gl_FrontFacing) {
		gl_FragColor = ambient + diffuse + specular;
	} else {
		gl_FragColor = ambient + backdiffuse + backspecular;
	}
}
