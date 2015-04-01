       //#include test2.h 
         // #include test.h    

//Lighting stuff
vec3 ambient; 
vec3 diffuse;
vec3 specular;
//float attenuation;

//Lighting parameters
uniform vec3 ambientLight; //between 0 and 1
uniform float shininess;
uniform vec3 specularColor;
uniform vec3 lightPos;
uniform bool hasTex;

varying vec3 point, normal;
varying vec2 texCoord;
uniform sampler2D samplera;
uniform sampler2D samplerd;
uniform sampler2D samplers;


void main() {
	vec4 dcolor;
	if (hasTex) {
		//dcolor = texture(samplerd, gl_TexCoord[0].xy);
		dcolor = vec4(1.0, 0.0, 0.0 ,1.0);
	} else {
		dcolor = vec4(0.0, 0.0, 1.0, 1.0);
	}
	vec3 pointToLight = vec3(lightPos - point);
	vec3 N = normalize(normal);
	vec3 L = normalize(pointToLight);

	ambient = vec3(ambientLight.x*dcolor.x, ambientLight.y*dcolor.y,ambientLight.z*dcolor.z);

	diffuse = max(0.0, dot(N, L))*vec3(dcolor);
	//diffuse = dot(N, L)*vec3(dcolor);
	//diffuse = clamp(dot(N, L), 0, 1)*vec3(dcolor);
	vec3 reflVec = reflect(-L, N);
	vec3 EV = normalize(-point);
	float cosAngle = max(0.0, dot(EV, reflVec));
	specular = vec3(0.0);
	if (max(0.0, dot (N, L)) > 0.0) {
		specular = pow(cosAngle, shininess)*specularColor;
	}

	//float distanceToLight = length(pointToLight);
	//float attenuation = float(1.0)/float(1.0 + pow(distanceToLight, 2.0));

	//gl_FragColor = vec4(diffuse, 1.0);
	//gl_FragColor = vec4(ambient + diffuse + specular, dcolor.w);
	//gl_FragColor = texture2D(sampler, gl_TexCoord[0].xy);
	
}
