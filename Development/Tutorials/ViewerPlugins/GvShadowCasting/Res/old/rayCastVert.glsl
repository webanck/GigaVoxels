#version 410

uniform vec3 viewPos;
uniform mat4 modelViewMat;

in vec4 vertexPos;

//layout(location = 0) out vec3 vertOutViewDir;

void main()
{
	gl_Position = vertexPos;

	//vec4 worldPos = vec4(vertexPos.xyz, 1.0f);
	//worldPos.z = -1.75f +0.001f; //-0.1f;

	//worldPos = inverse(modelViewMat) * worldPos;
	//worldPos.xyz = worldPos.xyz / worldPos.w;

	//vertOutViewDir = worldPos.xyz - viewPos;
}
