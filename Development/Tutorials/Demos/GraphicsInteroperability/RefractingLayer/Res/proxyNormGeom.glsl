#version 400 compatibility

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec3 newNormal[]; // for each vertex
in vec3 pos[];
in vec3 oldPos[];
in vec4 eyeVertexPosition[];
in float displacement[];
out vec3 Normal;

void main() {
	/*gl_Position = gl_in[0].gl_Position;
	vec3 normal1 = cross( pos[1] - pos[0], pos[2] - pos[0] );
	vec3 normal2 = cross( pos[2] - pos[0], pos[3] - pos[0] );
	vec3 normal3 = cross( pos[3] - pos[0], pos[4] - pos[0] );
	vec3 normal4 = cross( pos[4] - pos[0], pos[5] - pos[0] );
	vec3 normal5 = cross( pos[5] - pos[0], pos[6] - pos[0] );
	vec3 normal6 = cross( pos[6] - pos[0], pos[1] - pos[0] );

	Normal = normalize(normal1 + normal2 + normal3 + normal4 + normal5 + normal6);
	EmitVertex();*/

	//float angle01 = arccos( dot( normalize(oldPos[1] - oldPos[0]), normalize(pos[1] - pos[0]) ) );
	Normal = newNormal[0];

	gl_Position = gl_ProjectionMatrix*eyeVertexPosition[0];
	EmitVertex();

	Normal = newNormal[1];

	gl_Position = gl_ProjectionMatrix*eyeVertexPosition[1];
	EmitVertex();
		Normal = newNormal[2];

	gl_Position = gl_ProjectionMatrix*eyeVertexPosition[2];
	EmitVertex();


	EndPrimitive();
}