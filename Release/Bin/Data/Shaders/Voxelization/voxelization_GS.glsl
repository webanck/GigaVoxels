////////////////////////////////////////////////////////////////////////////////
//
// GEOMETRY SHADER
//
// On-the-fly voxelization
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// VERSION
////////////////////////////////////////////////////////////////////////////////

#version 410

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

// Input primitive type are triangles
layout( triangles ) in;

// ...
in Data
{
	// ...
    vec3 normal;

	// ...
	vec3 worldPosition;

} iData[];

////////////////////////////////////////////////////////////////////////////////
// UNIFORM
////////////////////////////////////////////////////////////////////////////////

// ...
uniform mat4 uProjectionMatX;

// ...
uniform mat4 uProjectionMatY;

// ...
uniform mat4 uProjectionMatZ;

// ...
//uniform vec2 hPixel;

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

// Geometry shader will generate a quad for each input point
layout( triangle_strip, max_vertices=3 ) out;

// ...
out Data
{
	// ...
    vec3 normal;

	// ...
	vec3 worldPosition;

	// ...
	//vec4 aabb;

	// ...
	//vec2 clipSpace;

} oData;

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
////////////////////////////////////////////////////////////////////////////////
//void main()
//{
//	// Compute normal (triangle edges cross product)
//	vec3 normal = cross( gl_in[ 1 ].gl_Position.xyz - gl_in[ 0 ].gl_Position.xyz, gl_in[ 2 ].gl_Position.xyz - gl_in[ 0 ].gl_Position.xyz );
//
//	// We select axes projection
//	if ( ( abs( normal.x ) > abs( normal.y ) ) && ( abs( normal.x ) > abs( normal.z ) ) )
//	{
//		// X is the max length normal component
//
//		// Along X
//		for ( int i = 0; i < gl_in.length(); i++ )
//		{
//			// copy attributes
//			gl_Position = uProjectionMatX * gl_in[ i ].gl_Position;
//			oData.normal = iData[ i ].normal;
//			oData.worldPosition = iData[ i ].worldPosition;
//
//			// Emit vertex
//			EmitVertex();
//		}
//	}
//	else if ( ( abs( normal.y ) > abs( normal.x ) ) && ( abs( normal.y ) > abs( normal.z ) ) )
//	{
//		// Y is the max length normal component
//
//		// Along Y
//		for ( int i = 0; i < gl_in.length(); i++ )
//		{
//			// copy attributes
//			gl_Position = uProjectionMatY * gl_in[ i ].gl_Position;
//			oData.normal = iData[ i ].normal;
//			oData.worldPosition = iData[ i ].worldPosition;
//
//			// Emit vertex
//			EmitVertex();
//		}
//	}
//	else
//	{
//		// Z is the max length normal component
//
//		// Along Z
//		for ( int i = 0; i < gl_in.length(); i++ )
//		{
//			// copy attributes
//			gl_Position = uProjectionMatZ * gl_in[ i ].gl_Position;
//			oData.normal = iData[ i ].normal;
//			oData.worldPosition = iData[ i ].worldPosition;
//
//			// Emit vertex
//			EmitVertex();
//		}
//	}
//}

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
////////////////////////////////////////////////////////////////////////////////
void main()
{
	// To store 3 well projected vertex
	vec4 v[ 3 ];

	// To store the bounding box
//	vec4 aabb;

	// We compute normal with cross product
	vec3 normal = cross( gl_in[ 1 ].gl_Position.xyz - gl_in[ 0 ].gl_Position.xyz, gl_in[ 2 ].gl_Position.xyz - gl_in[ 0 ].gl_Position.xyz );

	// We select axes projection
	if ( ( abs( normal.x ) > abs( normal.y ) ) && ( abs( normal.x ) > abs( normal.z ) ) )
	{
		// Along X
		for ( int i = 0; i < gl_in.length(); i++ )
		{
			// Copy attributes
			v[ i ] = uProjectionMatX * gl_in[ i ].gl_Position;

			//oData.normal = iData[i].normal;
			//oData.worldPosition = iData[i].worldPosition;

			// done with the vertex
			//EmitVertex();
		}
	}
	else if ( ( abs( normal.y ) > abs( normal.x ) ) && ( abs( normal.y ) > abs( normal.z ) ) )
	{
		// Along Y
		for ( int i = 0; i < gl_in.length(); i++ )
		{
			// Copy attributes
			v[ i ] = uProjectionMatY * gl_in[i].gl_Position;
			
			//oData.normal = iData[i].normal;
			//oData.worldPosition = iData[i].worldPosition;

			// done with the vertex
			//EmitVertex();
		}
	}
	else
	{
		// Along Z
		for ( int i = 0; i < gl_in.length(); i++ )
		{
			// Copy attributes
			v[i] = uProjectionMatZ * gl_in[i].gl_Position;

			//oData.normal = iData[i].normal;
			//oData.worldPosition = iData[i].worldPosition;

			// done with the vertex
			//EmitVertex();
		}
	}

	// Conservative rasterization : 

	// vec2 hPixel = vec2( 1.0 / 10.0, 1.0 / 10.0 );

	// // Compute bounding box
	// aabb.xy = v[0].xy;
	// aabb.zw = v[0].xy;

	// aabb.xy = min( aabb.xy, v[1].xy );
	// aabb.zw = max( aabb.zw, v[1].xy );

	// aabb.xy = min( aabb.xy, v[2].xy );
	// aabb.zw = max( aabb.zw, v[2].xy );

	// // Extend bounding box
	// aabb.xy -= vec2(hPixel);
	// aabb.zw += vec2(hPixel);

	// // Compute planes and translate them
	// vec3 plane[3];
	// plane[0] = cross(v[0].xyw - v[2].xyw, v[2].xyw);
	// plane[1] = cross(v[1].xyw - v[0].xyw, v[0].xyw);
	// plane[2] = cross(v[2].xyw - v[1].xyw, v[1].xyw);
	// plane[0].z -= dot(hPixel, abs(plane[0].xy));
	// plane[1].z -= dot(hPixel, abs(plane[1].xy));
	// plane[2].z -= dot(hPixel, abs(plane[2].xy));

	// // Compute the intersection point of the planes
	// v[0].xyw = cross(plane[0], plane[1]);
	// v[1].xyw = cross(plane[1], plane[2]);
	// v[2].xyw = cross(plane[2], plane[0]);
	// v[0].xyw /= v[0].w;
	// v[1].xyw /= v[1].w;
	// v[2].xyw /= v[2].w;

	// We emit vertex
	for ( int i = 0; i < gl_in.length(); i++ )
	{
		gl_Position = v[ i ];

		oData.normal = iData[ i ].normal;
		oData.worldPosition = iData[ i ].worldPosition;
		//oData.aabb = aabb;
		//oData.clipSpace = v[ i ].xy;

		EmitVertex();
	}
}
