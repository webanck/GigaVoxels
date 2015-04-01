/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2013 INRIA <http://www.inria.fr/>
 *
 * Authors : GigaVoxels Team
 *
 * GigaVoxels is distributed under a dual-license scheme.
 * You can obtain a specific license from Inria at gigavoxels-licensing@inria.fr.
 * Otherwise the default license is the GPL version 3.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/** 
 * @version 1.0
 */

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/
    

/**
* ...
*/
inline uint3 Scene::getFather( uint3 locCode ) const
{
	return locCode / make_uint3( 2 );
}

/**
* ...
*/
inline unsigned int Scene::getIndex( unsigned int d, uint3 locCode ) const
{
	// Compute the basis of the code
	unsigned int b = powf( 2, d );
	// Given a locCode and a depth there is only one index in the octree
	return ( powf( 8, d ) - 1 ) / (float)7 + ( locCode.x + locCode.y * b + locCode.z * b * b );
}

/**
* ...
*/
inline bool Scene::triangleIntersectBick( const float3 & brickPos, 
										  const float3 & brickSize,  
										  unsigned int triangleIndex, 
										  const std::vector<unsigned int> & IBO, 
										  const float *vertices )
{

    // Rmk = this function don't realy compute if a triangle intesect the brick : 
	// we needs to show triangles outside the brick but along axis because of the algorithm

	float3 A;
	A.x = vertices[ 3 * IBO[ triangleIndex + 0 ] + 0 ];
	A.y = vertices[ 3 * IBO[ triangleIndex + 0 ] + 1 ];
	A.z = vertices[ 3 * IBO[ triangleIndex + 0 ] + 2 ];

	float3 B;
	B.x = vertices[ 3 * IBO[ triangleIndex + 1 ] + 0 ];
	B.y = vertices[ 3 * IBO[ triangleIndex + 1 ] + 1 ];
	B.z = vertices[ 3 * IBO[ triangleIndex + 1 ] + 2 ];

	float3 C;
	C.x = vertices[ 3 * IBO[ triangleIndex + 2 ] + 0 ];
	C.y = vertices[ 3 * IBO[ triangleIndex + 2 ] + 1 ];
	C.z = vertices[ 3 * IBO[ triangleIndex + 2 ] + 2 ];

	// We test the projection along the 3 axis
	bool intersect = false;

	float2 a, b, c;
	float4 aabb;

	// Along X 
	a.x = A.y;
	a.y = A.z;
	b.x = B.y;
	b.y = B.z;
	c.x = C.y;
	c.y = C.z;

	aabb.x = brickPos.y; // left
	aabb.z = brickPos.y + brickSize.y; // rigth

	aabb.y = brickPos.z; // bottom
	aabb.w = brickPos.z + brickSize.z; //top

	intersect = intersect || triangleAabbIntersection2D( a, b, c, aabb );

	// Along Y
	a.x = A.z;
	a.y = A.x;
	b.x = B.z;
	b.y = B.x;
	c.x = C.z;
	c.y = C.x;

	aabb.x = brickPos.z; // left
	aabb.z = brickPos.z + brickSize.z; // rigth

	aabb.y = brickPos.x; // bottom
	aabb.w = brickPos.x + brickSize.x; //top

	intersect = intersect || triangleAabbIntersection2D( a, b, c, aabb );

	// Along Z
	a.x = A.x;
	a.y = A.y;
	b.x = B.x;
	b.y = B.y;
	c.x = C.x;
	c.y = C.y;

	aabb.x = brickPos.x; // left
	aabb.z = brickPos.x + brickSize.x; // rigth

	aabb.y = brickPos.y; // bottom
	aabb.w = brickPos.y + brickSize.y; //top

	intersect = intersect || triangleAabbIntersection2D( a, b, c, aabb );

	return intersect;

}



inline bool Scene::triangleAabbIntersection2D( const float2 & a, 
											   const float2 & b, 
											   const float2 & c,
											   const float4 & aabb )
{
	// Here, this a simplification of the intersection test between triangle and aabb in 2 dimension : 
	// we test intersection between aabb and the bounding box of the triangle.

	// We compute the bouding box of the triangle
	float4 aabbTriangle; 
	aabbTriangle.x = a.x ;
	aabbTriangle.y = a.y ;
	aabbTriangle.z = a.x ;
	aabbTriangle.w = a.y ;

	aabbTriangle.x = std::min( b.x, aabbTriangle.x );
	aabbTriangle.y = std::min( b.y, aabbTriangle.y ) ;
	aabbTriangle.z = std::max( b.x, aabbTriangle.z ) ;
	aabbTriangle.w = std::max( b.y, aabbTriangle.w ) ;

	aabbTriangle.x = std::min( c.x, aabbTriangle.x );
	aabbTriangle.y = std::min( c.y, aabbTriangle.y ) ;
	aabbTriangle.z = std::max( c.x, aabbTriangle.z ) ;
	aabbTriangle.w = std::max( c.y, aabbTriangle.w ) ;

	// We test the intersection between the rectangles
	float4 rectIntersection;
	rectIntersection.x = std::max( aabbTriangle.x, aabb.x );
	rectIntersection.y = std::max( aabbTriangle.y, aabb.y );
	rectIntersection.z = std::min( aabbTriangle.z, aabb.z );
	rectIntersection.w = std::min( aabbTriangle.w, aabb.w );

	return rectIntersection.x < rectIntersection.z && rectIntersection.y < rectIntersection.w;
}
