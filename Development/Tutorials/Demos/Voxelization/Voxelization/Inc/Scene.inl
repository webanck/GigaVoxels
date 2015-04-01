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

/******************************************************************************
 * ...
 ******************************************************************************/
inline uint3 Scene::getFather( const uint3& locCode ) const
{
	return locCode / make_uint3( 2 );
}

/******************************************************************************
 * Compute global index of a node in the node buffer given its depth and code localization info
 *
 * TODO : use generic code => only valid for octree...
 *
 * @param pDepth node's depth localization info
 * @param pCode node's code localization info
 *
 * return node's global index in the node buffer
 ******************************************************************************/
inline unsigned int Scene::getIndex( unsigned int pDepth, const uint3& pCode ) const
{
	// Compute the basis of the code
	const unsigned int b = static_cast< unsigned int >( powf( 2, pDepth ) );	// nb nodes at given depth d

	return ( powf( 8.f, static_cast< float >( pDepth ) ) - 1.f ) / static_cast< float >( 7.f ) + ( pCode.x + pCode.y * b + pCode.z * b * b );
}

/******************************************************************************
 * ...
 ******************************************************************************/
inline bool Scene::triangleIntersectBick( const float3& brickPos, 
										  const float3& brickSize,  
										  unsigned int triangleIndex, 
										  const std::vector< unsigned int >& IBO, 
										  const float* vertices )
{
	// We assume here that triangle are much smaller than brick ( at least as small as voxel's brick ) so we simplify the intersection test.
	// We only test if triangle's vertices are in the bick
	
	// Test vertices
	return vertexIsInBrick( brickPos, brickSize, IBO[ triangleIndex + 0 ], vertices ) || 
		   vertexIsInBrick( brickPos, brickSize, IBO[ triangleIndex + 1 ], vertices ) ||
		   vertexIsInBrick( brickPos, brickSize, IBO[ triangleIndex + 2 ], vertices ) ;
}

/******************************************************************************
 * ...
 ******************************************************************************/
inline bool Scene::vertexIsInBrick( const float3& brickPos, 
								   const float3& brickSize, 
							       unsigned int vertexIndex,
							       const float* vertices ) 
{
	return ( ( brickPos.x <= vertices[ 3 * vertexIndex + 0 ] && brickPos.x + brickSize.x >= vertices[ 3 * vertexIndex + 0 ] ) &&
		     ( brickPos.y <= vertices[ 3 * vertexIndex + 1 ] && brickPos.y + brickSize.y >= vertices[ 3 * vertexIndex + 1 ] ) &&
			 ( brickPos.z <= vertices[ 3 * vertexIndex + 2 ] && brickPos.z + brickSize.z >= vertices[ 3 * vertexIndex + 2 ] ) ) ;

}
