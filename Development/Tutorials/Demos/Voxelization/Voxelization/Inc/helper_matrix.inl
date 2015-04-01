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

//// glm
//#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtx/transform2.hpp>
//#include <glm/gtx/projection.hpp>
//#include <glm/gtc/type_ptr.hpp>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Define a viewing transformation
 *
 * @param eyeX specifies the position of the eye point
 * @param eyeY specifies the position of the eye point
 * @param eyeZ specifies the position of the eye point
 * @param centerX specifies the position of the reference point
 * @param centerY specifies the position of the reference point
 * @param centerZ specifies the position of the reference point
 * @param upX specifies the direction of the up vector
 * @param upY specifies the direction of the up vector
 * @param upZ specifies the direction of the up vector
 *
 * @return the resulting viewing transformation
 ******************************************************************************/
inline float4x4 MatrixHelper::lookAt( float eyeX, float eyeY, float eyeZ,
								   float centerX, float centerY, float centerZ,
								   float upX, float upY, float upZ )
{
	// TO DO : Can be optimized ?
	float3 f = make_float3( centerX - eyeX, centerY - eyeY, centerZ - eyeZ );
	float3 up = make_float3( upX, upY, upZ );
	
	f = normalize( f );
	up = normalize( up );
	float3 s = cross( f, up );
	//s = normalize( s );	// need to normalize ?
	float3 u = cross( s, f );

	float4x4 m;
	// row 1 
	m.element( 0, 0 ) = s.x;
	m.element( 0, 1 ) = s.y;
	m.element( 0, 2 ) = s.z;
	m.element( 0, 3 ) = 0.0f;
	// row 2
	m.element( 1, 0 ) = u.x;
	m.element( 1, 1 ) = u.y;
	m.element( 1, 2 ) = u.z;
	m.element( 1, 3 ) = 0.0f;
	// row 3
	m.element( 2, 0 ) = - f.x;
	m.element( 2, 1 ) = - f.y;
	m.element( 2, 2 ) = - f.z;
	m.element( 2, 3 ) = 0.0f;
	// row 4
	m.element( 3, 0 ) = 0.0f;
	m.element( 3, 1 ) = 0.0f;
	m.element( 3, 2 ) = 0.0f;
	m.element( 3, 3 ) = 1.0f;

	return mul( m, translate( -eyeX, -eyeY, -eyeZ ) );
}

/******************************************************************************
 * Define a translation matrix
 *
 * @param x specify the x, y, and z coordinates of a translation vector
 * @param y specify the x, y, and z coordinates of a translation vector
 * @param z specify the x, y, and z coordinates of a translation vector
 *
 * @return the resulting translation matrix
 ******************************************************************************/
inline float4x4 MatrixHelper::translate( float x, float y, float z)
{
	// TO DO : Can be optimized ?

	float4x4 res;

	// row 1 
	res.element( 0, 0 ) = 1.0f;
	res.element( 0, 1 ) = 0.0f;
	res.element( 0, 2 ) = 0.0f;
	res.element( 0, 3 ) = x;
	
	// row 2
	res.element( 1, 0 ) = 0.0f;
	res.element( 1, 1 ) = 1.0f;
	res.element( 1, 2 ) = 0.0f;
	res.element( 1, 3 ) = y;
	
	// row 3
	res.element( 2, 0 ) = 0.0f;
	res.element( 2, 1 ) = 0.0f;
	res.element( 2, 2 ) = 1.0f;
	res.element( 2, 3 ) = z;
	
	// row 4
	res.element( 3, 0 ) = 0.0f;
	res.element( 3, 1 ) = 0.0f;
	res.element( 3, 2 ) = 0.0f;
	res.element( 3, 3 ) = 1.0f;

	return res;
}

/******************************************************************************
 * Multiply two matrices
 *
 * @param a first matrix
 * @param b second matrix
 *
 * @return the resulting matrix
 ******************************************************************************/
inline float4x4 MatrixHelper::mul( const float4x4& a, const float4x4& b )
{
	float4x4 res;

	// row 1
	res.element( 0, 0 ) = a.element( 0, 0 ) * b.element( 0, 0 ) + 
					      a.element( 0, 1 ) * b.element( 1, 0 ) + 
						  a.element( 0, 2 ) * b.element( 2, 0 ) +
						  a.element( 0, 3 ) * b.element( 3, 0 ) ;

	res.element( 0, 1 ) = a.element( 0, 0 ) * b.element( 0, 1 ) + 
					      a.element( 0, 1 ) * b.element( 1, 1 ) + 
						  a.element( 0, 2 ) * b.element( 2, 1 ) +
						  a.element( 0, 3 ) * b.element( 3, 1 ) ;

	res.element( 0, 2 ) = a.element( 0, 0 ) * b.element( 0, 2 ) + 
					      a.element( 0, 1 ) * b.element( 1, 2 ) + 
						  a.element( 0, 2 ) * b.element( 2, 2 ) +
						  a.element( 0, 3 ) * b.element( 3, 2 ) ;

	res.element( 0, 3 ) = a.element( 0, 0 ) * b.element( 0, 3 ) + 
					      a.element( 0, 1 ) * b.element( 1, 3 ) + 
						  a.element( 0, 2 ) * b.element( 2, 3 ) +
						  a.element( 0, 3 ) * b.element( 3, 3 ) ;

	// row 1
	res.element( 1, 0 ) = a.element( 1, 0 ) * b.element( 0, 0 ) + 
					      a.element( 1, 1 ) * b.element( 1, 0 ) + 
						  a.element( 1, 2 ) * b.element( 2, 0 ) +
						  a.element( 1, 3 ) * b.element( 3, 0 ) ;

	res.element( 1, 1 ) = a.element( 1, 0 ) * b.element( 0, 1 ) + 
					      a.element( 1, 1 ) * b.element( 1, 1 ) + 
						  a.element( 1, 2 ) * b.element( 2, 1 ) +
						  a.element( 1, 3 ) * b.element( 3, 1 ) ;

	res.element( 1, 2 ) = a.element( 1, 0 ) * b.element( 0, 2 ) + 
					      a.element( 1, 1 ) * b.element( 1, 2 ) + 
						  a.element( 1, 2 ) * b.element( 2, 2 ) +
						  a.element( 1, 3 ) * b.element( 3, 2 ) ;

	res.element( 1, 3 ) = a.element( 1, 0 ) * b.element( 0, 3 ) + 
					      a.element( 1, 1 ) * b.element( 1, 3 ) + 
						  a.element( 1, 2 ) * b.element( 2, 3 ) +
						  a.element( 1, 3 ) * b.element( 3, 3 ) ;

	// row 2
	res.element( 2, 0 ) = a.element( 2, 0 ) * b.element( 0, 0 ) + 
					      a.element( 2, 1 ) * b.element( 1, 0 ) + 
						  a.element( 2, 2 ) * b.element( 2, 0 ) +
						  a.element( 2, 3 ) * b.element( 3, 0 ) ;

	res.element( 2, 1 ) = a.element( 2, 0 ) * b.element( 0, 1 ) + 
					      a.element( 2, 1 ) * b.element( 1, 1 ) + 
						  a.element( 2, 2 ) * b.element( 2, 1 ) +
						  a.element( 2, 3 ) * b.element( 3, 1 ) ;

	res.element( 2, 2 ) = a.element( 2, 0 ) * b.element( 0, 2 ) + 
					      a.element( 2, 1 ) * b.element( 1, 2 ) + 
						  a.element( 2, 2 ) * b.element( 2, 2 ) +
						  a.element( 2, 3 ) * b.element( 3, 2 ) ;

	res.element( 2, 3 ) = a.element( 2, 0 ) * b.element( 0, 3 ) + 
					      a.element( 2, 1 ) * b.element( 1, 3 ) + 
						  a.element( 2, 2 ) * b.element( 2, 3 ) +
						  a.element( 2, 3 ) * b.element( 3, 3 ) ;
	// row 3
	res.element( 3, 0 ) = a.element( 3, 0 ) * b.element( 0, 0 ) + 
					      a.element( 3, 1 ) * b.element( 1, 0 ) + 
						  a.element( 3, 2 ) * b.element( 2, 0 ) +
						  a.element( 3, 3 ) * b.element( 3, 0 ) ;

	res.element( 3, 1 ) = a.element( 3, 0 ) * b.element( 0, 1 ) + 
					      a.element( 3, 1 ) * b.element( 1, 1 ) + 
						  a.element( 3, 2 ) * b.element( 2, 1 ) +
						  a.element( 3, 3 ) * b.element( 3, 1 ) ;

	res.element( 3, 2 ) = a.element( 3, 0 ) * b.element( 0, 2 ) + 
					      a.element( 3, 1 ) * b.element( 1, 2 ) + 
						  a.element( 3, 2 ) * b.element( 2, 2 ) +
						  a.element( 3, 3 ) * b.element( 3, 2 ) ;

	res.element( 3, 3 ) = a.element( 3, 0 ) * b.element( 0, 3 ) + 
					      a.element( 3, 1 ) * b.element( 1, 3 ) + 
						  a.element( 3, 2 ) * b.element( 2, 3 ) +
						  a.element( 3, 3 ) * b.element( 3, 3 ) ;
	return res;

}

/******************************************************************************
 * Copy data from input matrix to output matrix
 *
 * @param pInputMatrix input matrix
 * @param pOutpuMatrix outpu matrix
 ******************************************************************************/
inline void MatrixHelper::copy( const float4x4& pInputMatrix, float4x4& pOutpuMatrix )
{
	memcpy( pOutpuMatrix._array, pInputMatrix._array, sizeof( float ) * 16 );
}

/******************************************************************************
 * Define a transformation that produces a parallel projection (i.e. orthographic)
 *
 * @param left specify the coordinates for the left and right vertical clipping planes
 * @param right specify the coordinates for the left and right vertical clipping planes
 * @param bottom specify the coordinates for the bottom and top horizontal clipping planes
 * @param top specify the coordinates for the bottom and top horizontal clipping planes
 * @param nearVal specify the distances to the nearer and farther depth clipping planes. These values are negative if the plane is to be behind the viewer.
 * @param farVal specify the distances to the nearer and farther depth clipping planes. These values are negative if the plane is to be behind the viewer.
 *
 * @return the resulting orthographic matrix
 ******************************************************************************/
inline float4x4 MatrixHelper::ortho( float left, float right, float bottom, float top,	float nearVal, float farVal )
{
	// TO DO : Can be optimized ?
	float4x4 res;

	// row 1 
	res.element( 0, 0 ) = 2.0 / ( right - left );
	res.element( 0, 1 ) = 0.0;
	res.element( 0, 2 ) = 0.0;
	res.element( 0, 3 ) = - ( right + left ) / ( right - left );

	// row 2
	res.element( 1, 0 ) = 0.0;
	res.element( 1, 1 ) = 2.0 / ( top - bottom );
	res.element( 1, 2 ) = 0.0;
	res.element( 1, 3 ) = - ( top + bottom ) / ( top - bottom );

	// row 3
	res.element( 2, 0 ) = 0.0;
	res.element( 2, 1 ) = 0.0;
	res.element( 2, 2 ) = - 2.0 / ( farVal - nearVal );
	res.element( 2, 3 ) = - ( farVal + nearVal ) / ( farVal - nearVal );

	// row 4
	res.element( 3, 0 ) = 0.0;
	res.element( 3, 1 ) = 0.0;
	res.element( 3, 2 ) = 0.0;
	res.element( 3, 3 ) = 1.0;

	return res;
}

/******************************************************************************
 * Create matrix used to change of reference frame matrix associated to a brick
 *
 * ...
 *
 * @param pBrickPos position of the brick (same as the position of the node minus the border)
 * @param pXSize x size of the brick ( same as the size of the node plus the border )
 * @param pYSize y size of the brick ( same as the size of the node plus the border )
 * @param pZSize z size of the brick ( same as the size of the node plus the border )
 *
 * @return ...
 ******************************************************************************/
inline float4x4 MatrixHelper::brickBaseMatrix( const float3& pBrickPos, float pXSize, float pYSize, float pZSize )
{
	float4x4 res;
	
	//		|  x  0  0  Tx  |
	//		|  0  y  0  Ty  |
	//		|  0  0  z  Tz  |
	//		|  0  0  0   1  |

	// Row 1 
	res.element( 0, 0 ) = 1.0f / pXSize;	// scale
	res.element( 0, 1 ) = 0.0f;
	res.element( 0, 2 ) = 0.0f;
	res.element( 0, 3 ) = - ( 1.0f / pXSize ) * pBrickPos.x;	// translate
	
	// Row 2
	res.element( 1, 0 ) = 0.0f;
	res.element( 1, 1 ) = 1.0f / pYSize;	// scale
	res.element( 1, 2 ) = 0.0f;
	res.element( 1, 3 ) = - ( 1.0f / pYSize ) * pBrickPos.y;	// translate

	// Row 3
	res.element( 2, 0 ) = 0.0f;
	res.element( 2, 1 ) = 0.0f;
	res.element( 2, 2 ) = 1.0f / pZSize;	// scale
	res.element( 2, 3 ) = - ( 1.0f / pZSize ) * pBrickPos.z;	// translate
	
	// Row 4
	res.element( 3, 0 ) = 0.0f;
	res.element( 3, 1 ) = 0.0f;
	res.element( 3, 2 ) = 0.0f;
	res.element( 3, 3 ) = 1.0f;

	return res;
}

/******************************************************************************
 * Methods that compute Change-of-basis matrices to project along the 3 axis.
 * Those matrices are the multiplication of the openGL modelViewMatrix with projectionMatrix
 * after a call to gluLookAt(...) and glortho(...).
 *
 * @param brickPos Origin of the brick's base
 * @param xSize Size of the brick along x axe
 * @param ySize Size of the brick along y axe
 * @param zSize Size of the brick along z axe
 * @param projectionMatX Change-of-basis matrix to project along X
 * @param projectionMatY Change-of-basis matrix to project along y
 * @param projectionMatZ Change-of-basis matrix to project along Z
 ******************************************************************************/
inline void MatrixHelper::projectionMatrix( const float3& brickPos, float xSize, float ySize, float zSize,
											float4x4& projectionMatX, float4x4& projectionMatY, float4x4& projectionMatZ ) 
{
	/*glm::mat4 projectionMatrix;
	glm::mat4 modelViewMatrix;
	glm::mat4 xAxisMVP;
	glm::mat4 yAxisMVP;
	glm::mat4 zAxisMVP;*/

	// In this method we only make what gluLookAt and glOrtho make, and we multiply the 2 matrices
	
	// X axis viewMatrix
	copy( /*input matrix*/mul( ortho( /*left*/0.f, /*right*/ySize, /*bottom*/0.f, /*top*/zSize, /*near*/0.f, /*far*/xSize ),
		        lookAt( /*eye*/brickPos.x + xSize, brickPos.y, brickPos.z,
						/*center*/brickPos.x, brickPos.y, brickPos.z,
						/*up*/0.0f, 0.0f, 1.0f ) ), 
		   /*output matrix*/projectionMatX );

	//-------------------------------
	//modelViewMatrix = glm::lookAt(	/*eye*/glm::vec3( brickPos.x + xSize, brickPos.y, brickPos.z ), 
	//								/*center*/glm::vec3( brickPos.x, brickPos.y, brickPos.z ), 
	//								/*up*/glm::vec3( 0.0f, 0.0f, 1.0f ) );
	//projectionMatrix = glm::ortho(	/*left*/0.f, /*right*/ySize,
	//								/*bottom*/0.f, /*top*/zSize,
	//								/*near*/0.f, /*far*/xSize );
	//xAxisMVP = projectionMatrix * modelViewMatrix;
	//-------------------------------

	// Y axis viewMatrix
	copy( mul( ortho( 0.f, zSize, 0.f, xSize, 0.f, ySize ),
		        lookAt( /*eye*/brickPos.x , brickPos.y + ySize, brickPos.z,
			            /*center*/brickPos.x, brickPos.y, brickPos.z,
			            /*up*/1.0f, 0.0f, 0.0f ) ), 
		   /*output matrix*/projectionMatY );

	//-------------------------------
	//modelViewMatrix = glm::lookAt(	/*eye*/glm::vec3( brickPos.x , brickPos.y + ySize, brickPos.z ), 
	//								/*center*/glm::vec3( brickPos.x, brickPos.y, brickPos.z ), 
	//								/*up*/glm::vec3( 1.0f, 0.0f, 0.0f ) );
	//projectionMatrix = glm::ortho( 0.f, zSize, 0.f, xSize, 0.f, ySize );
	//yAxisMVP = projectionMatrix * modelViewMatrix;
	//-------------------------------

	// Z axis viewMatrixs
	copy( mul( ortho( 0.f, xSize, 0.f, ySize, 0.f, zSize ),
		        lookAt( /*eye*/brickPos.x , brickPos.y , brickPos.z + zSize,
						/*center*/brickPos.x, brickPos.y, brickPos.z,
						/*up*/0.0f, 1.0f, 0.0f ) ), 
		   /*output matrix*/projectionMatZ );

	//-------------------------------
	//modelViewMatrix = glm::lookAt(	/*eye*/glm::vec3( brickPos.x , brickPos.y , brickPos.z + zSize ), 
	//								/*center*/glm::vec3( brickPos.x, brickPos.y, brickPos.z ), 
	//								/*up*/glm::vec3( 0.0f, 1.0f, 0.0f ) );
	//projectionMatrix = glm::ortho( 0.f, xSize, 0.f, ySize, 0.f, zSize );
	//zAxisMVP = projectionMatrix * modelViewMatrix;
	//-------------------------------
}
