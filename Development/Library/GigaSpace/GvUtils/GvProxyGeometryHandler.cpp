/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2012 INRIA <http://www.inria.fr/>
 *
 * Authors : GigaVoxels Team
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

#include "GvUtils/GvProxyGeometryHandler.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvError.h"

// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// System
#include <cassert>
#include <cstddef>
#include <iostream>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvUtils;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvProxyGeometryHandler::GvProxyGeometryHandler()
:	_vertexBuffer( 0 )
,	_indexBuffer( 0 )
,	_d_vertices( NULL )
,	_nbPoints( 0 )
{
}

/******************************************************************************
 * Desconstructor
 ******************************************************************************/
GvProxyGeometryHandler::~GvProxyGeometryHandler()
{
	finalize();
}

/******************************************************************************
 * Initialize
 *
 * @return a flag to tell wheter or not it succeeds.
 ******************************************************************************/
bool GvProxyGeometryHandler::initialize()
{
	assert( _vertexBuffer == 0 );
	assert( _indexBuffer == 0 );

	// [ 1 ] - Initialize the vertex buffer

	// Generate buffer
	glGenBuffers( 1, &_vertexBuffer );

	// Bind buffer
	glBindBuffer( GL_ARRAY_BUFFER, _vertexBuffer );

	// Allocate buffer
	//const unsigned int nbVertices = 0;
	//const unsigned int nbVertices = 8;	// TEST !!!!
	const unsigned int nbVertices = 1000;
	// Compute number of vertices
	//GLsizeiptr vertexBufferSize = sizeof( GLfloat ) * nbVertices * 3;	// only vertex data (no normals for the moment)
	GLsizeiptr vertexBufferSize = sizeof( GLfloat ) * nbVertices * 3;	// TEST !!!!!!!
	
	// Creates and initializes a buffer object's data store
	//glBufferData( GL_ARRAY_BUFFER, vertexBufferSize, NULL, GL_STATIC_DRAW );
	glBufferData( GL_ARRAY_BUFFER, vertexBufferSize, NULL, GL_DYNAMIC_DRAW );

	/// Fill data
//	GLfloat* vertexBufferData = static_cast< GLfloat* >( glMapBuffer( GL_ARRAY_BUFFER, GL_WRITE_ONLY ) );
//	unsigned int index = 0;
//	for ( size_t i = 0; i < nbVertices; i++ )
//	{
//		vertexBufferData[ index++ ] = 0.5f + 0.5f * cosf( 10.f * 2.f * 3.141592f * i / nbVertices );
//		vertexBufferData[ index++ ] = 0.5f + 0.5f * sinf( 2.f * 3.141592f * i / nbVertices );
//		vertexBufferData[ index++ ] = 0.5f + 0.5f * cosf( 50.f * 2.f * 3.141592f * i / nbVertices );
//	}
////	vertexBufferData[ 0 ] = 0.5f;
////	vertexBufferData[ 1 ] = 0.5;
////	vertexBufferData[ 2 ] = 0.5f;
//	glUnmapBuffer( GL_ARRAY_BUFFER );
	
	// Unbind buffer
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	// [ 2 ] - Initialize the index buffer
	//
	// ...

	// [ 3 ] - Register the graphics resource
	cudaError_t error = cudaGraphicsGLRegisterBuffer( &_d_vertices, _vertexBuffer, cudaGraphicsMapFlagsWriteDiscard );
	//cudaError_t error = cudaGraphicsGLRegisterBuffer( &_d_vertices, _vertexBuffer, cudaGraphicsMapFlagsNone );
	
	GV_CHECK_GL_ERROR();

	return false;
}

/******************************************************************************
 * Finalize
 *
 * @return a flag to tell wheter or not it succeeds.
 ******************************************************************************/
bool GvProxyGeometryHandler::finalize()
{
	// Unregister graphics resources
	cudaGraphicsUnregisterResource( _d_vertices );

	// Delete buffers
	glDeleteBuffers( 1, &_vertexBuffer );
	glDeleteBuffers( 1, &_indexBuffer );

	return false;
}

/******************************************************************************
 * Render
 ******************************************************************************/
void GvProxyGeometryHandler::render()
{
	//// --------------------------------------------------------------
	//// DEBUG
	//glBindBuffer( GL_ARRAY_BUFFER, _vertexBuffer );
	//GLfloat* vertexBufferData = static_cast< GLfloat* >( glMapBuffer( GL_ARRAY_BUFFER, GL_READ_ONLY ) );
	//int j = 0;
	//for ( int i = 0; i < _nbPoints; i++ )
	//{
	//	//printf( "\nvertexBufferData[ %d ] = ( %f, %f, %f, %f )", i, vertexBufferData[ j ], vertexBufferData[ j + 1 ], vertexBufferData[ j + 2 ], vertexBufferData[ j + 3 ] );
	//	//j += 4;
	//	printf( "\nvertexBufferData[ %d ] = ( %f, %f, %f )", i, vertexBufferData[ j ], vertexBufferData[ j + 1 ], vertexBufferData[ j + 2 ] );
	//	j += 3;
	//}
	//glBindBuffer( GL_ARRAY_BUFFER, 0 );
	//// --------------------------------------------------------------

	// Render from buffer object

	//glColor4f( 1.0f, 1.0f, 1.0f, 1.0f );
	glColor3f( 1.0f, 1.0f, 1.0f );
	glPointSize( 5.0f );

	// Bind buffer
	glBindBuffer( GL_ARRAY_BUFFER, _vertexBuffer );
	glEnableClientState( GL_VERTEX_ARRAY );
	glVertexPointer( 3, GL_FLOAT, 0, 0 );

	// Render
	glDrawArrays( GL_POINTS, 0, _nbPoints );
	//glDrawArrays( GL_POINTS, 0, 1000 );

	// Unbind buffer
	glDisableClientState( GL_VERTEX_ARRAY );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
}
