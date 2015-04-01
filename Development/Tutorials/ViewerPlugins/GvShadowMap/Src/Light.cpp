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

#include "Light.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <vector_types.h>

// STL
#include <iostream>

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// STL
using namespace std;

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
Light::Light()
:	_intensity( make_float3( 0.f, 0.f, 0.f ) )
,	_direction( make_float3( 0.f, 0.f, 0.f ) )
,	_position( make_float4( 0.f, 0.f, 0.f, 0.f ) )
,	_vertexArray( 0 )
,	_vertexBuffer( 0 )
,	_indexBuffer( 0 )
,	_nbVertices( 0 )
,	_nbFaces( 0 )
,	_color( make_float3( 1.f, 1.f, 1.f ) )
,	_eye( glm::vec3( 0.f ) )
,	_center( glm::vec3( 0.f ) )
,	_up( glm::vec3( 0.f ) )
,	_fovY( 0.f )
,	_aspect( 0.f )
,	_zNear( 0.f )
,	_zFar( 0.f )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
Light::~Light()
{
	finalize();
}

/******************************************************************************
 * Initialize
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
bool Light::initialize()
{
	// Initialize graphics resources
	std::vector< float3 > vertices;
	std::vector< unsigned int > indices;
	bool statusOK = initializeGraphicsResources( vertices, indices );
	assert( statusOK );
	if ( ! statusOK )
	{
		// Clean data
		//...

		return false;
	}

	return true;
}

/******************************************************************************
 * Finalize
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
bool Light::finalize()
{
	glDeleteBuffers( 1, &_indexBuffer );
	glDeleteBuffers( 1, &_vertexBuffer );
	glDeleteVertexArrays( 1, &_vertexArray );
	
	return true;
}

/******************************************************************************
 * Render
 ******************************************************************************/
void Light::render() const
{
	glBindVertexArray( _vertexArray );
	glDrawElements( GL_TRIANGLES, _nbFaces * 3, GL_UNSIGNED_INT, NULL );
	glBindVertexArray( 0 );
}

/******************************************************************************
 * Initialize graphics resources
 ******************************************************************************/
bool Light::initializeGraphicsResources( const std::vector< float3 >& pVertices, const std::vector< unsigned int >& pIndices )
{
	// Vertex buffer initialization
	assert( pVertices.size() > 0 );
	glGenBuffers( 1, &_vertexBuffer );
	glBindBuffer( GL_ARRAY_BUFFER, _vertexBuffer );
	glBufferData( GL_ARRAY_BUFFER, sizeof( float3 ) * pVertices.size(), &pVertices[ 0 ], GL_STATIC_DRAW );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	// Index buffer initialization
	assert( pIndices.size() > 0 );
	glGenBuffers( 1, &_indexBuffer );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _indexBuffer );
	glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( unsigned int ) * pIndices.size(), &pIndices[ 0 ], GL_STATIC_DRAW );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
	
	// Vertex array initialization
	glGenVertexArrays( 1, &_vertexArray );
	glBindVertexArray( _vertexArray );
	// Vertex position attribute
	glEnableVertexAttribArray( 0 );
	glBindBuffer( GL_ARRAY_BUFFER, _vertexBuffer );
	glVertexAttribPointer( 0/*attribute index*/, 3/*nb components per vertex*/, GL_FLOAT/*type*/, GL_FALSE/*un-normalized*/, 0/*memory stride*/, static_cast< GLubyte* >( NULL )/*byte offset from buffer*/ );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
	// Required for indexed rendering
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _indexBuffer );
	glBindVertexArray( 0 );

	return true;
}

/******************************************************************************
 * Get the associated view matrix
 *
 * @return the associated view matrix
 ******************************************************************************/
glm::mat4 Light::getViewMatrix() const
{
	return glm::lookAt( _eye, _center, _up );
}

/******************************************************************************
 * Get the associated projection matrix
 *
 * @return the associated projection matrix
 ******************************************************************************/
glm::mat4 Light::getProjectionMatrix() const
{
	return glm::perspective( _fovY, _aspect, _zNear, _zFar );
}
