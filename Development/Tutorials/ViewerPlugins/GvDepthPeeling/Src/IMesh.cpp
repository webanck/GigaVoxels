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

#include "IMesh.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <vector_types.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>

// STL
#include <iostream>

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvUtils;

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
IMesh::IMesh()
:	_shaderProgram( NULL )
,	_vertexArray( 0 )
,	_vertexBuffer( 0 )
,	_normalBuffer( 0 )
,	_texCoordsBuffer( 0 )
,	_indexBuffer( 0 )
,	_useInterleavedBuffers( false )
,	_nbVertices( 0 )
,	_nbFaces( 0 )
,	_hasNormals( false )
,	_hasTextureCoordinates( false )
,	_useIndexedRendering( true )
,	_color( make_float3( 1.f, 1.f, 1.f ) )
,	_isWireframeEnabled( false )
,	_wireframeColor( make_float3( 0.f, 0.f, 0.f ) )
,	_wireframeLineWidth( 1.f )
,	_shaderProgramConfiguration()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
IMesh::~IMesh()
{
	finalize();
}

/******************************************************************************
 * Initialize
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
bool IMesh::initialize()
{
	// Initialize shader program
	_shaderProgram = new GvShaderProgram();
		
	return true;
}

/******************************************************************************
 * Finalize
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
bool IMesh::finalize()
{
	delete _shaderProgram;
	_shaderProgram = NULL;

	glDeleteBuffers( 1, &_indexBuffer );
	glDeleteBuffers( 1, &_texCoordsBuffer );
	glDeleteBuffers( 1, &_normalBuffer );
	glDeleteBuffers( 1, &_vertexBuffer );
	glDeleteVertexArrays( 1, &_vertexArray );
	
	return true;
}

/******************************************************************************
 * Load mesh
 ******************************************************************************/
bool IMesh::load( const char* pFilename )
{
	assert( pFilename != NULL );
	if ( pFilename == NULL )
	{
		return false;
	}

	// Read mesh data
	std::vector< float3 > vertices;
	std::vector< float3 > normals;
	std::vector< float2 > texCoords;
	std::vector< unsigned int > indices;
	bool statusOK = read( pFilename, vertices, normals, texCoords, indices );
	assert( statusOK );
	if ( ! statusOK )
	{
		// Clean data
		//...

		return false;
	}

	// Initialize graphics resources
	statusOK = initializeGraphicsResources( vertices, normals, texCoords, indices );
	assert( statusOK );
	if ( ! statusOK )
	{
		// Clean data
		//...

		return false;
	}

	// Initialize shader program
	statusOK = initializeShaderProgram();
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
 * Set shader program configuration
 ******************************************************************************/
void IMesh::setShaderProgramConfiguration( const IMesh::ShaderProgramConfiguration& pShaderProgramConfiguration )
{
	// TO DO
	// - clean internal state
	// - shader program too ?
	_shaderProgramConfiguration.reset();

	_shaderProgramConfiguration = pShaderProgramConfiguration;
}

/******************************************************************************
 * Initialize shader program
 ******************************************************************************/
bool IMesh::initializeShaderProgram()
{
	/*shaders[ GvShaderProgram::eVertexShader ] = "";
	shaders[ GvShaderProgram::eTesselationControlShader ] = "";
	shaders[ GvShaderProgram::eTesselationEvaluationShader ] = "";
	shaders[ GvShaderProgram::eGeometryShader ] = "";
	shaders[ GvShaderProgram::eFragmentShader ] = "";
	shaders[ GvShaderProgram::eComputeShader ] = "";*/

	bool statusOK;

	// Initialize shader program
	_shaderProgram = new GvShaderProgram();
	assert( _shaderProgram != NULL );
	
	// Iterate through shaders
	std::map< GvShaderProgram::ShaderType, std::string >::const_iterator shaderIt = _shaderProgramConfiguration._shaders.begin();
	for ( ; shaderIt != _shaderProgramConfiguration._shaders.end(); ++shaderIt )
	{
		statusOK = _shaderProgram->addShader( shaderIt->first, shaderIt->second );
		assert( statusOK );
	}
	statusOK = _shaderProgram->link();
	assert( statusOK );

	return true;
}

/******************************************************************************
 * Read mesh data
 ******************************************************************************/
bool IMesh::read( const char* pFilename, std::vector< float3 >& pVertices, std::vector< float3 >& pNormals, std::vector< float2 >& pTexCoords, std::vector< unsigned int >& pIndices )
{
	return true;
}

/******************************************************************************
 * Read mesh
 ******************************************************************************/
bool IMesh::initializeGraphicsResources( std::vector< float3 >& pVertices, std::vector< float3 >& pNormals, std::vector< float2 >& pTexCoords, std::vector< unsigned int >& pIndices )
{
	// Vertex buffer initialization
	assert( pVertices.size() > 0 );
	glGenBuffers( 1, &_vertexBuffer );
	glBindBuffer( GL_ARRAY_BUFFER, _vertexBuffer );
	glBufferData( GL_ARRAY_BUFFER, sizeof( float3 ) * pVertices.size(), &pVertices[ 0 ], GL_STATIC_DRAW );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	// Normal buffer initialization
	if ( _hasNormals )
	{
		assert( pNormals.size() > 0 );
		glGenBuffers( 1, &_normalBuffer );
		glBindBuffer( GL_ARRAY_BUFFER, _normalBuffer );
		glBufferData( GL_ARRAY_BUFFER, sizeof( float3 ) * pNormals.size(), &pNormals[ 0 ], GL_STATIC_DRAW );
		glBindBuffer( GL_ARRAY_BUFFER, 0 );
	}

	// Textute coordinates buffer initialization
	if ( _hasTextureCoordinates )
	{
		assert( pTexCoords.size() > 0 );
		glGenBuffers( 1, &_texCoordsBuffer );
		glBindBuffer( GL_ARRAY_BUFFER, _texCoordsBuffer );
		glBufferData( GL_ARRAY_BUFFER, sizeof( float2 ) * pTexCoords.size(), &pTexCoords[ 0 ], GL_STATIC_DRAW );
		glBindBuffer( GL_ARRAY_BUFFER, 0 );
	}

	// Index buffer initialization
	if ( _useIndexedRendering )
	{
		assert( pIndices.size() > 0 );
		glGenBuffers( 1, &_indexBuffer );
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _indexBuffer );
		glBufferData( GL_ELEMENT_ARRAY_BUFFER, sizeof( unsigned int ) * pIndices.size(), &pIndices[ 0 ], GL_STATIC_DRAW );
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
	}

	// Vertex array initialization
	glGenVertexArrays( 1, &_vertexArray );
	glBindVertexArray( _vertexArray );
	// Vertex position attribute
	glEnableVertexAttribArray( 0 );
	glBindBuffer( GL_ARRAY_BUFFER, _vertexBuffer );
	glVertexAttribPointer( 0/*attribute index*/, 3/*nb components per vertex*/, GL_FLOAT/*type*/, GL_FALSE/*un-normalized*/, 0/*memory stride*/, static_cast< GLubyte* >( NULL )/*byte offset from buffer*/ );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
	// Vertex normal attribute
	if ( _hasNormals )
	{
		glEnableVertexAttribArray( 1 );
		glBindBuffer( GL_ARRAY_BUFFER, _normalBuffer );
		glVertexAttribPointer( 1/*attribute index*/, 3/*nb components per vertex*/, GL_FLOAT/*type*/, GL_FALSE/*un-normalized*/, 0/*memory stride*/, static_cast< GLubyte* >( NULL )/*byte offset from buffer*/ );
		glBindBuffer( GL_ARRAY_BUFFER, 0 );
	}
	// Vertex texture coordinates attribute
	if ( _hasTextureCoordinates )
	{
		glEnableVertexAttribArray( 2 );
		glBindBuffer( GL_ARRAY_BUFFER, _texCoordsBuffer );
		glVertexAttribPointer( 2/*attribute index*/, 2/*nb components per vertex*/, GL_FLOAT/*type*/, GL_FALSE/*un-normalized*/, 0/*memory stride*/, static_cast< GLubyte* >( NULL )/*byte offset from buffer*/ );
		glBindBuffer( GL_ARRAY_BUFFER, 0 );
	}
	// Required for indexed rendering
	if ( _useIndexedRendering )
	{
		glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _indexBuffer );
	}
	glBindVertexArray( 0 );

	return true;
}

/******************************************************************************
 * Render
 ******************************************************************************/
void IMesh::render( const float4x4& pModelViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	// Configure rendering state
	_shaderProgram->use();
	glBindVertexArray( _vertexArray );

	// Set uniforms
	GLuint location = glGetUniformLocation( _shaderProgram->_program, "uModelViewMatrix" );
	if ( location >= 0 )
	{
		glUniformMatrix4fv( location, 1, GL_FALSE, pModelViewMatrix._array );
	}
	location = glGetUniformLocation( _shaderProgram->_program, "uProjectionMatrix" );
	if ( location >= 0 )
	{
		glUniformMatrix4fv( location, 1, GL_FALSE, pProjectionMatrix._array );
	}
	location = glGetUniformLocation( _shaderProgram->_program, "uMeshColor" );
	if ( location >= 0 )
	{
		float4 color = make_float4( 1.f, 0.f, 0.f, 1.f );
		glUniform4f( location, color.x, color.y, color.z, color.w );
	}
	if ( _hasNormals )
	{
		location = glGetUniformLocation( _shaderProgram->_program, "uNormalMatrix" );
		if ( location >= 0 )
		{
			// TO DO
			// - retrieve or compute Normal matrix
			//
			float normalMatrix[ 9 ];
			glUniformMatrix3fv( location, 1, GL_FALSE, normalMatrix );
		}
	}

	// Draw mesh
	//glDrawArrays( GL_TRIANGLES, 0, 0 );
	glDrawElements( GL_TRIANGLES, _nbFaces * 3, GL_UNSIGNED_INT, NULL );
	
	// Reset rendering state
	glBindVertexArray( 0 );
	glUseProgram( 0 );
}

/******************************************************************************
 * Get the shape color
 *
 * @return the shape color
 ******************************************************************************/
const float3& IMesh::getColor() const
{
	return _color;
}

/******************************************************************************
 * Set the shape color
	 *
	 * @param pColor the shape color
 ******************************************************************************/
void IMesh::setColor( const float3& pColor )
{
	_color = pColor;
}

/******************************************************************************
 * Get the shape color
 *
 * @return the shape color
 ******************************************************************************/
bool IMesh::isWireframeEnabled() const
{
	return _isWireframeEnabled;
}

/******************************************************************************
 * Set the shape color
	 *
	 * @param pColor the shape color
 ******************************************************************************/
void IMesh::setWireframeEnabled( bool pFlag )
{
	_isWireframeEnabled = pFlag;
}

/******************************************************************************
 * Get the shape color
 *
 * @return the shape color
 ******************************************************************************/
const float3& IMesh::getWireframeColor() const
{
	return _wireframeColor;
}

/******************************************************************************
 * Set the shape color
	 *
	 * @param pColor the shape color
 ******************************************************************************/
void IMesh::setWireframeColor( const float3& pColor )
{
	_wireframeColor = pColor;
}

/******************************************************************************
 * Get the shape color
 *
 * @return the shape color
 ******************************************************************************/
float IMesh::getWireframeLineWidth() const
{
	return _wireframeLineWidth;
}

/******************************************************************************
 * Set the shape color
	 *
	 * @param pColor the shape color
 ******************************************************************************/
void IMesh::setWireframeLineWidth( float pValue )
{
	_wireframeLineWidth = pValue;
}
