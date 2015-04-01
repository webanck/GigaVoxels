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

#include "ParticleSystem.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// System
#include <cassert>

// STL
#include <iostream>

// Qt
#include <QFile>
#include <QDomDocument>
#include <QDomElement>

// Cuda SDK
#include <helper_math.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

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
ParticleSystem::ParticleSystem()
{
	// TEST
	_vertexBuffer = 0;
	_hasBrickDrawOneSlice = true;
	_brickNbPoints = 0;
	_pointSize = 1.0f;
	const unsigned int brickResolution = 8/*BrickRes::x*/;
	for ( unsigned int z = 0; z < brickResolution; z++ )
	{
		for ( unsigned int y = 0; y < brickResolution; y++ )
		{
			for ( unsigned int x = 0; x < brickResolution; x++ )
			{
				_brickPresenceFlags[ x ][ y ][ z ] = 0;
			}
		}
	}

	//initialize();
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
ParticleSystem::~ParticleSystem()
{
	finalize();
}

/******************************************************************************
 * Initialize
 ******************************************************************************/
bool ParticleSystem::initialize()
{
	// Vertex buffer
	glGenBuffers( 1, &_vertexBuffer );

	return false;
}

/******************************************************************************
 * Finalize
 ******************************************************************************/
bool ParticleSystem::finalize()
{
	// Vertex buffer
	glDeleteBuffers( 1, &_vertexBuffer );

	return false;
}

/******************************************************************************
 * Load data from file
 *
 * @param pFilename the filename
 ******************************************************************************/
bool ParticleSystem::load()
{
	// Vertex buffer
	if ( _vertexBuffer != 0 )
	{
		glDeleteBuffers( 1, &_vertexBuffer );
		glGenBuffers( 1, &_vertexBuffer );
		_brick.clear();
		_points.clear();
	}

	const int nbRepeat = 128;
	const float levelSize = 1.0f / static_cast< float >( nbRepeat );

	// Build one brick
	const unsigned int brickResolution = 8/*BrickRes::x*/;
	for ( unsigned int z = 0; z < brickResolution; z++ )
	{
		for ( unsigned int y = 0; y < brickResolution; y++ )
		{
			for ( unsigned int x = 0; x < brickResolution; x++ )
			{
				if ( _brickPresenceFlags[ x ][ y ][ z ] == 1 )
				{
					float3 point;
					point.x = static_cast< float >( x ) / ( brickResolution - 1 ) * levelSize;
					point.y = static_cast< float >( y ) / ( brickResolution - 1 ) * levelSize;
					point.z = static_cast< float >( z ) / ( brickResolution - 1 ) * levelSize;

					_brick.push_back( point );
				}
			}
		}
	}

	// Duplicate the brick
	unsigned int nbZRepeat = nbRepeat;
	if ( _hasBrickDrawOneSlice )
	{
		nbZRepeat = 1;
	}
	for ( unsigned int z = 0; z < nbZRepeat; z++ )
	{
		for ( unsigned int y = 0; y < nbRepeat; y++ )
		{
			for ( unsigned int x = 0; x < nbRepeat; x++ )
			{
				// Iterate through brick
				for ( unsigned int i = 0; i < _brick.size(); i++ )
				{
					float3 point = _brick[ i ];

					// Translation
					point.x += x * levelSize;
					point.y += y * levelSize;
					point.z += z * levelSize;

					_points.push_back( point );
				}
			}
		}
	}

	//// Duplicate the brick
	//if ( _hasBrickDrawOneSlice )
	//{
	//	unsigned int z = 0;

	//	for ( unsigned int y = 0; y < 128; y++ )
	//	{
	//		for ( unsigned int x = 0; x < 128; x++ )
	//		{
	//			for ( unsigned int k = 0; k < brickResolution; k++ )
	//			{
	//				for ( unsigned int j = 0; j < brickResolution; j++ )
	//				{
	//					for ( unsigned int i = 0; i < brickResolution; i++ )
	//					{
	//						if ( _brickPresenceFlags[ i ][ j ][ k ] == 1 )
	//						{
	//							float3 point;
	//							point.x = static_cast< float >( x ) / ( brickResolution - 1 ) * levelSize;
	//							point.y = static_cast< float >( y ) / ( brickResolution - 1 ) * levelSize;
	//							point.z = static_cast< float >( z ) / ( brickResolution - 1 ) * levelSize;

	//							// Translation
	//							point.x += x * levelSize;
	//							point.y += y * levelSize;
	//							point.z += z * levelSize;

	//							_points.push_back( point );
	//						}
	//					}
	//				}
	//			}
	//		}
	//	}
	//}
	//else
	//{
	//	for ( unsigned int z = 0; z < 128; z++ )
	//	{
	//		for ( unsigned int y = 0; y < 128; y++ )
	//		{
	//			for ( unsigned int x = 0; x < 128; x++ )
	//			{
	//				for ( unsigned int k = 0; k < brickResolution; k++ )
	//				{
	//					for ( unsigned int j = 0; j < brickResolution; j++ )
	//					{
	//						for ( unsigned int i = 0; i < brickResolution; i++ )
	//						{
	//							if ( _brickPresenceFlags[ i ][ j ][ k ] == 1 )
	//							{
	//								float3 point;
	//								point.x = static_cast< float >( x ) / ( brickResolution - 1 ) * levelSize;
	//								point.y = static_cast< float >( y ) / ( brickResolution - 1 ) * levelSize;
	//								point.z = static_cast< float >( z ) / ( brickResolution - 1 ) * levelSize;

	//								// Translation
	//								point.x += x * levelSize;
	//								point.y += y * levelSize;
	//								point.z += z * levelSize;

	//								_points.push_back( point );
	//							}
	//						}
	//					}
	//				}
	//			}
	//		}
	//	}
	//}

	// Fill Vertex buffer
	glBindBuffer( GL_ARRAY_BUFFER, _vertexBuffer );
	GLsizeiptr vertexBufferSize = sizeof( GLfloat ) * _points.size() * 3;
	//GLsizeiptr vertexBufferSize = sizeof( GLfloat ) * _brick.size() * 3;
	glBufferData( GL_ARRAY_BUFFER, vertexBufferSize, NULL, GL_STATIC_DRAW );
	GLfloat* vertexBufferData = static_cast< GLfloat* >( glMapBuffer( GL_ARRAY_BUFFER, GL_WRITE_ONLY ) );
	unsigned int index = 0;
	for ( size_t i = 0; i < _points.size(); i++ )
	{
		const float3& point = _points[ i ];

		vertexBufferData[ index++ ] = point.x;
		vertexBufferData[ index++ ] = point.y;
		vertexBufferData[ index++ ] = point.z;
	}
	glUnmapBuffer( GL_ARRAY_BUFFER );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
	
	return true;
}

/******************************************************************************
 * Render the particle system
 ******************************************************************************/
void ParticleSystem::render()
{
	//float test = 0;
	//glGetFloat( GL_MAX_ELEMENTS_VERTICES, test ); 

	// Set color
	//glColor3f( 0.0f, 0.0f, 1.0f );
	glColor4f( 0.0f, 0.0f, 1.0f, 0.5f );

	glEnable( GL_POINT_SMOOTH ); 

	// Set point size
	glPointSize( _pointSize );
	
	// Deactivate z-test
	glEnable( GL_DEPTH_TEST );
	glDepthMask( GL_FALSE );

	// Activate blending
	glEnable( GL_BLEND );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	
	// Apply translation
	glPushMatrix();
	glTranslatef( -0.5f, -0.5f, -0.5f );
	{
		glBindBuffer( GL_ARRAY_BUFFER, _vertexBuffer );
		glEnableClientState( GL_VERTEX_ARRAY );
		glVertexPointer( 3, GL_FLOAT, 0, 0 );

		// Render points
		glDrawArrays( GL_POINTS, 0, _points.size() );

		glDisableClientState( GL_VERTEX_ARRAY );
		glBindBuffer( GL_ARRAY_BUFFER, 0 );
	}
	glPopMatrix();

	// Activate z-test
	glDepthMask( GL_TRUE );
	glDisable( GL_DEPTH_TEST );
	
	// Deactivate blending
	glDisable( GL_BLEND );
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool ParticleSystem::hasBrickDrawOneSlice() const
{
	return _hasBrickDrawOneSlice;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ParticleSystem::setBrickDrawOneSlice( bool pFlag )
{
	_hasBrickDrawOneSlice = pFlag;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ParticleSystem::setBrickPresenceFlags( unsigned int pBrickPresenceFlags[][ 8 ][ 8 ] )
{
	const unsigned int brickResolution = 8/*BrickRes::x*/;

	for ( unsigned int z = 0; z < brickResolution; z++ )
	{
		for ( unsigned int y = 0; y < brickResolution; y++ )
		{
			for ( unsigned int x = 0; x < brickResolution; x++ )
			{
				_brickPresenceFlags[ x ][ y ][ z ] = pBrickPresenceFlags[ x ][ y ][ z ];
			}
		}
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
unsigned int ParticleSystem::getBrickNbPoints() const
{
	return _brickNbPoints;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ParticleSystem::setBrickNbPoints( unsigned int pValue )
{
	_brickNbPoints = pValue;
}

/******************************************************************************
 * ...
 ******************************************************************************/
float ParticleSystem::getBrickPointSize() const
{
	return _pointSize;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ParticleSystem::setBrickPointSize( float pValue )
{
	_pointSize = pValue;
}
