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

// STL
#include <iostream>

// System
#include <cstdlib>
#include <ctime>

// Cuda
#include <cuda_runtime.h>

// GigaVoxels
#include <GvCore/vector_types_ext.h>
#include <GvCore/GvError.h>
#include <GvUtils/GvShaderManager.h>
#include <GvRendering/GvGraphicsResource.h>
#include <GsGraphics/GsShaderProgram.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>
#include <QImage>
#include <QGLWidget>

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvRendering;
using namespace GvUtils;
using namespace GsGraphics;

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
ParticleSystem::ParticleSystem( const float3& pPoint1, const float3& pPoint2 )
:	_p1( pPoint1 )
,	_p2( pPoint2 )
,	_nbParticles( 0 )
,	_d_particleBuffer( NULL )
,	_pointSizeFader( 1.f )
,	_fixedSizePointSize( 0.f )
,	_pointsShaderProgram( 0 )
//,	_pointSpritesShaderProgram( 0 )
,	_nbRenderablePoints( 0 )
,	_graphicsResource( NULL )
,	_textureFilename()
,	_shaderUseUniformColor( false )
,	_shaderUniformColor( make_float4( 1.f, 0.f, 0.f, 1.f ) )
,	 _shaderAnimation( false )
,	_hasTexture( false )
,	_shaderProgramPoints( NULL )
,	_shaderProgramPointSprite( NULL )
,	_shaderProgramParticleSystem( NULL )
{
	// Initialize random seed
	srand( time( NULL ) );

	// Create and link a GLSL shader program
	QString dataRepository = QCoreApplication::applicationDirPath();
	dataRepository += QDir::separator();
	dataRepository += QString( "Data" );
	dataRepository += QDir::separator();
	dataRepository += QString( "Shaders" );
	dataRepository += QDir::separator();
	dataRepository += QString( "GvVBOGenerator" );
	dataRepository += QDir::separator();

	// Initialize points shader program
	QString vertexShaderFilename = dataRepository + QString( "points_vert.glsl" );
	QString fragmentShaderFilename = dataRepository + QString( "points_frag.glsl" );
	//_pointsShaderProgram = GvShaderManager::createShaderProgram( vertexShaderFilename.toLatin1().constData(), NULL, fragmentShaderFilename.toLatin1().constData() );
	//GvShaderManager::linkShaderProgram( _pointsShaderProgram );

	// Initialize shader program
	_shaderProgramPoints = new GsShaderProgram();
	_shaderProgramPoints->addShader( GsShaderProgram::eVertexShader, vertexShaderFilename.toStdString() );
	_shaderProgramPoints->addShader( GsShaderProgram::eFragmentShader, fragmentShaderFilename.toStdString() );
	_shaderProgramPoints->link();

	// Initialize points shader program
	vertexShaderFilename = dataRepository + QString( "pointSprites_vert.glsl" );
	QString geometryShaderFilename = dataRepository + QString( "pointSprites_geom.glsl" );
	fragmentShaderFilename = dataRepository + QString( "pointSprites_frag.glsl" );
	//_pointSpritesShaderProgram = GvShaderManager::createShaderProgram( vertexShaderFilename.toLatin1().constData(), geometryShaderFilename.toLatin1().constData(), fragmentShaderFilename.toLatin1().constData() );
	//GvShaderManager::linkShaderProgram( _pointSpritesShaderProgram );

		// Initialize shader program
	_shaderProgramPointSprite = new GsShaderProgram();
	_shaderProgramPointSprite->addShader( GsShaderProgram::eVertexShader, vertexShaderFilename.toStdString() );
	_shaderProgramPointSprite->addShader( GsShaderProgram::eGeometryShader, geometryShaderFilename.toStdString() );
	_shaderProgramPointSprite->addShader( GsShaderProgram::eFragmentShader, fragmentShaderFilename.toStdString() );
	_shaderProgramPointSprite->link();
	
//	// Vertex buffer
//	glGenBuffers( 1, &_positionBuffer );
//	glBindBuffer( GL_ARRAY_BUFFER, _positionBuffer );
//	//---------------------------------------------------------------------------------------
//	//const unsigned int nbVertices = 1000;
//	const unsigned int nbVertices = 100000000;
//	//---------------------------------------------------------------------------------------
//	GLsizeiptr vertexBufferSize = sizeof( GLfloat ) * nbVertices * 3;
//	glBufferData( GL_ARRAY_BUFFER, vertexBufferSize, NULL, GL_DYNAMIC_DRAW );
//	glBindBuffer( GL_ARRAY_BUFFER, 0 );
//
//	// Vertex array object
//	glGenVertexArrays( 1, &_vao );
//	glBindVertexArray( _vao );
//	glEnableVertexAttribArray( 0 );	// vertex position
//	glBindBuffer( GL_ARRAY_BUFFER, _positionBuffer );
//	glVertexAttribPointer( 0/*attribute index*/, 3/*nb components per vertex*/, GL_FLOAT/*type*/, GL_FALSE/*un-normalized*/, 0/*memory stride*/, static_cast< GLubyte* >( NULL )/*byte offset from buffer*/ );
//	glBindBuffer( GL_ARRAY_BUFFER, 0 );
//	glBindVertexArray( 0 );
//
//	// Activation des textures
////	glEnable( GL_TEXTURE_2D );
//
//	// Sprite texture
//	glGenTextures( 1, &_spriteTexture );
//	//QString spriteTextureFilename = dataRepository + QString( "star_01.png" );
//	//QImage spriteImage = QGLWidget::convertToGLFormat( QImage( spriteTextureFilename, "PNG" ) );
//	//QString spriteTextureFilename = dataRepository + QString( "star_01.jpg" );
//	//QString spriteTextureFilename = dataRepository + QString( "flower.png" );
//	QString spriteTextureFilename = dataRepository + QString( "star_01.png" );
//	QImage spriteImage = QGLWidget::convertToGLFormat( QImage( spriteTextureFilename ) );
//	glActiveTexture( GL_TEXTURE0 );
//	glBindTexture( GL_TEXTURE_2D, _spriteTexture );
//	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, spriteImage.width(), spriteImage.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, spriteImage.bits() );
//	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
//	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
//	glBindTexture( GL_TEXTURE_2D, 0 );
//	
//	// Register graphics resource
//	_graphicsResource = new GvGraphicsResource();
//	cudaError_t error = _graphicsResource->registerBuffer( _positionBuffer, cudaGraphicsMapFlagsWriteDiscard );	// Beware, this map and not registered
//	if ( error != cudaSuccess )
//	{
//		assert( false );
//	}
	// Sprite texture initialization
	const QString spriteTextureFilename = dataRepository + QString( "star_01.png" );
	_textureFilename = spriteTextureFilename.toStdString();
	initGraphicsResources();

	_uniformColor = make_float4( 0.0f, 0.0f, 1.0f, 1.0f );
	_shaderUniformColor = make_float4( 0.0f, 0.0f, 1.0f, 1.0f );
}

/******************************************************************************
 * ...
 *
 * return ...
 ******************************************************************************/
bool ParticleSystem::initGraphicsResources()
{
	// Data repository
	QString dataRepository = QCoreApplication::applicationDirPath();
	dataRepository += QDir::separator();
	dataRepository += QString( "Data" );
	dataRepository += QDir::separator();
	dataRepository += QString( "Shaders" );
	dataRepository += QDir::separator();
	dataRepository += QString( "GvVBOGenerator" );
	dataRepository += QDir::separator();

	//@todo use a user customizable constant 
	const unsigned int nbVertices = 1000000;	// 1.000.000

	// Vertex position buffer initialization
	glGenBuffers( 1, &_positionBuffer );
	glBindBuffer( GL_ARRAY_BUFFER, _positionBuffer );
	GLsizeiptr vertexBufferSize = sizeof( GLfloat ) * nbVertices * 3;
	glBufferData( GL_ARRAY_BUFFER, vertexBufferSize, NULL, GL_DYNAMIC_DRAW );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	// Vertex array object initialization
	glGenVertexArrays( 1, &_vao );
	glBindVertexArray( _vao );
	glEnableVertexAttribArray( 0 );	// vertex position
	glBindBuffer( GL_ARRAY_BUFFER, _positionBuffer );
	glVertexAttribPointer( 0/*attribute index*/, 3/*nb components per vertex*/, GL_FLOAT/*type*/, GL_FALSE/*un-normalized*/, 0/*memory stride*/, static_cast< GLubyte* >( NULL )/*byte offset from buffer*/ );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
	glBindVertexArray( 0 );

	// Activation des textures
	//glEnable( GL_TEXTURE_2D );

	//// Sprite texture initialization
	//const QString spriteTextureFilename = dataRepository + QString( "star_01.png" );
	//_textureFilename = spriteTextureFilename.toStdString();

	glGenTextures( 1, &_spriteTexture );
	QImage spriteImage = QGLWidget::convertToGLFormat( QImage( _textureFilename.c_str() ) );
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D, _spriteTexture );
	glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, spriteImage.width(), spriteImage.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, spriteImage.bits() );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glBindTexture( GL_TEXTURE_2D, 0 );
	
	// Register graphics resource
	_graphicsResource = new GvGraphicsResource();
	cudaError_t error = _graphicsResource->registerBuffer( _positionBuffer, cudaGraphicsMapFlagsWriteDiscard );	// Beware, this map and not registered
	if ( error != cudaSuccess )
	{
		assert( false );
	}

	return false;
}

/******************************************************************************
 * ...
 *
 * return ...
 ******************************************************************************/
bool ParticleSystem::releaseGraphicsResources()
{
	if ( _graphicsResource )
	{
		delete _graphicsResource;
		_graphicsResource = NULL;
	}

	if ( _spriteTexture )
	{
		glDeleteTextures( 1, &_spriteTexture );
		GV_CHECK_GL_ERROR();
	}

	if ( _vao )
	{
		glDeleteVertexArrays( 1, &_vao );
		GV_CHECK_GL_ERROR();
	}

	if ( _positionBuffer )
	{
		glDeleteBuffers( 1, &_positionBuffer );
		GV_CHECK_GL_ERROR();
	}
	
	return true;
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
ParticleSystem::~ParticleSystem()
{
	// TO DO
	// Handle destruction
	// ...
	//assert( false );
	releaseGraphicsResources();
}

/******************************************************************************
 * Initialise le buffer GPU contenant les positions
 ******************************************************************************/
void ParticleSystem::initGPUBuf()
{
	if ( _d_particleBuffer != NULL )
	{
		GV_CUDA_SAFE_CALL( cudaFree( _d_particleBuffer ) );
		_d_particleBuffer = NULL;
	}

	float4* part_buf = new float4[ _nbParticles ];

//	std::cout << std::endl;
//	std::cout << std::endl;
//	std::cout << std::endl;

	for ( unsigned int i = 0; i < _nbParticles; ++i )
	{
		// Radius
		//float radius = .005f + static_cast< float >( rand() ) / ( static_cast< float >( RAND_MAX ) / ( 0.02f - 0.005f ) );	// rayon de l'etoile dans [0.005 : 0.02]
		float radius = 0.0001f;
		// Global size gain
		//radius *= _pointSizeFader;

		// Position (generee aleatoirement)
		float3 pos = genPos( rand() );
		part_buf[ i ] = make_float4( pos.x, pos.y, pos.z, radius );
		
	//	std::cout << part_buf[ i ] << std::endl;
	}

	// DEBUG ---------------------------
	//part_buf[ 0 ] = make_float4( 0.2f, 0.2f, 0.2f, 0.10f );
	//part_buf[ 0 ] = make_float4( 0.2f, 0.2f, 0.2f, 0.00001f );
	// DEBUG ---------------------------

//	std::cout << std::endl;
//	std::cout << std::endl;
//	std::cout << std::endl;

	size_t size = _nbParticles * sizeof( float4 );

	//_d_particleBuffer = new GvCore::Array3DGPULinear( make_int3( _nbParticles, 1, 1 ), 0 );
	if ( cudaSuccess != cudaMalloc( &_d_particleBuffer, size ) )
	{
		return;
	}

	GV_CUDA_SAFE_CALL( cudaMemcpy( _d_particleBuffer, part_buf, size, cudaMemcpyHostToDevice ) );

	// TO DO
	// Delete the temporary buffer : part_buf
	// ...
}

/******************************************************************************
 * Get the buffer of data (sphere positions and radius)
 *
 * @return the buffer of data (sphere positions and radius)
 ******************************************************************************/
float4* ParticleSystem::getGPUBuf()
{
	return _d_particleBuffer;
}

/******************************************************************************
 * Genere une position aleatoire
 *
 * @param pSeed ...
 ******************************************************************************/
float3 ParticleSystem::genPos( int pSeed )
{
	float3 p;

	//srand( pSeed );

	float min;	// min de l'interval des valeurs sur l'axe
	float max;	// max de l'interval des valeurs sur l'axe

	// genere la coordonnee en X
	if ( _p1.x < _p2.x )
	{
		min = _p1.x;
		max = _p2.x;
	}
	else
	{
		min = _p2.x;
		max = _p1.x;
	}
	p.x = min + (float)rand() / ((float)RAND_MAX / (max-min));

	// genere la coordonnee en Y
	if ( _p1.y < _p2.y )
	{
		min = _p1.y;
		max = _p2.y;
	}
	else
	{
		min = _p2.y;
		max = _p1.y;
	}
	p.y = min + (float)rand() / ((float)RAND_MAX / (max-min));

	// genere la coordonnee en Z
	if ( _p1.z < _p2.z )
	{
		min = _p1.z;
		max = _p2.z;
	}
	else
	{
		min = _p2.z;
		max = _p1.z;
	}
	p.z = min + (float)rand() / ((float)RAND_MAX / (max-min));

	return p;
}

/******************************************************************************
 * Get the number of particles
 *
 * @return the number of particles
 ******************************************************************************/
unsigned int ParticleSystem::getNbParticles() const
{
	return _nbParticles;
}

/******************************************************************************
 * Set the number of particles
 *
 * @param pValue the number of particles
 ******************************************************************************/
void ParticleSystem::setNbParticles( unsigned int pValue )
{
	_nbParticles = pValue;
}

/******************************************************************************
 * ...
 ******************************************************************************/
float ParticleSystem::getPointSizeFader() const
{
	return _pointSizeFader;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ParticleSystem::setPointSizeFader( float pValue )
{
	_pointSizeFader = pValue;
}

/******************************************************************************
 * ...
 ******************************************************************************/
float ParticleSystem::getFixedSizePointSize() const
{
	return _fixedSizePointSize;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ParticleSystem::setFixedSizePointSize( float pValue )
{
	_fixedSizePointSize = pValue;
}

/******************************************************************************
 * Render the particle system
 ******************************************************************************/
void ParticleSystem::render( const float4x4& pModelViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	if ( ! _hasTexture )
	{
		//glUseProgram( _pointsShaderProgram );
		_shaderProgramPoints->use();
	
		glEnable( GL_PROGRAM_POINT_SIZE );
		//glPointSize( 5.0f );
	
		// The actual locations assigned to uniform variables are not known until the program object is linked successfully.
		// After linking has occurred, the command glGetUniformLocation can be used to obtain the location of a uniform variable.
		// This location value can then be passed to glUniform to set the value of the uniform variable or to glGetUniform
		// in order to query the current value of the uniform variable. After a program object has been linked successfully,
		// the index values for uniform variables remain fixed until the next link command occurs.
		// Uniform variable locations and values can only be queried after a link if the link was successful.

		//GLuint location = glGetUniformLocation( _pointsShaderProgram, "ModelViewMatrix" );
		GLint location = glGetUniformLocation( _shaderProgramPoints->_program, "ModelViewMatrix" );
		if ( location >= 0 )
		{
			glUniformMatrix4fv( location, 1, GL_FALSE, pModelViewMatrix._array );
		}
		//location = glGetUniformLocation( _pointsShaderProgram, "ProjectionMatrix" );
		location = glGetUniformLocation( _shaderProgramPoints->_program, "ProjectionMatrix" );
		if ( location >= 0 )
		{
			glUniformMatrix4fv( location, 1, GL_FALSE, pProjectionMatrix._array );
		}

		//location = glGetUniformLocation( _pointsShaderProgram, "PointSize" );
		location = glGetUniformLocation( _shaderProgramPoints->_program, "PointSize" );
		if ( location >= 0 )
		{
			glUniform1f( location, _pointSizeFader );
		}

		//location = glGetUniformLocation( _pointsShaderProgram, "VertexColor" );
		location = glGetUniformLocation( _shaderProgramPoints->_program, "VertexColor" );
		if ( location >= 0 )
		{
			//glUniform4f( location, _uniformColor.x, _uniformColor.y, _uniformColor.z, _uniformColor.w );
			glUniform4f( location, _shaderUniformColor.x, _shaderUniformColor.y, _shaderUniformColor.z, _shaderUniformColor.w );
		}

		// ---- Animation ----
		static float time = 0.f;
		location = glGetUniformLocation( _shaderProgramPoints->_program, "uTime" );
		if ( location >= 0 )
		{
			glUniform1f( location, time );
		}
		time += 1.f;

		// ---- Varying screen size ----
		location = glGetUniformLocation( _shaderProgramPoints->_program, "uWindowSize" );
		if ( location >= 0 )
		{
			glUniform2f( location, pViewport.z, pViewport.w );
		}
	}
	else
	{
		// Activation des textures
		glEnable( GL_TEXTURE_2D );
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_2D, _spriteTexture );

		//glUseProgram( _pointSpritesShaderProgram );
		_shaderProgramPointSprite->use();
	
		// The actual locations assigned to uniform variables are not known until the program object is linked successfully.
		// After linking has occurred, the command glGetUniformLocation can be used to obtain the location of a uniform variable.
		// This location value can then be passed to glUniform to set the value of the uniform variable or to glGetUniform
		// in order to query the current value of the uniform variable. After a program object has been linked successfully,
		// the index values for uniform variables remain fixed until the next link command occurs.
		// Uniform variable locations and values can only be queried after a link if the link was successful.

		//GLuint location = glGetUniformLocation( _pointSpritesShaderProgram, "ModelViewMatrix" );
		GLint location = glGetUniformLocation( _shaderProgramPointSprite->_program, "ModelViewMatrix" );
		if ( location >= 0 )
		{
			glUniformMatrix4fv( location, 1, GL_FALSE, pModelViewMatrix._array );
		}
		//location = glGetUniformLocation( _pointSpritesShaderProgram, "ProjectionMatrix" );
		location = glGetUniformLocation( _shaderProgramPointSprite->_program, "ProjectionMatrix" );
		if ( location >= 0 )
		{
			glUniformMatrix4fv( location, 1, GL_FALSE, pProjectionMatrix._array );
		}

		//location = glGetUniformLocation( _pointSpritesShaderProgram, "HalfSize" );
		location = glGetUniformLocation( _shaderProgramPointSprite->_program, "HalfSize" );
		if ( location >= 0 )
		{
			//glUniform1f( location, 0.01f );
			glUniform1f( location, 0.01f * _pointSizeFader );
		}	

		//location = glGetUniformLocation( _pointSpritesShaderProgram, "SpriteTex" );
		location = glGetUniformLocation( _shaderProgramPointSprite->_program, "SpriteTex" );
		if ( location >= 0 )
		{
			glUniform1i( location, 0 );
		}

		//location = glGetUniformLocation( _pointSpritesShaderProgram, "BackgroundColor" );
		location = glGetUniformLocation( _shaderProgramPointSprite->_program, "BackgroundColor" );
		if ( location >= 0 )
		{
			const float color = 128.f / 255.f;
			glUniform4f( location, color, color, color, color );
		}
	}
		
	glBindVertexArray( _vao );
	glDrawArrays( GL_POINTS, 0, _nbRenderablePoints );
	glBindVertexArray( 0 );

	//glDisable( GL_PROGRAM_POINT_SIZE );
	glDisable( GL_TEXTURE_2D );

	glUseProgram( 0 );
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool ParticleSystem::hasShaderUniformColor() const
{
	return _shaderUseUniformColor;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ParticleSystem::setShaderUniformColorMode( bool pFlag )
{
	_shaderUseUniformColor = pFlag;
}

/******************************************************************************
 * ...
 ******************************************************************************/
const float4& ParticleSystem::getShaderUniformColor() const
{
	return _shaderUniformColor;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ParticleSystem::setShaderUniformColor( float pR, float pG, float pB, float pA )
{
	_shaderUniformColor = make_float4( pR, pG, pB, pA );

	//GLuint location = glGetUniformLocation( _pointsShaderProgram, "VertexColor" );
	//if ( location >= 0 )
	//{
	//	//glUniform4f( location, _uniformColor.x, _uniformColor.y, _uniformColor.z, _uniformColor.w );
	//	glUniform4f( location, _shaderUniformColor.x, _shaderUniformColor.y, _shaderUniformColor.z, _shaderUniformColor.w );
	//}
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool ParticleSystem::hasShaderAnimation() const
{
	return _shaderAnimation;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ParticleSystem::setShaderAnimation( bool pFlag )
{
	_shaderAnimation = pFlag;
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool ParticleSystem::hasTexture() const
{
	return _hasTexture;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ParticleSystem::setTexture( bool pFlag )
{
	_hasTexture = pFlag;
}

/******************************************************************************
 * ...
 ******************************************************************************/
const std::string& ParticleSystem::getTextureFilename() const
{
	return _textureFilename;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ParticleSystem::setTextureFilename( const std::string& pFilename )
{
	_textureFilename = pFilename;

	releaseGraphicsResources();
	initGraphicsResources();
}
