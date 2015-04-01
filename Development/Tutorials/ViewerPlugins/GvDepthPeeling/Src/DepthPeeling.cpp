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

#include "DepthPeeling.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvUtils/GvShaderProgram.h>

// Cuda
#include <vector_types.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>

// STL
#include <iostream>

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
DepthPeeling::DepthPeeling()
:	_meshShaderProgram( NULL )
,	_frontToBackPeelingShaderProgram( NULL )
,	_blenderShaderProgram( NULL )
,	_finalShaderProgram( NULL )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
DepthPeeling::~DepthPeeling()
{
}

/******************************************************************************
 * Initialize
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
bool DepthPeeling::initialize()
{
	// Create and link a GLSL shader program
	QString dataRepository = QCoreApplication::applicationDirPath();
	dataRepository += QDir::separator();
	dataRepository += QString( "Data" );
	dataRepository += QDir::separator();
	dataRepository += QString( "Shaders" );
	dataRepository += QDir::separator();
	dataRepository += QString( "GvDepthPeeling" );
	dataRepository += QDir::separator();

	// Initialize shader program
	QString meshVertexShaderFilename = dataRepository + QString( "fullscreenQuad_vert.glsl" );
	QString meshFragmentShaderFilename = dataRepository + QString( "fullscreenQuad_frag.glsl" );
	_meshShaderProgram = new GvShaderProgram();
	_meshShaderProgram->addShader( GvShaderProgram::eVertexShader, meshVertexShaderFilename.toStdString() );
	_meshShaderProgram->addShader( GvShaderProgram::eFragmentShader, meshFragmentShaderFilename.toStdString() );
	_meshShaderProgram->link();

	// Initialize shader program
	QString frontToBackPeelingVertexShaderFilename = dataRepository + QString( "fullscreenQuad_vert.glsl" );
	QString frontToBackPeelingFragmentShaderFilename = dataRepository + QString( "fullscreenQuad_frag.glsl" );
	_frontToBackPeelingShaderProgram = new GvShaderProgram();
	_frontToBackPeelingShaderProgram->addShader( GvShaderProgram::eVertexShader, frontToBackPeelingVertexShaderFilename.toStdString() );
	_frontToBackPeelingShaderProgram->addShader( GvShaderProgram::eFragmentShader, frontToBackPeelingFragmentShaderFilename.toStdString() );
	_frontToBackPeelingShaderProgram->link();

	// Initialize shader program
	QString blenderVertexShaderFilename = dataRepository + QString( "fullscreenQuad_vert.glsl" );
	QString blenderFragmentShaderFilename = dataRepository + QString( "fullscreenQuad_frag.glsl" );
	_blenderShaderProgram = new GvShaderProgram();
	_blenderShaderProgram->addShader( GvShaderProgram::eVertexShader, blenderVertexShaderFilename.toStdString() );
	_blenderShaderProgram->addShader( GvShaderProgram::eFragmentShader, blenderFragmentShaderFilename.toStdString() );
	_blenderShaderProgram->link();

	// Initialize shader program
	QString finalVertexShaderFilename = dataRepository + QString( "fullscreenQuad_vert.glsl" );
	QString finalFragmentShaderFilename = dataRepository + QString( "fullscreenQuad_frag.glsl" );
	_finalShaderProgram = new GvShaderProgram();
	_finalShaderProgram->addShader( GvShaderProgram::eVertexShader, finalVertexShaderFilename.toStdString() );
	_finalShaderProgram->addShader( GvShaderProgram::eFragmentShader, finalFragmentShaderFilename.toStdString() );
	_finalShaderProgram->link();

	glGenFramebuffers( 2, _frameBuffers );
	glGenTextures( 2, _textures );
	glGenTextures( 2, _depthTextures );

	GLsizei width = 0;
	GLsizei height = 0;

	for ( unsigned int i = 0; i < 2; i++ )
	{
		glBindTexture( GL_TEXTURE_RECTANGLE, _textures[ i ] );
		glTexImage2D( GL_TEXTURE_RECTANGLE, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, NULL );
		glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

		glBindTexture( GL_TEXTURE_RECTANGLE, _depthTextures[ i ] );
		glTexImage2D( GL_TEXTURE_RECTANGLE, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL );
		glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

		glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffers[ i ] );
		glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, _textures[ i ], 0 );
		glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_RECTANGLE, _depthTextures[ i ], 0 );
		glBindFramebuffer( GL_FRAMEBUFFER, 0 );
	}

	glGenTextures( 1, &_blenderTexture );
	glBindTexture( GL_TEXTURE_RECTANGLE, _blenderTexture );
	glTexImage2D( GL_TEXTURE_RECTANGLE, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, 0 );
	glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

	glGenFramebuffers( 1, &_blenderFrameBuffer );
	glBindFramebuffer( GL_FRAMEBUFFER, _blenderFrameBuffer );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, _blenderTexture, 0 );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_RECTANGLE, _depthTextures[ 0 ], 0 );
	GLenum status = glCheckFramebufferStatus( GL_FRAMEBUFFER );
	if ( status != GL_FRAMEBUFFER_COMPLETE )
	{
		cout << "Error : problem with FBO configuration" << endl;
	}
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );

	// -------- Mesh initialization --------

	// Vertex buffer initialization
	glGenBuffers( 1, &_meshVertexBuffer );
	glBindBuffer( GL_ARRAY_BUFFER, _meshVertexBuffer );
	unsigned int nbVertices = 0;
	float3* vertexBuffer = new float3[ nbVertices ];
	GLsizeiptr vertexBufferSize = sizeof( GLfloat ) * nbVertices/*nb Vertices*/ * 3/*nb components per vertex*/;
	// Fill buffer
	// ...
	glBufferData( GL_ARRAY_BUFFER, vertexBufferSize, &vertexBuffer[ 0 ], GL_STATIC_DRAW );
	delete[] vertexBuffer;
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	// Index buffer initialization
	glGenBuffers( 1, &_meshIndexBuffer );

	// Vertex array initialization
	glGenVertexArrays( 1, &_meshVertexArray );
	glBindVertexArray( _meshVertexArray );
	glEnableVertexAttribArray( 0 ); // vertex position
	glBindBuffer( GL_ARRAY_BUFFER, _meshVertexBuffer );
	glVertexAttribPointer( 0/*attribute index*/, 3/*nb components per vertex*/, GL_FLOAT/*type*/, GL_FALSE/*un-normalized*/, 0/*memory stride*/, static_cast< GLubyte* >( NULL )/*byte offset from buffer*/ );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _meshIndexBuffer ); // Required for indexed rendering
	glBindVertexArray( 0 );
	
	return true;
}

/******************************************************************************
 * Finalize
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
bool DepthPeeling::finalize()
{
	delete _meshShaderProgram;
	_meshShaderProgram = NULL;

	delete _frontToBackPeelingShaderProgram;
	_frontToBackPeelingShaderProgram = NULL;

	delete _blenderShaderProgram;
	_blenderShaderProgram = NULL;

	delete _finalShaderProgram;
	_finalShaderProgram = NULL;
	
	return true;
}

/******************************************************************************
 * Render
 ******************************************************************************/
void DepthPeeling::render()
{
	GLenum colorBuffers[] = { GL_COLOR_ATTACHMENT0 };

	// -------- [ 1 ] --------
	glBindFramebuffer( GL_FRAMEBUFFER, _blenderFrameBuffer );
	glDrawBuffers( 1, colorBuffers );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	glEnable( GL_DEPTH_TEST );
	// Draw scene
	// ...
	_meshShaderProgram->use();
	glUseProgram( 0 );
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );

	// -------- [ 2 ] --------
	const unsigned int nbPasses = 6;
	const unsigned int nbLayers = ( nbPasses - 1 ) * 2;
	for ( unsigned int i = 1; i < nbLayers; i++ )
	{
		unsigned int currentId = i % 2;
		unsigned int previousId = 1 - currentId;

		glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffers[ currentId ] );
		glDrawBuffers( 1, colorBuffers );
		glClearColor( 0.f, 0.f, 0.f, 0.f );
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
		glDisable( GL_BLEND );
		glEnable( GL_DEPTH_TEST );
		glBindTexture( GL_TEXTURE_RECTANGLE, _depthTextures[ previousId ] );
		// Draw scene
		// ...
		_frontToBackPeelingShaderProgram->use();
		glUseProgram( 0 );
		glBindTexture( GL_TEXTURE_RECTANGLE, 0 );
		glBindFramebuffer( GL_FRAMEBUFFER, 0 );
	
		// -------- [ 3 ] --------
		glBindFramebuffer( GL_FRAMEBUFFER, _blenderFrameBuffer );
		glDrawBuffers( 1, colorBuffers );
		glDisable( GL_DEPTH_TEST );
		glEnable( GL_BLEND );
		glBlendEquation( GL_FUNC_ADD );
		glBlendFuncSeparate( GL_DST_ALPHA, GL_ONE, GL_ZERO, GL_ONE_MINUS_SRC_ALPHA );
		glBindTexture( GL_TEXTURE_RECTANGLE, _textures[ currentId ] );
		// Draw full screen quad
		// ...
		_blenderShaderProgram->use();
		glUseProgram( 0 );
		glDisable( GL_BLEND );
		glBindTexture( GL_TEXTURE_RECTANGLE, 0 );
		glBindFramebuffer( GL_FRAMEBUFFER, 0 );
	}
	
	// -------- [ 4 ] --------
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );
	glDrawBuffer( GL_BACK_LEFT );
	glDisable( GL_DEPTH_TEST );
	glDisable( GL_BLEND );
	glBindTexture( GL_TEXTURE_RECTANGLE, _blenderTexture );
	// Draw full screen quad
	// ...
	_finalShaderProgram->use();
	glUseProgram( 0 );
	glBindTexture( GL_TEXTURE_RECTANGLE, 0 );
}
