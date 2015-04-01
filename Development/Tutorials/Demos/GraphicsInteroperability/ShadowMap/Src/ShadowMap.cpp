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

#include "ShadowMap.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvUtils/GvShaderProgram.h>

// Project
#include "Mesh.h"

//// glm
//#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtx/transform2.hpp>
//#include <glm/gtx/projection.hpp>

// Cuda
#include <vector_types.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>
#include <QImage>

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

/**
 * Scale and bias matrix used to transform NDC space coordinates from [ -1 ; 1 ]
 * to Texture space coordinates in [ 0 ; 1 ]
 */
const glm::mat4 ShadowMap::sScaleBiasMatrix = glm::mat4(glm::vec4( 0.5f, 0.0f, 0.0f, 0.0f ) /*1st column - scale*/,
														glm::vec4( 0.0f, 0.5f, 0.0f, 0.0f ) /*2nd column - scale*/,
														glm::vec4( 0.0f, 0.0f, 0.5f, 0.0f ) /*3rd column - scale*/,
														glm::vec4( 0.5f, 0.5f, 0.5f, 1.0f ) /*4th column - bias*/ );

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
ShadowMap::ShadowMap()
:	_shaderProgramFirstPass( NULL )
,	_shaderProgramSecondPass( NULL )
,	_shadowMapFBO( 0 )
,	_shadowMapWidth( 0 )
,	_shadowMapHeight( 0 )
,	_depthTexture( 0 )
,	_mesh( NULL )
,	_width( 0 )
,	_height( 0 )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
ShadowMap::~ShadowMap()
{
	finalize();
}

/******************************************************************************
 * Initialize
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
bool ShadowMap::initialize()
{
	bool statusOK = false;

	// Initialize shader program
	_shaderProgramFirstPass = new GvShaderProgram();
	statusOK = _shaderProgramFirstPass->addShader( GvShaderProgram::eVertexShader, "D:\\Projects\\INRIA\\LastGV\\Release\\Bin\\Data\\Shaders\\GvShadowMap\\shadowMapping_1stPass_vert.glsl" );
	assert( statusOK );
	statusOK = _shaderProgramFirstPass->addShader( GvShaderProgram::eFragmentShader, "D:\\Projects\\INRIA\\LastGV\\Release\\Bin\\Data\\Shaders\\GvShadowMap\\shadowMapping_1stPass_frag.glsl" );
	assert( statusOK );
	statusOK = _shaderProgramFirstPass->link();
	assert( statusOK );

	// Initialize shader program
	_shaderProgramSecondPass = new GvShaderProgram();
	statusOK = _shaderProgramSecondPass->addShader( GvShaderProgram::eVertexShader, "D:\\Projects\\INRIA\\LastGV\\Release\\Bin\\Data\\Shaders\\GvShadowMap\\shadowMapping_2ndPass_vert.glsl" );
	assert( statusOK );
	statusOK = _shaderProgramSecondPass->addShader( GvShaderProgram::eFragmentShader, "D:\\Projects\\INRIA\\LastGV\\Release\\Bin\\Data\\Shaders\\GvShadowMap\\shadowMapping_2ndPass_frag.glsl" );
	assert( statusOK );
	statusOK = _shaderProgramSecondPass->link();
	assert( statusOK );

	// Shadow map parameters
	_shadowMapWidth = 512;
	_shadowMapHeight = 512;

	// TO DO
	// - handle that
	// ...
//	_width = 512;
//	_height = 512;

	// Initialize graphics resource
	//
	// TO DO : choose a format that CUDA can handle in its cudaGraphicsResource
	// - depth formats are not well handled...
	glGenTextures( 1, &_depthTexture );
	glBindTexture( GL_TEXTURE_2D, _depthTexture );
	glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, _shadowMapWidth, _shadowMapHeight, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL );
	// - set texture object parameters
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER );
	// - configure texture for shadow mapping : texture fetch will not retrieve value but test compairison
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LESS );
	GLfloat borderColor[] = { 1.f, 0.f, 0.f, 0.f };
	glTexParameterfv( GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor );
	glBindTexture( GL_TEXTURE_2D, 0 );
	
	// Initialize graphics resource
	glGenFramebuffers( 1, &_shadowMapFBO );
	glBindFramebuffer( GL_FRAMEBUFFER, _shadowMapFBO );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, _depthTexture, 0 );
	const GLenum drawBuffers[] = { GL_NONE };
    glDrawBuffers( 1, drawBuffers );
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );
	// TO DO
	// - test completeness
	// ...

	// Initialize mesh
	_mesh = new Mesh();
	IMesh::ShaderProgramConfiguration shaderProgramConfiguration;
	//shaderProgramConfiguration._shaders[ GvUtils::GvShaderProgram::eVertexShader ] = "D://Projects//INRIA//LastGV//Release//Bin//Data//Shaders//GvShadowMap//meshRendering_vert.glsl";
	//shaderProgramConfiguration._shaders[ GvUtils::GvShaderProgram::eFragmentShader ] = "D://Projects//INRIA//LastGV//Release//Bin//Data//Shaders//GvShadowMap//meshRendering_frag.glsl";
	//_mesh->setShaderProgramConfiguration( shaderProgramConfiguration );
	statusOK = _mesh->load( "D:\\Projects\\INRIA\\LastGV\\Release\\Bin\\Data\\3DModels\\stanford_bunny\\bunny.obj" );
	
	// Set Light viewing system parameters
	//setLightEye( glm::vec3( 0.0f, 0.0f, 0.0f ) );
	//setLightCenter( glm::vec3( 0.0f, 0.0f, 0.0f ) );
	//setLightUp( glm::vec3( 0.0f, 1.0f, 0.0f ) );
	//setLightFovY( 0.f );
	//setLightAspectRatio( 0.f );
	//setLightZNear( 0.1f );
	//setLightZFar( 1000.f );

	//// Set Camera viewing system parameters
	////setCameraEye( glm::vec3( 0.0f, 0.0f, 0.0f ) );
	//setCameraCenter( glm::vec3( 0.0f, 0.0f, 0.0f ) );
	//setCameraUp( glm::vec3( 0.0f, 1.0f, 0.0f ) );
	//setCameraFovY( 0.f );
	//setCameraAspectRatio( 1.f );
	//setCameraZNear( 0.1f );
	//setCameraZFar( 100.f );

	_lightIntensity = glm::vec3( 0.85f );

	glm::vec3 color = glm::vec3( 0.7f, 0.5f, 0.3f );
	_materialKa = color * 0.05f;
	_materialKd = color;
	_materialKs = glm::vec3( 0.9f, 0.9f, 0.9f );
	_materialShininess = 150.f;
	
	

	return true;
}

/******************************************************************************
 * Finalize
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
bool ShadowMap::finalize()
{
	delete _shaderProgramFirstPass;
	_shaderProgramFirstPass = NULL;

	delete _shaderProgramSecondPass;
	_shaderProgramSecondPass = NULL;

	glDeleteFramebuffers( 1, &_shadowMapFBO );
	glDeleteTextures( 1, &_depthTexture );

	delete _mesh;
	_mesh = NULL;

	return true;
}

/******************************************************************************
 * Render
 ******************************************************************************/
void ShadowMap::render( const float4x4& pModelViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	//// Render the result to the screen
	//glMatrixMode( GL_MODELVIEW );
	//glPushMatrix();
	//glLoadIdentity();

	//glMatrixMode( GL_PROJECTION );
	//glPushMatrix();
	//glLoadIdentity();

	// Build model matrix
	glm::mat4 modelMatrix = glm::mat4( 1.0f );
	//modelMatrix *= glm::translate( glm::vec3( 0.0f, 0.0f, 0.0f ) );
	//modelMatrix *= glm::rotate( -90.0f, glm::vec3( 1.0f, 0.0f, 0.0f ) );

	// 1st pass : shadow map generation
	//
	// The scene is rendered from the point of view of the light source.
	// Depth is written automatically in the shadow map.

	// Use shader program
	_shaderProgramFirstPass->use();
	
	//----------------------------------------------------------------
	//setLightEye( glm::vec3( 0.f, 5.f, 15.f ) );
	//setLightCenter( glm::vec3( 0.f, 3.f, 0.f ) );
	//setLightUp( glm::vec3( 0.f, 1.f, 0.f ) );
	//setLightZFar( 1000.f );
	//----------------------------------------------------------------


	// Retrieve light's viewing system parameters
	glm::mat4 lightViewMatrix = glm::lookAt( _lightEye, _lightCenter, _lightUp );
	glm::mat4 lightProjectionMatrix = glm::perspective( _lightFovY, _lightAspectRatio, _lightZNear, _lightZFar );
	glm::mat4 lightModelViewMatrix = lightViewMatrix * modelMatrix;

	//// Set uniform(s)
	//glm::mat4 lightModelViewProjectionMatrix = lightModelViewMatrix * modelMatrix;
	//GLint location = glGetUniformLocation( _shaderProgramFirstPass->_program, "uModelViewProjectionMatrix" );
	//if ( location >= 0 )
	//{
	//	glUniformMatrix4fv( location, 1, GL_FALSE, &lightModelViewProjectionMatrix[ 0 ][ 0 ] );
	//}

	// Configure OpenGL pipeline
	glViewport( 0, 0, _shadowMapWidth, _shadowMapHeight );
	glEnable( GL_CULL_FACE );
	glCullFace( GL_FRONT );
	
	// Render mesh
	glBindFramebuffer( GL_FRAMEBUFFER, _shadowMapFBO );
	glClear( GL_DEPTH_BUFFER_BIT );

	// Set uniform(s)
	glm::mat4 lightModelViewProjectionMatrix;

	lightModelViewProjectionMatrix = lightProjectionMatrix * lightModelViewMatrix;
	GLint location = glGetUniformLocation( _shaderProgramFirstPass->_program, "uModelViewProjectionMatrix" );
	if ( location >= 0 )
	{
		glUniformMatrix4fv( location, 1, GL_FALSE, &lightModelViewProjectionMatrix[ 0 ][ 0 ] );
	}
	_mesh->render( float4x4(), float4x4(), int4() );	// TO DO : think about API => what parameters do we need in render() ?
	
	//---------------------------------------
	// Occluder
	modelMatrix *= glm::rotate( 90.f, glm::vec3( 0.0f, 1.0f, 0.0f ) );
	modelMatrix *= glm::translate( glm::vec3( 0.0f, 2.0f, 10.0f ) );
	lightModelViewProjectionMatrix = lightProjectionMatrix * lightViewMatrix * modelMatrix;
	location = glGetUniformLocation( _shaderProgramFirstPass->_program, "uModelViewProjectionMatrix" );
	if ( location >= 0 )
	{
		glUniformMatrix4fv( location, 1, GL_FALSE, &lightModelViewProjectionMatrix[ 0 ][ 0 ] );
	}
	float planeSize = 5.25f;
	glBegin( GL_QUADS );
	glVertexAttrib3f( 0, -planeSize * 0.5f, -planeSize * 0.5f, 0.0f );
	glVertexAttrib3f( 0, planeSize * 0.5f, -planeSize * 0.5f, 0.0f );
	glVertexAttrib3f( 0, planeSize * 0.5f, planeSize * 0.5f, 0.0f );
	glVertexAttrib3f( 0, -planeSize * 0.5f, planeSize * 0.5f, 0.0f );
	glEnd();
	//---------------------------------------

	//glBindFramebuffer( GL_FRAMEBUFFER, 0 );

	glUseProgram( 0 );

	// Dump depth texture
	glFlush();
	glFinish();
	
	/*static int testPascal = 0;
	testPascal++;
	if ( testPascal > 2000 )
	{
		spitOutDepthBuffer();
		exit( 1978 );
	}*/

	// 2nd pass
	//
	// The scene is rendered from the point of view of the camera.

	// Use shader program
	_shaderProgramSecondPass->use();

	// Retrieve camera's viewing system parameters
	glm::mat4 viewMatrix = glm::lookAt( _cameraEye, _cameraCenter, _cameraUp );
	glm::mat4 projectionMatrix = glm::perspective( _cameraFovY, _cameraAspectRatio, _cameraZNear, _cameraZFar );
	
	//// Build shadow matrix
	//glm::mat4 shadowMatrix = sScaleBiasMatrix * lightModelViewMatrix * modelMatrix;
	//
	//// Set uniform(s)
	//glm::mat4 modelViewProjectionMatrix = projectionMatrix * viewMatrix * modelMatrix;
	//location = glGetUniformLocation( _shaderProgramSecondPass->_program, "uModelViewProjectionMatrix" );
	//if ( location >= 0 )
	//{
	//	glUniformMatrix4fv( location, 1, GL_FALSE, &modelViewProjectionMatrix[ 0 ][ 0 ] );
	//}
	//location = glGetUniformLocation( _shaderProgramSecondPass->_program, "uShadowMatrix" );
	//if ( location >= 0 )
	//{
	//	glUniformMatrix4fv( location, 1, GL_FALSE, &shadowMatrix[ 0 ][ 0 ] );
	//}

	// Configure OpenGL pipeline
	glViewport( 0, 0, _width, _height );
	glDisable( GL_CULL_FACE );

	// Render mesh
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ) ;

	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_2D, _depthTexture );
	location = glGetUniformLocation( _shaderProgramSecondPass->_program, "uShadowMap" );
	if ( location >= 0 )
	{
		glUniform1i( location, 0 );
	}
	//------------------
	//glColor4f( 1.0f, 0.0, 0.0, 1.0f );
	//------------------
	//glDrawBuffer( GL_BACK );

	modelMatrix = glm::mat4( 1.0f );
	glm::mat4 shadowMatrix;
	glm::mat4 modelViewMatrix;
	glm::mat4 modelViewProjectionMatrix;
	glm::mat3 normalMatrix;
	// Build shadow matrix
	shadowMatrix = sScaleBiasMatrix * lightProjectionMatrix * lightViewMatrix * modelMatrix;
	// Set uniform(s)
	modelViewMatrix = viewMatrix * modelMatrix;
	modelViewProjectionMatrix = projectionMatrix * modelViewMatrix;
	location = glGetUniformLocation( _shaderProgramSecondPass->_program, "uModelViewMatrix" );
	if ( location >= 0 )
	{
		glUniformMatrix4fv( location, 1, GL_FALSE, &modelViewMatrix[ 0 ][ 0 ] );
	}
	normalMatrix = glm::mat3( glm::vec3( modelViewMatrix[ 0 ] ), glm::vec3( modelViewMatrix[ 1 ] ), glm::vec3( modelViewMatrix[ 2 ] ) );
	location = glGetUniformLocation( _shaderProgramSecondPass->_program, "uNormalMatrix" );
	if ( location >= 0 )
	{
		glUniformMatrix3fv( location, 1, GL_FALSE, &normalMatrix[ 0 ][ 0 ] );
	}
	location = glGetUniformLocation( _shaderProgramSecondPass->_program, "uModelViewProjectionMatrix" );
	if ( location >= 0 )
	{
		glUniformMatrix4fv( location, 1, GL_FALSE, &modelViewProjectionMatrix[ 0 ][ 0 ] );
	}
	location = glGetUniformLocation( _shaderProgramSecondPass->_program, "uShadowMatrix" );
	if ( location >= 0 )
	{
		glUniformMatrix4fv( location, 1, GL_FALSE, &shadowMatrix[ 0 ][ 0 ] );
	}
	location = glGetUniformLocation( _shaderProgramSecondPass->_program, "uLightPosition" );
	if ( location >= 0 )
	{
		glm::vec4 tmp = viewMatrix * glm::vec4( _lightEye, 1.f );
		glUniform4f( location, tmp.x, tmp.y, tmp.z, tmp.w );
	}
	location = glGetUniformLocation( _shaderProgramSecondPass->_program, "uLightIntensity" );
	if ( location >= 0 )
	{
		glUniform3f( location, _lightIntensity.x, _lightIntensity.y, _lightIntensity.z );
	}
	glm::vec3 color = glm::vec3( 0.7f, 0.5f, 0.3f );
	location = glGetUniformLocation( _shaderProgramSecondPass->_program, "uKa" );
	if ( location >= 0 )
	{
		glUniform3f( location, _materialKa.x, _materialKa.y, _materialKa.z );
	}
	location = glGetUniformLocation( _shaderProgramSecondPass->_program, "uKd" );
	if ( location >= 0 )
	{
		glUniform3f( location, _materialKd.x, _materialKd.y, _materialKd.z );
	}
	location = glGetUniformLocation( _shaderProgramSecondPass->_program, "uKs" );
	if ( location >= 0 )
	{
		glUniform3f( location, _materialKs.x, _materialKs.y, _materialKs.z );
	}
	location = glGetUniformLocation( _shaderProgramSecondPass->_program, "uShininess" );
	if ( location >= 0 )
	{
		glUniform1f( location, _materialShininess );
	}

	// Draw mesh
	_mesh->render( float4x4(), float4x4(), int4() );	// TO DO : think about API => what parameters do we need in render() ?

	//---------------------------------------
	// Occluder
	modelMatrix *= glm::rotate( 90.f, glm::vec3( 0.0f, 1.0f, 0.0f ) );
	modelMatrix *= glm::translate( glm::vec3( 0.0f, 2.0f, 10.0f ) );
	// Build shadow matrix
	shadowMatrix = sScaleBiasMatrix * lightProjectionMatrix * lightViewMatrix * modelMatrix;
	// Set uniform(s)
	modelViewProjectionMatrix = projectionMatrix * viewMatrix * modelMatrix;
	location = glGetUniformLocation( _shaderProgramSecondPass->_program, "uModelViewProjectionMatrix" );
	if ( location >= 0 )
	{
		glUniformMatrix4fv( location, 1, GL_FALSE, &modelViewProjectionMatrix[ 0 ][ 0 ] );
	}
	location = glGetUniformLocation( _shaderProgramSecondPass->_program, "uShadowMatrix" );
	if ( location >= 0 )
	{
		glUniformMatrix4fv( location, 1, GL_FALSE, &shadowMatrix[ 0 ][ 0 ] );
	}

	//float planeSize = 5.25f;
	glBegin( GL_QUADS );
	glVertexAttrib3f( 0, -planeSize * 0.5f, -planeSize * 0.5f, 0.0f );
	glVertexAttrib3f( 0, planeSize * 0.5f, -planeSize * 0.5f, 0.0f );
	glVertexAttrib3f( 0, planeSize * 0.5f, planeSize * 0.5f, 0.0f );
	glVertexAttrib3f( 0, -planeSize * 0.5f, planeSize * 0.5f, 0.0f );
	glEnd();
	//---------------------------------------

	glUseProgram( 0 );

	//glPopMatrix();
	//glMatrixMode( GL_MODELVIEW );
	//glPopMatrix();
}

/******************************************************************************
 * Render
 ******************************************************************************/
void ShadowMap::spitOutDepthBuffer()
{
	int size = _shadowMapWidth * _shadowMapHeight;
	float* buffer = new float[ size ];
	unsigned char * imgBuffer = new unsigned char[ size * 4 ];
	
	//---------------------------------------------------------
	// Assign the depth buffer texture to texture channel 0
    glActiveTexture( GL_TEXTURE0 );
    glBindTexture( GL_TEXTURE_2D, _depthTexture );
	//---------------------------------------------------------
	glGetTexImage( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT, buffer );

	for( int i = 0; i < _shadowMapHeight; i++ )
	{
		for( int j = 0; j < _shadowMapWidth; j++ )
		{
			int imgIdx = 4 * ( ( i * _shadowMapWidth ) + j );
			int bufIdx = ( ( _shadowMapHeight - i - 1 ) * _shadowMapWidth ) + j;

			// This is just to make a more visible image.  Scale so that
			// the range (minVal, 1.0) maps to (0.0, 1.0).  This probably should
			// be tweaked for different light configurations.
			float minVal = 0.88f;
			float scale = ( buffer[ bufIdx ] - minVal ) / ( 1.0f - minVal );
			unsigned char val = (unsigned char)(scale * 255);
			imgBuffer[ imgIdx] = val;
			imgBuffer[ imgIdx + 1 ] = val;
			imgBuffer[ imgIdx + 2 ] = val;
			imgBuffer[ imgIdx + 3 ] = 0xff;
		}
	}

	QImage img( imgBuffer, _shadowMapWidth, _shadowMapHeight, QImage::Format_RGB32 );
	img.save("depth.png", "PNG");

	delete [] buffer;
	delete [] imgBuffer;

	//exit( 1 );
}

/******************************************************************************
 * Light viewing system parameters
 ******************************************************************************/
void ShadowMap::setLightEye( const glm::vec3& pValue )
{
	_lightEye = pValue;
}
void ShadowMap::setLightCenter( const glm::vec3& pValue )
{
	_lightCenter = pValue;
}
void ShadowMap::setLightUp( const glm::vec3& pValue )
{
	_lightUp = pValue;
}
void ShadowMap::setLightFovY( float pValue )
{
	_lightFovY = pValue;
}
void ShadowMap::setLightAspectRatio( float pValue )
{
	_lightAspectRatio = pValue;
}	
void ShadowMap::setLightZNear( float pValue )
{
	_lightZNear = pValue;
}
void ShadowMap::setLightZFar( float pValue )
{
	_lightZFar = pValue;
}

/******************************************************************************
 * Camera viewing system parameters
 ******************************************************************************/
void ShadowMap::setCameraEye( const glm::vec3& pValue )
{
	_cameraEye = pValue;
}
void ShadowMap::setCameraCenter( const glm::vec3& pValue )
{
	_cameraCenter = pValue;
}
void ShadowMap::setCameraUp( const glm::vec3& pValue )
{
	_cameraUp = pValue;
}
void ShadowMap::setCameraFovY( float pValue )
{
	_cameraFovY = pValue;
}
void ShadowMap::setCameraAspectRatio( float pValue )
{
	_cameraAspectRatio = pValue;
}	
void ShadowMap::setCameraZNear( float pValue )
{
	_cameraZNear = pValue;
}
void ShadowMap::setCameraZFar( float pValue )
{
	_cameraZFar = pValue;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ShadowMap::setWindowsSize( int pWidth, int pHeight )
{
	_width = pWidth;
	_height = pHeight;
}
