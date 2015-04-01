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

#ifndef _PARTICLE_SYSTEM_H_
#define _PARTICLE_SYSTEM_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/vector_types_ext.h>

// GL
#include <GL/glew.h>

// Cuda
#include <vector_types.h>
#include <driver_types.h>

// STL
#include <string>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GigaVoxels
namespace GvRendering
{
	class GvGraphicsResource;
}
namespace GsGraphics
{
	class GsShaderProgram;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class ParticleSystem
 *
 * @brief The ParticleSystem class provides the mecanisms to generate star positions.
 *
 * ...
 */
class ParticleSystem
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Rendering type
	 */
	enum RenderingType
	{
		ePoints = 0,		// points
		ePointSprite,		// textured quads
		eParticleSystem		// instanced 3D models
	};

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Current number of points in the buffer to be drawned
	 */
	unsigned int _nbRenderablePoints;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * Recupere deux points de la diagonale du cube pour connaitre l'intervalle de valeurs
	 */
	ParticleSystem( const float3& pPoint1, const float3& pPoint2 );

	/**
	 * Destructor
	 */
	~ParticleSystem();

	/**
	 * Initialise le buffer GPU contenant les positions
	 */
	void initGPUBuf();

	/**
	  * Get the buffer of data (sphere positions and radius)
	 *
	 * @return the buffer of data (sphere positions and radius)
	 */
	// Array3DGPULinear< float4 >* getGPUBuf();
	float4* getGPUBuf();

	/**
	 * Get the number of particles
	 *
	 * @return the number of particles
	 */
	unsigned int getNbParticles() const;

	/**
	 * Set the number of particles
	 *
	 * @param pValue the number of particles
	 */
	void setNbParticles( unsigned int pValue );

	/**
	 * Spheres ray-tracing methods
	 */
	float getPointSizeFader() const;
	void setPointSizeFader( float pValue );
	float getFixedSizePointSize() const;
	void setFixedSizePointSize( float pValue );
	//bool hasMeanSizeOfSpheres() const;
	//void setMeanSizeOfSpheres( bool pFlag );

	/**
	 * Render the particle system
	 */
	void render( const float4x4& pModelViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport );

	/**
	 * Get the associated graphics resource
	 * 
	 * return the associated graphics resource
	 */
	GvRendering::GvGraphicsResource* getGraphicsResource();

	/**
	 * ...
	 *
	 * return ...
	 */
	bool initGraphicsResources();

	/**
	 * ...
	 *
	 * return ...
	 */
	bool releaseGraphicsResources();

	bool hasShaderUniformColor() const;
	void setShaderUniformColorMode( bool pFlag );
	const float4& getShaderUniformColor() const;
	void setShaderUniformColor( float pR, float pG, float pB, float pA );
	bool hasShaderAnimation() const;
	void setShaderAnimation( bool pFlag );
	bool hasTexture() const;
	void setTexture( bool pFlag );
	const std::string& getTextureFilename() const;
	void setTextureFilename( const std::string& pFilename );
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Rendering type
	 */
	RenderingType _renderingType;

	/**
	 * Point rendering's GLSL shader program
	 */
	GLuint _pointsShaderProgram;

	/**
	 * Point sprites rendering's GLSL shader program
	 */
	//GLuint _pointSpritesShaderProgram;

	/**
	 * Vertex buffer
	 */
	GLuint _positionBuffer;

	/**
	 * Vertex array object
	 */
	GLuint _vao;

	/**
	 * Sprite texture
	 */
	GLuint _spriteTexture;

	///**
	// * Current number of points in the buffer to be drawned
	// */
	//unsigned int _nbRenderablePoints;

	/**
	 * Graphics resource
	 */
	GvRendering::GvGraphicsResource* _graphicsResource;

	/**
	 * Uniform color
	 */
	float4 _uniformColor;

	/**
	 * Texture filename
	 */
	std::string _textureFilename;

	bool _shaderUseUniformColor;
	float4 _shaderUniformColor;
	bool _shaderAnimation;
	bool _hasTexture;

	/**
	 * Shader program
	 */
	GsGraphics::GsShaderProgram* _shaderProgramPoints;
	GsGraphics::GsShaderProgram* _shaderProgramPointSprite;
	GsGraphics::GsShaderProgram* _shaderProgramParticleSystem;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private :

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Point devant en bas a gauche du cube
	 */
	float3 _p1;

	/**
	 * Point derriere en haut a droite du cube
	 */
	float3 _p2;

	/**
	 * Max number of particles
	 */
	unsigned int _nbParticles;

	/**
	 * The buffer of data (sphere positions and radius)
	 */
	float4* _d_particleBuffer;

	/**
	 * Spheres ray-tracing parameters
	 */
	float _pointSizeFader;
	float _fixedSizePointSize;

	/******************************** METHODS *********************************/

	/**
	 * Genere une position aleatoire
	 *
	 * @param pSeed ...
	 */
	float3 genPos( int pSeed );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "ParticleSystem.inl"

#endif
