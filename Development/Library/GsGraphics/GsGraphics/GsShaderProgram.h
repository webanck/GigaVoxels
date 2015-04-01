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

#ifndef _GS_SHADER_PROGRAM_H_
#define _GS_SHADER_PROGRAM_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaSpace
#include "GsGraphics/GsGraphicsCoreConfig.h"

// OpenGL
#include <GL/glew.h>

// System
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

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GsGraphics
{

/** 
 * @class GsShaderProgram
 *
 * @brief The GsShaderProgram class provides interface to handle a ray map.
 *
 * Ray map is a container of ray initialized for the rendering phase.
 */
class GSGRAPHICS_EXPORT GsShaderProgram
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Shader type enumeration
	 */
	enum ShaderType
	{
		eVertexShader = 0,
		eTesselationControlShader,
		eTesselationEvaluationShader,
		eGeometryShader,
		eFragmentShader,
		eComputeShader,
		eNbShaderTypes
	};

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Main shader program
	 */
	GLuint _program;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GsShaderProgram();

	/**
	 * Destructor
	 */
	virtual ~GsShaderProgram();

	/**
	 * Initialize
	 *
	 * @return a flag to tell wheter or not it succeeds.
	 */
	bool initialize();

	/**
	 * Finalize
	 *
	 * @return a flag to tell wheter or not it succeeds.
	 */
	bool finalize();

	/**
	 * Compile shader
	 */
	bool addShader( ShaderType pShaderType, const std::string& pShaderFileName );

	/**
	 * Link program
	 */
	bool link();

	/**
	 * Use program
	 */
	inline void use();

	/**
	 * Unuse program
	 */
	static inline void unuse();

	/**
	 * Set fixed pipeline
	 */
	static inline void setFixedPipeline();

	/**
	 * Tell wheter or not pipeline has a given type of shader
	 *
	 * @param pShaderType the type of shader to test
	 *
	 * @return a flag telling wheter or not pipeline has a given type of shader
	 */
	bool hasShaderType( ShaderType pShaderType ) const;

	/**
	 * Get the source code associated to a given type of shader
	 *
	 * @param pShaderType the type of shader
	 *
	 * @return the associated shader source code
	 */
	std::string getShaderSourceCode( ShaderType pShaderType ) const;

	/**
	 * Get the filename associated to a given type of shader
	 *
	 * @param pShaderType the type of shader
	 *
	 * @return the associated shader filename
	 */
	std::string getShaderFilename( ShaderType pShaderType ) const;

	/**
	 * ...
	 *
	 * @param pShaderType the type of shader
	 *
	 * @return ...
	 */
	bool reloadShader( ShaderType pShaderType );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Vertex shader file name
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::string _vertexShaderFilename;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * Tesselation Control shader file name
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::string _tesselationControlShaderFilename;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * Tesselation Evaluation shader file name
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::string _tesselationEvaluationShaderFilename;
#if defined _MSC_VER
#pragma warning( pop )
#endif
	/**
	 * Geometry shader file name
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::string _geometryShaderFilename;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * Fragment shader file name
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::string _fragmentShaderFilename;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * Compute shader file name
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::string _computeShaderFilename;
#if defined _MSC_VER
#pragma warning( pop )
#endif
	
	/**
	 * Vertex shader source code
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::string _vertexShaderSourceCode;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * Tesselation Control shader source code
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::string _tesselationControlShaderSourceCode;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * Tesselation Evaluation shader source code
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::string _tesselationEvaluationShaderSourceCode;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * Geometry shader source code
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::string _geometryShaderSourceCode;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * Fragment shader source code
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::string _fragmentShaderSourceCode;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * Compute shader source code
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::string _computeShaderSourceCode;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	///**
	// * Main shader program
	// */
	//GLuint _program;

	/**
	 * Vertex shader
	 */
	GLuint _vertexShader;

	/**
	 * Tesselation Control shader
	 */
	GLuint _tesselationControlShader;

	/**
	 * Tesselation Evaluation shader
	 */
	GLuint _tesselationEvaluationShader;

	/**
	 * Geometry shader
	 */
	GLuint _geometryShader;

	/**
	 * Fragment shader
	 */
	GLuint _fragmentShader;

	/**
	 * Compute shader
	 */
	GLuint _computeShader;

	/**
	 * ...
	 */
	bool _linked;

	/******************************** METHODS *********************************/

	/**
	 * ...
	 *
	 * @param pFilename ...
	 * @param pFileContent ...
	 *
	 * @return ...
	 */
	static bool getFileContent( const std::string& pFilename, std::string& pFileContent );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GsShaderProgram( const GsShaderProgram& );

	/**
	 * Copy operator forbidden.
	 */
	GsShaderProgram& operator=( const GsShaderProgram& );

};

} // namespace GsGraphics

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GsShaderProgram.inl"

#endif
