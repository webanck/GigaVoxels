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

#ifndef _SAMPLE_CORE_H_
#define _SAMPLE_CORE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <cuda_runtime.h>
#include <vector_types.h>

// GigaVoxels
#include <GvCore/vector_types_ext.h>
#include <GvUtils/GvForwardDeclarationHelper.h>

// Cuda GPU Computing SDK
#include <helper_math.h>

// Loki
#include <loki/Typelist.h>

// OpenGL
#include <GL/glew.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

namespace GvUtils
{
	// Transfer function
	class GvTransferFunction;
}

// Custom Producer
template< typename TDataStructureType >
class ProducerKernel;

// Custom Shader
class ShaderKernel;

// Volume data
namespace VolumeData
{
	class vdCube3D4;
}

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

// Defines the type list representing the content of one voxel
typedef Loki::TL::MakeTypelist< uchar >::Result DataType;

// Defines the size of a node tile
typedef GvCore::StaticRes1D< 2 > NodeRes;

// Defines the size of a brick
typedef GvCore::StaticRes1D< 8 > BrickRes;

// Defines the type of structure we want to use
typedef GvStructure::GvVolumeTree
<
	DataType,
	NodeRes, BrickRes
>
DataStructureType;

// Defines the type of the producer
typedef GvUtils::GvSimpleHostProducer
<
	ProducerKernel< DataStructureType >,
	DataStructureType
>
ProducerType;

// Defines the type of the shader
typedef GvUtils::GvSimpleHostShader
<
	ShaderKernel
>
ShaderType;

// Simple Pipeline
typedef GvUtils::GvSimplePipeline
<
	ProducerType,
	ShaderType,
	DataStructureType
>
PipelineType;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class SampleCore
 *
 * @brief The SampleCore class provides a helper class containing a	GifgaVoxels pipeline.
 *
 * A simple GigaVoxels pipeline consists of :
 * - a data structure
 * - a cache
 * - a custom producer
 * - a renderer
 *
 * The custom shader is pass as a template argument.
 *
 * Besides, this class enables the interoperability with OpenGL graphics library.
 */
class SampleCore
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	SampleCore();

	/**
	 * Destructor
	 */
	~SampleCore();

	/**
	 * Initialize the GigaVoxels pipeline
	 */
	void init();

	/**
	 * Draw function called of frame
	 */
	void draw();

	/**
	 * Resize the frame
	 *
	 * @param width the new width
	 * @param height the new height
	 */
	void resize( int width, int height );

	/**
	 * Clear the GigaVoxels cache
	 */
	void clearCache();

	/**
	 * Toggle the display of the N-tree (octree) of the data structure
	 */
	void toggleDisplayOctree();

	/**
	 * Toggle the GigaVoxels dynamic update mode
	 */
	void toggleDynamicUpdate();

	/**
	 * Toggle the display of the performance monitor utility if
	 * GigaVoxels has been compiled with the Performance Monitor option
	 *
	 * @param mode The performance monitor mode (1 for CPU, 2 for DEVICE)
	 */
	void togglePerfmonDisplay( uint mode );

	/**
	 * Increment the max resolution of the data structure
	 */
	void incMaxVolTreeDepth();

	/**
	 * Decrement the max resolution of the data structure
	 */
	void decMaxVolTreeDepth();

	/**
	 * Specify color to clear the color buffer
	 *
	 * @param pRed red component
	 * @param pGreen green component
	 * @param pBlue blue component
	 * @param pAlpha alpha component
	 */
	void setClearColor( unsigned char pRed, unsigned char pGreen, unsigned char pBlue, unsigned char pAlpha );

	/**
	 * Update the associated transfer function
	 *
	 * @param pData the new transfer function data
	 * @param pSize the size of the transfer function
	 */
	void updateTransferFunction( float* pData, unsigned int pSize );

	/**
	 * Set the light position
	 *
	 * @param pX the X light position
	 * @param pY the Y light position
	 * @param pZ the Z light position
	 */
	void setLightPosition( float pX, float pY, float pZ );

	/**
	 * Set the noise parameters
	 *
	 * @param p1 the first noise parameter
	 * @param p2 the second noise parameter
	 */
	void setNoiseParameters( float p1, float p2 );


	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Initialize the GigaVoxels pipeline
	 *
	 * @return flag to tell wheter or not it succeeded
	 */
	bool initializePipeline();

	/**
	 * Finalize the GigaVoxels pipeline
	 *
	 * @return flag to tell wheter or not it succeeded
	 */
	bool finalizePipeline();

	/**
	 * Initialize the 3D model
	 *
	 * @return flag to tell wheter or not it succeeded
	 */
	bool initialize3DModel();

	/**
	 * Finalize the 3D model
	 *
	 * @return flag to tell wheter or not it succeeded
	 */
	bool finalize3DModel();

	/**
	 * Initialize the transfer function
	 *
	 * @return flag to tell wheter or not it succeeded
	 */
	bool initializeTransferFunction();

	/**
	 * Finalize the transfer function
	 *
	 * @return flag to tell wheter or not it succeeded
	 */
	bool finalizeTransferFunction();

	/**
	 * Finalize graphics resources
	 *
	 * @return flag to tell wheter or not it succeeded
	 */
	bool finalizeGraphicsResources();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * GigaVoxels pipeline
	 */
	PipelineType* _pipeline;

	/**
	 * Depth buffer
	 */
	GLuint _depthBuffer;

	/**
	 * Color texture
	 */
	GLuint _colorTex;

	/**
	 * Depth texture
	 */
	GLuint _depthTex;

	/**
	 * Frame buffer
	 */
	GLuint _frameBuffer;

	/**
	 * Frame width
	 */
	int _width;

	/**
	 * Frame height
	 */
	int _height;

	/**
	 * Flag to tell wheter or not to display the N-tree (octree) of the data structure
	 */
	bool _displayOctree;

	/**
	 * Flag to tell wheter or not to display the performance monitor utility
	 */
	uint _displayPerfmon;

	/**
	 * Max resolution of the data structure
	 */
	uint _maxVolTreeDepth;

	/**
	 * 3D model as signed distance field( normal + distance)
	 */
	VolumeData::vdCube3D4* _signedDistanceField;

	/**
	 * Transfer function
	 */
	GvUtils::GvTransferFunction* _transferFunction;

	/******************************** METHODS *********************************/

};

#endif // !_SAMPLE_CORE_H_
