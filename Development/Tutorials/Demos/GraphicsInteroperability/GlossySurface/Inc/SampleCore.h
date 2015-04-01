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
class SampleViewer;
#include "GlossyObject.h"
#include "Mesh.h"
#include "CubeMap.h"

// Custom Producer
template< typename TDataStructureType, typename TDataProductionManager >
class Producer;
//template< typename TDataStructureType >
//class ProducerTorusKernel;

// Custom Shader
class Shader {};	// Define a NullType and/or EmptyType

// Custom Renderer
template
<
	typename VolumeTreeType, typename VolumeTreeCacheType,
	typename SampleShader
>
class VolumeTreeRendererGLSL;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

// Defines the type list representing the content of one voxel
typedef Loki::TL::MakeTypelist< uchar4 >::Result DataType;

// Defines the size of a node tile
typedef GvCore::StaticRes1D< 2 > NodeRes;

// Defines the size of a brick
typedef GvCore::StaticRes1D< 8 > BrickRes;

// Defines the size of the border around a brick
enum { BrickBorderSize = 1 };

// Defines the total size of a brick
typedef GvCore::StaticRes1D< 8 + 2 * BrickBorderSize > RealBrickRes;

// Defines the type of structure we want to use.
typedef GvStructure::GvVolumeTree
<
	DataType,
	NodeRes, BrickRes
>
DataStructureType;

// Defines the type of the producer
typedef GvStructure::GvDataProductionManager< DataStructureType > DataProductionManagerType;
typedef Producer< DataStructureType, DataProductionManagerType > ProducerType;

// Defines the type of the producer
//typedef GvUtils::GvSimpleHostProducer
//<
//	ProducerTorusKernel< DataStructureType >,
//	DataStructureType
//>
//ProducerType;

// Defines the type of the cache we want to use.
//typedef GvStructure::GvDataProductionManager
//<
//	DataStructureType
//>
//CacheType;

// Defines the type of the shader
typedef Shader ShaderType;

// Defines the type of the renderer we want to use.
typedef VolumeTreeRendererGLSL
<
	DataStructureType,
	DataProductionManagerType,
	ShaderType
>
RendererType;

// Simple Pipeline
typedef GvUtils::GvSimplePipeline
<
	ProducerType,
	ShaderType,
	DataStructureType,
	DataProductionManagerType,
	RendererType
>
PipelineType;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class SampleCore
 *
 * @brief The SampleCore class provides...
 *
 * ...
 */
class SampleCore
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

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
	 * ...
	 */
	void init(SampleViewer* sv);
	/**
	 * ...
	 */
	void draw();
	/**
	 * ...
	 *
	 * @param width ...
	 * @param height ...
	 */
	void resize( int width, int height );

	/**
	 * ...
	 */
	void clearCache();

	/**
	 * ...
	 */
	void toggleDisplayOctree();
	/**
	 * ...
	 */
	void toggleDynamicUpdate();
	/**
	 * ...
	 *
	 * @param mode ...
	 */
	void togglePerfmonDisplay( uint mode );

	/**
	 * ...
	 */
	void incMaxVolTreeDepth();
	/**
	 * ...
	 */
	void decMaxVolTreeDepth();

	/**
	 * Set the camera light position
	 *
	 * @param pX the X light position
	 * @param pY the Y light position
	 * @param pZ the Z light position
	 */
	void setLightPosition( float pX, float pY, float pZ );

	/**
	 * Set the world light position
	 *
	 * @param pX the X light position
	 * @param pY the Y light position
	 * @param pZ the Z light position
	 */
	void setWorldLight( float pX, float pY, float pZ );

	/**
	 * Get the OBJ file's name.
	 */
	string getShadowCasterFile();

	/**
	 * Set the world camera position
	 *
	 * @param x the X camera position
	 * @param y the Y camera position
	 * @param z the Z camera position
	 */
	void setWorldCamera(float x, float y, float z);

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Sample Viewer
	 */ 
	SampleViewer* sviewer;

	/**
	 * OBJ file to voxelize and whose shadow to cast
	 */
	string shadowCasterFile;

	/**
	 * Camera position in world coordinate system.
	 */
	float3 worldCamPos;

	/**
	 * GigaVoxels pipeline
	 */
	PipelineType* _pipeline;

	/**
	 * ...
	 */
	int				mWidth;

	/**
	 * ...
	 */
	int				mHeight;

	/**
	 * ...
	 */
	bool			mDisplayOctree;

	/**
	 * ...
	 */
	uint			mDisplayPerfmon;

	/**
	 * ...
	 */
	uint			mMaxVolTreeDepth;

	/**
	 * Light's position in camera coordinates
	 */
	float3          lightPos;

	/**
	 * Light's position in world coordinates
	 */
	float3          worldLight;

	/******************************** METHODS *********************************/

};

#endif // !_SAMPLE_CORE_H_
