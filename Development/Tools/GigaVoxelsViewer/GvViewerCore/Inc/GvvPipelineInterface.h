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

#ifndef GVVPIPELINEINTERFACE_H
#define GVVPIPELINEINTERFACE_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvCoreConfig.h"
#include "GvvBrowsable.h"
#include "GvvDataType.h"

// STL
#include <string>
#include <vector>

// glm
#include <glm/glm.hpp>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GvViewer
namespace GvViewerCore
{
	class GvvTransferFunctionInterface;
	class GvvMeshInterface;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerCore
{

/** 
 * @class SampleCore
 *
 * @brief The SampleCore class provides...
 *
 * ...
 */
class GVVIEWERCORE_EXPORT GvvPipelineInterface : public GvvBrowsable
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* INNER TYPES *******************************/

	///**
	// * Shader type enumeration
	// */
	//enum ShaderType
	//{
	//	eVertexShader = 0,
	//	eTesselationControlShader,
	//	eTesselationEvaluationShader,
	//	eGeometryShader,
	//	eFragmentShader,
	//	eComputeShader
	//};

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Type name
	 */
	static const char* cTypeName;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvvPipelineInterface();

	/**
	 * Destructor
	 */
	virtual ~GvvPipelineInterface();

	/**
	 * Returns the type of this browsable. The type is used for retrieving
	 * the context menu or when requested or assigning an icon to the
	 * corresponding item
	 *
	 * @return the type name of this browsable
	 */
	virtual const char* getTypeName() const;

	/**
     * Gets the name of this browsable
     *
     * @return the name of this browsable
     */
	virtual const char* getName() const;

	/**
	 * Initialize the GigaVoxels pipeline
	 */
	virtual void init();

	/**
	 * Draw function called every frame
	 */
	virtual void draw();

	/**
	 * Resize the frame
	 *
	 * @param width the new width
	 * @param height the new height
	 */
	virtual void resize( int width, int height );

	/**
	 * Clear the GigaVoxels cache
	 */
	virtual void clearCache();

	/**
	 * Toggle the display of the N-tree (octree) of the data structure
	 */
	virtual void toggleDisplayOctree();

	/**
	 * Get the appearance of the N-tree (octree) of the data structure
	 */
	virtual void getDataStructureAppearance( bool& pShowNodeHasBrickTerminal, bool& pShowNodeHasBrickNotTerminal, bool& pShowNodeIsBrickNotInCache, bool& pShowNodeEmptyOrConstant
											, float& pNodeHasBrickTerminalColorR, float& pNodeHasBrickTerminalColorG, float& pNodeHasBrickTerminalColorB, float& pNodeHasBrickTerminalColorA
											, float& pNodeHasBrickNotTerminalColorR, float& pNodeHasBrickNotTerminalColorG, float& pNodeHasBrickNotTerminalColorB, float& pNodeHasBrickNotTerminalColorA
											, float& pNodeIsBrickNotInCacheColorR, float& pNodeIsBrickNotInCacheColorG, float& pNodeIsBrickNotInCacheColorB, float& pNodeIsBrickNotInCacheColorA
											, float& pNodeEmptyOrConstantColorR, float& pNodeEmptyOrConstantColorG, float& pNodeEmptyOrConstantColorB, float& pNodeEmptyOrConstantColorA ) const;

	/**
	 * Set the appearance of the N-tree (octree) of the data structure
	 */
	virtual void setDataStructureAppearance( bool pShowNodeHasBrickTerminal, bool pShowNodeHasBrickNotTerminal, bool pShowNodeIsBrickNotInCache, bool pShowNodeEmptyOrConstant
											, float pNodeHasBrickTerminalColorR, float pNodeHasBrickTerminalColorG, float pNodeHasBrickTerminalColorB, float pNodeHasBrickTerminalColorA
											, float pNodeHasBrickNotTerminalColorR, float pNodeHasBrickNotTerminalColorG, float pNodeHasBrickNotTerminalColorB, float pNodeHasBrickNotTerminalColorA
											, float pNodeIsBrickNotInCacheColorR, float pNodeIsBrickNotInCacheColorG, float pNodeIsBrickNotInCacheColorB, float pNodeIsBrickNotInCacheColorA
											, float pNodeEmptyOrConstantColorR, float pNodeEmptyOrConstantColorG, float pNodeEmptyOrConstantColorB, float pNodeEmptyOrConstantColorA );

	/**
	 * Toggle the GigaVoxels dynamic update mode
	 */
	virtual void toggleDynamicUpdate();

	/**
	 * Get the dynamic update state
	 *
	 * @return the dynamic update state
	 */
	virtual bool hasDynamicUpdate() const;

	/**
	 * Set the dynamic update state
	 *
	 * @param pFlag the dynamic update state
	 */
	virtual void setDynamicUpdate( bool pFlag );

	/**
	 * Toggle the display of the performance monitor utility if
	 * GigaVoxels has been compiled with the Performance Monitor option
	 *
	 * @param mode The performance monitor mode (1 for CPU, 2 for DEVICE)
	 */
	virtual void togglePerfmonDisplay( unsigned int mode );

	/**
	 * Increment the max resolution of the data structure
	 */
	virtual void incMaxVolTreeDepth();

	/**
	 * Decrement the max resolution of the data structure
	 */
	virtual void decMaxVolTreeDepth();

	/**
	 * Get the node tile resolution of the data structure.
	 *
	 * @param pX the X node tile resolution
	 * @param pY the Y node tile resolution
	 * @param pZ the Z node tile resolution
	 */
	virtual void getDataStructureNodeTileResolution( unsigned int& pX, unsigned int& pY, unsigned int& pZ ) const;

	/**
	 * Get the brick resolution of the data structure (voxels).
	 *
	 * @param pX the X brick resolution
	 * @param pY the Y brick resolution
	 * @param pZ the Z brick resolution
	 */
	virtual void getDataStructureBrickResolution( unsigned int& pX, unsigned int& pY, unsigned int& pZ ) const;
	
	/**
	 * Get the max depth.
	 *
	 * @return the max depth
	 */
	virtual unsigned int getRendererMaxDepth() const;
	
	/**
	 * Set the max depth.
	 *
	 * @param pValue the max depth
	 */
	virtual void setRendererMaxDepth( unsigned int pValue );

	/**
	 * Get the max number of requests of node subdivisions the cache has to handle.
	 *
	 * @return the max number of requests
	 */
	virtual unsigned int getCacheMaxNbNodeSubdivisions() const;

	/**
	 * Set the max number of requests of node subdivisions.
	 *
	 * @param pValue the max number of requests
	 */
	virtual void setCacheMaxNbNodeSubdivisions( unsigned int pValue );

	/**
	 * Get the max number of requests of brick of voxel loads.
	 *
	 * @return the max number of requests
	 */
	virtual unsigned int getCacheMaxNbBrickLoads() const;
	
	/**
	 * Set the max number of requests of brick of voxel loads.
	 *
	 * @param pValue the max number of requests
	 */
	virtual void setCacheMaxNbBrickLoads( unsigned int pValue );
	
	/**
	 * Get the request strategy indicating if, during data structure traversal,
	 * priority of requests is set on brick loads or on node subdivisions first.
	 *
	 * @return the flag indicating the request strategy
	 */
	virtual bool hasRendererPriorityOnBricks() const;

	/**
	 * Set the request strategy indicating if, during data structure traversal,
	 * priority of requests is set on brick loads or on node subdivisions first.
	 *
	 * @param pFlag the flag indicating the request strategy
	 */
	virtual void setRendererPriorityOnBricks( bool pFlag );

	/**
	 * Tell wheter or not the pipeline has a transfer function.
	 *
	 * @return the flag telling wheter or not the pipeline has a transfer function
	 */
	virtual bool hasTransferFunction() const;

	/**
	 * Get the transfer function if it has one.
	 *
	 * @param pIndex the index of the transfer function
	 *
	 * @return the transfer function
	 */
	virtual GvvTransferFunctionInterface* getTransferFunction( unsigned int pIndex = 0 ) const;

	/**
	 * Get the transfer function if it has one.
	 *
	 * @param pIndex the index of the transfer function
	 *
	 * @return the transfer function
	 */
	virtual GvvTransferFunctionInterface* editTransferFunction( unsigned int pIndex = 0 );

	/**
	 * Get the transfer function filename if it has one.
	 *
	 * @param pIndex the index of the transfer function
	 *
	 * @return the transfer function
	 */
	virtual const char* getTransferFunctionFilename( unsigned int pIndex = 0 ) const;

	/**
	 * Set the transfer function filename if it has one.
	 *
	 * @param pFilename the transfer function's filename
	 * @param pIndex the index of the transfer function
	 *
	 * @return the transfer function
	 */
	virtual void setTransferFunctionFilename( const char* pFilename, unsigned int pIndex = 0 );

	/**
	 * Update the associated transfer function
	 *
	 * @param the new transfer function data
	 * @param the size of the transfer function
	 */
	virtual void updateTransferFunction( float* pData, unsigned int pSize );

	/**
	 * Tell wheter or not the pipeline has a light.
	 *
	 * @return the flag telling wheter or not the pipeline has a light
	 */
	virtual bool hasLight() const;

	/**
	 * Get the light position
	 *
	 * @param pX the X light position
	 * @param pY the Y light position
	 * @param pZ the Z light position
	 */
	virtual void getLightPosition( float& pX, float& pY, float& pZ ) const;

	/**
	 * Set the light position
	 *
	 * @param pX the X light position
	 * @param pY the Y light position
	 * @param pZ the Z light position
	 */
	virtual void setLightPosition( float pX, float pY, float pZ );

	/**
	 * Tell wheter or not the pipeline has a 3D model to load.
	 *
	 * @return the flag telling wheter or not the pipeline has a 3D model to load
	 */
	virtual bool has3DModel() const;

	/**
	 * Get the 3D model filename to load
	 *
	 * @return the 3D model filename to load
	 */
	virtual std::string get3DModelFilename() const;

	/**
	 * Set the 3D model filename to load
	 *
	 * @param pFilename the 3D model filename to load
	 */
	virtual void set3DModelFilename( const std::string& pFilename );

	/**
	 * Specify color to clear the color buffer
	 *
	 * @param pRed red component
	 * @param pGreen green component
	 * @param pBlue blue component
	 * @param pAlpha alpha component
	 */
	virtual void setClearColor( unsigned char pRed, unsigned char pGreen, unsigned char pBlue, unsigned char pAlpha );

	/**
	 * Get the data type list used to store voxels in the data structure
	 *
	 * @return the data type list of voxels
	 */
	const GvvDataType& getDataTypes() const;

	/**
	 * Get the data type list used to store voxels in the data structure
	 *
	 * @return the data type list of voxels
	 */
	GvvDataType& editDataTypes();

	/**
	 * Get the translation
	 *
	 * @param pX the translation on x axis
	 * @param pY the translation on y axis
	 * @param pZ the translation on z axis
	 */
	virtual void getTranslation( float& pX, float& pY, float& pZ ) const;

	/**
	 * Set the translation
	 *
	 * @param pX the translation on x axis
	 * @param pY the translation on y axis
	 * @param pZ the translation on z axis
	 */
	virtual void setTranslation( float pX, float pY, float pZ );

	/**
	 * Get the rotation
	 *
	 * @param pAngle the rotation angle (in degree)
	 * @param pX the x component of the rotation vector
	 * @param pY the y component of the rotation vector
	 * @param pZ the z component of the rotation vector
	 */
	virtual void getRotation( float& pAngle, float& pX, float& pY, float& pZ ) const;

	/**
	 * Set the rotation
	 *
	 * @param pAngle the rotation angle (in degree)
	 * @param pX the x component of the rotation vector
	 * @param pY the y component of the rotation vector
	 * @param pZ the z component of the rotation vector
	 */
	virtual void setRotation( float pAngle, float pX, float pY, float pZ );

	/**
	 * Get the uniform scale
	 *
	 * @param pValue the uniform scale
	 */
	virtual void getScale( float& pValue ) const;

	/**
	 * Set the uniform scale
	 *
	 * @param pValue the uniform scale
	 */
	virtual void setScale( float pValue );

	/**
	 * Get the number of requests of node subdivisions the cache has handled.
	 *
	 * @return the number of requests
	 */
	virtual unsigned int getCacheNbNodeSubdivisionRequests() const;

	/**
	 * Get the number of requests of brick of voxel loads the cache has handled.
	 *
	 * @return the number of requests
	 */
	virtual unsigned int getCacheNbBrickLoadRequests() const;

	/**
	 * Get the cache policy
	 *
	 * @return the cache policy
	 */
	virtual unsigned int getCachePolicy() const;

	/**
	 * Set the cache policy
	 *
	 * @param pValue the cache policy
	 */
	virtual void setCachePolicy( unsigned int pValue );

	/**
	 * Get the node cache memory
	 *
	 * @return the node cache memory
	 */
	virtual unsigned int getNodeCacheMemory() const;

	/**
	 * Set the node cache memory
	 *
	 * @param pValue the node cache memory
	 */
	virtual void setNodeCacheMemory( unsigned int pValue );

	/**
	 * Get the brick cache memory
	 *
	 * @return the brick cache memory
	 */
	virtual unsigned int getBrickCacheMemory() const;

	/**
	 * Set the brick cache memory
	 *
	 * @param pValue the brick cache memory
	 */
	virtual void setBrickCacheMemory( unsigned int pValue );

	/**
	 * Get the node cache capacity
	 *
	 * @return the node cache capacity
	 */
	virtual unsigned int getNodeCacheCapacity() const;

	/**
	 * Set the node cache capacity
	 *
	 * @param pValue the node cache capacity
	 */
	virtual void setNodeCacheCapacity( unsigned int pValue );

	/**
	 * Get the brick cache capacity
	 *
	 * @return the brick cache capacity
	 */
	virtual unsigned int getBrickCacheCapacity() const;

	/**
	 * Set the brick cache capacity
	 *
	 * @param pValue the brick cache capacity
	 */
	virtual void setBrickCacheCapacity( unsigned int pValue );

	/**
	 * Get the number of unused nodes in cache
	 *
	 * @return the number of unused nodes in cache
	 */
	virtual unsigned int getCacheNbUnusedNodes() const;

	/**
	 * Get the number of unused bricks in cache
	 *
	 * @return the number of unused bricks in cache
	 */
	virtual unsigned int getCacheNbUnusedBricks() const;

	/**
	 * Get the nodes cache usage
	 *
	 * @return the nodes cache usage
	 */
	virtual unsigned int getNodeCacheUsage() const;

	/**
	 * Get the bricks cache usage
	 *
	 * @return the bricks cache usage
	 */
	virtual unsigned int getBrickCacheUsage() const;

	/**
	 * Get the flag telling wheter or not tree data dtructure monitoring is activated
	 *
	 * @return the flag telling wheter or not tree data dtructure monitoring is activated
	 */
	virtual bool hasTreeDataStructureMonitoring() const;

	/**
	 * Set the flag telling wheter or not tree data dtructure monitoring is activated
	 *
	 * @param pFlag the flag telling wheter or not tree data dtructure monitoring is activated
	 */
	virtual void setTreeDataStructureMonitoring( bool pFlag );

	/**
	 * Get the number of tree leaf nodes
	 *
	 * @return the number of tree leaf nodes
	 */
	virtual unsigned int getNbTreeLeafNodes() const;

	/**
	 * Get the number of tree nodes
	 *
	 * @return the number of tree nodes
	 */
	virtual unsigned int getNbTreeNodes() const;

	/**
	 * Tell wheter or not the pipeline uses image downscaling.
	 *
	 * @return the flag telling wheter or not the pipeline uses image downscaling
	 */
	virtual bool hasImageDownscaling() const;

	/**
	 * Set the flag telling wheter or not the pipeline uses image downscaling
	 *
	 * @param pFlag the flag telling wheter or not the pipeline uses image downscaling
	 */
	virtual void setImageDownscaling( bool pFlag );

	/**
	 * Get the internal graphics buffer size
	 *
	 * @param pWidth the internal graphics buffer width
	 * @param pHeight the internal graphics buffer height
	 */
	virtual void getViewportSize( unsigned int& pWidth, unsigned int& pHeight ) const;

	/**
	 * Set the internal graphics buffer size
	 *
	 * @param pWidth the internal graphics buffer width
	 * @param pHeight the internal graphics buffer height
	 */
	virtual void setViewportSize( unsigned int pWidth, unsigned int pHeight );

	/**
	 * Get the internal graphics buffer size
	 *
	 * @param pWidth the internal graphics buffer width
	 * @param pHeight the internal graphics buffer height
	 */
	virtual void getGraphicsBufferSize( unsigned int& pWidth, unsigned int& pHeight ) const;

	/**
	 * Set the internal graphics buffer size
	 *
	 * @param pWidth the internal graphics buffer width
	 * @param pHeight the internal graphics buffer height
	 */
	virtual void setGraphicsBufferSize( unsigned int pWidth, unsigned int pHeight );

	/**
	 * Tell wheter or not pipeline uses programmable shaders
	 *
	 * @return a flag telling wheter or not pipeline uses programmable shaders
	 */
	virtual bool hasProgrammableShaders() const;

	/**
	 * Tell wheter or not pipeline has a given type of shader
	 *
	 * @param pShaderType the type of shader to test
	 *
	 * @return a flag telling wheter or not pipeline has a given type of shader
	 */
	virtual bool hasShaderType( unsigned int pShaderType ) const;

	/**
	 * Get the source code associated to a given type of shader
	 *
	 * @param pShaderType the type of shader
	 *
	 * @return the associated shader source code
	 */
	virtual std::string getShaderSourceCode( unsigned int pShaderType ) const;

	/**
	 * Get the filename associated to a given type of shader
	 *
	 * @param pShaderType the type of shader
	 *
	 * @return the associated shader filename
	 */
	virtual std::string getShaderFilename( unsigned int pShaderType ) const;

	/**
	 * ...
	 *
	 * @param pShaderType the type of shader
	 *
	 * @return ...
	 */
	virtual bool reloadShader( unsigned int pShaderType );

	/**
	 * Get the flag indicating wheter or not data production monitoring is activated
	 *
	 * @return the flag indicating wheter or not data production monitoring is activated
	 */
	virtual bool hasDataProductionMonitoring() const;

	/**
	 * Set the the flag indicating wheter or not data production monitoring is activated
	 *
	 * @param pFlag the flag indicating wheter or not data production monitoring is activated
	 */
	virtual void setDataProductionMonitoring( bool pFlag );

	/**
	 * Get the flag indicating wheter or not cache monitoring is activated
	 *
	 * @return the flag indicating wheter or not cache monitoring is activated
	 */
	virtual bool hasCacheMonitoring() const;

	/**
	 * Set the the flag indicating wheter or not cache monitoring is activated
	 *
	 * @param pFlag the flag indicating wheter or not cache monitoring is activated
	 */
	virtual void setCacheMonitoring( bool pFlag );

	/**
	 * Get the flag indicating wheter or not time budget monitoring is activated
	 *
	 * @return the flag indicating wheter or not time budget monitoring is activated
	 */
	virtual bool hasTimeBudgetMonitoring() const;

	/**
	 * Set the the flag indicating wheter or not time budget monitoring is activated
	 *
	 * @param pFlag the flag indicating wheter or not time budget monitoring is activated
	 */
	virtual void setTimeBudgetMonitoring( bool pFlag );

	/**
	 * Tell wheter or not time budget is acivated
	 *
	 * @return a flag to tell wheter or not time budget is activated
	 */
	virtual bool hasRenderingTimeBudget() const;

	/**
	 * Set the flag telling wheter or not time budget is acivated
	 *
	 * @param pFlag a flag to tell wheter or not time budget is activated
	 */
	virtual void setRenderingTimeBudgetActivated( bool pFlag );
	
	/**
	 * Get the user requested time budget
	 *
	 * @return the user requested time budget
	 */
	virtual unsigned int getRenderingTimeBudget() const;

	/**
	 * Set the user requested time budget
	 *
	 * @param pValue the user requested time budget
	 */
	virtual void setRenderingTimeBudget( unsigned int pValue );

	/**
	 * This method return the duration of the timer event between start and stop event
	 *
	 * @return the duration of the event in milliseconds
	 */
	virtual float getRendererElapsedTime() const;

	/**
	 * Get the flag telling wheter or not it has meshes
	 *
	 * @return the flag telling wheter or not it has meshes
	 */
	virtual bool hasMesh() const;

	/**
	 * Add a mesh
	 *
	 * @param pMesh a mesh
	 */
	virtual void addMesh( GvvMeshInterface* pMesh );

	/**
	 * Remove a mesh
	 *
	 * @param pMesh a mesh
	 */
	virtual void removeMesh( GvvMeshInterface* pMesh );

	/**
	 * Get the i-th mesh
	 *
	 * @param pIndex index of the mesh
	 *
	 * @return the i-th mesh
	 */
	virtual const GvvMeshInterface* getMesh( unsigned int pIndex ) const;
	
	/**
	 * Get the i-th mesh
	 *
	 * @param pIndex index of the mesh
	 *
	 * @return the i-th mesh
	 */
	virtual GvvMeshInterface* editMesh( unsigned int pIndex );

	/**
	 * Get the ModelView matrix
	 *
	 * return the ModelView matrix
	 */
	const float* getModelViewMatrix() const;

	/**
	 * Get the ModelView matrix
	 *
	 * return the ModelView matrix
	 */
	float* editModelViewMatrix();

	/**
	 * Get the Projection matrix
	 *
	 * return the Projection matrix
	 */
	const float* getProjectionMatrix() const;

	/**
	 * Get the Projection matrix
	 *
	 * return the Projection matrix
	 */
	float* editProjectionMatrix();

	/**
	 * Set or unset the flag used to tell whether or not the production time is limited.
	 *
	 * @param pLimit the flag value.
	 */
	virtual void useProductionTimeLimit( bool pLimit );

	/**
	 * Set the time limit for the production.
	 *
	 * @param pLimit the time limit (in ms).
	 */
	virtual void setProductionTimeLimit( float pLimit );

	/**
	 * Get the flag telling whether or not the production time limit is activated.
	 *
	 * @return the flag telling whether or not the production time limit is activated.
	 */
	virtual bool isProductionTimeLimited() const;

	/**
	 * Get the time limit actually in use.
	 * 
	 * @return the time limit.	
	 */
	virtual float getProductionTimeLimit() const;
				
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Data type list used to store voxels in the data structure
	 */
	GvvDataType _dataTypes;

	/**
	 * ModelView matrix
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	glm::mat4 _modelViewMatrix;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * Projection matrix
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	glm::mat4 _projectionMatrix;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} // namespace GvViewerCore

#endif // !GVVPIPELINEINTERFACE_H
