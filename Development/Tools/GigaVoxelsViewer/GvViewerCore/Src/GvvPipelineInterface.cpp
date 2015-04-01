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

#include "GvvPipelineInterface.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvMeshInterface.h"

// GLM
#include <glm/gtc/type_ptr.hpp>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;

// STL
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Tag name identifying a space profile element
 */
const char* GvvPipelineInterface::cTypeName = "Pipeline";

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvvPipelineInterface::GvvPipelineInterface()
:	GvvBrowsable()
,	_dataTypes()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvPipelineInterface::~GvvPipelineInterface()
{
}

/******************************************************************************
 * Returns the type of this browsable. The type is used for retrieving
 * the context menu or when requested or assigning an icon to the
 * corresponding item
 *
 * @return the type name of this browsable
 ******************************************************************************/
const char* GvvPipelineInterface::getTypeName() const
{
	return cTypeName;
}

/******************************************************************************
 * Gets the name of this browsable
 *
 * @return the name of this browsable
 ******************************************************************************/
const char* GvvPipelineInterface::getName() const
{
	return "GigaVoxelsPipeline";
}

/******************************************************************************
 * Initialize the GigaVoxels pipeline
 ******************************************************************************/
void GvvPipelineInterface::init()
{
}

/******************************************************************************
 * Draw function called every frame
 ******************************************************************************/
void GvvPipelineInterface::draw()
{
}

/******************************************************************************
 * Resize the frame
 *
 * @param width the new width
 * @param height the new height
 ******************************************************************************/
void GvvPipelineInterface::resize( int width, int height )
{
}

/******************************************************************************
 * Clear the GigaVoxels cache
 ******************************************************************************/
void GvvPipelineInterface::clearCache()
{
}

/******************************************************************************
 * Toggle the display of the N-tree (octree) of the data structure
 ******************************************************************************/
void GvvPipelineInterface::toggleDisplayOctree()
{
}

/******************************************************************************
 * Get the appearance of the N-tree (octree) of the data structure
 ******************************************************************************/
void GvvPipelineInterface::getDataStructureAppearance( bool& pShowNodeHasBrickTerminal, bool& pShowNodeHasBrickNotTerminal, bool& pShowNodeIsBrickNotInCache, bool& pShowNodeEmptyOrConstant
											, float& pNodeHasBrickTerminalColorR, float& pNodeHasBrickTerminalColorG, float& pNodeHasBrickTerminalColorB, float& pNodeHasBrickTerminalColorA
											, float& pNodeHasBrickNotTerminalColorR, float& pNodeHasBrickNotTerminalColorG, float& pNodeHasBrickNotTerminalColorB, float& pNodeHasBrickNotTerminalColorA
											, float& pNodeIsBrickNotInCacheColorR, float& pNodeIsBrickNotInCacheColorG, float& pNodeIsBrickNotInCacheColorB, float& pNodeIsBrickNotInCacheColorA
											, float& pNodeEmptyOrConstantColorR, float& pNodeEmptyOrConstantColorG, float& pNodeEmptyOrConstantColorB, float& pNodeEmptyOrConstantColorA ) const
{
}

/******************************************************************************
 * Set the appearance of the N-tree (octree) of the data structure
 ******************************************************************************/
void GvvPipelineInterface::setDataStructureAppearance( bool pShowNodeHasBrickTerminal, bool pShowNodeHasBrickNotTerminal, bool pShowNodeIsBrickNotInCache, bool pShowNodeEmptyOrConstant
											, float pNodeHasBrickTerminalColorR, float pNodeHasBrickTerminalColorG, float pNodeHasBrickTerminalColorB, float pNodeHasBrickTerminalColorA
											, float pNodeHasBrickNotTerminalColorR, float pNodeHasBrickNotTerminalColorG, float pNodeHasBrickNotTerminalColorB, float pNodeHasBrickNotTerminalColorA
											, float pNodeIsBrickNotInCacheColorR, float pNodeIsBrickNotInCacheColorG, float pNodeIsBrickNotInCacheColorB, float pNodeIsBrickNotInCacheColorA
											, float pNodeEmptyOrConstantColorR, float pNodeEmptyOrConstantColorG, float pNodeEmptyOrConstantColorB, float pNodeEmptyOrConstantColorA )
{
}

/******************************************************************************
 * Toggle the GigaVoxels dynamic update mode
 ******************************************************************************/
void GvvPipelineInterface::toggleDynamicUpdate()
{
}

/******************************************************************************
 * Get the dynamic update state
 *
 * @return the dynamic update state
 ******************************************************************************/
bool GvvPipelineInterface::hasDynamicUpdate() const
{
	return false;
}

/******************************************************************************
 * Set the dynamic update state
 *
 * @param pFlag the dynamic update state
 ******************************************************************************/
void GvvPipelineInterface::setDynamicUpdate( bool pFlag )
{
}

/******************************************************************************
 * Toggle the display of the performance monitor utility if
 * GigaVoxels has been compiled with the Performance Monitor option
 *
 * @param mode The performance monitor mode (1 for CPU, 2 for DEVICE)
 ******************************************************************************/
void GvvPipelineInterface::togglePerfmonDisplay( unsigned int mode )
{
}

/******************************************************************************
 * Increment the max resolution of the data structure
 ******************************************************************************/
void GvvPipelineInterface::incMaxVolTreeDepth()
{
}

/******************************************************************************
 * Decrement the max resolution of the data structure
 ******************************************************************************/
void GvvPipelineInterface::decMaxVolTreeDepth()
{
}

/******************************************************************************
 * Get the node tile resolution of the data structure.
 *
 * @param pX the X node tile resolution
 * @param pY the Y node tile resolution
 * @param pZ the Z node tile resolution
 ******************************************************************************/
void GvvPipelineInterface::getDataStructureNodeTileResolution( unsigned int& pX, unsigned int& pY, unsigned int& pZ ) const
{
}

/******************************************************************************
 * Get the brick resolution of the data structure (voxels).
 *
 * @param pX the X brick resolution
 * @param pY the Y brick resolution
 * @param pZ the Z brick resolution
 ******************************************************************************/
void GvvPipelineInterface::getDataStructureBrickResolution( unsigned int& pX, unsigned int& pY, unsigned int& pZ ) const
{
}

/******************************************************************************
 * Get the max depth.
 *
 * @return the max depth
 ******************************************************************************/
unsigned int GvvPipelineInterface::getRendererMaxDepth() const
{
	return 0;
}

/******************************************************************************
 * Set the max depth.
 *
 * @param pValue the max depth
 ******************************************************************************/
void GvvPipelineInterface::setRendererMaxDepth( unsigned int pValue )
{
}

/******************************************************************************
 * Get the max number of requests of node subdivisions.
 *
 * @return the max number of requests
 ******************************************************************************/
unsigned int GvvPipelineInterface::getCacheMaxNbNodeSubdivisions() const
{
	return 0;
}

/******************************************************************************
 * Set the max number of requests of node subdivisions.
 *
 * @param pValue the max number of requests
 ******************************************************************************/
void GvvPipelineInterface::setCacheMaxNbNodeSubdivisions( unsigned int pValue )
{
}

/******************************************************************************
 * Get the max number of requests of brick of voxel loads.
 *
 * @return the max number of requests
 ******************************************************************************/
unsigned int GvvPipelineInterface::getCacheMaxNbBrickLoads() const
{
	return 0;
}

/******************************************************************************
 * Set the max number of requests of brick of voxel loads.
 *
 * @param pValue the max number of requests
 ******************************************************************************/
void GvvPipelineInterface::setCacheMaxNbBrickLoads( unsigned int pValue )
{
}

/******************************************************************************
 * Get the request strategy indicating if, during data structure traversal,
 * priority of requests is set on brick loads or on node subdivisions first.
 *
 * @return the flag indicating the request strategy
 ******************************************************************************/
bool GvvPipelineInterface::hasRendererPriorityOnBricks() const
{
	return true;
}

/******************************************************************************
 * Set the request strategy indicating if, during data structure traversal,
 * priority of requests is set on brick loads or on node subdivisions first.
 *
 * @param pFlag the flag indicating the request strategy
 ******************************************************************************/
void GvvPipelineInterface::setRendererPriorityOnBricks( bool pFlag )
{
}

/******************************************************************************
 * Tell wheter or not the pipeline has a transfer function.
 *
 * @return the flag telling wheter or not the pipeline has a transfer function
 ******************************************************************************/
bool GvvPipelineInterface::hasTransferFunction() const
{
	return false;
}

/******************************************************************************
 * Get the transfer function if it has one.
 *
 * @param pIndex the index of the transfer function
 *
 * @return the transfer function
 ******************************************************************************/
GvvTransferFunctionInterface* GvvPipelineInterface::getTransferFunction( unsigned int pIndex ) const
{
	return NULL;
}

/******************************************************************************
 * Get the transfer function if it has one.
 *
 * @param pIndex the index of the transfer function
 *
 * @return the transfer function
 ******************************************************************************/
GvvTransferFunctionInterface* GvvPipelineInterface::editTransferFunction( unsigned int pIndex )
{
	return NULL;
}

/******************************************************************************
 * Get the transfer function filename if it has one.
 *
 * @param pIndex the index of the transfer function
 *
 * @return the transfer function
 ******************************************************************************/
const char* GvvPipelineInterface::getTransferFunctionFilename( unsigned int pIndex ) const
{
	return NULL;
}

/******************************************************************************
 * Set the transfer function filename  if it has one.
 *
 * @param pFilename the transfer function's filename
 * @param pIndex the index of the transfer function
 *
 * @return the transfer function
 ******************************************************************************/
void GvvPipelineInterface::setTransferFunctionFilename( const char* pFilename, unsigned int pIndex )
{
}

/******************************************************************************
 * Update the associated transfer function
 *
 * @param the new transfer function data
 * @param the size of the transfer function
 ******************************************************************************/
void GvvPipelineInterface::updateTransferFunction( float* pData, unsigned int pSize )
{
}

/******************************************************************************
 * Tell wheter or not the pipeline has a light.
 *
 * @return the flag telling wheter or not the pipeline has a light
 ******************************************************************************/
bool GvvPipelineInterface::hasLight() const
{
	return false;
}

/******************************************************************************
 * Get the light position
 *
 * @param pX the X light position
 * @param pY the Y light position
 * @param pZ the Z light position
 ******************************************************************************/
void GvvPipelineInterface::getLightPosition( float& pX, float& pY, float& pZ ) const
{
}

/******************************************************************************
 * Set the light position
 *
 * @param pX the X light position
 * @param pY the Y light position
 * @param pZ the Z light position
 ******************************************************************************/
void GvvPipelineInterface::setLightPosition( float pX, float pY, float pZ )
{
}

/******************************************************************************
 * Tell wheter or not the pipeline has a light.
 *
 * @return the flag telling wheter or not the pipeline has a light
 ******************************************************************************/
bool GvvPipelineInterface::has3DModel() const
{
	return false;
}

/******************************************************************************
 * Get the 3D model filename to load
 *
 * @return the 3D model filename to load
 ******************************************************************************/
string GvvPipelineInterface::get3DModelFilename() const
{
	return string();
}

/******************************************************************************
 * Set the 3D model filename to load
 *
 * @param pFilename the 3D model filename to load
 ******************************************************************************/
void GvvPipelineInterface::set3DModelFilename( const string& pFilename )
{
}

/******************************************************************************
 * Specify color to clear the color buffer
 *
 * @param pRed red component
 * @param pGreen green component
 * @param pBlue blue component
 * @param pAlpha alpha component
 ******************************************************************************/
void GvvPipelineInterface::setClearColor( unsigned char pRed, unsigned char pGreen, unsigned char pBlue, unsigned char pAlpha )
{
}

/******************************************************************************
 * Get the data type list used to store voxels in the data structure
 *
 * @return the data type list of voxels
 ******************************************************************************/
const GvvDataType& GvvPipelineInterface::getDataTypes() const
{
	return _dataTypes;
}

/******************************************************************************
 * Get the data type list used to store voxels in the data structure
 *
 * @return the data type list of voxels
 ******************************************************************************/
GvvDataType& GvvPipelineInterface::editDataTypes()
{
	return _dataTypes;
}

/******************************************************************************
 * Get the translation
 *
 * @param pX the translation on x axis
 * @param pY the translation on y axis
 * @param pZ the translation on z axis
 ******************************************************************************/
void GvvPipelineInterface::getTranslation( float& pX, float& pY, float& pZ ) const
{
}

/******************************************************************************
 * Set the translation
 *
 * @param pX the translation on x axis
 * @param pY the translation on y axis
 * @param pZ the translation on z axis
 ******************************************************************************/
void GvvPipelineInterface::setTranslation( float pX, float pY, float pZ )
{
}

/******************************************************************************
 * Get the rotation
 *
 * @param pAngle the rotation angle (in degree)
 * @param pX the x component of the rotation vector
 * @param pY the y component of the rotation vector
 * @param pZ the z component of the rotation vector
 ******************************************************************************/
void GvvPipelineInterface::getRotation( float& pAngle, float& pX, float& pY, float& pZ ) const
{
}

/******************************************************************************
 * Set the rotation
 *
 * @param pAngle the rotation angle (in degree)
 * @param pX the x component of the rotation vector
 * @param pY the y component of the rotation vector
 * @param pZ the z component of the rotation vector
 ******************************************************************************/
void GvvPipelineInterface::setRotation( float pAngle, float pX, float pY, float pZ )
{
}

/******************************************************************************
 * Get the uniform scale
 *
 * @param pValue the uniform scale
 ******************************************************************************/
void GvvPipelineInterface::getScale( float& pValue ) const
{
}

/******************************************************************************
 * Set the uniform scale
 *
 * @param pValue the uniform scale
 ******************************************************************************/
void GvvPipelineInterface::setScale( float pValue )
{
}

/******************************************************************************
 * Get the number of requests of node subdivisions the cache has handled.
 *
 * @return the number of requests
 ******************************************************************************/
unsigned int GvvPipelineInterface::getCacheNbNodeSubdivisionRequests() const
{
	return 0;
}

/******************************************************************************
 * Get the number of requests of brick of voxel loads the cache has handled.
 *
 * @return the number of requests
 ******************************************************************************/
unsigned int GvvPipelineInterface::getCacheNbBrickLoadRequests() const
{
	return 0;
}

/******************************************************************************
 * Get the cache policy
 *
 * @return the cache policy
 ******************************************************************************/
unsigned int GvvPipelineInterface::getCachePolicy() const
{
	return 0;
}

/******************************************************************************
 * Set the cache policy
 *
 * @param pValue the cache policy
 ******************************************************************************/
void GvvPipelineInterface::setCachePolicy( unsigned int pValue )
{
}

/******************************************************************************
	 * Get the node cache memory
	 *
	 * @return the node cache memory
 ******************************************************************************/
unsigned int GvvPipelineInterface::getNodeCacheMemory() const
{
	return 0;
}

/******************************************************************************
	 * Set the node cache memory
	 *
	 * @param pValue the node cache memory
 ******************************************************************************/
void GvvPipelineInterface::setNodeCacheMemory( unsigned int pValue )
{
}

/******************************************************************************
	 * Get the brick cache memory
	 *
	 * @return the brick cache memory
 ******************************************************************************/
unsigned int GvvPipelineInterface::getBrickCacheMemory() const
{
	return 0;
}

/******************************************************************************
	 * Set the brick cache memory
	 *
	 * @param pValue the brick cache memory
 ******************************************************************************/
void GvvPipelineInterface::setBrickCacheMemory( unsigned int pValue )
{
}

/******************************************************************************
 * Get the node cache capacity
 *
 * @return the node cache capacity
 ******************************************************************************/
unsigned int GvvPipelineInterface::getNodeCacheCapacity() const
{
	return 0;
}

/******************************************************************************
 * Set the node cache capacity
 *
 * @param pValue the node cache capacity
 ******************************************************************************/
void GvvPipelineInterface::setNodeCacheCapacity( unsigned int pValue )
{
}

/******************************************************************************
 * Get the brick cache capacity
 *
 * @return the brick cache capacity
 ******************************************************************************/
unsigned int GvvPipelineInterface::getBrickCacheCapacity() const
{
	return 0;
}

/******************************************************************************
 * Set the brick cache capacity
 *
 * @param pValue the brick cache capacity
 ******************************************************************************/
void GvvPipelineInterface::setBrickCacheCapacity( unsigned int pValue )
{
}

/******************************************************************************
 * Get the number of unused nodes in cache
 *
 * @return the number of unused nodes in cache
 ******************************************************************************/
unsigned int GvvPipelineInterface::getCacheNbUnusedNodes() const
{
	return 0;
}

/******************************************************************************
 * Get the number of unused bricks in cache
 *
 * @return the number of unused bricks in cache
 ******************************************************************************/
unsigned int GvvPipelineInterface::getCacheNbUnusedBricks() const
{
	return 0;
}

/******************************************************************************
 * Get the nodes cache usage
 *
 * @return the nodes cache usage
 ******************************************************************************/
unsigned int GvvPipelineInterface::getNodeCacheUsage() const
{
	return 0;
}

/******************************************************************************
 * Get the bricks cache usage
 *
 * @return the bricks cache usage
 ******************************************************************************/
unsigned int GvvPipelineInterface::getBrickCacheUsage() const
{
	return 0;
}

/******************************************************************************
 * Get the flag telling wheter or not tree data dtructure monitoring is activated
 *
 * @return the flag telling wheter or not tree data dtructure monitoring is activated
 ******************************************************************************/
bool GvvPipelineInterface::hasTreeDataStructureMonitoring() const
{
	return false;
}

/******************************************************************************
 * Set the flag telling wheter or not tree data dtructure monitoring is activated
 *
 * @param pFlag the flag telling wheter or not tree data dtructure monitoring is activated
 ******************************************************************************/
void GvvPipelineInterface::setTreeDataStructureMonitoring( bool pFlag )
{
}

/******************************************************************************
 * Get the number of tree leaf nodes
 *
 * @return the number of tree leaf nodes
 ******************************************************************************/
unsigned int GvvPipelineInterface::getNbTreeLeafNodes() const
{
	return 0;
}

/******************************************************************************
 * Get the number of tree leaf nodes
 *
 * @return the number of tree leaf nodes
 ******************************************************************************/
unsigned int GvvPipelineInterface::getNbTreeNodes() const
{
	return 0;
}

/******************************************************************************
 * Tell wheter or not the pipeline uses image downscaling.
 *
 * @return the flag telling wheter or not the pipeline uses image downscaling
 ******************************************************************************/
bool GvvPipelineInterface::hasImageDownscaling() const
{
	return false;
}

/******************************************************************************
 * Set the flag telling wheter or not the pipeline uses image downscaling
 *
 * @param pFlag the flag telling wheter or not the pipeline uses image downscaling
 ******************************************************************************/
void GvvPipelineInterface::setImageDownscaling( bool pFlag )
{
}

/******************************************************************************
 * Get the internal graphics buffer size
 *
 * @param pWidth the internal graphics buffer width
 * @param pHeight the internal graphics buffer height
 ******************************************************************************/
void GvvPipelineInterface::getViewportSize( unsigned int& pWidth, unsigned int& pHeight ) const
{
}

/******************************************************************************
 * Set the internal graphics buffer size
 *
 * @param pWidth the internal graphics buffer width
 * @param pHeight the internal graphics buffer height
 ******************************************************************************/
void GvvPipelineInterface::setViewportSize( unsigned int pWidth, unsigned int pHeight )
{
}

/******************************************************************************
 * Get the internal graphics buffer size
 *
 * @param pWidth the internal graphics buffer width
 * @param pHeight the internal graphics buffer height
 ******************************************************************************/
void GvvPipelineInterface::getGraphicsBufferSize( unsigned int& pWidth, unsigned int& pHeight ) const
{
}

/******************************************************************************
 * Set the internal graphics buffer size
 *
 * @param pWidth the internal graphics buffer width
 * @param pHeight the internal graphics buffer height
 ******************************************************************************/
void GvvPipelineInterface::setGraphicsBufferSize( unsigned int pWidth, unsigned int pHeight )
{
}

/******************************************************************************
 * Tell wheter or not pipeline uses programmable shaders
 *
 * @return a flag telling wheter or not pipeline uses programmable shaders
 ******************************************************************************/
bool GvvPipelineInterface::hasProgrammableShaders() const
{
	return false;
}

/******************************************************************************
 * Tell wheter or not pipeline has a given type of shader
 *
 * @param pShaderType the type of shader to test
 *
 * @return a flag telling wheter or not pipeline has a given type of shader
 ******************************************************************************/
bool GvvPipelineInterface::hasShaderType( unsigned int pShaderType ) const
{
	return false;
}

/******************************************************************************
 * Get the source code associated to a given type of shader
 *
 * @param pShaderType the type of shader
 *
 * @return the associated shader source code
 ******************************************************************************/
std::string GvvPipelineInterface::getShaderSourceCode( unsigned int pShaderType ) const
{
	std::string shaderSourceCode( "" );

	return shaderSourceCode;
}

/******************************************************************************
 * Get the filename associated to a given type of shader
 *
 * @param pShaderType the type of shader
 *
 * @return the associated shader filename
 ******************************************************************************/
std::string GvvPipelineInterface::getShaderFilename( unsigned int pShaderType ) const
{
	std::string shaderFilename( "" );

	return shaderFilename;
}

/******************************************************************************
 * ...
 *
 * @param pShaderType the type of shader
 *
 * @return ...
 ******************************************************************************/
bool GvvPipelineInterface::reloadShader( unsigned int pShaderType )
{
	return false;
}

/******************************************************************************
* Get the flag indicating wheter or not data production monitoring is activated
*
* @return the flag indicating wheter or not data production monitoring is activated
 ******************************************************************************/
bool GvvPipelineInterface::hasDataProductionMonitoring() const
{
	return false;
}

/******************************************************************************
* Set the the flag indicating wheter or not data production monitoring is activated
*
* @param pFlag the flag indicating wheter or not data production monitoring is activated
 ******************************************************************************/
void GvvPipelineInterface::setDataProductionMonitoring( bool pFlag )
{
}

/******************************************************************************
* Get the flag indicating wheter or not cache monitoring is activated
*
* @return the flag indicating wheter or not cache monitoring is activated
 ******************************************************************************/
bool GvvPipelineInterface::hasCacheMonitoring() const
{
	return false;
}

/******************************************************************************
* Set the the flag indicating wheter or not cache monitoring is activated
*
* @param pFlag the flag indicating wheter or not cache monitoring is activated
 ******************************************************************************/
void GvvPipelineInterface::setCacheMonitoring( bool pFlag )
{
}

/******************************************************************************
* Get the flag indicating wheter or not time budget monitoring is activated
*
* @return the flag indicating wheter or not time budget monitoring is activated
 ******************************************************************************/
bool GvvPipelineInterface::hasTimeBudgetMonitoring() const
{
	return false;
}

/******************************************************************************
* Set the the flag indicating wheter or not time budget monitoring is activated
*
* @param pFlag the flag indicating wheter or not time budget monitoring is activated
 ******************************************************************************/
void GvvPipelineInterface::setTimeBudgetMonitoring( bool pFlag )
{
}

/******************************************************************************
 *Tell wheter or not time budget is acivated
 *
 * @return a flag to tell wheter or not time budget is activated
 ******************************************************************************/
bool GvvPipelineInterface::hasRenderingTimeBudget() const
{
	return false;
}

/******************************************************************************
 * Set the flag telling wheter or not time budget is acivated
 *
 * @param pFlag a flag to tell wheter or not time budget is activated
 ******************************************************************************/
void GvvPipelineInterface::setRenderingTimeBudgetActivated( bool pFlag )
{
}

/******************************************************************************
 * Get the user requested time budget
 *
 * @return the user requested time budget
 ******************************************************************************/
unsigned int GvvPipelineInterface::getRenderingTimeBudget() const
{
	return false;
}

/******************************************************************************
 * Set the user requested time budget
 *
 * @param pValue the user requested time budget
 ******************************************************************************/
void GvvPipelineInterface::setRenderingTimeBudget( unsigned int pValue )
{
}

/******************************************************************************
 * This method return the duration of the timer event between start and stop event
 *
 * @return the duration of the event in milliseconds
 ******************************************************************************/
float GvvPipelineInterface::getRendererElapsedTime() const
{
	return 0.f;
}

/******************************************************************************
 * Get the flag telling wheter or not it has meshes
 *
 * @return the flag telling wheter or not it has meshes
 ******************************************************************************/
bool GvvPipelineInterface::hasMesh() const
{
	return false;
}

/******************************************************************************
 * Add a mesh
 *
 * @param pMesh a mesh
 ******************************************************************************/
void GvvPipelineInterface::addMesh( GvvMeshInterface* pMesh )
{
}

/******************************************************************************
 * Remove a mesh
 *
 * @param pMesh a mesh
 ******************************************************************************/
void GvvPipelineInterface::removeMesh( GvvMeshInterface* pMesh )
{
}

/******************************************************************************
 * Get the i-th mesh
 *
 * @param pIndex index of the mesh
 *
 * @return the i-th mesh
 ******************************************************************************/
const GvvMeshInterface* GvvPipelineInterface::getMesh( unsigned int pIndex ) const
{
	return NULL;
}

/******************************************************************************
 * Get the i-th mesh
 *
 * @param pIndex index of the mesh
 *
 * @return the i-th mesh
 ******************************************************************************/
GvvMeshInterface* GvvPipelineInterface::editMesh( unsigned int pIndex )
{
	return NULL;
}

/******************************************************************************
 * Get the ModelView matrix
 *
 * return the ModelView matrix
 ******************************************************************************/
const float* GvvPipelineInterface::getModelViewMatrix() const
{
	return glm::value_ptr( _modelViewMatrix );
}

/******************************************************************************
 * Get the ModelView matrix
 *
 * return the ModelView matrix
 ******************************************************************************/
float* GvvPipelineInterface::editModelViewMatrix()
{
	return glm::value_ptr( _modelViewMatrix );
}

/******************************************************************************
 * Get the Projection matrix
 *
 * return the Projection matrix
 ******************************************************************************/
const float* GvvPipelineInterface::getProjectionMatrix() const
{
	return glm::value_ptr( _projectionMatrix );
}

/******************************************************************************
 * Get the Projection matrix
 *
 * return the Projection matrix
 ******************************************************************************/
float* GvvPipelineInterface::editProjectionMatrix()
{
	return glm::value_ptr( _projectionMatrix );
}

/******************************************************************************
 * Set or unset the flag used to tell whether or not the production time is limited.
 *
 * @param pLimit the flag value.
 ******************************************************************************/
void GvvPipelineInterface::useProductionTimeLimit( bool pLimit )
{
}

/******************************************************************************
 * Set the time limit for the production.
 *
 * @param pLimit the time limit (in ms).
 ******************************************************************************/
void GvvPipelineInterface::setProductionTimeLimit( float pLimit )
{
}

/******************************************************************************
 * Get the flag telling whether or not the production time limit is activated.
 *
 * @return the flag telling whether or not the production time limit is activated.
 ******************************************************************************/
bool GvvPipelineInterface::isProductionTimeLimited() const
{
	return false;
}

/******************************************************************************
 * Get the time limit actually in use.
 *
 * @return the time limit.
 ******************************************************************************/
float GvvPipelineInterface::getProductionTimeLimit() const
{
	return 0.f;
}
