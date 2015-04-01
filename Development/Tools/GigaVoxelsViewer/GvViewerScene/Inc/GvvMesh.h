/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2014 INRIA <http://www.inria.fr/>
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

#ifndef _GVV_MESH_H_
#define _GVV_MESH_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvSceneConfig.h"
#include "GvvMeshInterface.h"

// STL
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
	class GvvProgrammableShaderInterface;
}
namespace GvViewerScene
{
	class GvvGraphicsObject;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerScene
{

/** 
 * @class GvvMesh
 *
 * @brief The GvvMesh class provides info on a device.
 *
 * ...
 */
class GVVIEWERSCENE_EXPORT GvvMesh : public GvViewerCore::GvvMeshInterface
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
	GvvMesh();

	/**
	 * Destructor
	 */
	virtual ~GvvMesh();

	/**
	 * Get the flag telling wheter or not it has programmable shaders
	 *
	 * @return the flag telling wheter or not it has programmable shaders
	 */
	virtual bool hasProgrammableShader() const;

	/**
	 * Add a programmable shader
	 */
	virtual void addProgrammableShader( GvViewerCore::GvvProgrammableShaderInterface* pShader );

	/**
	 * Get the associated programmable shader
	 *
	 * @param pIndex shader index
	 *
	 * @return the associated programmable shader
	 */
	virtual void removeProgrammableShader( GvViewerCore::GvvProgrammableShaderInterface* pShader );

	/**
	 * Get the associated programmable shader
	 *
	 * @param pIndex shader index
	 *
	 * @return the associated programmable shader
	 */
	virtual const GvViewerCore::GvvProgrammableShaderInterface* getProgrammableShader( unsigned int pIndex ) const;

	/**
	 * Get the associated programmable shader
	 *
	 * @param pIndex shader index
	 *
	 * @return the associated programmable shader
	 */
	virtual GvViewerCore::GvvProgrammableShaderInterface* editProgrammableShader( unsigned int pIndex );

	/**
	 * Load 3D object/scene
	 *
	 * @param pFilename filename
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool load( const char* pFilename );

	/**
	 * This function is the specific implementation method called
	 * by the parent GvIRenderer::render() method during rendering.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 * @param pViewport the viewport configuration
	 */
	virtual void render( const glm::mat4& pModelViewMatrix, const glm::mat4& pProjectionMatrix, const glm::uvec4& pViewport );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Graphics object
	 */
	GvViewerScene::GvvGraphicsObject* _graphicsObject;

	/**
	 * Programmable shader
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::vector< GvViewerCore::GvvProgrammableShaderInterface* > _programmableShaders;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvvMesh( const GvvMesh& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvMesh& operator=( const GvvMesh& );

};

} // namespace GvViewerScene

#endif // !_GVV_MESH_H_
