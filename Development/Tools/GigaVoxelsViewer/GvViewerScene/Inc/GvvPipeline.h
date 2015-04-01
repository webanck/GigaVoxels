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

#ifndef _GVV_PIPELINE_H_
#define _GVV_PIPELINE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvSceneConfig.h"
#include "GvvPipelineInterface.h"

// STL
#include <vector>

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
	class GvvMeshInterface;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerScene
{

/** 
 * @class GvvPipeline
 *
 * @brief The GvvPipeline class provides info on a device.
 *
 * ...
 */
class GVVIEWERSCENE_EXPORT GvvPipeline : public GvViewerCore::GvvPipelineInterface
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
	GvvPipeline();

	/**
	 * Destructor
	 */
	virtual ~GvvPipeline();

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
	virtual void addMesh( GvViewerCore::GvvMeshInterface* pMesh );

	/**
	 * Remove a mesh
	 *
	 * @param pMesh a mesh
	 */
	virtual void removeMesh( GvViewerCore::GvvMeshInterface* pMesh );

	/**
	 * Get the i-th mesh
	 *
	 * @param pIndex index of the mesh
	 *
	 * @return the i-th mesh
	 */
	virtual const GvViewerCore::GvvMeshInterface* getMesh( unsigned int pIndex ) const;
	
	/**
	 * Get the i-th mesh
	 *
	 * @param pIndex index of the mesh
	 *
	 * @return the i-th mesh
	 */
	virtual GvViewerCore::GvvMeshInterface* editMesh( unsigned int pIndex );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Programmable shader
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::vector< GvViewerCore::GvvMeshInterface* > _meshes;
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
	GvvPipeline( const GvvPipeline& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvPipeline& operator=( const GvvPipeline& );

};

} // namespace GvViewerScene

#endif // !_GVV_PIPELINE_H_
