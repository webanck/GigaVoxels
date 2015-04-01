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

#include "GvvPipeline.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvMeshInterface.h"

// STL
#include <algorithm>

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerScene;

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
GvvPipeline::GvvPipeline()
:	GvvPipelineInterface()
,	_meshes()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvPipeline::~GvvPipeline()
{
}

/******************************************************************************
 * Get the flag telling wheter or not it has meshes
 *
 * @return the flag telling wheter or not it has meshes
 ******************************************************************************/
bool GvvPipeline::hasMesh() const
{
	return ( _meshes.size() > 0 );
}

/******************************************************************************
 * Add a mesh
 *
 * @param pMesh a mesh
 ******************************************************************************/
void GvvPipeline::addMesh( GvvMeshInterface* pMesh )
{
	// TO DO : check in already there ?

	_meshes.push_back( pMesh );
}

/******************************************************************************
 * Remove a mesh
 *
 * @param pMesh a mesh
 ******************************************************************************/
void GvvPipeline::removeMesh( GvvMeshInterface* pMesh )
{
	vector< GvvMeshInterface* >::iterator itMesh;
	itMesh = find( _meshes.begin(), _meshes.end(), pMesh );
	if ( itMesh != _meshes.end() )
	{
		// Remove pipeline
		_meshes.erase( itMesh );
	}
}

/******************************************************************************
 * Get the i-th mesh
 *
 * @param pIndex index of the mesh
 *
 * @return the i-th mesh
 ******************************************************************************/
const GvvMeshInterface* GvvPipeline::getMesh( unsigned int pIndex ) const
{
	assert( pIndex < _meshes.size() );
	if ( pIndex < _meshes.size() )
	{
		return _meshes[ pIndex ];
	}

	return NULL;
}
	
/******************************************************************************
 * Get the i-th mesh
 *
 * @param pIndex index of the mesh
 *
 * @return the i-th mesh
 ******************************************************************************/
GvvMeshInterface* GvvPipeline::editMesh( unsigned int pIndex )
{
	return const_cast< GvvMeshInterface* >( getMesh( pIndex ) );
}
