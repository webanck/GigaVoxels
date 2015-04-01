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

#include "GvvMesh.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvProgrammableShaderInterface.h"
#include "GvvGraphicsObject.h"

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
GvvMesh::GvvMesh()
:	GvvMeshInterface()
,	_graphicsObject( NULL )
,	_programmableShaders()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvMesh::~GvvMesh()
{
	// TO DO

	// Release resources
	delete _graphicsObject;
	_graphicsObject = NULL;
}

/******************************************************************************
 * Get the flag telling wheter or not it has programmable shaders
 *
 * @return the flag telling wheter or not it has programmable shaders
 ******************************************************************************/
bool GvvMesh::hasProgrammableShader() const
{
	return ( _programmableShaders.size() > 0 );
}

/******************************************************************************
 * Add a programmable shader
 ******************************************************************************/
void GvvMesh::addProgrammableShader( GvvProgrammableShaderInterface* pShader )
{
	// TO DO : check in already there ?

	_programmableShaders.push_back( pShader );
}

/******************************************************************************
 * Get the associated programmable shader
 *
 * @param pIndex shader index
 *
 * @return the associated programmable shader
 ******************************************************************************/
void GvvMesh::removeProgrammableShader( GvvProgrammableShaderInterface* pShader )
{
	vector< GvvProgrammableShaderInterface* >::iterator itShader;
	itShader = find( _programmableShaders.begin(), _programmableShaders.end(), pShader );
	if ( itShader != _programmableShaders.end() )
	{
		// Remove pipeline
		_programmableShaders.erase( itShader );
	}
}

/******************************************************************************
 * Get the associated programmable shader
 *
 * @param pIndex shader index
 *
 * @return the associated programmable shader
 ******************************************************************************/
const GvvProgrammableShaderInterface* GvvMesh::getProgrammableShader( unsigned int pIndex ) const
{
	assert( pIndex < _programmableShaders.size() );
	if ( pIndex < _programmableShaders.size() )
	{
		return _programmableShaders[ pIndex ];
	}

	return NULL;
}

/******************************************************************************
 * Get the associated programmable shader
 *
 * @param pIndex shader index
 *
 * @return the associated programmable shader
 ******************************************************************************/
GvvProgrammableShaderInterface* GvvMesh::editProgrammableShader( unsigned int pIndex )
{
	assert( pIndex < _programmableShaders.size() );
	if ( pIndex < _programmableShaders.size() )
	{
		return _programmableShaders[ pIndex ];
	}

	return NULL;
}

/******************************************************************************
 * Load 3D object/scene
 *
 * @param pFilename filename
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
bool GvvMesh::load( const char* pFilename )
{
	assert( pFilename != NULL );

	// TODO : use ResourceManager::get()

	// Release resources
	delete _graphicsObject;
	_graphicsObject = NULL;

	// Create graphics object
	_graphicsObject = new GvvGraphicsObject();
	_graphicsObject->initialize();
	_graphicsObject->load( pFilename );

	return false;
}

/******************************************************************************
 * This function is the specific implementation method called
 * by the parent GvIRenderer::render() method during rendering.
 *
 * @param pModelMatrix the current model matrix
 * @param pViewMatrix the current view matrix
 * @param pProjectionMatrix the current projection matrix
 * @param pViewport the viewport configuration
 ******************************************************************************/
void GvvMesh::render( const glm::mat4& pModelViewMatrix, const glm::mat4& pProjectionMatrix, const glm::uvec4& pViewport )
{
	_graphicsObject->render( pModelViewMatrix, pProjectionMatrix, pViewport );
}
