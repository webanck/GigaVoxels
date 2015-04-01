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

#include "GvvMeshInterface.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvProgrammableShaderInterface.h"

// STL
#include <algorithm>

// System
#include <cassert>

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
const char* GvvMeshInterface::cTypeName = "Mesh";

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvvMeshInterface::GvvMeshInterface()
:	GvvBrowsable()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvMeshInterface::~GvvMeshInterface()
{
}

/******************************************************************************
 * Returns the type of this browsable. The type is used for retrieving
 * the context menu or when requested or assigning an icon to the
 * corresponding item
 *
 * @return the type name of this browsable
 ******************************************************************************/
const char* GvvMeshInterface::getTypeName() const
{
	return cTypeName;
}

/******************************************************************************
 * Gets the name of this browsable
 *
 * @return the name of this browsable
 ******************************************************************************/
const char* GvvMeshInterface::getName() const
{
	return "Mesh";
}

/******************************************************************************
 * Get the flag telling wheter or not it has programmable shaders
 *
 * @return the flag telling wheter or not it has programmable shaders
 ******************************************************************************/
bool GvvMeshInterface::hasProgrammableShader() const
{
	return false;
}

/******************************************************************************
 * Add a programmable shader
 ******************************************************************************/
void GvvMeshInterface::addProgrammableShader( GvvProgrammableShaderInterface* pShader )
{
}

/******************************************************************************
 * Get the associated programmable shader
 *
 * @param pIndex shader index
 *
 * @return the associated programmable shader
 ******************************************************************************/
void GvvMeshInterface::removeProgrammableShader( GvvProgrammableShaderInterface* pShader )
{
}

/******************************************************************************
 * Get the associated programmable shader
 *
 * @param pIndex shader index
 *
 * @return the associated programmable shader
 ******************************************************************************/
const GvvProgrammableShaderInterface* GvvMeshInterface::getProgrammableShader( unsigned int pIndex ) const
{
	return NULL;
}

/******************************************************************************
 * Get the associated programmable shader
 *
 * @param pIndex shader index
 *
 * @return the associated programmable shader
 ******************************************************************************/
GvvProgrammableShaderInterface* GvvMeshInterface::editProgrammableShader( unsigned int pIndex )
{
	return NULL;
}
