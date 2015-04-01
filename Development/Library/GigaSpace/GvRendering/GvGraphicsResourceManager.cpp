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

#include "GvRendering/GvGraphicsResourceManager.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvRendering/GvGraphicsResource.h"

// CUDA toolkit
#include <cuda_runtime.h>

// System
#include <cassert>
#include <iostream>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// Gigavoxels
using namespace GvRendering;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * The unique device manager
 */
GvGraphicsResourceManager* GvGraphicsResourceManager::msInstance = NULL;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Get the device manager.
 *
 * @return the device manager
 ******************************************************************************/
GvGraphicsResourceManager& GvGraphicsResourceManager::get()
{
	if ( msInstance == NULL )
	{
		msInstance = new GvGraphicsResourceManager();
	}
	assert( msInstance != NULL );
	return *msInstance;
}

/******************************************************************************
 * Constructor.
 ******************************************************************************/
GvGraphicsResourceManager::GvGraphicsResourceManager()
:	_graphicsResources()
,	_isInitialized( false )
{
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvGraphicsResourceManager::~GvGraphicsResourceManager()
{
	finalize();
}

/******************************************************************************
 * Initialize the device manager
 ******************************************************************************/
bool GvGraphicsResourceManager::initialize()
{
	return false;
}

/******************************************************************************
 * Finalize the device manager
 ******************************************************************************/
void GvGraphicsResourceManager::finalize()
{
}

/******************************************************************************
 * Get the number of devices
 *
 * @return the number of devices
 ******************************************************************************/
size_t GvGraphicsResourceManager::getNbResources() const
{
	return _graphicsResources.size();
}

/******************************************************************************
 * Get the device given its index
 *
 * @param the index of the requested device
 *
 * @return the requested device
 ******************************************************************************/
const GvGraphicsResource* GvGraphicsResourceManager::getResource( int pIndex ) const
{
	assert( pIndex < _graphicsResources.size() );
	return _graphicsResources[ pIndex ];
}

/******************************************************************************
 * Get the device given its index
 *
 * @param the index of the requested device
 *
 * @return the requested device
 ******************************************************************************/
GvGraphicsResource* GvGraphicsResourceManager::editResource( int pIndex )
{
	assert( pIndex < _graphicsResources.size() );
	return _graphicsResources[ pIndex ];
}
