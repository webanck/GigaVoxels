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

#include "GvvContextManager.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvBrowsable.h"

// STL
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerGui;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * The unique instance of the context
 */
GvvContextManager* GvvContextManager::msInstance = NULL;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Returns the context
 ******************************************************************************/
GvvContextManager* GvvContextManager::get()
{
	if ( msInstance == NULL )
	{
		msInstance = new GvvContextManager();
	}
	assert( msInstance != NULL );
	return msInstance;
}

/******************************************************************************
 * Default constructor.
 ******************************************************************************/
GvvContextManager::GvvContextManager()
:	QObject()
,	_currentBrowsable( NULL )
{
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvContextManager::~GvvContextManager()
{
}

/******************************************************************************
 * Sets the current browsable
 ******************************************************************************/
void GvvContextManager::setCurrentBrowsable( GvvBrowsable* pBrowsable )
{
	//** Sets the current browsable
	if ( _currentBrowsable != pBrowsable )
	{
		_currentBrowsable = pBrowsable;

		//** Emits the corresponding signal
		emit currentBrowsableChanged();
	}
}
