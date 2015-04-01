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

#include "GvvContextListenerProxy.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvContextListener.h"

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerGui;

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
 * Default constructor.
 ******************************************************************************/
GvvContextListenerProxy::GvvContextListenerProxy( GvvContextListener* pContextListener )
:	_contextListener( NULL )
{
	assert( pContextListener != NULL );
	_contextListener = pContextListener;
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvContextListenerProxy::~GvvContextListenerProxy()
{
	_contextListener = NULL;
}

/******************************************************************************
 ****************************** SLOT DEFINITION *******************************
 ******************************************************************************/

/******************************************************************************
* slot called when the current browsable changed
 ******************************************************************************/
void GvvContextListenerProxy::onCurrentBrowsableChanged()
{
	_contextListener->onCurrentBrowsableChanged();
}
