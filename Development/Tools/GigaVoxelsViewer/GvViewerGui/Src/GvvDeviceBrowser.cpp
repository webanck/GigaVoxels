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

#include "GvvDeviceBrowser.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvBrowserItem.h"
#include "GvvBrowsable.h"
#include "GvvContextMenu.h"
#include "GvvDeviceInterface.h"

// Qt
#include <QContextMenuEvent>
#include <QTreeWidget>

// System
#include <cassert>

// GigaSpace
#include <GsCompute/GsDeviceManager.h>
#include <GsCompute/GsDevice.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerGui;

// GigaSpace
using namespace GsCompute;

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
GvvDeviceBrowser::GvvDeviceBrowser( QWidget* pParent ) 
:	GvvBrowser( pParent )
{
	//// TEST --------------------------------------------
	//GvvBrowsable* light = new GvvBrowsable();
	//GvvBrowserItem* lightItem = createItem( light );
	//addTopLevelItem( lightItem );
	//// TEST --------------------------------------------

	//// attention, peut-être déjà fait ailleurs...
	GsCompute::GsDeviceManager::get().initialize();
	
	for ( int i = 0; i < GsCompute::GsDeviceManager::get().getNbDevices(); i++ )
	{
		const GsCompute::GsDevice* device = GsCompute::GsDeviceManager::get().getDevice( i );

		GvvDeviceInterface* deviceInterface = new GvvDeviceInterface();

		GvvBrowserItem* deviceItem = createItem( deviceInterface );

		deviceItem->setText( 0, device->_name.c_str() );
		deviceItem->setToolTip( 0, QString( "Compute capability : " ) + QString::number( device->mProperties._computeCapabilityMajor ) + QString( "." ) + QString::number( device->mProperties._computeCapabilityMinor ) );

		addTopLevelItem( deviceItem );
	}
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvDeviceBrowser::~GvvDeviceBrowser()
{
}
