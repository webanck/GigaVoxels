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

#include "GvvContextListener.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvContextManager.h"

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
GvvContextListener::GvvContextListener( unsigned int pSignals )
:	_contextListenerProxy( this )
{
	//** Listen to the given signals
	listen( pSignals );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvContextListener::~GvvContextListener()
{
	disconnectAll();
}

/******************************************************************************
 * Listen the given document
 *
 * @param pSignals specifies the signals to be activated
 ******************************************************************************/
void GvvContextListener::listen( unsigned int pSignals )
{
	assert( GvvContextManager::get() != NULL );
	disconnectAll();
	connect( pSignals );
}

/******************************************************************************
 * Connect this plugin to the specified signal
 *
 * @param	pSignal specifies the signal to be activated
 ******************************************************************************/
void GvvContextListener::connect( unsigned int pSignals )
{
	//** HACK
	//** Disconnect desired signal(s) because we don't know
	//** if the listener is already connected to this/these desired signal(s)
	disconnect( pSignals );

	// Connection(s)
	if ( pSignals & eBrowsableChanged )
	{
		QObject::connect( GvvContextManager::get(), SIGNAL( currentBrowsableChanged( ) ), &_contextListenerProxy, SLOT( onCurrentBrowsableChanged( ) ) );
	}
}

/******************************************************************************
 * Disconnects all the slots
 ******************************************************************************/
void GvvContextListener::connectAll()
{
	connect( eAllSignals );
}

/******************************************************************************
 * Disconnectes this plugin to the specified signal
 *
 * @param	pSignal specifies the signal to be activated
 ******************************************************************************/
void GvvContextListener::disconnect( unsigned int pSignals )
{
	if ( pSignals & eBrowsableChanged )
	{
		QObject::disconnect( GvvContextManager::get(), SIGNAL( currentBrowsableChanged( ) ), &_contextListenerProxy, SLOT( onCurrentBrowsableChanged( ) ) );
	}
}

/******************************************************************************
 * Disconnects all the slots
 ******************************************************************************/
void GvvContextListener::disconnectAll()
{
	disconnect( eAllSignals );
}

/******************************************************************************
 * Returns whether this plugin is connected to the specified signal
 *
 * @param	pSignal specifies the signal to be checked
 *
 * @return	true if the signal is handled
 ******************************************************************************/
bool GvvContextListener::isConnected( GvvSignal pSignal )
{
	return false;
}

/******************************************************************************
 ****************************** SLOT DEFINITION *******************************
 ******************************************************************************/

/******************************************************************************
 * This slot is called when an Current Browsable Changed
 *
 ******************************************************************************/
void GvvContextListener::onCurrentBrowsableChanged( )
{
}
