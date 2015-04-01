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

#ifndef GVVCONTEXTLISTENER_H
#define GVVCONTEXTLISTENER_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "GvvContextListenerProxy.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

class GVVIEWERGUI_EXPORT GvvContextListener
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	enum GvvSignal
	{
		eBrowsableChanged	= 1,
		eAllSignals			= (1 << 1) - 1
	};
	
	/******************************** METHODS *********************************/
	
	/**
	 * Default constructor.
	 */
	GvvContextListener( unsigned int pSignals = eAllSignals );
	
	/**
	 * Destructor.
	 */
	virtual ~GvvContextListener();
	
	/**
	 * Listen the context
	 *
	 * @param pSignals specifies the signals to be activated
	 */
	void listen( unsigned int pSignals );

    /**
     * Connects the given signals
     *
     * @param pSignals specifies the signals to be connected
     */
	void connect( unsigned int pSignals );

    /**
     * Connects all the signals
     */
	void connectAll();

    /**
     * Disconnects the given signals
     *
     * @param pSignals specifies the signals to be disconnected
     */
	void disconnect( unsigned int pSignals );

    /**
     * Disconnects all the signals
     */
	void disconnectAll();

    /**
     * Returns whether this plugin is connected to the specified signal
     *
     * @param	pSignal specifies the signal to be checked
     *
     * @return	true if the signal is handled
     */
	bool isConnected( GvvSignal pSignal );

	/**
	 * slot called when the current browsable changed
	 */
	virtual void onCurrentBrowsableChanged();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************* ATTRIBUTES *******************************/
	
	/**
	 * The context listener proxy
	 */
	GvvContextListenerProxy _contextListenerProxy;
    
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
	GvvContextListener( const GvvContextListener& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvContextListener& operator=( const GvvContextListener& );
	
};

} // namespace GvViewerGui

#endif
