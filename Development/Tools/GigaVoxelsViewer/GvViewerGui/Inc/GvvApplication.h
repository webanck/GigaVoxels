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

#ifndef GVVAPPICATION_H
#define GVVAPPICATION_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
 
// Qt
#include <QApplication>

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
namespace GvViewerGui
{
	class GvvMainWindow;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * @class GvQApplication
 *
 * @brief The GvQApplication class provides ...
 *
 * ...
 */
class GVVIEWERGUI_EXPORT GvvApplication : public QApplication
{
	// Qt Macro
	Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Initialize the application
	 *
	 * @param pArgc Number of arguments
	 * @param pArgv List of arguments
	 */
	static void initialize( int& pArgc, char** pArgv );

	/**
	 * Finalize the application
	 */
	static void finalize();

	/**
	 * Get the application
	 *
	 * return The application
	 */
	static GvvApplication& get();

	/**
	 * Execute the application
	 */
	int execute();

	/**
	 * Get the main window
	 *
	 * return The main window
	 */
	GvvMainWindow* getMainWindow();

	/**
	 *
	 */
	bool isGPUComputingInitialized() const;

	/**
	 *
	 */
	void setGPUComputingInitialized( bool pFlag );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 *
	 */
	bool mGPUComputingInitialized;
		
	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param pArgc Number of arguments
	 * @param pArgv List of arguments
	 */
	GvvApplication( int& pArgc, char** pArgv );

	/**
	 * Destructor
	 */
	virtual ~GvvApplication();

	/**
	 * Initialize the main wiondow
	 */
	void initializeMainWindow();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
     * The unique instance
     */
    static GvvApplication* msInstance;

	/**
	 * The main window
	 */
	GvvMainWindow* mMainWindow;
	
	/******************************** METHODS *********************************/

};

} // namespace GvViewerGui

#endif
