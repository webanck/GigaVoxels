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

#include "GvvApplication.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvMainWindow.h"
#include "GvvPluginManager.h"

// Qt
//#include <QSplashScreen>
#include <QImageReader>

// System
#include <cassert>

// STL
#include <iostream>

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
 * The unique instance of the singleton.
 */
GvvApplication* GvvApplication::msInstance = NULL;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Initialize the application
 *
 * @param pArgc Number of arguments
 * @param pArgv List of arguments
******************************************************************************/
void GvvApplication::initialize( int& pArgc, char** pArgv )
{
	assert( msInstance == NULL );
	if ( msInstance == NULL )
    {
		// Initialize application
        msInstance = new GvvApplication( pArgc, pArgv );

		//// Splash screen
		//QPixmap pixmap( "J:\\Projects\\Inria\\GigaVoxels\\Development\\Library\\doc\\GigaVoxelsLogo_div2.png" );
		//QSplashScreen* splash = new QSplashScreen( pixmap, Qt::WindowStaysOnTopHint );
		//splash->show();

		//// Loading some items
		//splash->showMessage( "Loaded modules" );

		//qApp->processEvents();

		//app.processEvents();
		//...
		//	QMainWindow window;
		//window.show();
		//splash.finish(&window);

		// Initialize main window
		msInstance->initializeMainWindow();
	}	
}

/******************************************************************************
 * Finalize the application
******************************************************************************/
void GvvApplication::finalize()
{
	delete msInstance;
	msInstance = NULL;
}

/******************************************************************************
 * Get the application
 *
 * @return The aplication
 ******************************************************************************/
GvvApplication& GvvApplication::get()
{
	assert( msInstance != NULL );
	
	return *msInstance;
}

/******************************************************************************
 * Constructor
 *
 * @param pArgc Number of arguments
 * @param pArgv List of arguments
 ******************************************************************************/
GvvApplication::GvvApplication( int& pArgc, char** pArgv )
:	QApplication( pArgc, pArgv )
,	mGPUComputingInitialized( false )
,	mMainWindow( NULL )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvApplication::~GvvApplication()
{
	delete mMainWindow;
	mMainWindow = NULL;
}

/******************************************************************************
 * Execute the application
 ******************************************************************************/
int GvvApplication::execute()
{	
	mMainWindow->show();

	// Test ----------------------------
	std::cout << "Qt supported image formats :" << std::endl;
	QList< QByteArray > supportedImageFormats = QImageReader::supportedImageFormats();
	for ( int i = 0; i < supportedImageFormats.size(); i++ )
	{
		std::cout << "- " <<supportedImageFormats.at( i ).constData() << std::endl;
	}
	// Test ----------------------------

	// Main Qt's event loop
	int result = exec();
	
	// Remove any plugin
	GvvPluginManager::get().unloadAll();

	// Destroy the main window
	delete mMainWindow;
	mMainWindow = NULL;

	return result;
}

/******************************************************************************
 * Initialize the main wiondow
 ******************************************************************************/
void GvvApplication::initializeMainWindow()
{
	mMainWindow = new GvvMainWindow();
	if ( mMainWindow != NULL )
	{
		mMainWindow->initialize();
	}
}

/******************************************************************************
 * Get the main window
 *
 * return The main window
 ******************************************************************************/
GvvMainWindow* GvvApplication::getMainWindow()
{
	return mMainWindow;
}

/******************************************************************************
 *
 ******************************************************************************/
bool GvvApplication::isGPUComputingInitialized() const
{
	return mGPUComputingInitialized;
}

/******************************************************************************
 *
 ******************************************************************************/
void GvvApplication::setGPUComputingInitialized( bool pFlag )
{
	mGPUComputingInitialized = pFlag;
}
