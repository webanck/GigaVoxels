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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

// Qt
#include <QApplication>

// Simple Sphere
#include "SampleViewer.h"

// GigaVoxels
#include <GvCore/GvVersion.h>
#include <GsCompute/GsDeviceManager.h>

// STL
#include <iostream>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvCore;
using namespace GsCompute;

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
 * Main entry program
 *
 * @param pArgc number of arguments
 * @param pArgv list of arguments
 *
 * @return code
 ******************************************************************************/
int main( int pArgc, char* pArgv[] )
{
	// Exit code
	int result = 0;

	// GLUT initialization
	glutInit( &pArgc, pArgv );

	// Qt main application
	QApplication app( pArgc, pArgv );

	// GigaVoxels API's version
	cout << "GigaVoxels API's version : " << GvVersion::getVersion() << endl;

	// Test client architecture
	// If harware is compliant with le GigaVoxels Engine, launch the demo
	SampleViewer* sampleViewer = NULL;
	if ( GsDeviceManager::get().initialize() )
	{
		// Create your QGLViewer custom widget
		sampleViewer = new SampleViewer();
		if ( sampleViewer != NULL )
		{
			sampleViewer->setWindowTitle( "Menger Sponge - Serpinski example" );
			sampleViewer->show();

			// Enter Qt main event loop
			result = app.exec();
		}
	}
	else
	{
		cout << "\nThe program will now exit" << endl;
	}
	
	// Release memory
	delete sampleViewer;
	GsDeviceManager::get().finalize();

	// CUDA tip: clean up to ensure correct profiling
	cudaError_t error = cudaDeviceReset();
	
	// Return exit code
	return result;
}
