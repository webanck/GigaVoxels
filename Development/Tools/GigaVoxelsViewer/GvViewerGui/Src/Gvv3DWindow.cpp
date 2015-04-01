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

#include "Gvv3DWindow.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvPipelineInterfaceViewer.h"

//---------------
#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include <QGroupBox>
//---------------

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerGui;

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
 * Default constructor
 ******************************************************************************/
Gvv3DWindow::Gvv3DWindow( QWidget* parent, Qt::WindowFlags flags )
:	QObject( parent )
,	mPipelineViewer( NULL )
{
	//mPipelineViewer = new GvvPipelineInterfaceViewer( parent, NULL, flags );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
Gvv3DWindow::~Gvv3DWindow()
{
	delete mPipelineViewer;
	mPipelineViewer = NULL;
}

/******************************************************************************
 * Get the pipeline viewer
 *
 * return The pipeline viewer
 ******************************************************************************/
GvvPipelineInterfaceViewer* Gvv3DWindow::getPipelineViewer()
{
	return mPipelineViewer;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Gvv3DWindow::addViewer()
{
//	delete mPipelineViewer;
//	mPipelineViewer = NULL;

//	GvvApplication& application = GvvApplication::get();
//	GvvMainWindow* mainWindow = application.getMainWindow();

	//GvvPipelineInterfaceViewer* viewer = new GvvPipelineInterfaceViewer( mainWindow );
	GvvPipelineInterfaceViewer* viewer = new GvvPipelineInterfaceViewer( NULL );

	//mainWindow->mUi._3DViewGroupBox->layout()->addWidget( viewer );

	mPipelineViewer = viewer;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Gvv3DWindow::removeViewer()
{
//	GvvApplication& application = GvvApplication::get();
	//GvvMainWindow* mainWindow = application.getMainWindow();

	//mainWindow->mUi._3DViewGroupBox->layout()->removeWidget( mPipelineViewer );

	delete mPipelineViewer;
	mPipelineViewer = NULL;
}
