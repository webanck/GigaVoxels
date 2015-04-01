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

#include "Plugin.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "PluginConfig.h"

// System
#include <cassert>
#include <iostream>

#include <GvUtils/GvPluginManager.h>

// STL
#include <sstream>

// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

// Project
#include "SampleCore.h"
#include "CustomEditor.h"
//#include "GvgMD2Model.h"

// GvViewer
#include <GvvApplication.h>
#include <GvvMainWindow.h>
#include <Gvv3DWindow.h>
#include <GvvPipelineManager.h>
#include <GvvEditorWindow.h>
#include <GvvDataLoaderDialog.h>

// Qt
#include <QFileDialog>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigqVoxels
using namespace GvUtils;

// VtViewer
using namespace GvViewerCore;
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
 * 
 ******************************************************************************/
extern "C" GVVIDEOGAME_EXPORT GvPluginInterface* createPlugin( GvPluginManager& pManager )
{
    //return new GvMyPlugin( pManager );
	GvMyPlugin* plugin = new GvMyPlugin( pManager );
	assert( plugin != NULL );

	return plugin;
}

/******************************************************************************
 * Default constructor
 ******************************************************************************/
GvMyPlugin::GvMyPlugin( GvPluginManager& pManager )
:	mManager( pManager )
,	mName( "GvVideoGamePlugin" )
,	mExportName( "Format A" )
,	mPipeline( NULL )
{
	initialize();
}

/******************************************************************************
 * Default constructor
 ******************************************************************************/
GvMyPlugin::~GvMyPlugin()
{
	finalize();
}

/******************************************************************************
 *
 ******************************************************************************/
void GvMyPlugin::initialize()
{
	//-------------------------------------------
	//GvgMD2Model* model = new GvgMD2Model();
	//model->load( "Weapon.md2" );
	////model->draw( 0 );
	//-------------------------------------------

	////-----------------------------------------------
	//// PROBLEM :
	//// le dialog semble provoquer un draw() sans qu'il y ait eu un resize donc un init du pipeline => crash OpenGL...
	//QString modelFilename;
	//unsigned int modelResolution;
	//{	// "{" is used to destroy the widget....
	///*if ( mPipeline != NULL )
	//{*/
	////	if ( mPipeline->has3DModel() )
	////	{
	//		GvvDataLoaderDialog dataLoaderDialog( NULL );
	//		dataLoaderDialog.exec();

	//		if ( dataLoaderDialog.result() == QDialog::Accepted )
	//		{
	//			//mPipeline->set3DModelFilename( dataLoaderDialog.get3DModelFilename().toLatin1().constData() );
	//			modelFilename = dataLoaderDialog.get3DModelFilename();
	//			modelResolution = dataLoaderDialog.get3DModelResolution();
	//		}
	////	}
	////}
	//} // "}" is used to destroy the widget....
	////-----------------------------------------------

	GvViewerGui::GvvApplication& application = GvViewerGui::GvvApplication::get();
	GvViewerGui::GvvMainWindow* mainWindow = application.getMainWindow();

	// Register custom editor's factory method
	GvViewerGui::GvvEditorWindow* editorWindow = mainWindow->getEditorWindow();
	editorWindow->registerEditorFactory( SampleCore::cTypeName, &CustomEditor::create );

	// Add the GigaVoxels pipeline in 3D view
	GvViewerGui::Gvv3DWindow* window3D = mainWindow->get3DWindow();
	window3D->addViewer();	// deplacer cet appel via le "GvvPipelineManager::get().addPipeline( mPipeline )" !!!!!!!!!!!!!!!!

	// Create the GigaVoxels pipeline
	mPipeline = new SampleCore();

	////-----------------------------------------------
	//if ( mPipeline != NULL )
	//{
	//	mPipeline->set3DModelFilename( modelFilename.toLatin1().constData() );
	//	mPipeline->set3DModelResolution( modelResolution );

	//	// TO DO
	//	// add resolution too !!!
	//}
	////-----------------------------------------------

	// Tell the viewer that a new pipeline has been added
	GvvPipelineManager::get().addPipeline( mPipeline );
}

/******************************************************************************
 *
 ******************************************************************************/
void GvMyPlugin::finalize()
{
	// Tell the viewer that a pipeline is about to be removed
	GvvPipelineManager::get().removePipeline( mPipeline );

	// Destroy the pipeline
	delete mPipeline;
	mPipeline = NULL;

	GvViewerGui::GvvApplication& application = GvViewerGui::GvvApplication::get();
	GvViewerGui::GvvMainWindow* mainWindow = application.getMainWindow();
	
	// deplacer cet appel via le "GvvPipelineManager::get().removePipeline( mPipeline )" !!!!!!!!!!!!!!!!
	GvViewerGui::Gvv3DWindow* window3D = mainWindow->get3DWindow();
	window3D->removeViewer();

	// Register custom editor's factory method
	GvViewerGui::GvvEditorWindow* editorWindow = mainWindow->getEditorWindow();
	editorWindow->unregisterEditorFactory( SampleCore::cTypeName );
}

/******************************************************************************
 * 
 ******************************************************************************/
const string& GvMyPlugin::getName()
{
    return mName;
}
