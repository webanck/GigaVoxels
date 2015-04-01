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

#include "GvvMainWindow.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qt
#include <QUrl>
#include <QFileDialog>
#include <QMessageBox>
#include <QVBoxLayout>
#include <QToolBar>
#include <QDockWidget>

// GvViewer
#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "Gvv3DWindow.h"
#include "GvvPipelineInterfaceViewer.h"
#include "GvvPluginManager.h"
#include "GvvCacheEditor.h"
#include "GvvToolBar.h"
#include "GvvAnalyzerToolBar.h"
#include "GvvCaptureToolBar.h"
#include "GvvOpenTransferFunctionEditor.h"
#include "GvvEditBrowsableAction.h"
#include "GvvRemoveBrowsableAction.h"
#include "GvvActionManager.h"
#include "GvvPipelineBrowser.h"
#include "GvvContextMenu.h"
#include "GvvPipelineInterface.h"
#include "GvvAddLightAction.h"
#include "GvvAddPipelineAction.h"
#include "GvvTransferFunctionInterface.h"
#include "GvvEditorWindow.h"
#include "GvvDisplayDataStructureAction.h"
#include "GvvDisplayPerformanceCountersAction.h"
#include "GvvEditCameraAction.h"
#include "GvvDeviceBrowser.h"
#include "GvvAboutDialog.h"
#include "GvvPreferencesDialog.h"
#include "GvvPipelineEditor.h"
#include "GvvTransferFunctionEditor.h"
#include "GvvGLSceneBrowser.h"
#include "GvvAddSceneAction.h"
#include "GvvGLSLSourceEditor.h"
#include "GvvCUDASourceEditor.h"
#include "GvvPlotView.h"
#include "GvvCacheUsageView.h"
#include "GvvTimeBudgetMonitoringEditor.h"
#include "GvvOpenProgrammableShaderEditor.h"
//#include "GvvSnapshotAction.h"
#include "GvvCaptureVideoAction.h"
#include "GvvMeshInterface.h"
#include "GvvProgrammableShaderInterface.h"
#include "GvvGLSceneInterface.h"
#include "GvvZoomToAction.h"
#include "GvvCameraEditor.h"

//// GigaVoxels
//#include "GvCore/GvDeviceManager.h"
//#include "GvCore/GvDevice.h"

// Qt
#include <QDesktopServices>
#include <QFile>

// STL
#include <iostream>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
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
 * Default constructor
 ******************************************************************************/
GvvMainWindow::GvvMainWindow( QWidget *parent, Qt::WindowFlags flags )
:	QMainWindow( parent, flags )
,	mFilename("")
//-------------------------
//,	_3DViewGroupBox( NULL )
//-------------------------
,	_3DWindow( NULL )
,	_transferFunctionEditor( NULL )
//,	_cacheEditor( NULL )
,	_toolBar( NULL )
,	_analyzerToolBar( NULL )
,	_captureToolBar( NULL )
,	_pipelineBrowser( NULL )
,	_deviceBrowser( NULL )
,	_editorWindow( NULL )
,	_sceneBrowser( NULL )
,	_GLSLSourceEditor( NULL )
,	_CUDASourceEditor( NULL )
,	_cachePlotView( NULL )
,	_cameraEditor( NULL )
{
	// Setup UI
	mUi.setupUi( this );
	
	// Title customization
#ifdef _DEBUG
	setWindowTitle( tr( "GigaVoxels Viewer - DEBUG" ) );
#else
	setWindowTitle( tr( "GigaVoxels Viewer" ) );
#endif

	// Add
//	_3DViewGroupBox = new QGroupBox();
	QVBoxLayout* vbox = new QVBoxLayout();
	vbox->setContentsMargins( 0, 0, 0, 0 );
	mUi._3DViewGroupBox->setLayout( vbox );
//	setCentralWidget( _3DViewGroupBox );

	// 3D window
	_3DWindow = new Gvv3DWindow( this, flags );
	//GvvPipelineInterfaceViewer* pipelineViewer = _3DWindow->getPipelineViewer();
	//setCentralWidget( pipelineViewer );

	//-----------------------------------
	_3DWindow->addViewer();
	//GvvApplication& application = GvvApplication::get();
	//GvvMainWindow* mainWindow = application.getMainWindow();
	//GvvPipelineInterfaceViewer* viewer = new GvvPipelineInterfaceViewer( mainWindow );
	//GvvPipelineInterfaceViewer* viewer = new GvvPipelineInterfaceViewer( NULL );
	/*mainWindow->*/mUi._3DViewGroupBox->layout()->addWidget( _3DWindow->getPipelineViewer() );
	//mPipelineViewer = viewer;
	//-----------------------------------

	//-----------------------------------------------------------------------------------
	// Add a default pipeline viewer
	/*GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	if ( mainWindow != NULL )
	{*/
	//	assert( mainWindow->mUi._3DViewGroupBox != NULL );
		//if ( mainWindow->mUi._3DViewGroupBox != NULL )
		//{
		//	GvvPipelineInterfaceViewer* viewer = new GvvPipelineInterfaceViewer( NULL );
		//	mUi._3DViewGroupBox->layout()->addWidget( viewer );
		//}
//	}
	//-----------------------------------------------------------------------------------

	// Transfer function editor
	_transferFunctionEditor = new GvvTransferFunctionEditor();

	//GvvPipelineInterfaceViewer* viewer = get3DWindow()->getPipelineViewer();
	//QObject::connect( _transferFunctionEditor, SIGNAL( functionChanged() ), viewer, SLOT( onTransferfunctionChanged() ) );
	
	//// Cache editor
	//_cacheEditor = new GvvCacheEditor( this, flags );
	//QDockWidget* dockWidget = new QDockWidget( tr( "Pipeline Editor" ), this, flags );
	//dockWidget->setAllowedAreas( Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea );
	//dockWidget->setWidget( _cacheEditor );
	//addDockWidget( Qt::RightDockWidgetArea, dockWidget );
	
	// Toolbar
	_toolBar = new GvvToolBar( tr( "Shader" ), this );
	addToolBar( _toolBar );

	// Analyzer toolbar
	_analyzerToolBar = new GvvAnalyzerToolBar( tr( "Analyzer" ), this );
	addToolBar( _analyzerToolBar );

	// Capture toolbar
	_captureToolBar = new GvvCaptureToolBar( tr( "Capture" ), this );
	addToolBar( _captureToolBar );

	// Device Browser
	_deviceBrowser = new GvvDeviceBrowser( this );
	QDockWidget* dockWidgetDeviceBrowser = new QDockWidget( tr( "Device Browser" ), this, flags );
	dockWidgetDeviceBrowser->setAllowedAreas( Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea );
	dockWidgetDeviceBrowser->setWidget( _deviceBrowser );
	addDockWidget( Qt::LeftDockWidgetArea, dockWidgetDeviceBrowser );

	// Pipeline Browser
	_pipelineBrowser = new GvvPipelineBrowser( this );
	QDockWidget* dockWidgetPipelineBrowser = new QDockWidget( tr( "Pipeline Browser" ), this, flags );
	dockWidgetPipelineBrowser->setAllowedAreas( Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea );
	dockWidgetPipelineBrowser->setWidget( _pipelineBrowser );
	addDockWidget( Qt::LeftDockWidgetArea, dockWidgetPipelineBrowser );

	// Scene Browser
	_sceneBrowser = new GvvGLSceneBrowser( this );
	QDockWidget* dockWidgetSceneBrowser = new QDockWidget( tr( "Scene Browser" ), this, flags );
	dockWidgetSceneBrowser->setAllowedAreas( Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea );
	dockWidgetSceneBrowser->setWidget( _sceneBrowser );
	addDockWidget( Qt::LeftDockWidgetArea, dockWidgetSceneBrowser );

	// Editor window
	_editorWindow = new GvvEditorWindow( this );
	//_editorWindow->registerEditorFactory( GvvPipelineInterface::cTypeName, &GvvCacheEditor::create );
	//_editorWindow->registerEditorFactory( GvvPipelineInterface::cTypeName, &GvvPipelineEditor::create );
	QDockWidget* dockWidgetEditorWindow = new QDockWidget( tr( "Pipeline Editor" ), this, flags );
	dockWidgetEditorWindow->setAllowedAreas( Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea );
	dockWidgetEditorWindow->setWidget( _editorWindow );
	addDockWidget( Qt::RightDockWidgetArea, dockWidgetEditorWindow );

	// GLSL source editor
	_GLSLSourceEditor = new GvvGLSLSourceEditor( this );
	//_GLSLSourceEditor->show();

	// CUDA source editor
	_CUDASourceEditor = new GvvCUDASourceEditor( this );
	//_CUDASourceEditor->show();

	// Cache plot view
	_cachePlotView = new GvvPlotView( this );
	//_cachePlotView->show();
	QDockWidget* dockWidgetCachePlotView = new QDockWidget( tr( "Cache Plot View" ), this, flags );
	dockWidgetCachePlotView->setAllowedAreas( Qt::BottomDockWidgetArea );
	dockWidgetCachePlotView->setWidget( _cachePlotView );
	addDockWidget( Qt::BottomDockWidgetArea, dockWidgetCachePlotView );
	dockWidgetCachePlotView->close();

	// Time budget monitoring view
	_timeBudgetMonitoringView = new GvvTimeBudgetMonitoringEditor( this );
	//_timeBudgetMonitoringView->show();
	QDockWidget* dockWidgetTimeBudgetMonitoringView = new QDockWidget( tr( "Time Budget Monitoring View" ), this, flags );
	dockWidgetTimeBudgetMonitoringView->setAllowedAreas( Qt::BottomDockWidgetArea );
	dockWidgetTimeBudgetMonitoringView->setWidget( _timeBudgetMonitoringView );
	addDockWidget( Qt::BottomDockWidgetArea, dockWidgetTimeBudgetMonitoringView );
	dockWidgetTimeBudgetMonitoringView->close();

	// Cache usage view
	_cacheUsageView = new GvvCacheUsageView( this );
	//_cacheUsageView->show();
	QDockWidget* dockWidgetCacheUsageView = new QDockWidget( tr( "Cache Usage View" ), this, flags );
	dockWidgetCacheUsageView->setAllowedAreas( Qt::LeftDockWidgetArea );
	dockWidgetCacheUsageView->setWidget( _cacheUsageView );
	addDockWidget( Qt::LeftDockWidgetArea, dockWidgetCacheUsageView );
	//dockWidgetCacheUsageView->close();

	// Camera editor
	_cameraEditor = new GvvCameraEditor( this, Qt::Dialog );
	
	// Organize dock widgets
	//tabifyDockWidget( dockWidgetPipelineBrowser, dockWidgetDeviceBrowser );
	tabifyDockWidget( dockWidgetPipelineBrowser, dockWidgetSceneBrowser );
	dockWidgetPipelineBrowser->raise();
	
	// new/save/open managenement
	connect( mUi.actionOpen, SIGNAL( triggered() ), this, SLOT( onActionOpenFile() ) );
	connect( mUi.actionExit, SIGNAL( triggered() ), this, SLOT( onActionExit() ) );
	connect( mUi.actionPreferences, SIGNAL( triggered() ), this, SLOT( onActionEditPreferences() ) );
	connect( mUi.actionFull_Screen, SIGNAL( triggered() ), this, SLOT( onActionFullScreen() ) );
	connect( mUi.actionHelp, SIGNAL( triggered() ), this, SLOT( onActionHelp() ) );
	connect( mUi.actionAbout, SIGNAL( triggered() ), this, SLOT( onActionAbout() ) );

	// TO DO - to tell whether or not cache is trashing
	if ( statusBar() )
	{
		QPushButton* masterAlarm = new QPushButton( tr( "Master Alarm" ), this );
		masterAlarm->setCheckable( false );
		//masterAlarm->setStyleSheet( "background-color: red" );
		statusBar()->addPermanentWidget( masterAlarm );
	}
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvMainWindow::~GvvMainWindow()
{
	// Delete the transfer function editor because it has no parent widget
	delete _transferFunctionEditor;

	//-----------------------------------
	mUi._3DViewGroupBox->layout()->removeWidget( _3DWindow->getPipelineViewer() );
	_3DWindow->removeViewer();
	//-----------------------------------
}

/******************************************************************************
 * Initialize the main wiondow
 ******************************************************************************/
void GvvMainWindow::initialize()
{
	////---------------------------------------------
	//GvCore::GvDeviceManager::get().initialize();
	//for ( int i = 0; i < GvCore::GvDeviceManager::get().getNbDevices(); i++ )
	//{
	//	GvCore::GvDevice* device = GvCore::GvDeviceManager::get().get
	//}
	////---------------------------------------------
	
	GvvAction* action = NULL;

	action = new GvvOpenTransferFunctionEditor( "" );
	action = new GvvEditBrowsableAction();
	action = new GvvRemoveBrowsableAction();
	action = new GvvAddLightAction( "" );	// enlever le premier paramètre
	action = new GvvAddPipelineAction();	// enlever le premier paramètre
	action = new GvvDisplayDataStructureAction( "" );	// enlever le premier paramètre
	action = new GvvDisplayPerformanceCountersAction( "" );	// enlever le premier paramètre
	action = new GvvAddSceneAction();	// enlever le premier paramètre
	action = new GvvEditCameraAction( "" );
	action = new GvvOpenProgrammableShaderEditor( "" );
	//action = new GvvSnapshotAction( "" );
	action = new GvvCaptureVideoAction( "" );
	action = new GvvZoomToAction( "" );
			
	// ToolBar
	_toolBar->addAction( GvvActionManager::get().editAction( GvvOpenTransferFunctionEditor::cName ) );
	_toolBar->addAction( GvvActionManager::get().editAction( GvvAddLightAction::cName ) );
	// Tmp
	_toolBar->addAction( GvvActionManager::get().editAction( GvvEditCameraAction::cName ) );

	// Analyzer toolBar
	_analyzerToolBar->addAction( GvvActionManager::get().editAction( GvvDisplayDataStructureAction::cName ) );
	_analyzerToolBar->addAction( GvvActionManager::get().editAction( GvvDisplayPerformanceCountersAction::cName ) );

	// Capture toolbar
	//_captureToolBar->addAction( GvvActionManager::get().editAction( GvvSnapshotAction::cName ) );
	_captureToolBar->addAction( GvvActionManager::get().editAction( GvvCaptureVideoAction::cName ) );

	//GvvActionManager::get().editAction( GvvDisplayPerformanceCountersAction::cName )->setEnabled(  );

	// Global root context menu
	GvvContextMenu* contextMenu = _pipelineBrowser->getContextMenu( "" );
	if ( contextMenu != NULL )
	{
		//contextMenu->addAction( GvvActionManager::get().editAction( GvvOpenTransferFunctionEditor::cName ) );
		contextMenu->addAction( GvvActionManager::get().editAction( GvvAddPipelineAction::cName ) );
	}

	// Pipeline context menu
	contextMenu = _pipelineBrowser->getContextMenu( QString( GvvPipelineInterface::cTypeName ) );
	if ( contextMenu != NULL)
	{
		contextMenu->addAction( GvvActionManager::get().editAction( GvvEditBrowsableAction::cName ) );
		contextMenu->addSeparator();
		contextMenu->addAction( GvvActionManager::get().editAction( GvvRemoveBrowsableAction::cName ) );
		contextMenu->addSeparator();
		contextMenu->addAction( GvvActionManager::get().editAction( GvvOpenProgrammableShaderEditor::cName ) );
	}

	// Transfer Function context menu
	contextMenu = _pipelineBrowser->getContextMenu( QString( GvvTransferFunctionInterface::cTypeName ) );
	if ( contextMenu != NULL)
	{
		contextMenu->addAction( GvvActionManager::get().editAction( GvvEditBrowsableAction::cName ) );
		contextMenu->addSeparator();
		contextMenu->addAction( GvvActionManager::get().editAction( GvvRemoveBrowsableAction::cName ) );
	}

	// Scene browser root context menu
	contextMenu = _sceneBrowser->getContextMenu( "" );
	if ( contextMenu != NULL )
	{
		contextMenu->addAction( GvvActionManager::get().editAction( GvvAddSceneAction::cName ) );
	}

	// Scene context menu
	contextMenu = _sceneBrowser->getContextMenu( QString( GvvGLSceneInterface::cTypeName ) );
	if ( contextMenu != NULL )
	{
		contextMenu->addAction( GvvActionManager::get().editAction( GvvRemoveBrowsableAction::cName ) );
		contextMenu->addSeparator();
		contextMenu->addAction( GvvActionManager::get().editAction( GvvZoomToAction::cName ) );
	}

	// Mesh menu
	contextMenu = _pipelineBrowser->getContextMenu( QString( GvvMeshInterface::cTypeName ) );
	if ( contextMenu != NULL)
	{
		contextMenu->addAction( GvvActionManager::get().editAction( "Zoom To" ) );
		contextMenu->addSeparator();
		contextMenu->addAction( "Show Normals" );
		contextMenu->addSeparator();
		contextMenu->addAction( "Voxelize" );
		contextMenu->addSeparator();
		contextMenu->addAction( GvvActionManager::get().editAction( GvvRemoveBrowsableAction::cName ) );
	}

	// Mesh menu
	contextMenu = _pipelineBrowser->getContextMenu( QString( GvvProgrammableShaderInterface::cTypeName ) );
	if ( contextMenu != NULL)
	{
		contextMenu->addAction( "Edit" );
		contextMenu->addSeparator();
		contextMenu->addAction( "Remove" );
	}
	
	// A deplacer + utiliser des settings
	this->resize( 1024, 768 );
}

/******************************************************************************
 * Open action
 ******************************************************************************/
void GvvMainWindow::onActionOpenFile()
{
	QString fileName = QFileDialog::getOpenFileName( this, "Choose a file", QString( "." ), tr( "Demo Files (*.gvp)" ) );
	if ( ! fileName.isEmpty() )
	{
		//loadModel( lFileName );

		//const std::string pluginFilename = "DynamicLoad.d.gvp";
		GvvPluginManager::get().unloadAll();
		GvvPluginManager::get().loadPlugin( fileName.toStdString() );

		mFilename = fileName;

		//_cacheEditor->
	}
}

/******************************************************************************
 * Exit action
 ******************************************************************************/
void GvvMainWindow::onActionExit()
{
	QMessageBox::information( this, tr( "Exit" ), tr( "Not yet implemented..." ) );
}

/******************************************************************************
 * Edit preferences action
 ******************************************************************************/
void GvvMainWindow::onActionEditPreferences()
{
	//QMessageBox::information( this, tr( "Edit Preferences" ), tr( "Not yet implemented..." ) );
	GvvPreferencesDialog preferencesDialog( this );
	preferencesDialog.exec();
}

/******************************************************************************
 * Display full screen action
 ******************************************************************************/
void GvvMainWindow::onActionFullScreen()
{
	/*if ( get3DWindow() != NULL && get3DWindow()->getPipelineViewer() )
	{
		get3DWindow()->getPipelineViewer()->setFullScreen( true );
	}*/
	QMessageBox::information( this, tr( "Display Full Screen" ), tr( "Not yet implemented..." ) );
}

/******************************************************************************
 * Display help action
 ******************************************************************************/
void GvvMainWindow::onActionHelp()
{
	//QMessageBox::information( this, tr( "Help" ), tr( "Not yet implemented..." ) );
	//** Retrieves the url to the help file
	//QDir lDir = GvvEnvironment::getSystemDirPath( GvvEnvironment::eManualsDir, false );
	QString path( "J:\\Projects\\Inria\\GigaVoxels\\Development\\Documents\\Doxygen\\html\\index.html" );
	if ( QFile::exists( path ) )
	{
		//** Laucnhs the default pdf viewer
		QDesktopServices::openUrl( QUrl::fromLocalFile( path ) );
	}
	else
	{
		//GvvApplication::warn( qApp->translate( "GvvMainWindow", "Unable to find the manual file " ) + lPath );
	}
}

/******************************************************************************
 * Open about dialog action
 ******************************************************************************/
void GvvMainWindow::onActionAbout()
{
	//QMessageBox::information( this, tr( "About" ), tr( "Not yet implemented..." ) );
	GvvAboutDialog aboutDialog( this );
	aboutDialog.exec();
}

/******************************************************************************
 * Get the 3D window.
 *
 * return the 3D window
 ******************************************************************************/
Gvv3DWindow* GvvMainWindow::get3DWindow()
{
	return _3DWindow;
}

/******************************************************************************
 * Get the pipeline browser
 *
 * return the pipeline browser
 ******************************************************************************/
GvvPipelineBrowser* GvvMainWindow::getPipelineBrowser()
{
	return _pipelineBrowser;
}

///******************************************************************************
// * Get the pipeline editor
// *
// * return the pipeline editor
// ******************************************************************************/
//GvvCacheEditor* GvvMainWindow::getPipelineEditor()
//{
//	return _cacheEditor;
//}

/******************************************************************************
 * Get the transfer function editor.
 *
 * return the transfer function editor
 ******************************************************************************/
GvvTransferFunctionEditor* GvvMainWindow::getTransferFunctionEditor()
{
	return _transferFunctionEditor;
}

/******************************************************************************
 * Get the editor window
 *
 * return the editor window
 ******************************************************************************/
GvvEditorWindow* GvvMainWindow::getEditorWindow()
{
	return _editorWindow;
}

/******************************************************************************
 * Get the scene browser
 *
 * return the scene browser
 ******************************************************************************/
GvvGLSceneBrowser* GvvMainWindow::getSceneBrowser()
{
	return _sceneBrowser;
}

/******************************************************************************
 * Get the cache plot viewer.
 *
 * return the cache plot viewer
 ******************************************************************************/
GvvPlotView* GvvMainWindow::getCachePlotView()
{
	return _cachePlotView;
}

/******************************************************************************
 * Get the cache usage view
 *
 * return the cache usage view
 ******************************************************************************/
GvvCacheUsageView* GvvMainWindow::getCacheUsageView()
{
	return _cacheUsageView;
}

/******************************************************************************
 * Get the time budget monitoring view
 *
 * return the time budget monitoring view
 ******************************************************************************/
GvvTimeBudgetMonitoringEditor* GvvMainWindow::getTimeBudgetMonitoringView()
{
	return _timeBudgetMonitoringView;
}

/******************************************************************************
 * Get the programmable shader browser.
 *
 * return the programmable shader browser.
 ******************************************************************************/
GvvGLSLSourceEditor* GvvMainWindow::getGLSLourceEditor()
{
	return _GLSLSourceEditor;
}

/******************************************************************************
 * Get the Camera editor
 *
 * return the Camera editor
 ******************************************************************************/
GvvCameraEditor* GvvMainWindow::getCameraEditor()
{
	return _cameraEditor;
}
