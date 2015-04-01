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

#ifndef GVVMAINWINDOW_H
#define GVVMAINWINDOW_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "UI_GvQMainWindow.h"

// Qt
#include <QMainWindow>

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
	class Gvv3DWindow;
	class GvvCacheEditor;
	class GvvToolBar;
	class GvvAnalyzerToolBar;
	class GvvCaptureToolBar;
	class GvvPipelineBrowser;
	class GvvEditorWindow;
	class GvvDeviceBrowser;
	class GvvTransferFunctionEditor;
	class GvvGLSceneBrowser;
	class GvvGLSLSourceEditor;
	class GvvCUDASourceEditor;
	class GvvPlotView;
	class GvvCacheUsageView;
	class GvvTimeBudgetMonitoringEditor;
	class GvvCameraEditor;
}

//// Qt
//class QGroupBox;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * @class GvQMainWindow
 *
 * @brief The GvQMainWindow class provides ...
 *
 * ...
 */
class GVVIEWERGUI_EXPORT GvvMainWindow : public QMainWindow
{
	// Qt Macro
	Q_OBJECT

	/**
	 * ...
	 */
	friend class Gvv3DWindow;

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	///**
	// * The 3D window
	// */
	//QGroupBox* _3DViewGroupBox;
	
	/******************************** METHODS *********************************/

	/**
	 * Default constructor.
	 */
	GvvMainWindow( QWidget *parent = 0, Qt::WindowFlags flags = 0 );

	/**
	 * Destructor.
	 */
	virtual ~GvvMainWindow();

	/**
	 * Initialize.
	 */
	void initialize();

	/**
	 * Get the 3D window.
	 *
	 * return the 3D window
	 */
	Gvv3DWindow* get3DWindow();

	/**
	 * Get the pipeline editor.
	 *
	 * return the pipeline editor
	 */
	//GvvCacheEditor* getPipelineEditor();
	//GvvPipelineEditor* getPipelineEditor();

	/**
	 * Get the pipeline browser.
	 *
	 * return the pipeline browser
	 */
	GvvPipelineBrowser* getPipelineBrowser();

	/**
	 * Get the pipeline editor.
	 *
	 * return the pipeline editor
	 */
	GvvEditorWindow* getEditorWindow();
	
	/**
	 * Get the transfer function editor.
	 *
	 * return the transfer function editor
	 */
	GvvTransferFunctionEditor* getTransferFunctionEditor();

	/**
	 * Get the scene browser.
	 *
	 * return the scene browser
	 */
	GvvGLSceneBrowser* getSceneBrowser();

	/**
	 * Get the programmable shader browser.
	 *
	 * return the programmable shader browser.
	 */
	GvvGLSLSourceEditor* getGLSLourceEditor();

	/**
	 * Get the scene browser.
	 *
	 * return the scene browser
	 */
	GvvCUDASourceEditor* getCUDASourceEditor();

	/**
	 * Get the cache plot viewer.
	 *
	 * return the cache plot viewer
	 */
	GvvPlotView* getCachePlotView();

	/**
	 * Get the cache usage view
	 *
	 * return the cache usage view
	 */
	GvvCacheUsageView* getCacheUsageView();

	/**
	 * Get the time budget monitoring view
	 *
	 * returnthe time budget monitoring view
	 */
	GvvTimeBudgetMonitoringEditor* getTimeBudgetMonitoringView();

	/**
	 * Get the Camera editor
	 *
	 * return the Camera editor
	 */
	GvvCameraEditor* getCameraEditor();
				
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************* ATTRIBUTES *******************************/

	/**
	 * The 3D window
	 */
	Gvv3DWindow* _3DWindow;

	/**
	 * The cache editor
	 */
	//GvvCacheEditor* _cacheEditor;
	//GvvPipelineEditor* _pipelineCacheEditor;
	
	/**
	 * The tool bar
	 */
	GvvToolBar* _toolBar;

	/**
	 * The analyzer tool bar
	 */
	GvvAnalyzerToolBar* _analyzerToolBar;

	/**
	 * The capture tool bar
	 */
	GvvCaptureToolBar* _captureToolBar;

	/**
	 * The pipeline browser
	 */
	GvvPipelineBrowser* _pipelineBrowser;

	/**
	 * The device browser
	 */
	GvvDeviceBrowser* _deviceBrowser;
	
	/**
	 * The transfer function editor
	 */
	GvvTransferFunctionEditor* _transferFunctionEditor;

	/**
	 * The editor window
	 */
	GvvEditorWindow* _editorWindow;

	/**
	 * The scene browser
	 */
	GvvGLSceneBrowser* _sceneBrowser;

	/**
	 * Shader source editor
	 */
	GvvGLSLSourceEditor* _GLSLSourceEditor;

	/**
	 * Shader source editor
	 */
	GvvCUDASourceEditor* _CUDASourceEditor;

	/**
	 * Cache plot viewer
	 */
	GvvPlotView* _cachePlotView;

	/**
	 * Cache usage view
	 */
	GvvCacheUsageView* _cacheUsageView;

	/**
	 * Time budget monitoring view
	 */
	GvvTimeBudgetMonitoringEditor* _timeBudgetMonitoringView;

	/**
	 * Camera editor
	 */
	GvvCameraEditor* _cameraEditor;
			
	/******************************** METHODS *********************************/
	
	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
	
	/**
	 * Ui designer class
	 */
	Ui::GvQMainWindow mUi;

	/**
	 * the opened filename
	 */
	QString mFilename;
	
	/******************************** METHODS *********************************/

private slots:

	/**
	 * Open file action
	 */
	void onActionOpenFile();

	/**
	 * Exit action
	 */
	void onActionExit();

	/**
	 * Edit preferences action
	 */
	void onActionEditPreferences();

	/**
	 * Display full screen action
	 */
	void onActionFullScreen();

	/**
	 * Display help action
	 */
	void onActionHelp();

	/**
	 * Open about dialog action
	 */
	void onActionAbout();

};

} // namespace GvViewerGui

#endif

