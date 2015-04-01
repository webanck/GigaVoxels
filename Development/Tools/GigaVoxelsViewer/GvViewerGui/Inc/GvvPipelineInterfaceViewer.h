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

#ifndef _GVV_PIPELINE_INTERFACE_VIEWER_H_
#define _GVV_PIPELINE_INTERFACE_VIEWER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "GvvPipelineManagerListener.h"
#include "GvvGLSceneManagerListener.h"

// OpenGL
#include <GL/glew.h>

// Qt
#include <QGLViewer/qglviewer.h>
#include <QKeyEvent>

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
namespace GvViewerCore
{
	class GvvPipelineInterface;
	class GvvGLSceneInterface;
}

// QGLViewer
namespace qglviewer
{
	class ManipulatedFrame;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * @class SampleViewer
 *
 * @brief The SampleViewer class provides...
 *
 * ...
 */
class GVVIEWERGUI_EXPORT GvvPipelineInterfaceViewer : public QGLViewer, public GvViewerCore::GvvPipelineManagerListener, public GvViewerCore::GvvGLSceneManagerListener
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
	 * Constructor
	 */
	GvvPipelineInterfaceViewer( QWidget* parent = 0, const QGLWidget* shareWidget = 0, Qt::WindowFlags flags = 0 );

	/**
	 * Destructor
	 */
	virtual ~GvvPipelineInterfaceViewer();

	/**
	 * Set the pipeline.
	 *
	 * @param pPipeline The pipeline
	 */
	virtual void setPipeline( GvViewerCore::GvvPipelineInterface* pPipeline );

	/**
	 * Get the pipeline.
	 *
	 * @return The pipeline
	 */
	virtual const GvViewerCore::GvvPipelineInterface* getPipeline() const;

	/**
	 * Edit the pipeline.
	 *
	 * @return The pipeline
	 */
	virtual GvViewerCore::GvvPipelineInterface* editPipeline();

	/**
	 * Capture video by producing list of images
	 *
	 * @param flag to start and stop video generation
	 *
	 * @return flag to tell wheter or not it succeeds
	 */
	bool captureVideo( bool pFlag );

	/******************************** SIGNALS *********************************/

signals:

	/**
	 * The signal is emitted when the viewer has been resized
	 *
	 * @param pWidth new viewer width
	 * @param pHeight new viewr height
	 */
	void resized( int pWidth, int pHeight );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Flag to tell wheter or not to display FPS
	 */
	bool _showFPS;
	
	/**
	 * Flag to tell wheter or not to display grid
	 */
	bool _showGrid;

	/**
	 * Flag to tell wheter or not to display axis
	 */
	bool _showAxis;

	/**
	 * ModelView matrix
	 */
	GLdouble _modelViewMatrix[ 16 ];

	/**
	 * Projection matrix
	 */
	GLdouble _projectionMatrix[ 16 ];

	/******************************** METHODS *********************************/

	/**
	 * Init.
	 */
	virtual void init();

	/**
	 * Main paint method, inherited from \c QGLWidget.
	 *
	 * Calls the following methods, in that order:
	 * @param preDraw() (or preDrawStereo() if viewer displaysInStereo()) : places the camera in the world coordinate system.
	 * @param draw() (or fastDraw() when the camera is manipulated) : main drawing method. Should be overloaded.
	 * @param postDraw() : display of visual hints (world axis, FPS...)
	 */
	virtual void paintGL();

	/**
	 * Pre draw
	 */
	//virtual void preDraw();

	/**
	 * Fast draw
	 */
	//virtual void fastDraw();
	
	/**
	 * Draw
	 */
	virtual void draw();

	/**
	 * Post draw
	 */
	//virtual void postDraw();

	/**
	 * Resize GL
	 *
	 * @param width width
	 * @param height height
	 */
	virtual void resizeGL( int width, int height );

	/**
	 * Size hint.
	 *
	 * @return size hint
	 */
	virtual QSize sizeHint() const;

	/**
	 * Key press event.
	 *
	 * @param e the event
	 */
	virtual void keyPressEvent( QKeyEvent* e );

	/**
	 * Add a pipeline.
	 *
	 * @param the pipeline to add
	 */
	virtual void onPipelineAdded( GvViewerCore::GvvPipelineInterface* pPipeline );

	/**
	 * Remove a pipeline.
	 *
	 * @param the pipeline to remove
	 */
	virtual void onPipelineRemoved( GvViewerCore::GvvPipelineInterface* pPipeline );

	/**
	 * Add a scene.
	 *
	 * @param pScene the scene to add
	 */
	virtual void onGLSceneAdded( GvViewerCore::GvvGLSceneInterface* pScene );

	/**
	 * Remove a scene.
	 *
	 * @param pScene the scene to remove
	 */
	virtual void onGLSceneRemoved( GvViewerCore::GvvGLSceneInterface* pScene );

	/********************************* SLOTS **********************************/

protected slots:

	/**
	 * Slot called when the transfer function of the associated editor has been modified
	 */
	void onTransferfunctionChanged();

	/**
	 * Slot called when the ManipulatedFrame is manipulated
	 */
	void onFrameManipulated();
	
	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * GigaVoxels pipeline
	 */
	GvViewerCore::GvvPipelineInterface* mPipeline;

	/**
	 * QGLViewer frame
	 */
	qglviewer::ManipulatedFrame* _manipulatedFrame;

	/**
	 * GL scene
	 */
	GvViewerCore::GvvGLSceneInterface* _scene;

	/******************************** METHODS *********************************/

};

} // namespace GvViewerGui

#endif // !_GVV_PIPELINE_INTERFACE_VIEWER_H_
