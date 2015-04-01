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

#ifndef _GVV_TRANSFER_FUNCTION_EDITOR_H_
#define _GVV_TRANSFER_FUNCTION_EDITOR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "GvvPipelineInterface.h"
#include "GvvContextListener.h"

// Qtfe
#include "Qtfe.h"

// Qt
#include <QObject>
#include <QWidget>

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
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * ...
 *
 * @ingroup GvViewerGui
 */
class GVVIEWERGUI_EXPORT GvvTransferFunctionEditor : public QObject, public GvvContextListener
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
	GvvTransferFunctionEditor( QWidget* pParent = NULL, Qt::WindowFlags pFlags = 0 );

	/**
	 * Destructor
	 */
	virtual ~GvvTransferFunctionEditor();

	/**
	 * ...
	 */
	void show();

	///**
	// * ...
	// *
	// * @param pPipeline ...
	// * @param pFlag ...
	// */
	//void connect( GvViewerCore::GvvPipelineInterface* pPipeline, bool pFlag );

	/**
	 * Get the transfer function.
	 *
	 * @return the transfer function
	 */
	Qtfe* getTransferFunction();

	/**
	 * Set the pipeline.
	 *
	 * @param pPipeline The pipeline
	 */
	void setPipeline( GvViewerCore::GvvPipelineInterface* pPipeline );
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************* ATTRIBUTES *******************************/

	/**
	 * The cache editor
	 */
	Qtfe* _editor;

	/**
	 *
	 */
	GvViewerCore::GvvPipelineInterface* _pipeline;

	/******************************** METHODS *********************************/

	/**
	 * This slot is called when the current browsable is changed
	 */
	virtual void onCurrentBrowsableChanged();

	/********************************* SLOTS **********************************/

protected slots:

	/**
	 * Slot called when at least one canal changed
	 */
	void onFunctionChanged();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Copy constructor forbidden.
	 */
	GvvTransferFunctionEditor( const GvvTransferFunctionEditor& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvTransferFunctionEditor& operator=( const GvvTransferFunctionEditor& );
	
};

} // namespace GvViewerGui

#endif
