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

#include "GvvPreferencesDialog.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "Gvv3DWindow.h"
#include "GvvPipelineInterfaceViewer.h"
#include "GvvPipelineInterface.h"

// Qt
#include <QColorDialog>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerGui;

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
 * Default constructor.
 ******************************************************************************/
GvvPreferencesDialog::GvvPreferencesDialog( QWidget* pParent ) 
:	QDialog( pParent )
{
	//** Set the name
	setAccessibleName( qApp->translate( "GvvPreferencesDialog", "Preferences Dialog" ) );

	//** Initalizes the dialog
	setupUi( this );

	// Initialize widget
	blockSignals( true );
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			bool showNodeHasBrickTerminal;
			bool showNodeHasBrickNotTerminal;
			bool showNodeIsBrickNotInCache;
			bool showNodeEmptyOrConstant;
			float nodeHasBrickTerminalColorR;
			float nodeHasBrickTerminalColorG;
			float nodeHasBrickTerminalColorB;
			float nodeHasBrickTerminalColorA;
			float nodeHasBrickNotTerminalColorR;
			float nodeHasBrickNotTerminalColorG;
			float nodeHasBrickNotTerminalColorB;
			float nodeHasBrickNotTerminalColorA;
			float nodeIsBrickNotInCacheColorR;
			float nodeIsBrickNotInCacheColorG;
			float nodeIsBrickNotInCacheColorB;
			float nodeIsBrickNotInCacheColorA;
			float nodeEmptyOrConstantColorR;
			float nodeEmptyOrConstantColorG;
			float nodeEmptyOrConstantColorB;
			float nodeEmptyOrConstantColorA;

			pipeline->getDataStructureAppearance( showNodeHasBrickTerminal, showNodeHasBrickNotTerminal, showNodeIsBrickNotInCache, showNodeEmptyOrConstant
				, nodeHasBrickTerminalColorR, nodeHasBrickTerminalColorG, nodeHasBrickTerminalColorB, nodeHasBrickTerminalColorA
				, nodeHasBrickNotTerminalColorR, nodeHasBrickNotTerminalColorG, nodeHasBrickNotTerminalColorB, nodeHasBrickNotTerminalColorA
				, nodeIsBrickNotInCacheColorR, nodeIsBrickNotInCacheColorG, nodeIsBrickNotInCacheColorB, nodeIsBrickNotInCacheColorA
				, nodeEmptyOrConstantColorR, nodeEmptyOrConstantColorG, nodeEmptyOrConstantColorB, nodeEmptyOrConstantColorA );

			_nodeHasBrickTerminalCheckBox->setChecked( showNodeHasBrickTerminal );
			_nodeHasBrickNotTerminalCheckBox->setChecked( showNodeHasBrickTerminal );
			_nodeIsBrickNotInCacheCheckBox->setChecked( showNodeHasBrickTerminal );
			_nodeEmptyOrConstantCheckBox->setChecked( showNodeEmptyOrConstant );
		}
	}
	blockSignals( false );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvPreferencesDialog::~GvvPreferencesDialog()
{
}

/******************************************************************************
 * Slot called when 3D window background color tool button is released
 ******************************************************************************/
void GvvPreferencesDialog::on__3DWindowBackgroundColorToolButton_released()
{
	QColor color = QColorDialog::getColor( Qt::white, this, tr( "Background Color" ), QColorDialog::ShowAlphaChannel );
	if ( color.isValid() )
	{
		GvvApplication& application = GvvApplication::get();
		GvvMainWindow* mainWindow = application.getMainWindow();
		Gvv3DWindow* window3D = mainWindow->get3DWindow();
		GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
		if ( pipelineViewer != NULL )
		{
			pipelineViewer->setBackgroundColor( color );
		}
	}
}

/******************************************************************************
 * Slot called when data structure appearance color tool button is released
 ******************************************************************************/
void GvvPreferencesDialog::on__nodeHasBrickTerminalColorToolButton_released()
{
	QColor color = QColorDialog::getColor( Qt::white, this );
	if ( color.isValid() )
	{
		GvvApplication& application = GvvApplication::get();
		GvvMainWindow* mainWindow = application.getMainWindow();
		Gvv3DWindow* window3D = mainWindow->get3DWindow();
		GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
		if ( pipelineViewer != NULL )
		{
			GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
			if ( pipeline != NULL )
			{
				bool showNodeHasBrickTerminal;
				bool showNodeHasBrickNotTerminal;
				bool showNodeIsBrickNotInCache;
				bool showNodeEmptyOrConstant;
				float nodeHasBrickTerminalColorR;
				float nodeHasBrickTerminalColorG;
				float nodeHasBrickTerminalColorB;
				float nodeHasBrickTerminalColorA;
				float nodeHasBrickNotTerminalColorR;
				float nodeHasBrickNotTerminalColorG;
				float nodeHasBrickNotTerminalColorB;
				float nodeHasBrickNotTerminalColorA;
				float nodeIsBrickNotInCacheColorR;
				float nodeIsBrickNotInCacheColorG;
				float nodeIsBrickNotInCacheColorB;
				float nodeIsBrickNotInCacheColorA;
				float nodeEmptyOrConstantColorR;
				float nodeEmptyOrConstantColorG;
				float nodeEmptyOrConstantColorB;
				float nodeEmptyOrConstantColorA;
				
				pipeline->getDataStructureAppearance( showNodeHasBrickTerminal, showNodeHasBrickNotTerminal, showNodeIsBrickNotInCache, showNodeEmptyOrConstant
													, nodeHasBrickTerminalColorR, nodeHasBrickTerminalColorG, nodeHasBrickTerminalColorB, nodeHasBrickTerminalColorA
													, nodeHasBrickNotTerminalColorR, nodeHasBrickNotTerminalColorG, nodeHasBrickNotTerminalColorB, nodeHasBrickNotTerminalColorA
													, nodeIsBrickNotInCacheColorR, nodeIsBrickNotInCacheColorG, nodeIsBrickNotInCacheColorB, nodeIsBrickNotInCacheColorA
													, nodeEmptyOrConstantColorR, nodeEmptyOrConstantColorG, nodeEmptyOrConstantColorB, nodeEmptyOrConstantColorA );

				nodeHasBrickTerminalColorR = static_cast< float >( color.red() ) / 255.f;
				nodeHasBrickTerminalColorG = static_cast< float >( color.green() ) / 255.f;
				nodeHasBrickTerminalColorB = static_cast< float >( color.blue() ) / 255.f;
				nodeHasBrickTerminalColorA = static_cast< float >( color.alpha() ) / 255.f;

				// Set the appearance of the N-tree (octree) of the data structure
				pipeline->setDataStructureAppearance( showNodeHasBrickTerminal, showNodeHasBrickNotTerminal, showNodeIsBrickNotInCache, showNodeEmptyOrConstant
													, nodeHasBrickTerminalColorR, nodeHasBrickTerminalColorG, nodeHasBrickTerminalColorB, nodeHasBrickTerminalColorA
													, nodeHasBrickNotTerminalColorR, nodeHasBrickNotTerminalColorG, nodeHasBrickNotTerminalColorB, nodeHasBrickNotTerminalColorA
													, nodeIsBrickNotInCacheColorR, nodeIsBrickNotInCacheColorG, nodeIsBrickNotInCacheColorB, nodeIsBrickNotInCacheColorA
													, nodeEmptyOrConstantColorR, nodeEmptyOrConstantColorG, nodeEmptyOrConstantColorB, nodeEmptyOrConstantColorA );
			}
		}
	}
}

/******************************************************************************
 * Slot called when data structure appearance color tool button is released
 ******************************************************************************/
void GvvPreferencesDialog::on__nodeHasBrickNotTerminalColorToolButton_released()
{
	QColor color = QColorDialog::getColor( Qt::white, this );
	if ( color.isValid() )
	{
		GvvApplication& application = GvvApplication::get();
		GvvMainWindow* mainWindow = application.getMainWindow();
		Gvv3DWindow* window3D = mainWindow->get3DWindow();
		GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
		if ( pipelineViewer != NULL )
		{
			GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
			if ( pipeline != NULL )
			{
				bool showNodeHasBrickTerminal;
				bool showNodeHasBrickNotTerminal;
				bool showNodeIsBrickNotInCache;
				bool showNodeEmptyOrConstant;
				float nodeHasBrickTerminalColorR;
				float nodeHasBrickTerminalColorG;
				float nodeHasBrickTerminalColorB;
				float nodeHasBrickTerminalColorA;
				float nodeHasBrickNotTerminalColorR;
				float nodeHasBrickNotTerminalColorG;
				float nodeHasBrickNotTerminalColorB;
				float nodeHasBrickNotTerminalColorA;
				float nodeIsBrickNotInCacheColorR;
				float nodeIsBrickNotInCacheColorG;
				float nodeIsBrickNotInCacheColorB;
				float nodeIsBrickNotInCacheColorA;
				float nodeEmptyOrConstantColorR;
				float nodeEmptyOrConstantColorG;
				float nodeEmptyOrConstantColorB;
				float nodeEmptyOrConstantColorA;
				
				pipeline->getDataStructureAppearance( showNodeHasBrickTerminal, showNodeHasBrickNotTerminal, showNodeIsBrickNotInCache, showNodeEmptyOrConstant
													, nodeHasBrickTerminalColorR, nodeHasBrickTerminalColorG, nodeHasBrickTerminalColorB, nodeHasBrickTerminalColorA
													, nodeHasBrickNotTerminalColorR, nodeHasBrickNotTerminalColorG, nodeHasBrickNotTerminalColorB, nodeHasBrickNotTerminalColorA
													, nodeIsBrickNotInCacheColorR, nodeIsBrickNotInCacheColorG, nodeIsBrickNotInCacheColorB, nodeIsBrickNotInCacheColorA
													, nodeEmptyOrConstantColorR, nodeEmptyOrConstantColorG, nodeEmptyOrConstantColorB, nodeEmptyOrConstantColorA );

				nodeHasBrickNotTerminalColorR = static_cast< float >( color.red() ) / 255.f;
				nodeHasBrickNotTerminalColorG = static_cast< float >( color.green() ) / 255.f;
				nodeHasBrickNotTerminalColorB = static_cast< float >( color.blue() ) / 255.f;
				nodeHasBrickNotTerminalColorA = static_cast< float >( color.alpha() ) / 255.f;

				// Set the appearance of the N-tree (octree) of the data structure
				pipeline->setDataStructureAppearance( showNodeHasBrickTerminal, showNodeHasBrickNotTerminal, showNodeIsBrickNotInCache, showNodeEmptyOrConstant
													, nodeHasBrickTerminalColorR, nodeHasBrickTerminalColorG, nodeHasBrickTerminalColorB, nodeHasBrickTerminalColorA
													, nodeHasBrickNotTerminalColorR, nodeHasBrickNotTerminalColorG, nodeHasBrickNotTerminalColorB, nodeHasBrickNotTerminalColorA
													, nodeIsBrickNotInCacheColorR, nodeIsBrickNotInCacheColorG, nodeIsBrickNotInCacheColorB, nodeIsBrickNotInCacheColorA
													, nodeEmptyOrConstantColorR, nodeEmptyOrConstantColorG, nodeEmptyOrConstantColorB, nodeEmptyOrConstantColorA );
			}
		}
	}
}

/******************************************************************************
 * Slot called when data structure appearance color tool button is released
 ******************************************************************************/
void GvvPreferencesDialog::on__nodeIsBrickNotInCacheColorToolButton_released()
{
	QColor color = QColorDialog::getColor( Qt::white, this );
	if ( color.isValid() )
	{
		GvvApplication& application = GvvApplication::get();
		GvvMainWindow* mainWindow = application.getMainWindow();
		Gvv3DWindow* window3D = mainWindow->get3DWindow();
		GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
		if ( pipelineViewer != NULL )
		{
			GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
			if ( pipeline != NULL )
			{
				bool showNodeHasBrickTerminal;
				bool showNodeHasBrickNotTerminal;
				bool showNodeIsBrickNotInCache;
				bool showNodeEmptyOrConstant;
				float nodeHasBrickTerminalColorR;
				float nodeHasBrickTerminalColorG;
				float nodeHasBrickTerminalColorB;
				float nodeHasBrickTerminalColorA;
				float nodeHasBrickNotTerminalColorR;
				float nodeHasBrickNotTerminalColorG;
				float nodeHasBrickNotTerminalColorB;
				float nodeHasBrickNotTerminalColorA;
				float nodeIsBrickNotInCacheColorR;
				float nodeIsBrickNotInCacheColorG;
				float nodeIsBrickNotInCacheColorB;
				float nodeIsBrickNotInCacheColorA;
				float nodeEmptyOrConstantColorR;
				float nodeEmptyOrConstantColorG;
				float nodeEmptyOrConstantColorB;
				float nodeEmptyOrConstantColorA;
				
				pipeline->getDataStructureAppearance( showNodeHasBrickTerminal, showNodeHasBrickNotTerminal, showNodeIsBrickNotInCache, showNodeEmptyOrConstant
													, nodeHasBrickTerminalColorR, nodeHasBrickTerminalColorG, nodeHasBrickTerminalColorB, nodeHasBrickTerminalColorA
													, nodeHasBrickNotTerminalColorR, nodeHasBrickNotTerminalColorG, nodeHasBrickNotTerminalColorB, nodeHasBrickNotTerminalColorA
													, nodeIsBrickNotInCacheColorR, nodeIsBrickNotInCacheColorG, nodeIsBrickNotInCacheColorB, nodeIsBrickNotInCacheColorA
													, nodeEmptyOrConstantColorR, nodeEmptyOrConstantColorG, nodeEmptyOrConstantColorB, nodeEmptyOrConstantColorA );

				nodeIsBrickNotInCacheColorR = static_cast< float >( color.red() ) / 255.f;
				nodeIsBrickNotInCacheColorG = static_cast< float >( color.green() ) / 255.f;
				nodeIsBrickNotInCacheColorB = static_cast< float >( color.blue() ) / 255.f;
				nodeIsBrickNotInCacheColorA = static_cast< float >( color.alpha() ) / 255.f;

				// Set the appearance of the N-tree (octree) of the data structure
				pipeline->setDataStructureAppearance( showNodeHasBrickTerminal, showNodeHasBrickNotTerminal, showNodeIsBrickNotInCache, showNodeEmptyOrConstant
													, nodeHasBrickTerminalColorR, nodeHasBrickTerminalColorG, nodeHasBrickTerminalColorB, nodeHasBrickTerminalColorA
													, nodeHasBrickNotTerminalColorR, nodeHasBrickNotTerminalColorG, nodeHasBrickNotTerminalColorB, nodeHasBrickNotTerminalColorA
													, nodeIsBrickNotInCacheColorR, nodeIsBrickNotInCacheColorG, nodeIsBrickNotInCacheColorB, nodeIsBrickNotInCacheColorA
													, nodeEmptyOrConstantColorR, nodeEmptyOrConstantColorG, nodeEmptyOrConstantColorB, nodeEmptyOrConstantColorA );
			}
		}
	}
}

/******************************************************************************
 * Slot called when data structure appearance color tool button is released
 ******************************************************************************/
void GvvPreferencesDialog::on__nodeEmptyOrConstantColorToolButton_released()
{
	QColor color = QColorDialog::getColor( Qt::white, this );
	if ( color.isValid() )
	{
		GvvApplication& application = GvvApplication::get();
		GvvMainWindow* mainWindow = application.getMainWindow();
		Gvv3DWindow* window3D = mainWindow->get3DWindow();
		GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
		if ( pipelineViewer != NULL )
		{
			GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
			if ( pipeline != NULL )
			{
				bool showNodeHasBrickTerminal;
				bool showNodeHasBrickNotTerminal;
				bool showNodeIsBrickNotInCache;
				bool showNodeEmptyOrConstant;
				float nodeHasBrickTerminalColorR;
				float nodeHasBrickTerminalColorG;
				float nodeHasBrickTerminalColorB;
				float nodeHasBrickTerminalColorA;
				float nodeHasBrickNotTerminalColorR;
				float nodeHasBrickNotTerminalColorG;
				float nodeHasBrickNotTerminalColorB;
				float nodeHasBrickNotTerminalColorA;
				float nodeIsBrickNotInCacheColorR;
				float nodeIsBrickNotInCacheColorG;
				float nodeIsBrickNotInCacheColorB;
				float nodeIsBrickNotInCacheColorA;
				float nodeEmptyOrConstantColorR;
				float nodeEmptyOrConstantColorG;
				float nodeEmptyOrConstantColorB;
				float nodeEmptyOrConstantColorA;
				
				pipeline->getDataStructureAppearance( showNodeHasBrickTerminal, showNodeHasBrickNotTerminal, showNodeIsBrickNotInCache, showNodeEmptyOrConstant
													, nodeHasBrickTerminalColorR, nodeHasBrickTerminalColorG, nodeHasBrickTerminalColorB, nodeHasBrickTerminalColorA
													, nodeHasBrickNotTerminalColorR, nodeHasBrickNotTerminalColorG, nodeHasBrickNotTerminalColorB, nodeHasBrickNotTerminalColorA
													, nodeIsBrickNotInCacheColorR, nodeIsBrickNotInCacheColorG, nodeIsBrickNotInCacheColorB, nodeIsBrickNotInCacheColorA
													, nodeEmptyOrConstantColorR, nodeEmptyOrConstantColorG, nodeEmptyOrConstantColorB, nodeEmptyOrConstantColorA );

				nodeEmptyOrConstantColorR = static_cast< float >( color.red() ) / 255.f;
				nodeEmptyOrConstantColorG = static_cast< float >( color.green() ) / 255.f;
				nodeEmptyOrConstantColorB = static_cast< float >( color.blue() ) / 255.f;
				nodeEmptyOrConstantColorA = static_cast< float >( color.alpha() ) / 255.f;

				// Set the appearance of the N-tree (octree) of the data structure
				pipeline->setDataStructureAppearance( showNodeHasBrickTerminal, showNodeHasBrickNotTerminal, showNodeIsBrickNotInCache, showNodeEmptyOrConstant
													, nodeHasBrickTerminalColorR, nodeHasBrickTerminalColorG, nodeHasBrickTerminalColorB, nodeHasBrickTerminalColorA
													, nodeHasBrickNotTerminalColorR, nodeHasBrickNotTerminalColorG, nodeHasBrickNotTerminalColorB, nodeHasBrickNotTerminalColorA
													, nodeIsBrickNotInCacheColorR, nodeIsBrickNotInCacheColorG, nodeIsBrickNotInCacheColorB, nodeIsBrickNotInCacheColorA
													, nodeEmptyOrConstantColorR, nodeEmptyOrConstantColorG, nodeEmptyOrConstantColorB, nodeEmptyOrConstantColorA );
			}
		}
	}
}

/******************************************************************************
 * Slot called when data structure appearance check box is toggled
 ******************************************************************************/
void GvvPreferencesDialog::on__nodeHasBrickTerminalCheckBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			bool showNodeHasBrickTerminal;
			bool showNodeHasBrickNotTerminal;
			bool showNodeIsBrickNotInCache;
			bool showNodeEmptyOrConstant;
			float nodeHasBrickTerminalColorR;
			float nodeHasBrickTerminalColorG;
			float nodeHasBrickTerminalColorB;
			float nodeHasBrickTerminalColorA;
			float nodeHasBrickNotTerminalColorR;
			float nodeHasBrickNotTerminalColorG;
			float nodeHasBrickNotTerminalColorB;
			float nodeHasBrickNotTerminalColorA;
			float nodeIsBrickNotInCacheColorR;
			float nodeIsBrickNotInCacheColorG;
			float nodeIsBrickNotInCacheColorB;
			float nodeIsBrickNotInCacheColorA;
			float nodeEmptyOrConstantColorR;
			float nodeEmptyOrConstantColorG;
			float nodeEmptyOrConstantColorB;
			float nodeEmptyOrConstantColorA;

			pipeline->getDataStructureAppearance( showNodeHasBrickTerminal, showNodeHasBrickNotTerminal, showNodeIsBrickNotInCache, showNodeEmptyOrConstant
													, nodeHasBrickTerminalColorR, nodeHasBrickTerminalColorG, nodeHasBrickTerminalColorB, nodeHasBrickTerminalColorA
													, nodeHasBrickNotTerminalColorR, nodeHasBrickNotTerminalColorG, nodeHasBrickNotTerminalColorB, nodeHasBrickNotTerminalColorA
													, nodeIsBrickNotInCacheColorR, nodeIsBrickNotInCacheColorG, nodeIsBrickNotInCacheColorB, nodeIsBrickNotInCacheColorA
													, nodeEmptyOrConstantColorR, nodeEmptyOrConstantColorG, nodeEmptyOrConstantColorB, nodeEmptyOrConstantColorA );

			// Set the appearance of the N-tree (octree) of the data structure
			pipeline->setDataStructureAppearance( pChecked, showNodeHasBrickNotTerminal, showNodeIsBrickNotInCache, showNodeEmptyOrConstant
													, nodeHasBrickTerminalColorR, nodeHasBrickTerminalColorG, nodeHasBrickTerminalColorB, nodeHasBrickTerminalColorA
													, nodeHasBrickNotTerminalColorR, nodeHasBrickNotTerminalColorG, nodeHasBrickNotTerminalColorB, nodeHasBrickNotTerminalColorA
													, nodeIsBrickNotInCacheColorR, nodeIsBrickNotInCacheColorG, nodeIsBrickNotInCacheColorB, nodeIsBrickNotInCacheColorA
													, nodeEmptyOrConstantColorR, nodeEmptyOrConstantColorG, nodeEmptyOrConstantColorB, nodeEmptyOrConstantColorA );
		}
	}
}
	
/******************************************************************************
 * Slot called when data structure appearance check box is toggled
 ******************************************************************************/
void GvvPreferencesDialog::on__nodeHasBrickNotTerminalCheckBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			bool showNodeHasBrickTerminal;
			bool showNodeHasBrickNotTerminal;
			bool showNodeIsBrickNotInCache;
			bool showNodeEmptyOrConstant;
			float nodeHasBrickTerminalColorR;
			float nodeHasBrickTerminalColorG;
			float nodeHasBrickTerminalColorB;
			float nodeHasBrickTerminalColorA;
			float nodeHasBrickNotTerminalColorR;
			float nodeHasBrickNotTerminalColorG;
			float nodeHasBrickNotTerminalColorB;
			float nodeHasBrickNotTerminalColorA;
			float nodeIsBrickNotInCacheColorR;
			float nodeIsBrickNotInCacheColorG;
			float nodeIsBrickNotInCacheColorB;
			float nodeIsBrickNotInCacheColorA;
			float nodeEmptyOrConstantColorR;
			float nodeEmptyOrConstantColorG;
			float nodeEmptyOrConstantColorB;
			float nodeEmptyOrConstantColorA;

			pipeline->getDataStructureAppearance( showNodeHasBrickTerminal, showNodeHasBrickNotTerminal, showNodeIsBrickNotInCache, showNodeEmptyOrConstant
													, nodeHasBrickTerminalColorR, nodeHasBrickTerminalColorG, nodeHasBrickTerminalColorB, nodeHasBrickTerminalColorA
													, nodeHasBrickNotTerminalColorR, nodeHasBrickNotTerminalColorG, nodeHasBrickNotTerminalColorB, nodeHasBrickNotTerminalColorA
													, nodeIsBrickNotInCacheColorR, nodeIsBrickNotInCacheColorG, nodeIsBrickNotInCacheColorB, nodeIsBrickNotInCacheColorA
													, nodeEmptyOrConstantColorR, nodeEmptyOrConstantColorG, nodeEmptyOrConstantColorB, nodeEmptyOrConstantColorA );

			// Set the appearance of the N-tree (octree) of the data structure
			pipeline->setDataStructureAppearance( showNodeHasBrickTerminal, pChecked, showNodeIsBrickNotInCache, showNodeEmptyOrConstant
				, nodeHasBrickTerminalColorR, nodeHasBrickTerminalColorG, nodeHasBrickTerminalColorB, nodeHasBrickTerminalColorA
				, nodeHasBrickNotTerminalColorR, nodeHasBrickNotTerminalColorG, nodeHasBrickNotTerminalColorB, nodeHasBrickNotTerminalColorA
				, nodeIsBrickNotInCacheColorR, nodeIsBrickNotInCacheColorG, nodeIsBrickNotInCacheColorB, nodeIsBrickNotInCacheColorA
				, nodeEmptyOrConstantColorR, nodeEmptyOrConstantColorG, nodeEmptyOrConstantColorB, nodeEmptyOrConstantColorA );
		}
	}
}

/******************************************************************************
 * Slot called when data structure appearance check box is toggled
 ******************************************************************************/
void GvvPreferencesDialog::on__nodeIsBrickNotInCacheCheckBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			bool showNodeHasBrickTerminal;
			bool showNodeHasBrickNotTerminal;
			bool showNodeIsBrickNotInCache;
			bool showNodeEmptyOrConstant;
			float nodeHasBrickTerminalColorR;
			float nodeHasBrickTerminalColorG;
			float nodeHasBrickTerminalColorB;
			float nodeHasBrickTerminalColorA;
			float nodeHasBrickNotTerminalColorR;
			float nodeHasBrickNotTerminalColorG;
			float nodeHasBrickNotTerminalColorB;
			float nodeHasBrickNotTerminalColorA;
			float nodeIsBrickNotInCacheColorR;
			float nodeIsBrickNotInCacheColorG;
			float nodeIsBrickNotInCacheColorB;
			float nodeIsBrickNotInCacheColorA;
			float nodeEmptyOrConstantColorR;
			float nodeEmptyOrConstantColorG;
			float nodeEmptyOrConstantColorB;
			float nodeEmptyOrConstantColorA;

			pipeline->getDataStructureAppearance( showNodeHasBrickTerminal, showNodeHasBrickNotTerminal, showNodeIsBrickNotInCache, showNodeEmptyOrConstant
													, nodeHasBrickTerminalColorR, nodeHasBrickTerminalColorG, nodeHasBrickTerminalColorB, nodeHasBrickTerminalColorA
													, nodeHasBrickNotTerminalColorR, nodeHasBrickNotTerminalColorG, nodeHasBrickNotTerminalColorB, nodeHasBrickNotTerminalColorA
													, nodeIsBrickNotInCacheColorR, nodeIsBrickNotInCacheColorG, nodeIsBrickNotInCacheColorB, nodeIsBrickNotInCacheColorA
													, nodeEmptyOrConstantColorR, nodeEmptyOrConstantColorG, nodeEmptyOrConstantColorB, nodeEmptyOrConstantColorA );

			// Set the appearance of the N-tree (octree) of the data structure
			pipeline->setDataStructureAppearance( showNodeHasBrickTerminal, showNodeHasBrickNotTerminal, pChecked, showNodeEmptyOrConstant
				, nodeHasBrickTerminalColorR, nodeHasBrickTerminalColorG, nodeHasBrickTerminalColorB, nodeHasBrickTerminalColorA
				, nodeHasBrickNotTerminalColorR, nodeHasBrickNotTerminalColorG, nodeHasBrickNotTerminalColorB, nodeHasBrickNotTerminalColorA
				, nodeIsBrickNotInCacheColorR, nodeIsBrickNotInCacheColorG, nodeIsBrickNotInCacheColorB, nodeIsBrickNotInCacheColorA
				, nodeEmptyOrConstantColorR, nodeEmptyOrConstantColorG, nodeEmptyOrConstantColorB, nodeEmptyOrConstantColorA );
		}
	}
}

/******************************************************************************
 * Slot called when data structure appearance check box is toggled
 ******************************************************************************/
void GvvPreferencesDialog::on__nodeEmptyOrConstantCheckBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			bool showNodeHasBrickTerminal;
			bool showNodeHasBrickNotTerminal;
			bool showNodeIsBrickNotInCache;
			bool showNodeEmptyOrConstant;
			float nodeHasBrickTerminalColorR;
			float nodeHasBrickTerminalColorG;
			float nodeHasBrickTerminalColorB;
			float nodeHasBrickTerminalColorA;
			float nodeHasBrickNotTerminalColorR;
			float nodeHasBrickNotTerminalColorG;
			float nodeHasBrickNotTerminalColorB;
			float nodeHasBrickNotTerminalColorA;
			float nodeIsBrickNotInCacheColorR;
			float nodeIsBrickNotInCacheColorG;
			float nodeIsBrickNotInCacheColorB;
			float nodeIsBrickNotInCacheColorA;
			float nodeEmptyOrConstantColorR;
			float nodeEmptyOrConstantColorG;
			float nodeEmptyOrConstantColorB;
			float nodeEmptyOrConstantColorA;

			pipeline->getDataStructureAppearance( showNodeHasBrickTerminal, showNodeHasBrickNotTerminal, showNodeIsBrickNotInCache, showNodeEmptyOrConstant
													, nodeHasBrickTerminalColorR, nodeHasBrickTerminalColorG, nodeHasBrickTerminalColorB, nodeHasBrickTerminalColorA
													, nodeHasBrickNotTerminalColorR, nodeHasBrickNotTerminalColorG, nodeHasBrickNotTerminalColorB, nodeHasBrickNotTerminalColorA
													, nodeIsBrickNotInCacheColorR, nodeIsBrickNotInCacheColorG, nodeIsBrickNotInCacheColorB, nodeIsBrickNotInCacheColorA
													, nodeEmptyOrConstantColorR, nodeEmptyOrConstantColorG, nodeEmptyOrConstantColorB, nodeEmptyOrConstantColorA );

			// Set the appearance of the N-tree (octree) of the data structure
			pipeline->setDataStructureAppearance( showNodeHasBrickTerminal, showNodeHasBrickNotTerminal, showNodeIsBrickNotInCache, pChecked
				, nodeHasBrickTerminalColorR, nodeHasBrickTerminalColorG, nodeHasBrickTerminalColorB, nodeHasBrickTerminalColorA
				, nodeHasBrickNotTerminalColorR, nodeHasBrickNotTerminalColorG, nodeHasBrickNotTerminalColorB, nodeHasBrickNotTerminalColorA
				, nodeIsBrickNotInCacheColorR, nodeIsBrickNotInCacheColorG, nodeIsBrickNotInCacheColorB, nodeIsBrickNotInCacheColorA
				, nodeEmptyOrConstantColorR, nodeEmptyOrConstantColorG, nodeEmptyOrConstantColorB, nodeEmptyOrConstantColorA );
		}
	}
}
