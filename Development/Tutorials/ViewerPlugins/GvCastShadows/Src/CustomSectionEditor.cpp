/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2013 INRIA <http://www.inria.fr/>
 *
 * Authors : GigaVoxels Team
 *
 * GigaVoxels is distributed under a dual-license scheme.
 * You can obtain a specific license from Inria at gigavoxels-licensing@inria.fr.
 * Otherwise the default license is the GPL version 3.
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

#include "CustomSectionEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qt
#include <QtCore/QUrl>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QtGui/QVBoxLayout>
#include <QtGui/QToolBar>

// GvViewer
#include "Gvv3DWindow.h"
#include "GvvPipelineInterfaceViewer.h"
#include "GvvPluginManager.h"

#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "Gvv3DWindow.h"
#include "GvvPipelineInterfaceViewer.h"
#include "GvvPipelineInterface.h"

// Project
#include "SampleCore.h"

// STL
#include <iostream>

// System
#include <cassert>

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
 *
 * @param pParent parent widget
 * @param pFlags the window flags
 ******************************************************************************/
CustomSectionEditor::CustomSectionEditor( QWidget* pParent, Qt::WindowFlags pFlags )
:	GvvSectionEditor( pParent, pFlags )
{
	setupUi( this );

	// Editor name
	setName( tr( "Shadow Casting" ) );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
CustomSectionEditor::~CustomSectionEditor()
{
}

/******************************************************************************
 * Populates this editor with the specified browsable
 *
 * @param pBrowsable specifies the browsable to be edited
 ******************************************************************************/
void CustomSectionEditor::populate( GvViewerCore::GvvBrowsable* pBrowsable )
{
	assert( pBrowsable != NULL );
	SampleCore* pipeline = dynamic_cast< SampleCore* >( pBrowsable );
	assert( pipeline != NULL );
	if ( pipeline != NULL )
	{
		// 3D model parameters
		_3DModelLineEdit->setText( pipeline->get3DModelFilename().c_str() );

		// -- [ Transform ] --
		float x;
		float y;
		float z;
		float w;
		// Translation
		pipeline->getShadowReceiverTranslation( x, y, z );
		_xTranslationSpinBox->setValue( x );
		_yTranslationSpinBox->setValue( y );
		_zTranslationSpinBox->setValue( z );
		// Rotation
		pipeline->getShadowReceiverRotation( x, y, z, w );
		_angleRotationSpinBox->setValue( x );
		_xRotationSpinBox->setValue( y );
		_yRotationSpinBox->setValue( z );
		_zRotationSpinBox->setValue( w );
		// Scale
		pipeline->getShadowReceiverScale( x );
		_uniformScaleSpinBox->setValue( x );

		// Shadow Caster

		// 3D model parameters
		_3DModelLineEdit_2->setText( pipeline->get3DModelFilename().c_str() );

		// -- [ Transform ] --
		/*float x;
		float y;
		float z;
		float w;*/
		// Translation
		pipeline->getShadowCasterTranslation( x, y, z );
		_xTranslationSpinBox_2->setValue( x );
		_yTranslationSpinBox_2->setValue( y );
		_zTranslationSpinBox_2->setValue( z );
		// Rotation
		pipeline->getShadowCasterRotation( x, y, z, w );
		_angleRotationSpinBox_2->setValue( x );
		_xRotationSpinBox_2->setValue( y );
		_yRotationSpinBox_2->setValue( z );
		_zRotationSpinBox_2->setValue( w );
		// Scale
		pipeline->getShadowCasterScale( x );
		_uniformScaleSpinBox_2->setValue( x );
	}
}

/******************************************************************************
 * Slot called when the 3D model file button has been clicked (released)
 ******************************************************************************/
void CustomSectionEditor::on__3DModelToolButton_released()
{
	// Try to open 3D model
	QString filename = QFileDialog::getOpenFileName( this, "Choose a 3D model/scene file", QString( "." ), tr( "3D Model Files (*.obj *.dae *.3ds)" ) );
	if ( ! filename.isEmpty() )
	{
		_3DModelLineEdit->setText( filename );
		_3DModelLineEdit->setToolTip( filename );

		// Update the GigaVoxels pipeline
		GvvApplication& application = GvvApplication::get();
		GvvMainWindow* mainWindow = application.getMainWindow();
		Gvv3DWindow* window3D = mainWindow->get3DWindow();
		GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

		SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
		assert( sampleCore != NULL );

		sampleCore->set3DModelFilename( filename.toStdString() );
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void CustomSectionEditor::on__xTranslationSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
			assert( sampleCore != NULL );
			
			float x;
			float y;
			float z;
			sampleCore->getShadowReceiverTranslation( x, y, z );
			sampleCore->setShadowReceiverTranslation( pValue, y, z );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void CustomSectionEditor::on__yTranslationSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
			assert( sampleCore != NULL );

			float x;
			float y;
			float z;
			sampleCore->getShadowReceiverTranslation( x, y, z );
			sampleCore->setShadowReceiverTranslation( x, pValue, z );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void CustomSectionEditor::on__zTranslationSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
			assert( sampleCore != NULL );

			float x;
			float y;
			float z;
			sampleCore->getShadowReceiverTranslation( x, y, z );
			sampleCore->setShadowReceiverTranslation( x, y, pValue );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void CustomSectionEditor::on__xRotationSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
			assert( sampleCore != NULL );

			float angle;
			float x;
			float y;
			float z;
			sampleCore->getShadowReceiverRotation( angle, x, y, z );
			sampleCore->setShadowReceiverRotation( angle, pValue, y, z );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void CustomSectionEditor::on__yRotationSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
			assert( sampleCore != NULL );

			float angle;
			float x;
			float y;
			float z;
			sampleCore->getShadowReceiverRotation( angle, x, y, z );
			sampleCore->setShadowReceiverRotation( angle, x, pValue, z );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void CustomSectionEditor::on__zRotationSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
			assert( sampleCore != NULL );

			float angle;
			float x;
			float y;
			float z;
			sampleCore->getShadowReceiverRotation( angle, x, y, z );
			sampleCore->setShadowReceiverRotation( angle, x, y, pValue );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void CustomSectionEditor::on__angleRotationSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
			assert( sampleCore != NULL );

			float angle;
			float x;
			float y;
			float z;
			sampleCore->getShadowReceiverRotation( angle, x, y, z );
			sampleCore->setShadowReceiverRotation( pValue, x, y, z );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void CustomSectionEditor::on__uniformScaleSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
			assert( sampleCore != NULL );

			sampleCore->setShadowReceiverScale( pValue );
		}
	}
}

// Shadow Caster

/******************************************************************************
 * Slot called when the 3D model file button has been clicked (released)
 ******************************************************************************/
void CustomSectionEditor::on__3DModelToolButton_2_released()
{
	// Try to open 3D model
	QString filename = QFileDialog::getOpenFileName( this, "Choose a 3D model/scene file", QString( "." ), tr( "3D Model Files (*.obj *.dae *.3ds)" ) );
	if ( ! filename.isEmpty() )
	{
		_3DModelLineEdit->setText( filename );
		_3DModelLineEdit->setToolTip( filename );

		// Update the GigaVoxels pipeline
		GvvApplication& application = GvvApplication::get();
		GvvMainWindow* mainWindow = application.getMainWindow();
		Gvv3DWindow* window3D = mainWindow->get3DWindow();
		GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

		SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
		assert( sampleCore != NULL );

		sampleCore->setShadowCaster3DModelFilename( filename.toStdString() );
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void CustomSectionEditor::on__xTranslationSpinBox_2_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
			assert( sampleCore != NULL );
			
			float x;
			float y;
			float z;
			sampleCore->getShadowCasterTranslation( x, y, z );
			sampleCore->setShadowCasterTranslation( pValue, y, z );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void CustomSectionEditor::on__yTranslationSpinBox_2_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
			assert( sampleCore != NULL );

			float x;
			float y;
			float z;
			sampleCore->getShadowCasterTranslation( x, y, z );
			sampleCore->setShadowCasterTranslation( x, pValue, z );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void CustomSectionEditor::on__zTranslationSpinBox_2_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
			assert( sampleCore != NULL );

			float x;
			float y;
			float z;
			sampleCore->getShadowCasterTranslation( x, y, z );
			sampleCore->setShadowCasterTranslation( x, y, pValue );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void CustomSectionEditor::on__xRotationSpinBox_2_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
			assert( sampleCore != NULL );

			float angle;
			float x;
			float y;
			float z;
			sampleCore->getShadowCasterRotation( angle, x, y, z );
			sampleCore->setShadowCasterRotation( angle, pValue, y, z );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void CustomSectionEditor::on__yRotationSpinBox_2_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
			assert( sampleCore != NULL );

			float angle;
			float x;
			float y;
			float z;
			sampleCore->getShadowCasterRotation( angle, x, y, z );
			sampleCore->setShadowCasterRotation( angle, x, pValue, z );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void CustomSectionEditor::on__zRotationSpinBox_2_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
			assert( sampleCore != NULL );

			float angle;
			float x;
			float y;
			float z;
			sampleCore->getShadowCasterRotation( angle, x, y, z );
			sampleCore->setShadowCasterRotation( angle, x, y, pValue );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void CustomSectionEditor::on__angleRotationSpinBox_2_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
			assert( sampleCore != NULL );

			float angle;
			float x;
			float y;
			float z;
			sampleCore->getShadowCasterRotation( angle, x, y, z );
			sampleCore->setShadowCasterRotation( pValue, x, y, z );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void CustomSectionEditor::on__uniformScaleSpinBox_2_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
			assert( sampleCore != NULL );

			sampleCore->setShadowCasterScale( pValue );
		}
	}
}
