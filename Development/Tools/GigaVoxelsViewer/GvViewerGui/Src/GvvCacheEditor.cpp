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

#include "GvvCacheEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qt
#include <QUrl>
#include <QFileDialog>
#include <QMessageBox>
#include <QVBoxLayout>
#include <QToolBar>

// GvViewer
#include "Gvv3DWindow.h"
#include "GvvPipelineInterfaceViewer.h"
#include "GvvPluginManager.h"

#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "Gvv3DWindow.h"
#include "GvvPipelineInterfaceViewer.h"
#include "GvvPipelineInterface.h"

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

///******************************************************************************
// * ...
// *
// * @param pParent ...
// * @param pBrowsable ...
// *
// * @return ...
// ******************************************************************************/
//GvvEditor* GvvCacheEditor::create( QWidget* pParent, GvViewerCore::GvvBrowsable* pBrowsable )
//{
//	return new GvvCacheEditor();
//}

/******************************************************************************
 * Default constructor
 ******************************************************************************/
GvvCacheEditor::GvvCacheEditor( QWidget *parent, Qt::WindowFlags flags )
:	GvvSectionEditor( parent, flags )
{
	setupUi( this );

	//_dataTypeGroupBox->setHidden( true );

	// Editor name
	setName( tr( "Data Structure - Cache" ) );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvCacheEditor::~GvvCacheEditor()
{
}

/******************************************************************************
 * Populates this editor with the specified browsable
 *
 * @param pBrowsable specifies the browsable to be edited
 ******************************************************************************/
void GvvCacheEditor::populate( GvViewerCore::GvvBrowsable* pBrowsable )
{
	assert( pBrowsable != NULL );
	GvvPipelineInterface* pipeline = dynamic_cast< GvvPipelineInterface* >( pBrowsable );
	assert( pipeline != NULL );
	if ( pipeline != NULL )
	{
		// Voxels data types
		_dataTableWidget->clear();

		const GvvDataType& dataTypes = pipeline->getDataTypes();
		// TO DO : check sizes...
		const vector< string >& types = dataTypes.getTypes();
		const vector< string >& names = dataTypes.getNames();
		const vector< string >& info = dataTypes.getInfo();

		_dataTableWidget->setColumnCount( 2 );
		QStringList headerLabels;
		headerLabels <<  tr( "Type" ) << tr( "Name" );
		_dataTableWidget->setHorizontalHeaderLabels( headerLabels );

		if ( types.size() > 0 )
		{
			_dataTableWidget->setRowCount( types.size() );
		}
		for ( size_t i = 0; i < types.size(); i++ )
		{
			QTableWidgetItem* typeItem = new QTableWidgetItem( types[ i ].c_str() );
			_dataTableWidget->setItem( i, 0, typeItem );

			QTableWidgetItem* nameItem = NULL;
			if ( names.size() > i )
			{
				nameItem = new QTableWidgetItem( names[ i ].c_str() );
				_dataTableWidget->setItem( i, 1, nameItem );
			}

			if ( info.size() > i )
			{
				typeItem->setToolTip( QString( info[ i ].c_str() ) );
				if ( nameItem != NULL )
				{
					nameItem->setToolTip( QString( info[ i ].c_str() ) );
				}
			}
		}

		{
			unsigned int x;
			unsigned int y;
			unsigned int z;
			pipeline->getDataStructureNodeTileResolution( x, y, z );
			_nodeTileResolutionLineEdit->setText( QString( "%1 x %2 x %3" ).arg( QString::number( x ) ).arg( QString::number( y ) ).arg( QString::number( z ) ) );
			pipeline->getDataStructureBrickResolution( x, y, z );
			_brickResolutionLineEdit->setText( QString( "%1 x %2 x %3" ).arg( QString::number( x ) ).arg( QString::number( y ) ).arg( QString::number( z ) ) );
		}

		_nbSubdivisionsSpinBox->setValue( pipeline->getCacheMaxNbNodeSubdivisions() );
		_nbLoadsSpinBox->setValue( pipeline->getCacheMaxNbBrickLoads() );

		const unsigned int cachePolicy = pipeline->getCachePolicy();
		if ( cachePolicy == 0 )
		{
			_preventReplacingUsedElementsCachePolicyCheckBox->setChecked( false );
			_smoothLoadingCachePolicyGroupBox->setChecked( false );
		}
		else if ( cachePolicy == 1 )
		{
			_preventReplacingUsedElementsCachePolicyCheckBox->setChecked( true );
			_smoothLoadingCachePolicyGroupBox->setChecked( false );
		}
		else if ( cachePolicy == 2 )
		{
			_preventReplacingUsedElementsCachePolicyCheckBox->setChecked( false );
			_smoothLoadingCachePolicyGroupBox->setChecked( true );
		}
		else if ( cachePolicy == 3 )
		{
			_preventReplacingUsedElementsCachePolicyCheckBox->setChecked( true );
			_smoothLoadingCachePolicyGroupBox->setChecked( true );
		}
		else
		{
			assert( false );
		}

		_nodeCacheMemoryLineEdit->setText( QString::number( pipeline->getNodeCacheMemory() ) + " Mo" );
		_brickCacheMemoryLineEdit->setText( QString::number( pipeline->getBrickCacheMemory() ) + " Mo" );

		_nodeCacheCapacityLineEdit->setText( QString::number( pipeline->getNodeCacheCapacity() ) );
		_brickCacheCapacityLineEdit->setText( QString::number( pipeline->getBrickCacheCapacity() ) );

		// Time limit on production
		_timeLimitDoubleSpinBox->setValue( pipeline->getProductionTimeLimit() );
		_timeLimitGroupBox->setChecked( pipeline->isProductionTimeLimited() );
	}
}

/******************************************************************************
 * Slot called when custom cache policy value has changed
 ******************************************************************************/
void GvvCacheEditor::on__preventReplacingUsedElementsCachePolicyCheckBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			unsigned int cachePolicy = 0;
			if ( pChecked )
			{
				if ( _smoothLoadingCachePolicyGroupBox->isChecked() )
				{
					cachePolicy = 3;
				}
				else
				{
					cachePolicy = 1;
				}
			}
			else
			{
				if ( _smoothLoadingCachePolicyGroupBox->isChecked() )
				{
					cachePolicy = 2;
				}
				else
				{
					cachePolicy = 0;
				}
			}
			pipeline->setCachePolicy( cachePolicy );
		}
	}
}

/******************************************************************************
 * Slot called when custom cache policy value has changed
 ******************************************************************************/
void GvvCacheEditor::on__smoothLoadingCachePolicyGroupBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{unsigned int cachePolicy = 0;
			if ( pChecked )
			{
				if ( _preventReplacingUsedElementsCachePolicyCheckBox->isChecked() )
				{
					cachePolicy = 3;
				}
				else
				{
					cachePolicy = 2;
				}
			}
			else
			{
				if ( _preventReplacingUsedElementsCachePolicyCheckBox->isChecked() )
				{
					cachePolicy = 1;
				}
				else
				{
					cachePolicy = 0;
				}
			}
			pipeline->setCachePolicy( cachePolicy );
		}
	}
}

/******************************************************************************
 * Slot called when number of node subdivision value has changed
 ******************************************************************************/
void GvvCacheEditor::on__nbSubdivisionsSpinBox_valueChanged( int i )
{
	// Temporary, waiting for the global context listener...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			pipeline->setCacheMaxNbNodeSubdivisions( i );
		}
	}
}

/******************************************************************************
 * Slot called when number of brick loads value has changed
 ******************************************************************************/
void GvvCacheEditor::on__nbLoadsSpinBox_valueChanged( int i )
{
	// Temporary, waiting for the global context listener...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			pipeline->setCacheMaxNbBrickLoads( i );
		}
	}
}

/******************************************************************************
 * Slot called when custom cache policy value has changed
 ******************************************************************************/
void GvvCacheEditor::on__timeLimitGroupBox_toggled( bool pChecked )
{
	// Temporary, waiting for the global context listener...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			pipeline->useProductionTimeLimit( pChecked );
		}
	}
}

/******************************************************************************
 * Slot called when the time limit value has changed
 ******************************************************************************/
void GvvCacheEditor::on__timeLimitDoubleSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listener...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			pipeline->setProductionTimeLimit( static_cast< float >( pValue ) );
		}
	}
}
