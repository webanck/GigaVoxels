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

#include "GvvTimeBudgetMonitoringEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "GvvTimeBudgetPlotView.h"
#include "GvvPipelineInterface.h"

// STL
#include <iostream>

// System
#include <cassert>

// Qt
#include <QString>

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
 * Constructor
 *
 * pParent ...
 * pFlags ...
 ******************************************************************************/
GvvTimeBudgetMonitoringEditor::GvvTimeBudgetMonitoringEditor( QWidget* pParent, Qt::WindowFlags pFlags )
:	QWidget( pParent, pFlags )
,	_plotView( NULL )
,	_pipeline( NULL )
{
	setupUi( this );

	// Editor name
	setObjectName( tr( "Frame Time Budget Monitor" ) );

	// ------------------------------------------------------

	_plotView =  new GvvTimeBudgetPlotView( this, "Frame Time Budget Monitor" );
	
	QHBoxLayout* layout = new QHBoxLayout();
	_frameTimeViewGroupBox->setLayout( layout );
	assert( layout != NULL );
	if ( layout != NULL )
	{
		layout->addWidget( _plotView );
	}

	// ------------------------------------------------------
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvTimeBudgetMonitoringEditor::~GvvTimeBudgetMonitoringEditor()
{
}

/******************************************************************************
 * Populates this editor with the specified browsable
 *
 * @param pPipeline specifies the pipeline to be edited
 ******************************************************************************/
void GvvTimeBudgetMonitoringEditor::populate( GvvPipelineInterface* pPipeline )
{
	assert( pPipeline != NULL );
	if ( pPipeline != NULL )
	{
		_pipeline = pPipeline;
	
		blockSignals( true );

//		_timeBudgetParametersGroupBox->setChecked( pPipeline->hasRenderingTimeBudget() );
//		_timeBudgetSpinBox->setValue( pPipeline->getRenderingTimeBudget() );
//		_timeBudgetLineEdit->setText( QString::number( 1.f / static_cast< float >( pPipeline->getRenderingTimeBudget() ) * 1000.f ) + QString( " ms" ) );
		_plotView->setTimeBudget( pPipeline->getRenderingTimeBudget() );

		blockSignals( false );
	}
}

/******************************************************************************
 * Draw the specified curve
 *
 * @param pCurve specifies the curve to be drawn
 ******************************************************************************/
void GvvTimeBudgetMonitoringEditor::onCurveChanged( unsigned int pFrame, float pFrameTime )
{
	assert( _plotView != NULL );

	_plotView->onCurveChanged( pFrame, pFrameTime );
}

///******************************************************************************
// * Slot called when time budget parameters group box state has changed
// ******************************************************************************/
//void GvvTimeBudgetMonitoringEditor::on__timeBudgetParametersGroupBox_toggled( bool pChecked )
//{
//	assert( _pipeline  != NULL );
//
//	_pipeline->setTimeBudgetActivated( pChecked );
//}
//
///******************************************************************************
// * Slot called when user requested time budget value has changed
// ******************************************************************************/
//void GvvTimeBudgetMonitoringEditor::on__timeBudgetSpinBox_valueChanged( int pValue )
//{
//	assert( _pipeline  != NULL );
//
//	_pipeline->setRenderingTimeBudget( pValue );
//	_plotView->setTimeBudget( pValue );
//	_timeBudgetLineEdit->setText( QString::number( 1.f / static_cast< float >( pValue ) * 1000.f ) + QString( " ms" ) );
//}
