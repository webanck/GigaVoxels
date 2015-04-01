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

#include "TimeBudgetView.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "PlotView.h"
#include "SampleCore.h"

// STL
#include <iostream>

// System
#include <cassert>

// Qt
#include <QString>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

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
TimeBudgetView::TimeBudgetView( QWidget* pParent, Qt::WindowFlags pFlags )
:	QWidget( pParent, pFlags )
,	_plotView( NULL )
,	_pipeline( NULL )
{
	setupUi( this );

	// Editor name
	setObjectName( tr( "Frame Time Budget Monitor" ) );

	// ------------------------------------------------------

	_plotView =  new PlotView( this, "Frame Time Budget Monitor" );
	
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
TimeBudgetView::~TimeBudgetView()
{
}

/******************************************************************************
 * Populates this editor with the specified browsable
 *
 * @param pBrowsable specifies the browsable to be edited
 ******************************************************************************/
void TimeBudgetView::populate( SampleCore* pPipeline )
{
	assert( pPipeline != NULL );
	if ( pPipeline != NULL )
	{
		_pipeline = pPipeline;
	
		blockSignals( true );

		_timeBudgetParametersGroupBox->setChecked( pPipeline->hasTimeBudget() );
		_timeBudgetSpinBox->setValue( pPipeline->getTimeBudget() );
		_timeBudgetLineEdit->setText( QString::number( 1.f / static_cast< float >( pPipeline->getTimeBudget() ) * 1000.f ) + QString( " ms" ) );
		_plotView->setTimeBudget( pPipeline->getTimeBudget() );

		blockSignals( false );
	}
}

/******************************************************************************
 * Draw the specified curve
 *
 * @param pCurve specifies the curve to be drawn
 ******************************************************************************/
void TimeBudgetView::onCurveChanged( unsigned int pFrame, float pFrameTime )
{
	assert( _plotView != NULL );

	_plotView->onCurveChanged( pFrame, pFrameTime );
}

/******************************************************************************
 * Slot called when time budget parameters group box state has changed
 ******************************************************************************/
void TimeBudgetView::on__timeBudgetParametersGroupBox_toggled( bool pChecked )
{
	assert( _pipeline  != NULL );

	_pipeline->setTimeBudgetActivated( pChecked );
}

/******************************************************************************
 * Slot called when user requested time budget value has changed
 ******************************************************************************/
void TimeBudgetView::on__timeBudgetSpinBox_valueChanged( int pValue )
{
	assert( _pipeline  != NULL );

	_pipeline->setTimeBudget( pValue );
	_plotView->setTimeBudget( pValue );
	_timeBudgetLineEdit->setText( QString::number( 1.f / static_cast< float >( pValue ) * 1000.f ) + QString( " ms" ) );
}
