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

#include "GvvCacheUsageView.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvPipelineInterface.h"
#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "Gvv3DWindow.h"
#include "GvvPipelineInterfaceViewer.h"

// Qwt
#include <qwt_thermo.h>

// System
#include <cassert>

// Qt
#include <QHBoxLayout>
#include <QGridLayout>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// XGraph
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
 * Default constructor
 ******************************************************************************/
GvvCacheUsageView::GvvCacheUsageView( QWidget* pParent, const char* pName ) 
:	QWidget( pParent )
,	_nodesCacheUsage( NULL )
,	_bricksCacheUsage( NULL )
,	_treeNodeSparseness( NULL )
,	_treeBrickSparseness( NULL )
{
	setupUi( this );

	// Editor name
	setObjectName( tr( "Cache Usage View" ) );

	_nodesCacheUsage = new QwtThermo();
	_bricksCacheUsage = new QwtThermo();

	_nodesCacheUsage->setMinValue( 0.0f );
	_nodesCacheUsage->setMaxValue( 100.0f );
	//_nodesCacheUsage->setLowerBound( 0.0f );
	//_nodesCacheUsage->setUpperBound( 100.0 );
	_nodesCacheUsage->setAlarmLevel( 75.0f );
	_nodesCacheUsage->setFillBrush( Qt::green );
	_nodesCacheUsage->setAlarmBrush( Qt::red );
	_nodesCacheUsage->setOrientation( Qt::Horizontal, QwtThermo::TopScale );
	//_nodesCacheUsage->setOrientation( Qt::Horizontal );
	//_nodesCacheUsage->setScalePosition( QwtThermo::TrailingScale );

	_bricksCacheUsage->setMinValue( 0.0f );
	_bricksCacheUsage->setMaxValue( 100.0f );
	//_bricksCacheUsage->setLowerBound( 0.0f );
	//_bricksCacheUsage->setUpperBound( 100.0 );
	_bricksCacheUsage->setAlarmLevel( 75.0f );
	_bricksCacheUsage->setFillBrush( Qt::green );
	_bricksCacheUsage->setAlarmBrush( Qt::red );
	_bricksCacheUsage->setOrientation( Qt::Horizontal, QwtThermo::TopScale );
	//_bricksCacheUsage->setOrientation( Qt::Horizontal );
	//_bricksCacheUsage->setScalePosition( QwtThermo::TrailingScale );

	QGridLayout* gridLayout = dynamic_cast< QGridLayout* >( _nodeCacheGroupBox->layout() );
	assert( gridLayout != NULL );
	if ( gridLayout != NULL )
	{
		gridLayout->addWidget( _nodesCacheUsage, 3, 0, 1, 3 );
	}

	gridLayout = dynamic_cast< QGridLayout* >( _dataCacheGroupBox->layout() );
	assert( gridLayout != NULL );
	if ( gridLayout != NULL )
	{
		gridLayout->addWidget( _bricksCacheUsage, 3, 0, 1, 3 );
	}

	_treeNodeSparseness = new QwtThermo();
	_treeNodeSparseness->setMinValue( 0.0 );
	_treeNodeSparseness->setMaxValue( 100.0 );
	//_treeNodeSparseness->setLowerBound( 0.0 );
	//_treeNodeSparseness->setUpperBound( 100.0 );
	_treeNodeSparseness->setAlarmLevel( 75.0 );
	_treeNodeSparseness->setFillBrush( Qt::green );
	_treeNodeSparseness->setAlarmBrush( Qt::red );
	_treeNodeSparseness->setOrientation( Qt::Horizontal, QwtThermo::TopScale );
	//_treeNodeSparseness->setOrientation( Qt::Horizontal );
	//_treeNodeSparseness->setScalePosition( QwtThermo::TrailingScale );

	gridLayout = dynamic_cast< QGridLayout* >( _treeNodeGroupBox->layout() );
	assert( gridLayout != NULL );
	if ( gridLayout != NULL )
	{
		gridLayout->addWidget( _treeNodeSparseness, 1, 0, 2, 3 );
	}

	_treeBrickSparseness = new QwtThermo();
	_treeBrickSparseness->setMinValue( 0.0 );
	_treeBrickSparseness->setMaxValue( 100.0 );
	//_treeBrickSparseness->setLowerBound( 0.0 );
	//_treeBrickSparseness->setUpperBound( 100.0 );
	_treeBrickSparseness->setAlarmLevel( 75.0 );
	_treeBrickSparseness->setFillBrush( Qt::green );
	_treeBrickSparseness->setAlarmBrush( Qt::red );
	_treeBrickSparseness->setOrientation( Qt::Horizontal, QwtThermo::TopScale );
	//_treeBrickSparseness->setOrientation( Qt::Horizontal );
	//_treeBrickSparseness->setScalePosition( QwtThermo::TrailingScale );

	gridLayout = dynamic_cast< QGridLayout* >( _treeBrickGroupBox->layout() );
	assert( gridLayout != NULL );
	if ( gridLayout != NULL )
	{
		gridLayout->addWidget( _treeBrickSparseness, 1, 0, 1, 3 );
	}
}

/******************************************************************************
 * Default destructor
 ******************************************************************************/
GvvCacheUsageView::~GvvCacheUsageView()
{
}

/******************************************************************************
 * Update view
 *
 * @param pNodeCacheUsage node cache usage (%)
 * @param pBrickCacheUsage brick cache usage (%)
 ******************************************************************************/
void GvvCacheUsageView::update( const GvViewerCore::GvvPipelineInterface* pPipeline )
{
	_nodesCacheUsage->setValue( pPipeline->getNodeCacheUsage() );
	_bricksCacheUsage->setValue( pPipeline->getBrickCacheUsage() );

	// Node cache
	_nodeCacheCapacityLineEdit->setText( QString::number( pPipeline->getNodeCacheCapacity() ) );
	_nodeCacheFillingRatioLineEdit->setText( QString::number( pPipeline->getNodeCacheUsage() ) );
	_nodeCacheNbElementsLineEdit->setText( QString::number( pPipeline->getNodeCacheCapacity() - pPipeline->getCacheNbUnusedNodes() ) );

	// Brick Cache
	_dataCacheCapacityLineEdit->setText( QString::number( pPipeline->getBrickCacheCapacity() ) );
	_dataCacheFillingRatioLineEdit->setText( QString::number( pPipeline->getBrickCacheUsage() ) );
	_dataCacheNbElementsLineEdit->setText( QString::number( pPipeline->getBrickCacheCapacity() - pPipeline->getCacheNbUnusedBricks() ) );

	// Tree data structure monitoring
	//if ( pPipeline->hasTreeDataStructureMonitoring() )
	if ( _treeMonitoringGroupBox->isChecked() )
	{
		// Tree
		_treeEmptyNodeRatioLineEdit->setText( QString::number( static_cast< float >( pPipeline->getNbTreeLeafNodes() ) / static_cast< float >( pPipeline->getNbTreeNodes() ) * 100.f  ) );
		_treeNbEmptyNodeLineEdit->setText( QString::number( pPipeline->getNbTreeLeafNodes() ) );
	//	_treeBrickSparsenessRatioLineEdit->setText( QString::number( pPipeline->getNodeCacheCapacity() - pPipeline->getCacheNbUnusedNodes() ) );

		_treeNodeSparseness->setValue( static_cast< float >( pPipeline->getNbTreeLeafNodes() ) / static_cast< float >( pPipeline->getNbTreeNodes() ) * 100.f  );
		//_treeBrickSparseness->setValue( static_cast< float >( pPipeline->getNbTreeLeafBricks() ) / static_cast< float >( pPipeline->getNbTreeNodes() * 1000/*nbVoxels*/ ) * 100.f  );
	}
}

/******************************************************************************
 * Slot called when tree data structure monitoring's state has changed
 ******************************************************************************/
void GvvCacheUsageView::on__treeMonitoringGroupBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	assert( pipeline != NULL );
	pipeline->setTreeDataStructureMonitoring( pChecked );

	// Reset
	if ( ! pChecked )
	{
		_treeEmptyNodeRatioLineEdit->setText( "" );
		_treeNbEmptyNodeLineEdit->setText( "" );
		//_treeBrickSparsenessRatioLineEdit->setText( "" );

		_treeNodeSparseness->setValue( 0.0 );
		//_treeBrickSparseness->setValue( static_cast< float >( pPipeline->getNbTreeLeafBricks() ) / static_cast< float >( pPipeline->getNbTreeNodes() * 1000/*nbVoxels*/ ) * 100.f  );
	}
}
