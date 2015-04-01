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

#include "InspectorView.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "SampleCore.h"

// GigaVoxels
#include <GvStructure/GvVolumeTreeAddressType.h>

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
InspectorView::InspectorView( QWidget* pParent, Qt::WindowFlags pFlags )
:	QWidget( pParent, pFlags )
,	_dataStructureChildArray( NULL )
,	_dataStructureDataArray( NULL )
,	_nodeCacheTimeStampList( NULL )
,	_nodeCacheElementAddressList( NULL )
,	_nodeCacheNbUnusedElements( 0 )
,	_brickCacheTimeStampList( NULL )
,	_brickCacheElementAddressList( NULL )
,	_brickCacheNbUnusedElements( 0 )
{
	setupUi( this );

	// Editor name
	setObjectName( tr( "Inspector View" ) );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
InspectorView::~InspectorView()
{
}

/******************************************************************************
 * Initialize this editor with the specified browsable
 *
 * @param pBrowsable specifies the browsable to be edited
 ******************************************************************************/
void InspectorView::initialize( SampleCore* pPipeline )
{
	assert( pPipeline != NULL );
	if ( pPipeline != NULL )
	{
		pPipeline->analyse( _dataStructureChildArray, _dataStructureDataArray,
							_nodeCacheTimeStampList, _nodeCacheElementAddressList, _nodeCacheNbUnusedElements,
							_brickCacheTimeStampList, _brickCacheElementAddressList, _brickCacheNbUnusedElements );

		// Node Pool
		_nodePoolTableWidget->setColumnCount( _dataStructureChildArray->getNumElements() );
		for ( unsigned int i = 0; i < _dataStructureChildArray->getNumElements(); i++ )
		{
			// Extremely slow !!!!!!!!!!
			//_nodePoolTableWidget->setHorizontalHeaderItem( i, new QTableWidgetItem( QString::number( i ) ) );

			_nodePoolTableWidget->setItem( 0, i, new QTableWidgetItem() );
			_nodePoolTableWidget->setItem( 1, i, new QTableWidgetItem() );

			if ( i % 8 == 0 )
			{
				_nodePoolTableWidget->item( 0, i )->setBackground( QColor( 255, 127, 127 ) );
			}
		}
		_nodePoolTableWidget->setItem( 0, 0, new QTableWidgetItem( tr( "0 - NOT USED" ) ) );
		_nodePoolTableWidget->setItem( 1, 0, new QTableWidgetItem( tr( "0 - NOT USED" ) ) );
		_nodePoolTableWidget->verticalHeaderItem( 0 )->setToolTip( tr( "For a given node of the nodepool, it shows the starting address of its children and its flag (terminal and empty or data inside)" ) );
		_nodePoolTableWidget->verticalHeaderItem( 1 )->setToolTip( tr( "For a given node of the nodepool, it shows the starting address of its associated brick of data in the datapool (i.e. 3D texture), shifted by the border size." ) );

		// Data Pool
		//_dataPoolTableWidget->setColumnCount( 1000000 );

		// Node Cache Manager
		_nodeCacheManagerTableWidget->setColumnCount( _nodeCacheTimeStampList->getNumElements() );
		for ( unsigned int i = 0; i < _nodeCacheTimeStampList->getNumElements(); i++ )
		{
			_nodeCacheManagerTableWidget->setItem( 0, i, new QTableWidgetItem() );
			_nodeCacheManagerTableWidget->setItem( 1, i, new QTableWidgetItem() );
		}
		_nodeCacheManagerTableWidget->setItem( 0, 0, new QTableWidgetItem( tr( "0 - NOT USED" ) ) );
		_nodeCacheManagerTableWidget->setItem( 1, 0, new QTableWidgetItem( tr( "0 - NOT USED" ) ) );
		_nodeCacheManagerTableWidget->setItem( 1, 1, new QTableWidgetItem( tr( "1 - ROOT" ) ) );
		_nodeCacheManagerTableWidget->verticalHeaderItem( 0 )->setToolTip( tr( "For each nodetile of the nodepool, it shows its associated time (i.e. the last frame index it has been flagged as used)" ) );
		_nodeCacheManagerTableWidget->verticalHeaderItem( 0 )->setToolTip( tr( "Sorted list of node addresses in the nodepool (unused ones first then used ones at the end [in green])" ) );
		_nodeCacheManagerTableWidget->item( 0, 1 )->setBackground( QColor( 127, 255, 127 ) );
		_nodeCacheManagerTableWidget->item( 1, 1 )->setBackground( QColor( 127, 255, 127 ) );

		// Data Cache Manager
		_dataCacheManagerTableWidget->setColumnCount( _brickCacheTimeStampList->getNumElements() );
		for ( unsigned int i = 0; i < _brickCacheTimeStampList->getNumElements(); i++ )
		{
			_dataCacheManagerTableWidget->setItem( 0, i, new QTableWidgetItem() );
			_dataCacheManagerTableWidget->setItem( 1, i, new QTableWidgetItem() );
		}
		_dataCacheManagerTableWidget->setItem( 0, 0, new QTableWidgetItem( tr( "0 - NOT USED" ) ) );
		_dataCacheManagerTableWidget->setItem( 1, 0, new QTableWidgetItem( tr( "0 - NOT USED" ) ) );
		_dataCacheManagerTableWidget->setItem( 1, 1, new QTableWidgetItem( tr( "1 - NOT USED" ) ) );
		_dataCacheManagerTableWidget->verticalHeaderItem( 0 )->setToolTip( tr( "For each brick of the datapool, it shows its associated time (i.e. the last frame index it has been flagged as used)" ) );
	}
}

/******************************************************************************
 * Populates this editor with the specified browsable
 *
 * @param pBrowsable specifies the browsable to be edited
 ******************************************************************************/
void InspectorView::populate( SampleCore* pPipeline )
{
	assert( pPipeline != NULL );
	if ( pPipeline != NULL )
	{
		pPipeline->analyse( _dataStructureChildArray, _dataStructureDataArray,
							_nodeCacheTimeStampList, _nodeCacheElementAddressList, _nodeCacheNbUnusedElements,
							_brickCacheTimeStampList, _brickCacheElementAddressList, _brickCacheNbUnusedElements );

		for ( unsigned int i = 0; i < _dataStructureChildArray->getNumElements(); i++ )
		{
			// child array
			QTableWidgetItem* item = _nodePoolTableWidget->item( 0, i );
			if ( item != NULL )
			{
				//item->setText( QString::number( _pipeline->editDataStructure()->_childArraySync->get( i ) ) );

				const uint address = _dataStructureChildArray->get( i );
				//if ( ! GvStructure::VolTreeNodeAddress::isNull( address ) )
				if ( address )
				{
					//const uint3 unpackedAddress = GvStructure::VolTreeNodeAddress::unpackAddress( address );
					const uint unpackedAddress = GvStructure::VolTreeNodeAddress::unpackAddress( address ).x;

					// address

					QString text = "( ";
					text += QString::number( unpackedAddress/*.x*/ );
					/*text += " | ";
					text += QString::number( unpackedAddress.y );
					text += " | ";
					text += QString::number( unpackedAddress.z );*/
					text += " ) ";

					// flags
					text += " - ";
					text += "[ ";
					text += QString::number( ( address & 0x80000000 ) >> 31 );
					text += " | ";
					text += QString::number( ( address & 0x40000000 ) >> 30 );
					text += " ] ";

					item->setText( text );
				}
				else
				{
					item->setText( "" );
				}
			}

			// data array
			item = _nodePoolTableWidget->item( 1, i );

			if ( item != NULL )
			{
				//item->setText( QString::number( _pipeline->editDataStructure()->_dataArraySync->get( i ) ) );

				const uint address = _dataStructureDataArray->get( i );
				//if ( ! GvStructure::VolTreeBrickAddress::isNull( address ) )
				if ( address )
				{
					const uint3 unpackedAddress = GvStructure::VolTreeBrickAddress::unpackAddress( address );
					QString text = QString::number( unpackedAddress.x );
					text += " | ";
					text += QString::number( unpackedAddress.y );
					text += " | ";
					text += QString::number( unpackedAddress.z );
					item->setText( text );
				}
				else
				{
					item->setText( "" );
				}
			}
		}

		// Node cache manager
		{
			const size_t nbElements = _nodeCacheTimeStampList->getNumElements();

			// Node cache manager
			{
				GvCore::Array3DGPULinear< uint >* timeStampList = _nodeCacheTimeStampList;
				GvCore::Array3D< uint > hostElementAddressList( timeStampList->getResolution(), GvCore::Array3D< uint >::StandardHeapMemory );
				GvCore::memcpyArray( &hostElementAddressList, timeStampList );
				for ( unsigned int i = 0; i < nbElements; i++ )
				{
					// time stamps
					QTableWidgetItem* item = _nodeCacheManagerTableWidget->item( 0, i );
					if ( item != NULL )
					{
						item->setText( QString::number( hostElementAddressList.get( i ) ) );
					}
				}
			}
					
			{
				thrust::device_vector< uint >* elementAddressList = _nodeCacheElementAddressList;
				thrust::host_vector< uint > hostElementAddressList( elementAddressList->size() );
				thrust::copy( elementAddressList->begin(), elementAddressList->end(), hostElementAddressList.begin() );
				for ( unsigned int i = 0; i < nbElements - 2/*locked elements*/; i++ )
				{
					// elements address
					QTableWidgetItem* item = _nodeCacheManagerTableWidget->item( 1, i + 2/*locked elements*/ );
					if ( item != NULL )
					{
						//item->setText( QString::number( hostElementAddressList[ i ] ) );

						//const uint3 unpackedAddress = GvStructure::VolTreeNodeAddress::unpackAddress( hostElementAddressList[ i ] );
						const uint unpackedAddress = GvStructure::VolTreeNodeAddress::unpackAddress( hostElementAddressList[ i ] ).x;
						QString text = QString::number( unpackedAddress/*.x*/ );
						/*text += " | ";
						text += QString::number( unpackedAddress.y );
						text += " | ";
						text += QString::number( unpackedAddress.z );*/
						item->setText( text );

						if ( i >= _nodeCacheNbUnusedElements )
						{
							item->setBackground( QColor( 127, 255, 127 ) );
						}
						else
						{
							item->setBackground( QColor( 255, 255, 255 ) );
						}
					}
				}
			}

			const uint nbElementAddresses = _nodeCacheElementAddressList->size();
			const uint nbUnusedElementAddresses = _nodeCacheNbUnusedElements;
			const uint nbUsedElementAddresses = nbElementAddresses - nbUnusedElementAddresses;
			_nodeCacheNbElementsLineEdit->setText( QString::number( nbElementAddresses ) );
			_nodeCacheUnusedVsUsedNbElementsLineEdit->setText( QString::number( nbUnusedElementAddresses ) + " - " + QString::number( nbUsedElementAddresses ) );
		}

		// Data cache manager
		{
			const size_t nbElements = _brickCacheTimeStampList->getNumElements();

			{
				GvCore::Array3DGPULinear< uint >* timeStampList = _brickCacheTimeStampList;
				GvCore::Array3D< uint > hostElementAddressList( timeStampList->getResolution(), GvCore::Array3D< uint >::StandardHeapMemory );
				GvCore::memcpyArray( &hostElementAddressList, timeStampList );
				for ( unsigned int i = 0; i < nbElements; i++ )
				{
					// time stamps
					QTableWidgetItem* item = _dataCacheManagerTableWidget->item( 0, i );
					if ( item != NULL )
					{
						item->setText( QString::number( hostElementAddressList.get( i ) ) );
					}
				}
			}

			{
				thrust::device_vector< uint >* elementAddressList = _brickCacheElementAddressList;
				thrust::host_vector< uint > hostElementAddressList( elementAddressList->size() );
				thrust::copy( elementAddressList->begin(), elementAddressList->end(), hostElementAddressList.begin() );
				for ( unsigned int i = 0; i < nbElements - 2/*locked elements*/; i++ )
				{
					// elements address
					QTableWidgetItem* item = _dataCacheManagerTableWidget->item( 1, i + 2/*locked elements*/ );
					if ( item != NULL )
					{
						//item->setText( QString::number( hostElementAddressList[ i ] ) );

						const uint3 unpackedAddress = GvStructure::VolTreeBrickAddress::unpackAddress( hostElementAddressList[ i ] );
						QString text = QString::number( unpackedAddress.x );
						text += " | ";
						text += QString::number( unpackedAddress.y );
						text += " | ";
						text += QString::number( unpackedAddress.z );
						item->setText( text );

						if ( i >= _brickCacheNbUnusedElements )
						{
							item->setBackground( QColor( 127, 255, 127 ) );
						}
						else
						{
							item->setBackground( QColor( 255, 255, 255 ) );
						}
					}
				}
			}

			const uint nbElementAddresses = _brickCacheElementAddressList->size();
			const uint nbUnusedElementAddresses = _brickCacheNbUnusedElements;
			const uint nbUsedElementAddresses = nbElementAddresses - nbUnusedElementAddresses;
			_brickCacheNbElementsLineEdit->setText( QString::number( nbElementAddresses ) );
			_brickCacheUnusedVsUsedNbElementsLineEdit->setText( QString::number( nbUnusedElementAddresses ) + " - " + QString::number( nbUsedElementAddresses ) );
		}
	}
}
