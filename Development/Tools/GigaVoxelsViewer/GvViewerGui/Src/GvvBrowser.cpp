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

#include "GvvBrowser.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

#include "GvvBrowserItem.h"
#include "GvvBrowsable.h"
//#include "GvvResourceManager.h"
#include "GvvContextManager.h"
#include "GvvContextMenu.h"

// Qt
#include <QContextMenuEvent>
#include <QTreeWidget>

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
GvvBrowser::GvvBrowser( QWidget* pParent ) 
:	QTreeWidget( pParent )
,	mContextMenus()
{
	// TEST -----
	setHeaderHidden( true );
	// TEST -----

	//** Forbids edition
	setEditTriggers( NoEditTriggers );

	//** Specifies the selection mode
	setSelectionMode( SingleSelection );

	//** Setups connection
	QObject::connect( this, SIGNAL( itemClicked( QTreeWidgetItem*, int ) ),
					  this, SLOT( onItemClicked( QTreeWidgetItem*, int ) ) );

	QObject::connect( this, SIGNAL( itemChanged( QTreeWidgetItem*, int ) ),
					  this, SLOT( onItemChanged( QTreeWidgetItem*, int ) ) );

	QObject::connect( this, SIGNAL( currentItemChanged ( QTreeWidgetItem*, QTreeWidgetItem* ) ),
					  this, SLOT( onCurrentItemChanged( QTreeWidgetItem*, QTreeWidgetItem*) ) );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvBrowser::~GvvBrowser()
{
}

/******************************************************************************
 * Returns the specified contextual menu. If the menu does not exist then
 * the method creates it and adds it within the map.
 *
 * @param pName specifies the name of the menu to be 
 *
 * @return the corresponding context menu
 ******************************************************************************/
GvvContextMenu* GvvBrowser::getContextMenu( const QString& pName )
{
	//** Computes the menu key
	//-----------------------------------------------------------------------------
	// TO DO : check if the problem comes from the int instead of the uint
	//-----------------------------------------------------------------------------
	int lKey = (int)( qHash( pName ) );

	//** Retrieves the menu of the given name
	GvvContextMenuHash::iterator lMenuIt = mContextMenus.find( lKey );
	if ( lMenuIt != mContextMenus.end() )
	{
		return lMenuIt.value();
	}

	//** Creates the menu and adds it within the map
	GvvContextMenu* lMenu = new GvvContextMenu( this );
	if ( lMenu != NULL )
	{
		mContextMenus.insert( lKey, lMenu );
	}
	return lMenu;
}

/******************************************************************************
 * Returns the contextual menu with the given name. If the menu does not 
 * exist then the method creates it and adds it within the map.
 *
 * @param pName specifies the name of the menu to be access
 * @param pPath specifies the sub menu hierarchy
 *
 * @return the corresponding context menu
 ******************************************************************************/
GvvContextMenu* GvvBrowser::getContextSubMenu( const QString& pName, const QStringList& pMenuPath )
{
	//** Retrieves the menu
	GvvContextMenu* lMenu = getContextMenu( pName );
	if ( lMenu != NULL )
	{
		//** Builds the hierarchy menu
		for	( int i = 0; i < pMenuPath.size(); ++i )
		{
			GvvContextMenu* lSubMenu = lMenu->findChild< GvvContextMenu* >( pMenuPath.at( i ) );
			if ( lSubMenu == NULL )
			{
				lSubMenu = new GvvContextMenu( pMenuPath.at( i ), lMenu );
				lSubMenu->setObjectName( pMenuPath.at( i ) );
				lMenu->addMenu( lSubMenu );
			}
			lMenu = lSubMenu;
		}
	}

	return lMenu;
}

/******************************************************************************
 * Finds the item assigned to the given browsable
 *
 * @param pBrowsable specifies the browsable to be searched
 *
 * @return the corresponding item
 ******************************************************************************/
GvvBrowserItem* GvvBrowser::find( GvvBrowsable* pBrowsable )
{
	//** Iterates through the children and performs a recursive search
	for ( int i = 0; i < topLevelItemCount(); ++i )
	{
		GvvBrowserItem* lChildItem = dynamic_cast< GvvBrowserItem* >( topLevelItem( i ) );
		if ( lChildItem != NULL )
		{
			GvvBrowserItem* lFoundItem = lChildItem->find( pBrowsable );
			if ( lFoundItem != NULL )
			{
				return lFoundItem;
			}
		}
	}

	return NULL;
}

/******************************************************************************
 * Finds the first item whose name matches the given name
 *
 * @param pName specifies the name of the browsable to be searched
 *
 * @return the corresponding item
 ******************************************************************************/
GvvBrowserItem* GvvBrowser::find( const QString& pName )
{
	QList< QTreeWidgetItem* > lItems = findItems( pName, Qt::MatchExactly | Qt::MatchRecursive, 0 );
	if ( ! lItems.empty() )
	{
		return dynamic_cast< GvvBrowserItem* >( lItems.front() );
	}
	return NULL;
}

/******************************************************************************
 * Creates a default item for the given browsable
 *
 * @param pBrowsable specifies the browsable for which an item is required
 *
 * @return the corresponding item
 ******************************************************************************/
GvvBrowserItem* GvvBrowser::createItem( GvvBrowsable* pBrowsable )
{
	GvvBrowserItem* lItem = NULL;
	if ( pBrowsable != NULL )
	{
		lItem = new GvvBrowserItem( pBrowsable );
		if ( lItem != NULL )
		{
			//** Sets the text of this item
			lItem->setText( 0, pBrowsable->getName() );

			//** Sets the icon of this item
			//QIcon* lIcon = GvvResourceManager::get().queryIcon( pBrowsable->getBrowsableTypeName() + ".png" );
			QIcon* lIcon = new QIcon( QString( "Icons" ) + QString( "/" ) + QString( pBrowsable->getTypeName() ) + QString( ".png" ) );
			//QIcon* lIcon = NULL;
			if ( lIcon != NULL )
			{
				lItem->setIcon( 0, *lIcon );
			}

			//** Setups the flags of this item
			Qt::ItemFlags lFlags = lItem->flags();
			lFlags |= Qt::ItemIsEditable;
			if ( pBrowsable->isCheckable() )
			{
				lFlags |= Qt::ItemIsUserCheckable;
				lItem->setCheckState( 0, pBrowsable->isChecked() ? Qt::Checked : Qt::Unchecked );
			}
			lItem->setFlags( lFlags );
		}
	}

	return lItem;
}

/******************************************************************************
 * Handles the conext menu event
 *
 * @param pEvent the context menu event
 ******************************************************************************/
void GvvBrowser::contextMenuEvent( QContextMenuEvent* pEvent )
{
	const QPoint lDelta( 10, 0 );

	//** Finds the item "selected" by the event and computes the global position
	GvvBrowserItem* lItem = NULL;
	QPoint lGlobalPos;
	if ( pEvent->reason() == QContextMenuEvent::Keyboard )
	{
		//** Event was triggered by the context keyboard key
		lItem = dynamic_cast< GvvBrowserItem* >( currentItem() );
		if ( lItem != NULL )
		{
			QRect lRect = visualItemRect( lItem );
			lGlobalPos = viewport()->mapToGlobal( lRect.center() );
			lGlobalPos -= lDelta;
		}
	}
	else
	{
		lGlobalPos = pEvent->globalPos();
		lItem = dynamic_cast< GvvBrowserItem* >( itemAt( pEvent->pos() ) );
	}

	//** Retrieves the menu to be shown according the selected item and its state
	int lItemType = 0;
	if ( lItem != NULL )
	{
		if ( lItem->flags() & Qt::ItemIsEnabled )
		{
			lItemType = lItem->type();
		}
		else
		{
			lItemType = -1;
		}
	}
	
	//** Shows the menu
	if ( lItemType != -1 )
	{
		GvvContextMenuHash::iterator itContextMenu = mContextMenus.find( lItemType );
		if ( itContextMenu != mContextMenus.end() )
		{
			//** Popups the menu
			itContextMenu.value()->popup( lGlobalPos + lDelta );
		}
	}
}

/******************************************************************************
 * Handles the itemSelectionChanged signal
 *
 * @param pItem the clicked item
 * @param pColumn the clicked column
 ******************************************************************************/
void GvvBrowser::onItemClicked( QTreeWidgetItem* pItem, int pColumn )
{
	//* Updates the context with the first one
	GvvBrowserItem* lItem = dynamic_cast< GvvBrowserItem* >( pItem );
	if ( lItem != NULL )
	{
		GvvBrowsable* lBrowsable = lItem->editBrowsable();
		GvvContextManager::get()->setCurrentBrowsable( lBrowsable );
	}
	else
	{
		//** Clears the context
		GvvContextManager::get()->setCurrentBrowsable( NULL );
	}
}
/******************************************************************************
 * Handles the currentitemChanged signal
 *
 * @param pCurrent the Current item
 * @param pPrevious the  pPrevious current item
 ******************************************************************************/
void GvvBrowser::onCurrentItemChanged( QTreeWidgetItem* pCurrent , QTreeWidgetItem* pPrevious)
{
	//* Updates the context with the first one
	GvvBrowserItem* lItem = dynamic_cast< GvvBrowserItem* >( pCurrent );
	if ( lItem == NULL )
	{
		//** Clears the context
		GvvContextManager::get()->setCurrentBrowsable( NULL );
	}
}
/******************************************************************************
 * Handles the itemClicked signal
 *
 * @param pItem the clicked item
 * @param pColumn the clicked column
 ******************************************************************************/
void GvvBrowser::onItemChanged( QTreeWidgetItem* pItem, int pColumn )
{
	GvvBrowserItem* lItem = dynamic_cast< GvvBrowserItem* >( pItem );
	if ( ( lItem != NULL ) && ( lItem->flags() & Qt::ItemIsEnabled ) ) 
	{
		//** Handles whether the checkbox state has changed
		GvvBrowsable* lBrowsable = lItem->editBrowsable();
		if ( lBrowsable->isCheckable() && ( pColumn == 0 ) )
		{
			// Sets the browsable as checked or unchecked
			lBrowsable->setChecked( lItem->checkState( pColumn ) & Qt::Checked );
		}
	}
}
