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

#include "GvvBrowserItem.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvBrowsable.h"

// Qt
#include <QHash>

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
 * Constructs an item with the given browsable
 *
 * @param pBrowsable specifies the browsable to be assigned to this item 
 ******************************************************************************/
GvvBrowserItem::GvvBrowserItem( GvvBrowsable* pBrowsable )
//:	QTreeWidgetItem( qHash( pBrowsable->getTypeName() ) )
:	QTreeWidgetItem( qHash( QString( pBrowsable->getTypeName() ) ) )
,	mBrowsable( pBrowsable )
{
	//-----------------------------------------------------------------------------
	// TO DO :
	// - for QTreeWidgetItem( qHash( QString( pBrowsable->getTypeName() ) ) )
	// - check if the problem comes from the int instead of the uint
	//-----------------------------------------------------------------------------
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvBrowserItem::~GvvBrowserItem()
{
	mBrowsable = NULL;
}

/******************************************************************************
 * Returns the Browsable assigned to this item
 *
 * @return the Browsable assigned to this item
 ******************************************************************************/
const GvvBrowsable* GvvBrowserItem::getBrowsable() const
{
	return mBrowsable;
}

/******************************************************************************
 * Returns the Browsable assigned to this item
 *
 * @return the Browsable assigned to this item
 ******************************************************************************/
GvvBrowsable* GvvBrowserItem::editBrowsable()
{
	return mBrowsable;
}

/******************************************************************************
 * Finds the item assigned to the given browsable
 *
 * @param pBrowsable specifies the browsable to be searched
 *
 * @return the corresponding item
 ******************************************************************************/
GvvBrowserItem* GvvBrowserItem::find( GvvBrowsable* pBrowsable )
{
	//** Checks whether this item holds the given element
	if ( mBrowsable == pBrowsable )
	{
		return this;
	}

	//** Iterates through the children and performs a recursive search
	for	( int i = 0; i < childCount(); ++i )
	{
		GvvBrowserItem* lChildItem = dynamic_cast< GvvBrowserItem* >( child( i ) );
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
