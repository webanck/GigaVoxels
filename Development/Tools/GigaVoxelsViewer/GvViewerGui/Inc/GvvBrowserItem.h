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

#ifndef GVVBROWSERITEM_H
#define GVVBROWSERITEM_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"

// Qt
#include <QTreeWidgetItem>

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
	class GvvBrowsable;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/**
 * This class represents the abstract base class for a browsable item.
 *
 * @ingroup	GvViewerGui
 */
class GVVIEWERGUI_EXPORT GvvBrowserItem : public QTreeWidgetItem
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Default constructor.
	 */
	GvvBrowserItem( GvViewerCore::GvvBrowsable* pBrowsable );
	
	/**
	 * Destructor.
	 */
	virtual ~GvvBrowserItem();

	/**
	 * Returns the Browsable assigned to this item
	 *
	 * @return the Browsable assigned to this item
	 */
	const GvViewerCore::GvvBrowsable* getBrowsable() const;

	/**
	 * Returns the Browsable assigned to this item
	 *
	 * @return the Browsable assigned to this item
	 */
	GvViewerCore::GvvBrowsable* editBrowsable();

	/**
	 * Finds the item assigned to the given browsable
	 *
	 * @param pBrowsable specifies the browsable to be searched
	 *
	 * @return the corresponding item
	 */
	GvvBrowserItem* find( GvViewerCore::GvvBrowsable* pBrowsable );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************** TYPEDEFS ********************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * The browsable contained by this item
	 */
	GvViewerCore::GvvBrowsable* mBrowsable;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden
	 */
	GvvBrowserItem( const GvvBrowserItem& );

	/**
	 * Copy operator forbidden
	 */
	GvvBrowserItem& operator=( const GvvBrowserItem& );

};

} // namespace GvViewerGui

#endif
