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

#ifndef GVVCONTEXTMENU_H
#define GVVCONTEXTMENU_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"

// Qt
#include <QMenu>

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
namespace GvViewerGui
{
	class GvvBrowser;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * This class specializes a QMenu in order to handle the aboutToShow signal.
 * This allows to iterate throught the actions and update them according
 * the context
 *
 * @ingroup XBrowser
 */
class GVVIEWERGUI_EXPORT GvvContextMenu : public QMenu
{

	Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Default constructor.
	 * 
	 * @param pBrowser the parent widget
	 */
	GvvContextMenu( GvvBrowser* pBrowser );
	
	/**
	 * Contructs a menu with the given title and parent
	 * 
	 * @param pTitle the title menu
	 * @param pMenu the parent menu
	 */
	GvvContextMenu( const QString& pTitle, GvvContextMenu* pMenu );

	/**
	 * Destructor.
	 */
	virtual ~GvvContextMenu();
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************** TYPEDEFS ********************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/
	
	/********************************** SLOTS **********************************/

protected slots:

	/**
	 * Handles the aboutToShow signal
	 */
	void onAboutToShow();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Copy constructor forbidden.
	 */
	GvvContextMenu( const GvvContextMenu& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvContextMenu& operator=( const GvvContextMenu& );
	
};

} // namespace GvViewerGui

#endif
