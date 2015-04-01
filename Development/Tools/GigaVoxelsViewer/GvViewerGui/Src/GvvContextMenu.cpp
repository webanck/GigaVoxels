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

#include "GvvContextMenu.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvBrowser.h"
#include "GvvAction.h"

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
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
GvvContextMenu::GvvContextMenu( GvvBrowser* pBrowser ) 
:	QMenu( pBrowser )
{
	//** Setups connections
	QObject::connect( this, SIGNAL( aboutToShow() ), this, SLOT( onAboutToShow() ) );
}

/******************************************************************************
 * Contructs a menu with the given title and parent
 * 
 * @param pTitle the title menu
 * @param pMenu the parent menu
 ******************************************************************************/
GvvContextMenu::GvvContextMenu( const QString& pTitle, GvvContextMenu* pMenu )
:	QMenu( pTitle, pMenu )
{
	//** Setups connections
	QObject::connect( this, SIGNAL( aboutToShow() ), this, SLOT( onAboutToShow() ) );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvContextMenu::~GvvContextMenu()
{
}

/******************************************************************************
 * Handles the aboutToShow signal
 ******************************************************************************/
void GvvContextMenu::onAboutToShow()
{
	//** Retrieves the list of actions within this context menu
	QList< QAction* > lActions = actions();

	//** Iterates though the actions and update them
	for ( int i = 0; i < lActions.size(); ++i )
	{
		GvvAction* lAction = dynamic_cast< GvvAction* >( lActions[ i ] );
		if
			( lAction != NULL )
		{
			lAction->onAboutToShow();
		}
	}
}
