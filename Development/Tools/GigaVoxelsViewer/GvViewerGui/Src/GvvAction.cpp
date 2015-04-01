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

#include "GvvAction.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvActionManager.h"

// Qt
#include <QObject>

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
 * Constructs an action.
 * 
 * @param	pName		specifies the name of the action.
 * @param	pText		specifies the descriptive text of the action.
 * @param	pIconName	specifies the name of the icon for this action located in the icons application path
 *							Does nothing if the string is empty. A full file path can also be given.
 * @param	pIsToggled	specifies if the action is toggled (Not toggled by default)
 ******************************************************************************/
GvvAction::GvvAction( QObject* pParent, const QString& pName, const QString& pText, const QString& pIconName, bool pIsToggled )
:	QAction( pText, pParent )
,	mName()
{
	//** Set the action name
	setName( pName );

	//** Create the default signal/slot connection
	connect( this, SIGNAL( triggered( bool ) ), this, SLOT( execute() ) );

	//** If the action is toggled
	setCheckable( pIsToggled );
	
	//** Define the path of the icon to be assigned
	QString iconName = pIconName;
	if ( iconName.isEmpty() )
	{
		iconName = pName + ".png";
	}

	//** Assign the icon
	QIcon* icon = new QIcon( QString( "Icons" ) + QString( "/" ) + iconName );
	//QIcon* icon = NULL;
	if ( icon != NULL )
	{	
		setIcon( *icon );
	}

	//** Register this action 
	GvvActionManager::get().registerAction( this );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvAction::~GvvAction()
{
	//** Unregisters it 
	GvvActionManager::get().unregisterAction( this );
}

/******************************************************************************
 * Returns the name of this action.
 *
 * @return the name of this action.
 ******************************************************************************/
const QString& GvvAction::getName() const
{
	return mName;
}

/******************************************************************************
 * Returns the name of this action.
 *
 * @return the name of this action.
 ******************************************************************************/
void GvvAction::setName( const QString& pName )
{
	setObjectName( pName );
	mName = pName;
}

/******************************************************************************
 * Updates this action before being shown
 ******************************************************************************/
void GvvAction::onAboutToShow()
{
}

/******************************************************************************
 * Executes this action.
 ******************************************************************************/
void GvvAction::execute()
{
}
