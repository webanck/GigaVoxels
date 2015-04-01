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

#ifndef GVVACTION_H
#define GVVACTION_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"

// Qt
#include <QAction>
#include <QString>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// Qt
class QObject;

// GvViewer
namespace GvViewerGui
{
	class GvvContextMenu;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/**
 * Base class for actions
 *
 * @ingroup GvViewerGui
 */
class GVVIEWERGUI_EXPORT GvvAction : public QAction
{

	Q_OBJECT

	/**************************************************************************
	 ***************************** FRIEND SECTION *****************************
	 **************************************************************************/

	friend class GvvContextMenu;

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructs an action.
	 *	 
	 * @param	pName		specifies the name of the action
	 * @param	pText		specifies the descriptive text of the action.
	 * @param	pIconName	specifies the name of the icon for this action located in the icons application path
	 *							Does nothing if the string is empty. A full file path can also be given.
	 * @param	pIsToggled	specifies if the action is toggled (Not toggled by default)
	 */
	GvvAction( QObject* pParent, const QString& pName, const QString& pText, const QString& pIconName = QString::null, bool pIsToggled = false );

	/**
	 * Destructor.
	 */
	virtual ~GvvAction();

	/**
	 * Returns the name of this action
	 *
	 * @return the name of this action
	 */
	virtual const QString& getName() const;

	/********************************** SLOTS **********************************/

public slots:

	/**
	 * Execute this action
	 */
	virtual void execute();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Sets the name of this action
	 *
	 * @param	pName	specifies the name to be assigned to this action
	 */
	virtual void setName( const QString& pName );

	/**
	 * Updates this action before being shown
	 */
	virtual void onAboutToShow();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/** 
	 * The name of this action
	 */
	QString mName;

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvvAction( const GvvAction& );

	/**
	 * Copy operator forbidden.
	 */
	GvvAction& operator=( const GvvAction& );

};

} // namespace GvViewerGui

#endif
