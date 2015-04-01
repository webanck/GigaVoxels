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

#ifndef GVVACTIONMANAGER_H
#define GVVACTIONMANAGER_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "GvvAction.h"

// Qt
#include <QString>

// STL
#include <vector>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/**
 * Manager of actions
 *
 * @ingroup XBrowser
 */
class GVVIEWERGUI_EXPORT GvvActionManager
{

	/**************************************************************************
	 ***************************** FRIENDS SECTION ****************************
	 **************************************************************************/

	friend class GvvAction;

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Returns the manager
	 */
	static GvvActionManager& get();

	/**
	 * Returns whether this manager is empty
	 *
	 * @return	true if this manager is empty
	 */
	bool isEmpty() const;

	/**
	 * Returns the number of elements this manager contains
	 *
	 * @return	the number of elements this manager contains
	 */
	unsigned int getNbActions() const;

	/**
	 * Returns the i-th element
	 *
	 * @param	pIndex	specifies the index of the desired element
	 *
	 * @return	a const pointer to the pIndex-th element
	 */
	const GvvAction* getAction( unsigned int pIndex ) const;

	/**
	 * Returns the i-th element
	 *
	 * @param	pIndex	specifies the index of the desired element
	 *
	 * @return	a pointer to the pIndex-th element
	 */
	GvvAction* editAction( unsigned int pIndex );

	/**
	 * Returns the element represented by the specified name
	 *
	 * @param	pName	specifies the name of the desired element
	 *
	 * @return	a const pointer to the element or null if not found
	 */
	const GvvAction* getAction( const QString& pName ) const;

	/**
	 * Returns the element represented by the specified name
	 *
	 * @param	pName	specifies the name of the desired element
	 *
	 * @return	a pointer to the element or null if not found
	 */
	GvvAction* editAction( const QString& pName );

	/**
	 * Searchs for the specified element
	 *
	 * @param	pManageable	specifies the element to be found
	 *
	 * @return	the index of the element or -1 if not found
	 */
	unsigned int findAction( const GvvAction* pAction ) const;

	/**
	 * Searchs for the element with the specified name
	 *
	 * @param	pName	specifies the name of the element to be found
	 *
	 * @return	the index of the element or -1 if not found
	 */
	unsigned int findAction( const QString& pName ) const;

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************** TYPEDEFS ********************************/

	/**
	 *
	 */
	typedef std::vector< GvvAction* > GvvActionVector;

	/******************************* ATTRIBUTES *******************************/

	/**
	 * The unique action manager
	 */
	static GvvActionManager* msInstance;

	/**
	 * The container of actions
	 */
	GvvActionVector mActions;

	/**
	 * The auto-delete flag
	 */
	 bool mDeleteActions;

	/******************************** METHODS *********************************/

	/**
	 * Default constructor.
	 */
	GvvActionManager( bool pDeleteElements = false );

	/**
	 * Destructor.
	 */
	virtual ~GvvActionManager();

	/**
	 * Registers the specified element to this manager
	 *
	 * @param	pManageable	specifies the element to be registered
	 */
	void registerAction( GvvAction* pAction );

	/**
	 * Unregisters the specified element from this manager
	 *
	 * @param	pManageable	specifies the element to be unregistered
	 */
	void unregisterAction( GvvAction* pAction );

	/**
	 * Deletes the specified element from memory
	 *
	 * @param	pManageable	specifies the element to delete
	 */
	virtual void deleteAction( GvvAction* pAction );

	/**
	 * Deletes all the elements from memory and clears the manager
	 */
	void deleteActions();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvvActionManager( const GvvActionManager& );

	/**
	 * Copy operator forbidden.
	 */
	GvvActionManager& operator=( const GvvActionManager& );

};

} // namespace GvViewerGui

/******************************************************************************
 ****************************** INLINE SECTION ********************************
 ******************************************************************************/

#include "GvvActionManager.inl"

#endif
