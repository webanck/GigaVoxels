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

#ifndef GVVBROWSABLE_H
#define GVVBROWSABLE_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvCoreConfig.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerCore
{

/** 
 * Abstract base class of all browsable entities.
 *
 * @ingroup GvViewerCore
 */
class GVVIEWERCORE_EXPORT GvvBrowsable
{

	/**************************************************************************
	 ***************************** FRIEND SECTION *****************************
	 **************************************************************************/

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * Default constructor
	 */
	GvvBrowsable();

	/**
	 * Destructor.
	 */
	virtual ~GvvBrowsable();

	/**
	 * Returns the type of this browsable. The type is used for retrieving
	 * the context menu or when requested or assigning an icon to the
	 * corresponding item
	 *
	 * @return the type name of this browsable
	 */
	virtual const char* getTypeName() const = 0;

	/**
     * Gets the name of this browsable
     *
     * @return the name of this browsable
     */
	virtual const char* getName() const = 0;

	/**
	 * Returns whether this browsable is checkable
	 *
	 * @return true if this browsable is checkable
	 */
	virtual bool isCheckable() const;

	/**
	 * Returns whether this browsable is checked
	 *
	 * @return true if this browsable is checked
	 */
	virtual bool isChecked() const;

	/**
	 * Sets this browsable has checked or not
	 *
	 * @param pFlag specifies whether this browsable is checked or not
	 */
	virtual void setChecked( bool pFlag );

	/**
	 * Returns whether this browsable is editable. Returns false by default.
	 *
	 * @return true if this browsable is editable
	 */
	virtual bool isEditable() const;

	/**
	 * Returns whether this browsable is read-only. Returns false by default
	 *
	 * @return true if this browsable is read only
	 */
	virtual bool isReadOnly() const;

	/**
	 * Tell wheter or not the pipeline has a custom editor.
	 *
	 * @return the flag telling wheter or not the pipeline has a custom editor
	 */
	virtual bool hasCustomEditor() const;

   	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/	

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvvBrowsable( const GvvBrowsable& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvBrowsable& operator=( const GvvBrowsable& );
	
};

} // namespace GvViewerCore

#endif
