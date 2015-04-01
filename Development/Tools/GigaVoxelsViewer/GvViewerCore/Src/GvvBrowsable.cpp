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

#include "GvvBrowsable.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;

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
 * Default constructor
 ******************************************************************************/
GvvBrowsable::GvvBrowsable()
{
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvBrowsable::~GvvBrowsable()
{
}

/******************************************************************************
 * Returns whether this browsable is checkable
 *
 * @return true if this browsable is checkable
 ******************************************************************************/
bool GvvBrowsable::isCheckable() const
{
	return false;
}

/******************************************************************************
 * Returns whether this browsable is enabled
 *
 * @return true if this browsable is enabled
 ******************************************************************************/
bool GvvBrowsable::isChecked() const
{
	assert( isCheckable() );
	return true;
}

/******************************************************************************
 * Sets this browsable has checked or not
 *
 * @param pFlag specifies whether this browsable is checked or not
 ******************************************************************************/
void GvvBrowsable::setChecked( bool pFlag )
{
	assert( isCheckable() );
}

/******************************************************************************
 * Returns whether this browsable is editable. Returns false by default.
 *
 * @return true if this browsable is editable
 ******************************************************************************/
bool GvvBrowsable::isEditable() const
{
	return false;
}

/******************************************************************************
 * Returns whether this browsable is read-only
 *
 * @return true if this browsable is read only
 ******************************************************************************/
bool GvvBrowsable::isReadOnly() const
{
	return false;
}

/******************************************************************************
 * Tell wheter or not the pipeline has a custom editor.
 *
 * @return the flag telling wheter or not the pipeline has a custom editor
 ******************************************************************************/
bool GvvBrowsable::hasCustomEditor() const
{
	return false;
}
