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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

#include <cassert>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvViewerGui
{

/******************************************************************************
 * Returns whether this manager is empty
 *
 * @return	true if this manager is empty
 ******************************************************************************/
inline bool GvvActionManager::isEmpty() const
{
	return getNbActions() == 0;
}

/******************************************************************************
 * Returns the number of elements
 *
 * @return	the number of elements
 ******************************************************************************/
inline unsigned int GvvActionManager::getNbActions() const
{
	return mActions.size();
}

/******************************************************************************
 * Returns the i-th element
 *
 * @param	pIndex	specifies the index of the desired element
 *
 * @return	a const pointer to the pIndex-th element
 ******************************************************************************/
inline const GvvAction* GvvActionManager::getAction( unsigned int pIndex ) const
{
	assert( pIndex < getNbActions() );
	return mActions[ pIndex ];
}	

/******************************************************************************
 * Returns the i-th element
 *
 * @param	pIndex	specifies the index of the desired element
 *
 * @return	a pointer to the pIndex-th element
 ******************************************************************************/
inline GvvAction* GvvActionManager::editAction( unsigned int pIndex )
{
	return const_cast< GvvAction* >( getAction( pIndex ) );
}

/******************************************************************************
 * Returns the element represented by the specified name
 *
 * @param	pName specifies the name of the desired element
 *
 * @return	a const pointer to the element or null if not found
 ******************************************************************************/
inline const GvvAction* GvvActionManager::getAction( const QString& pName ) const
{
	unsigned int lIndex = findAction( pName );
	if
		( lIndex != -1 )
	{
		return getAction( lIndex );
	}
	return NULL;
}

/******************************************************************************
 * Returns the element represented by the specified name
 *
 * @param	pName specifies the name of the desired element
 *
 * @return	a pointer to the element or null if not found
 ******************************************************************************/
inline GvvAction* GvvActionManager::editAction( const QString& pName )
{
	return const_cast< GvvAction* >( getAction( pName ) );
}

} // namespace GvViewerGui
