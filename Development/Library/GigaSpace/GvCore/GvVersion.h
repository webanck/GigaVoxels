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

#ifndef GVVERSION_H
#define GVVERSION_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

 // STL
 #include <string>
 
 /******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

 /**
  * Version information used to query API's version at compile time
  */
#define GV_API_VERSION_MAJOR 1
#define GV_API_VERSION_MINOR 0
#define GV_API_VERSION_PATCH 0

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvCore
{

/** 
 * @class GvVersion
 *
 * @brief The GvVersion class provides version information for the GigaVoxels API
 *
 * @ingroup GvCore
 *
 * ...
 */
class GIGASPACE_EXPORT GvVersion
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/
		
public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Return the major version number, e.g., 1 for "1.2.3"
	 *
	 * @return the major version number
	 */
	static unsigned int getMajor();
	
	/**
	 * Return the minor version number, e.g., 2 for "1.2.3"
	 *
	 * @return the minor version number
	 */
	static unsigned int getMinor();
	
	/**
	 * Return the patch version number, e.g., 3 for "1.2.3"
	 *
	 * @return the patch version number
	 */
	static unsigned int getPatch();

	/**
	 * Return the full version number as a string, e.g., "1.2.3"
	 *
	 * @return the the full version number
	 */
	static std::string getVersion();

	/**
	 * Return true if the current version >= (major, minor, patch)
	 *
	 * @param pMajor The major version
	 * @param pMinor The minor version
	 * @param pPatch The patch version
	 *
	 * @return true if the current version >= (major, minor, patch)
	 */
	static bool isAtLeast( unsigned int pMajor, unsigned int pMinor, unsigned int pPatch );

	/**
	 * Return true if the named feature is available in this version
	 *
	 * @param pName The name of a feature
	 *
	 * @return true if the named feature is available in this version
	 */
	static bool hasFeature( const std::string& pName );
	
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
	
};

} // namespace GvCore

#endif // !GVVERSION_H
