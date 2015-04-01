/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2014 INRIA <http://www.inria.fr/>
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

#include "GvCore/GvVersion.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// STL
#include <string>
#include <sstream>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// Gigavoxels
using namespace GvCore;

// STL
using namespace std;

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
 * Return the major version number, e.g., 1 for "1.2.3"
 *
 * @return the major version number
 ******************************************************************************/
unsigned int GvVersion::getMajor()
{
	return static_cast< unsigned int >( GV_API_VERSION_MAJOR );
}

/******************************************************************************
 * Return the minor version number, e.g., 2 for "1.2.3"
 *
 * @return the minor version number
 ******************************************************************************/
unsigned int GvVersion::getMinor()
{
	return static_cast< unsigned int >( GV_API_VERSION_MINOR );
}

/******************************************************************************
 * Return the patch version number, e.g., 3 for "1.2.3"
 *
 * @return the patch version number
 ******************************************************************************/
unsigned int GvVersion::getPatch()
{
	return static_cast< unsigned int >( GV_API_VERSION_PATCH );
}

/******************************************************************************
 * Return the full version number as a string, e.g., "1.2.3"
 *
 * @return the the full version number
 ******************************************************************************/
string GvVersion::getVersion()
{
	static string version( "" );
	if ( version.empty() )
	{
		// Cache the version string
		ostringstream stream;
		stream << GV_API_VERSION_MAJOR << "."
			   << GV_API_VERSION_MINOR << "."
			   << GV_API_VERSION_PATCH;
		version = stream.str();
	}

	return version;
}

/******************************************************************************
 * Return true if the current version >= (major, minor, patch)
 *
 * @param pMajor ...
 * @param pMinor ...
 * @param pPatch ...
 *
 * @return true if the current version >= (major, minor, patch)
 ******************************************************************************/
bool GvVersion::isAtLeast( unsigned int pMajor, unsigned int pMinor, unsigned int pPatch )
{
	if ( GV_API_VERSION_MAJOR < pMajor )
	{
		return false;
	}

	if ( GV_API_VERSION_MAJOR > pMajor )
	{
		return true;
	}

	if ( GV_API_VERSION_MINOR < pMinor )
	{
		return false;
	}

	if ( GV_API_VERSION_MINOR > pMinor )
	{
		return true;
	}

	if ( GV_API_VERSION_PATCH < pPatch )
	{
		return false;
	}

	return true;
}

/******************************************************************************
 * Return true if the named feature is available in this version
 *
 * @param pName The name of a feature
 *
 * @return true if the named feature is available in this version
 ******************************************************************************/
bool GvVersion::hasFeature( const string& pName )
{
	// Not yet implemented
	return false;
}
