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

#ifndef _GVV_DATA_TYPE_H_
#define _GVV_DATA_TYPE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvCoreConfig.h"

// STL
#include <string>
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

namespace GvViewerCore
{

/** 
 * @class GvvDataType
 *
 * @brief The GvvDataType class provides info on voxels data types
 * stored in a GigaVoxels data structure.
 *
 * ...
 */
class GVVIEWERCORE_EXPORT GvvDataType
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvvDataType();

	/**
	 * Destructor
	 */
	virtual ~GvvDataType();

	/**
	 * Get the data type list used to store voxels in the data structure
	 *
	 * @return the data type list of voxels
	 */
	const std::vector< std::string >& getTypes() const;

	/**
	 * Set the data type list used to store voxels in the data structure
	 *
	 * @param pTypeList the data type list of voxels
	 */
	void setTypes( const std::vector< std::string >& pTypeList );

	/**
	 * Get the name of data type list used to store voxels in the data structure
	 *
	 * @return the names of data type list of voxels
	 */
	const std::vector< std::string >& getNames() const;

	/**
	 * Set the name of data type list used to store voxels in the data structure
	 *
	 * @param pNameList the name of the data type list of voxels
	 */
	void setNames( const std::vector< std::string >& pNameList );

	/**
	 * Get the info of data type list used to store voxels in the data structure
	 *
	 * @return the info of data type list of voxels
	 */
	const std::vector< std::string >& getInfo() const;

	/**
	 * Set the info of the data type list used to store voxels in the data structure
	 *
	 * @param pInfoList the info of the data type list of voxels
	 */
	void setInfo( const std::vector< std::string >& pInfoList );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Data type list used to store voxels in the data structure
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::vector< std::string > _typeList;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * Name of data type list used to store voxels in the data structure
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::vector< std::string > _nameList;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * Info on data type list used to store voxels in the data structure
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::vector< std::string > _infoList;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} // namespace GvViewerCore

#endif // !_GVV_DATA_TYPE_H_
