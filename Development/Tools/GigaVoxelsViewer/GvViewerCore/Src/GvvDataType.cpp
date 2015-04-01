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

#include "GvvDataType.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;

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
 * Constructor
 ******************************************************************************/
GvvDataType::GvvDataType()
:	_typeList()
,	_nameList()
,	_infoList()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvDataType::~GvvDataType()
{
}

/******************************************************************************
 * Get the data type list used to store voxels in the data structure
 *
 * @return the data type list of voxels
 ******************************************************************************/
const vector< string >& GvvDataType::getTypes() const
{
	return _typeList;
}

/******************************************************************************
 * Set the data type list used to store voxels in the data structure
 *
 * @return the data type list of voxels
 ******************************************************************************/
void GvvDataType::setTypes( const vector< string >& pTypeList )
{
	_typeList = pTypeList;
}

/******************************************************************************
 * Get the name of data type list used to store voxels in the data structure
 *
 * @return the names of data type list of voxels
 ******************************************************************************/
const vector< string >& GvvDataType::getNames() const
{
	return _nameList;
}

/******************************************************************************
 * Set the name of data type list used to store voxels in the data structure
 *
 * @param pNameList the name of the data type list of voxels
 ******************************************************************************/
void GvvDataType::setNames( const vector< string >& pNameList )
{
	_nameList = pNameList;
}

/******************************************************************************
 * Get the info of data type list used to store voxels in the data structure
 *
 * @return the info of data type list of voxels
 ******************************************************************************/
const vector< string >& GvvDataType::getInfo() const
{
	return _infoList;
}

/******************************************************************************
 * Set the info of the data type list used to store voxels in the data structure
 *
 * @param pInfoList the info of the data type list of voxels
 ******************************************************************************/
void GvvDataType::setInfo( const vector< string >& pInfoList )
{
	_infoList = pInfoList;
}
