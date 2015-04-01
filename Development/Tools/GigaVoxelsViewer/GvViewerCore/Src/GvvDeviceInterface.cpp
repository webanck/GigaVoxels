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

#include "GvvDeviceInterface.h"

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

/**
 * Tag name identifying a space profile element
 */
const char* GvvDeviceInterface::cTypeName = "Device";

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvvDeviceInterface::GvvDeviceInterface()
:	GvvBrowsable()
,	_deviceName()
,	_computeCapabilityMajor( 0 )
,	_computeCapabilityMinor( 0 )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvDeviceInterface::~GvvDeviceInterface()
{
}

/******************************************************************************
 * Returns the type of this browsable. The type is used for retrieving
 * the context menu or when requested or assigning an icon to the
 * corresponding item
 *
 * @return the type name of this browsable
 ******************************************************************************/
const char* GvvDeviceInterface::getTypeName() const
{
	return cTypeName;
}

/******************************************************************************
 * Gets the name of this browsable
 *
 * @return the name of this browsable
 ******************************************************************************/
const char* GvvDeviceInterface::getName() const
{
	return "Device";
}

/******************************************************************************
 * Get device name
 *
 * @return device name
 ******************************************************************************/
const string& GvvDeviceInterface::getDeviceName() const
{
	return _deviceName;
}

/******************************************************************************
 * Set device name
 *
 * @param pName device name
 ******************************************************************************/
void GvvDeviceInterface::setDeviceName( const string& pName )
{
	_deviceName = pName;
}

/******************************************************************************
 * Get compute capability (major number)
 *
 * @return major compute capability 
 ******************************************************************************/
int GvvDeviceInterface::getComputeCapabilityMajor() const
{
	return _computeCapabilityMajor;
}

/******************************************************************************
 * Set compute capability (major number)
 *
 * @param pValue major compute capability 
 ******************************************************************************/
void GvvDeviceInterface::setComputeCapabilityMajor( int pValue )
{
	_computeCapabilityMajor = pValue;
}

/******************************************************************************
 * Get compute capability (minor number)
 *
 * @return minor compute capability 
 ******************************************************************************/
int GvvDeviceInterface::getComputeCapabilityMinor() const
{
	return _computeCapabilityMinor;
}

/******************************************************************************
 * Set compute capability (minor number)
 *
 * @param pValue minor compute capability 
 ******************************************************************************/
void GvvDeviceInterface::setComputeCapabilityMinor( int pValue )
{
	_computeCapabilityMinor = pValue;
}
