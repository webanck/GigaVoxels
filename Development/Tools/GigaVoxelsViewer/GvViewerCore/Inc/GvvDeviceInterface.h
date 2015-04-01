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

#ifndef _GVV_DEVICE_INTERFACE_H_
#define _GVV_DEVICE_INTERFACE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvCoreConfig.h"
#include "GvvBrowsable.h"

// STL
#include <string>

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
 * @class GvvDeviceInterface
 *
 * @brief The GvvDeviceInterface class provides info on a device.
 *
 * ...
 */
class GVVIEWERCORE_EXPORT GvvDeviceInterface : public GvvBrowsable
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	static const char* cTypeName;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvvDeviceInterface();

	/**
	 * Destructor
	 */
	virtual ~GvvDeviceInterface();

	/**
	 * Returns the type of this browsable. The type is used for retrieving
	 * the context menu or when requested or assigning an icon to the
	 * corresponding item
	 *
	 * @return the type name of this browsable
	 */
	virtual const char* getTypeName() const;

	/**
     * Gets the name of this browsable
     *
     * @return the name of this browsable
     */
	virtual const char* getName() const;

	/**
	 * Get device name
	 *
	 * @return device name
	 */
	const std::string& getDeviceName() const;

	/**
	 * Set device name
	 *
	 * @param pName device name
	 */
	void setDeviceName( const std::string& pName );

	/**
	 * Get compute capability (major number)
	 *
	 * @return major compute capability 
	 */
	int getComputeCapabilityMajor() const;

	/**
	 * Set compute capability (major number)
	 *
	 * @param pValue major compute capability 
	 */
	void setComputeCapabilityMajor( int pValue );

	/**
	 * Get compute capability (minor number)
	 *
	 * @return minor compute capability 
	 */
	int getComputeCapabilityMinor() const;

	/**
	 * Set compute capability (minor number)
	 *
	 * @param pValue minor compute capability 
	 */
	void setComputeCapabilityMinor( int pValue );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Name
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::string _deviceName;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * Compute capability (major number)
	 */
	int _computeCapabilityMajor;

	/**
	 * Compute capability (minor number)
	 */
	int _computeCapabilityMinor;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} // namespace GvViewerCore

#endif // !_GVV_DEVICE_INTERFACE_H_
