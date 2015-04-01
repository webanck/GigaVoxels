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

#ifndef _GS_DEVICE_H_
#define _GS_DEVICE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaSpace
#include "GsCompute/GsComputeConfig.h"
#include "GsCompute/GsDeviceProperties.h"

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

// GigaSpace
namespace GsCompute
{
	class GsDeviceManager;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GsCompute
{
	
/** 
 * @class GsDevice
 *
 * @brief The GsDevice class provides an interface for accessing device properties.
 *
 * @ingroup GvCore
 *
 * It holds device properties.
 */
class GSCOMPUTE_EXPORT GsDevice
{

	/**************************************************************************
	 ***************************** FRIEND SECTION *****************************
	 **************************************************************************/

	/**
	 * Device manager is responsible of all device instantiations
	 */
	friend class GsDeviceManager;

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Device properties
	 */
	GsDeviceProperties mProperties;

	/**
	 * Warp size (in number of threads)
	 */
	static int _warpSize;

	/**
	 * Index of the device
	 */
	int _index;

	/**
	 * Name
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::string _name;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/******************************** METHODS *********************************/
				
	/**
	 * Print information about the device
	 */
	void printInfo() const;
		
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
		
	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GsDevice();

	/**
	 * Destructor
	 */
	virtual ~GsDevice();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} // namespace GsCompute

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#endif // !GVDEVICE_H
