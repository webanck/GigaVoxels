/*
 * GigaSpace is a ray-guided streaming library used for efficient
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

#ifndef _GS_DEVICE_MANAGER_H_
#define _GS_DEVICE_MANAGER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaSpace
#include "GsCompute/GsComputeConfig.h"

// STL
#include <vector>
#include <cstddef>

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
	class GsDevice;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GsCompute
{
	
/** 
 * @class GsDeviceManager
 *
 * @brief The GsDeviceManager class provides way to access all available devices.
 *
 * @ingroup GsCompute
 *
 * The GsDeviceManager class is the main accesor of all devices.
 */
class GSCOMPUTE_EXPORT GsDeviceManager
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Get the device manager
	 *
	 * @return the device manager
	 */
	static GsDeviceManager& get();

	/**
	 * Initialize the device manager
	 */
	bool initialize();

	/**
	 * Finalize the device manager
	 */
	void finalize();

	/**
	 * Get the number of devices
	 *
	 * @return the number of devices
	 */
	size_t getNbDevices() const;

	/**
	 * Get the device given its index
	 *
	 * @param the index of the requested device
	 *
	 * @return the requested device
	 */
	const GsDevice* getDevice( int pIndex ) const;

	/**
	 * Get the device given its index
	 *
	 * @param the index of the requested device
	 *
	 * @return the requested device
	 */
	GsDevice* editDevice( int pIndex );
		
	/**
	 * Get the current used device if set
	 *
	 * @return the current device
	 */
	const GsDevice* getCurrentDevice() const;

	/**
	 * Get the current used device if set
	 *
	 * @return the current device
	 */
	GsDevice* editCurrentDevice();

	/**
	 * Set the current device
	 *
	 * @param pDevice the device
	 */
	void setCurrentDevice( GsDevice* pDevice );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** TYPEDEFS ********************************/

	/**
	 * The unique device manager
	 */
	static GsDeviceManager* msInstance;

	/**
	 * The container of devices
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::vector< GsDevice* > _devices;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * The current device
	 */
	GsDevice* _currentDevice;

	/**
	 * Flag to tell wheter or not the device manager is initialized
	 */
	bool _isInitialized;

	/**
	 * Flag to tell wheter or not the device manager has found at least
	 * one compliant hardware
	 */
	bool _hasCompliantHardware;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GsDeviceManager();

	/**
	 * Destructor
	 */
	~GsDeviceManager();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GsDeviceManager( const GsDeviceManager& );

	/**
	 * Copy operator forbidden.
	 */
	GsDeviceManager& operator=( const GsDeviceManager& );

};

} // namespace GsCompute

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#endif // !_GS_DEVICE_MANAGER_H_
