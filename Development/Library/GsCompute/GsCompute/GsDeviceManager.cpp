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

#include "GsCompute/GsDeviceManager.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaSpace
#include "GsCompute/GsDevice.h"

// CUDA toolkit
#include <cuda_runtime.h>

// System
#include <cassert>
#include <iostream>
#include <cstdio>

// GL
#include <GL/glew.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// Gigavoxels
using namespace GsCompute;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * The unique device manager
 */
GsDeviceManager* GsDeviceManager::msInstance = NULL;

/**
 * Required compute capability
 */
#define GV_REQUIRED_COMPUTE_CAPABILITY_MAJOR 2
#define GV_REQUIRED_COMPUTE_CAPABILITY_MINOR 0

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Get the device manager.
 *
 * @return the device manager
 ******************************************************************************/
GsDeviceManager& GsDeviceManager::get()
{
	if ( msInstance == NULL )
	{
		msInstance = new GsDeviceManager();
	}
	assert( msInstance != NULL );
	return *msInstance;
}

/******************************************************************************
 * Constructor.
 ******************************************************************************/
GsDeviceManager::GsDeviceManager()
:	_devices()
,	_currentDevice( NULL )
,	_isInitialized( false )
,	_hasCompliantHardware( false )
{
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GsDeviceManager::~GsDeviceManager()
{
	finalize();
}

/******************************************************************************
 * Initialize the device manager
 ******************************************************************************/
bool GsDeviceManager::initialize()
{
	// Check the flag to tell wheter or not the device manager is initialized
	if ( _isInitialized )
	{
		return _hasCompliantHardware;
	}
	
	// Retrieve number of devices
	int nbDevices;
	cudaGetDeviceCount( &nbDevices );

	// Iterate through devices
	for ( int i = 0; i < nbDevices; i++ )
	{
		// Iterate device properties
		cudaDeviceProp cudaDeviceProperties;
		cudaGetDeviceProperties( &cudaDeviceProperties, i );

		// TO DO
		// - modify this, cause the value could be different from devices ?
		// - in practice, it should not occur
		// ...
		// Register warp size
		GsDevice::_warpSize = cudaDeviceProperties.warpSize;

		// Create GigaSpace device and fill its properties
		GsDevice* device = new GsDevice();
		if ( device != NULL )
		{
			device->_index = i;

			// Retrieve device property
			GsDeviceProperties& deviceProperties = device->mProperties;
			device->_name = cudaDeviceProperties.name;
			deviceProperties._computeCapabilityMajor = cudaDeviceProperties.major;
			deviceProperties._computeCapabilityMinor = cudaDeviceProperties.minor;
			deviceProperties._warpSize = cudaDeviceProperties.warpSize;
			
			// Store the GigaSpace device
			_devices.push_back( device );

			// TEST the architecture
			// The GigaSpace Engine require devices with at least compute capability
			if ( cudaDeviceProperties.major >= GV_REQUIRED_COMPUTE_CAPABILITY_MAJOR &&
				 cudaDeviceProperties.minor >= GV_REQUIRED_COMPUTE_CAPABILITY_MINOR )
			{
				// Update the flag to tell wheter or not the device manager has found at least
				// one compliant hardware
				_hasCompliantHardware = true;
			}
		}
	}

	// Architecture(s) report
	std::cout << "\nThe GigaVoxels-GigaSpace Engine requires devices with at least compute capability " << GV_REQUIRED_COMPUTE_CAPABILITY_MAJOR << "." << GV_REQUIRED_COMPUTE_CAPABILITY_MINOR << std::endl;
	for ( int i = 0; i < getNbDevices(); i++ )
	{
		getDevice( i )->printInfo();
	}
	if ( _hasCompliantHardware )
	{
		std::cout << "OK" << std::endl;
	}
	else
	{
		// Test failed
		// 
		// TO DO : exit program ?
		std::cout << "ERROR : " << "There is no compliant devices" << std::endl;
	}

	// Update the flag to tell wheter or not the device manager is initialized
	_isInitialized = true;

	return _hasCompliantHardware;
}

/******************************************************************************
 * Finalize the device manager
 ******************************************************************************/
void GsDeviceManager::finalize()
{
	for ( int i = 0; i < _devices.size(); i++ )
	{
		delete _devices[ i ];
		_devices[ i ] = NULL;
	}
	_devices.clear();

	// Update the flag to tell wheter or not the device manager is initialized
	_isInitialized = false;

	// Update the flag to tell wheter or not the device manager has found at least
	// one compliant hardware
	_hasCompliantHardware = false;
}

/******************************************************************************
 * Get the number of devices
 *
 * @return the number of devices
 ******************************************************************************/
size_t GsDeviceManager::getNbDevices() const
{
	return _devices.size();
}

/******************************************************************************
 * Get the device given its index
 *
 * @param the index of the requested device
 *
 * @return the requested device
 ******************************************************************************/
const GsDevice* GsDeviceManager::getDevice( int pIndex ) const
{
	assert( pIndex < _devices.size() );
	return _devices[ pIndex ];
}

/******************************************************************************
 * Get the device given its index
 *
 * @param the index of the requested device
 *
 * @return the requested device
 ******************************************************************************/
GsDevice* GsDeviceManager::editDevice( int pIndex )
{
	assert( pIndex < _devices.size() );
	return _devices[ pIndex ];
}

/******************************************************************************
 * Get the current used device if set
 *
 * @return the current device
 *******************************************************************************/
const GsDevice* GsDeviceManager::getCurrentDevice() const
{
	return _currentDevice;
}

/******************************************************************************
 * Get the current used device if set
 *
 * @return the current device
 ******************************************************************************/
GsDevice* GsDeviceManager::editCurrentDevice()
{
	return _currentDevice;
}

/******************************************************************************
 * Set the current device
 *
 * @param pDevice the device
 ******************************************************************************/
void GsDeviceManager::setCurrentDevice( GsDevice* pDevice )
{
	_currentDevice = pDevice;
}
