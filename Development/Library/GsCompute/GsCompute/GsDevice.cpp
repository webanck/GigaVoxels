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

#include "GsCompute/GsDevice.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// CUDA toolkit
#include <cuda_runtime.h>

// STL
#include <iostream>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// Gigavoxels
using namespace GsCompute;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Warp size (in number of threads)
 */
int GsDevice::_warpSize = 32;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
GsDevice::GsDevice()
:	mProperties()
,	_index( 0 )
,	_name()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GsDevice::~GsDevice()
{
}

/******************************************************************************
 * Print information about the device
 ******************************************************************************/
void GsDevice::printInfo() const
{
	// LOG GPU Computing device properties (i.e. CUDA)
	cudaDeviceProp cudaDeviceProperties;
	cudaGetDeviceProperties( &cudaDeviceProperties, _index );

	std::cout << "- device " << _index << " has compute capability " << cudaDeviceProperties.major << "." << cudaDeviceProperties.minor << std::endl;
}
