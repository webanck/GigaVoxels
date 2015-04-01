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

#ifndef _GV_TRANSFER_FUNCTION_H_
#define _GV_TRANSFER_FUNCTION_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

// Cuda
#include <vector_types.h>
#include <driver_types.h>

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

namespace GvUtils
{

/** 
 * @class GvTransferFunction
 *
 * @brief The GvTransferFunction class provides an implementation
 * of a transfer function on the device.
 *
 * Transfer function is a mathematical tool used tu map an input to an output.
 * In computer graphics, a volume renderer can use it to map a sampled density
 * value to an RGBA value.
 */
class GIGASPACE_EXPORT GvTransferFunction
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
	GvTransferFunction();

	/**
	 * Destructor
	 */
	virtual ~GvTransferFunction();

	/**
	 * Get the filename
	 *
	 * @return the filename
	 */
	const std::string& getFilename() const;

	/**
	 * Set the filename
	 *
	 * @param pName the filename
	 */
	void setFilename( const std::string& pName );

	/**
	 * Get the resolution
	 *
	 * @return the resolution
	 */
	unsigned int getResolution() const;

	///**
	// * Set the resolution
	// *
	// * @param pValue the resolution
	// */
	//void setResolution( unsigned int pValue );

	/**
	 * Get the transfer function's data
	 *
	 * @return the transfer function's data
	 */
	inline const float4* getData() const;

	/**
	 * Get the transfer function's data
	 *
	 * @return the transfer function's data
	 */
	inline float4* editData();

	/**
	 * Create the transfer function
	 *
	 * @param pResolution the dimension of the transfer function
	 */
	bool create( unsigned int pResolution );

	/**
	 * Update device memory
	 */
	void updateDeviceMemory();

	/**
	 * Bind the internal data to a specified texture
	 * that can be used to fetch data on device.
	 *
	 * @param pTexRefName name of the texture reference to bind
	 * @param pNormalizedAccess indicates whether texture reads are normalized or not
	 * @param pFilterMode type of texture filter mode
	 * @param pAddressMode type of texture access mode
	 */
	void bindToTextureReference( const void* pSymbol, const char* pTexRefName, bool pNormalizedAccess, cudaTextureFilterMode pFilterMode, cudaTextureAddressMode pAddressMode );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Transfer function file name
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::string _filename;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * Transfer function data
	 */
	float4* _data;

	/**
	 * Transfer function resolution
	 */
	unsigned int _resolution;

	/**
	 * Transfer function data in CUDA memory space
	 */
	cudaArray* _dataArray;

	/**
	 * Channel format descriptor
	 */
	cudaChannelFormatDesc _channelFormatDesc;
	
	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvTransferFunction( const GvTransferFunction& );

	/**
	 * Copy operator forbidden.
	 */
	GvTransferFunction& operator=( const GvTransferFunction& );

};

}

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvTransferFunction.inl"

#endif
