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

#include "GvUtils/GvTransferFunction.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvError.h"

// Cuda
#include <cuda_runtime.h>

// STL
#include <iostream>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvVoxelizer
using namespace GvUtils;

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
GvTransferFunction::GvTransferFunction()
:	_filename()
,	_data( NULL )
,	_resolution( 0 )
,	_dataArray( NULL )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvTransferFunction::~GvTransferFunction()
{
	// Free host memory
	delete [] _data;

	// Free device memory
	GV_CUDA_SAFE_CALL( cudaFreeArray( _dataArray ) );
}

/******************************************************************************
 * 3D model file name
 ******************************************************************************/
const std::string& GvTransferFunction::getFilename() const
{
	return _filename;
}

/******************************************************************************
 * 3D model file name
 ******************************************************************************/
void GvTransferFunction::setFilename( const std::string& pName )
{
	_filename = pName;
}
	
/******************************************************************************
 * Data resolution
 ******************************************************************************/
unsigned int GvTransferFunction::getResolution() const
{
	return _resolution;
}

///******************************************************************************
// * Data resolution
// ******************************************************************************/
//void GvTransferFunction::setResolution( unsigned int pValue )
//{
//	_resolution = pValue;
//}

/******************************************************************************
 * Create the transfer function
 *
 * @param pResolution the dimension of the transfer function
 ******************************************************************************/
bool GvTransferFunction::create( unsigned int pResolution )
{
	bool result = false;

	_resolution = pResolution;

	// Allocate data in host memory
	_data = new float4[ _resolution ];

	// Allocate CUDA array in device memory
	_channelFormatDesc = cudaCreateChannelDesc< float4 >();
	GV_CUDA_SAFE_CALL( cudaMallocArray( &_dataArray, &_channelFormatDesc, _resolution, 1 ) );

	result = true;

	return result;
}

/******************************************************************************
 * Update device memory
 ******************************************************************************/
void GvTransferFunction::updateDeviceMemory()
{
	// Copy to device memory some data located at address _data in host memory
	cudaMemcpyToArray( _dataArray, 0, 0, _data, _resolution * sizeof( float4 ), cudaMemcpyHostToDevice );
}

/******************************************************************************
 * Bind the internal data to a specified texture
 * that can be used to fetch data on device.
 *
 * @param pTexRefName name of the texture reference to bind
 * @param pNormalizedAccess indicates whether texture reads are normalized or not
 * @param pFilterMode type of texture filter mode
 * @param pAddressMode type of texture access mode
 ******************************************************************************/
void GvTransferFunction::bindToTextureReference( const void* pSymbol, const char* pTexRefName, bool pNormalizedAccess, cudaTextureFilterMode pFilterMode, cudaTextureAddressMode pAddressMode )
{
	std::cout << "bindToTextureReference : " << pTexRefName << std::endl;

	textureReference* texRefPtr;
	GV_CUDA_SAFE_CALL( cudaGetTextureReference( (const textureReference **)&texRefPtr, pSymbol ) );

	texRefPtr->normalized = pNormalizedAccess; // Access with normalized texture coordinates
	texRefPtr->filterMode = pFilterMode;
	texRefPtr->addressMode[ 0 ] = pAddressMode; // Wrap texture coordinates
	texRefPtr->addressMode[ 1 ] = pAddressMode;
	texRefPtr->addressMode[ 2 ] = pAddressMode;

	// Bind array to 3D texture
	GV_CUDA_SAFE_CALL( cudaBindTextureToArray( (const textureReference *)texRefPtr, _dataArray, &_channelFormatDesc ) );
}
