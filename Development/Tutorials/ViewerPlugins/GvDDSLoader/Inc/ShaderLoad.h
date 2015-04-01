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

#ifndef _SHADERLOAD_H_
#define _SHADERLOAD_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GvError.h>

// Project
#include "ShaderLoad.hcu"

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

/** 
 * @class ShaderLoad
 *
 * @brief The ShaderLoad class provides...
 *
 * ...
 */
class ShaderLoad
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition of the associated device-side object
	 */
	typedef ShaderLoadKernel KernelType;

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Get the associated device-side object
	 *
	 * @return the associated device-side object
	 */
	inline ShaderLoadKernel getKernelObject();

	/**
	 * ...
	 */
	static void createTransferFunction( uint resolution )
	{
		transferFuncRes = resolution;
		transferFunc = new float4[ transferFuncRes ];

		// Allocate cuda storage
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc< float4 >();
		GV_CUDA_SAFE_CALL( cudaMallocArray( &d_transferFuncArray, &channelDesc, transferFuncRes, 1 ) );

		// Bind array to cuda texture
		transerFunctionTexture.normalized = true;
		transerFunctionTexture.filterMode = cudaFilterModeLinear;
		transerFunctionTexture.addressMode[ 0 ] = cudaAddressModeClamp;
		transerFunctionTexture.addressMode[ 1 ] = cudaAddressModeClamp;
		transerFunctionTexture.addressMode[ 2 ] = cudaAddressModeClamp;
		GV_CUDA_SAFE_CALL( cudaBindTextureToArray( transerFunctionTexture, d_transferFuncArray, channelDesc ) );
	}

	/**
	 * ...
	 */
	static void transferFunctionUpdated()
	{
		cudaMemcpyToArray( d_transferFuncArray, 0, 0, transferFunc, transferFuncRes * sizeof( float4 ), cudaMemcpyHostToDevice );
	}

	/**
	 * ...
	 */
	static float4* getTransferFunction()
	{
		return transferFunc;
	}

	/**
	 * ...
	 */
	static uint getTransferFunctionRes()
	{
		return transferFuncRes;
	}

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * The associated device-side object
	 */
	ShaderLoadKernel kernelObject;

	// Transfer function
	static float4* transferFunc;
	static uint transferFuncRes;
	static cudaArray* d_transferFuncArray;

	/******************************** METHODS *********************************/

};

float4* ShaderLoad::transferFunc = NULL;
uint ShaderLoad::transferFuncRes = 0;
cudaArray* ShaderLoad::d_transferFuncArray = NULL;

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "ShaderLoad.inl"

#endif // !_SHADERLOAD_H_
