/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2013 INRIA <http://www.inria.fr/>
 *
 * Authors : GigaVoxels Team
 *
 * GigaVoxels is distributed under a dual-license scheme.
 * You can obtain a specific license from Inria at gigavoxels-licensing@inria.fr.
 * Otherwise the default license is the GPL version 3.
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

#ifndef _SHADERFRACTAL_H_
#define _SHADERFRACTAL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GvError.h>

// Project
#include "ShaderFractalKernel.hcu"

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
 * @class ShaderFractal
 *
 * @brief The ShaderFractal class provides...
 *
 * ...
 */
class ShaderFractal
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * ...
	 */
	typedef ShaderFractalKernel KernelType;

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	//ShaderFractalKernel getKernelObject()
	//{
	//	return kernelObject;
	//}

	/**
	 * ...
	 *
	 * @param resolution ...
	 */
	static void createTransferFunction( uint resolution )
	{
		transferFuncRes = resolution;
		transferFunc = new float4[transferFuncRes];

		// allocate cuda storage
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
		GV_CUDA_SAFE_CALL( cudaMallocArray(&d_transferFuncArray, &channelDesc, transferFuncRes, 1) );

		// bind array to cuda texture
		transFuncTex.normalized = true;
		transFuncTex.filterMode = cudaFilterModeLinear;
		transFuncTex.addressMode[0] = cudaAddressModeClamp;
		transFuncTex.addressMode[1] = cudaAddressModeClamp;
		transFuncTex.addressMode[2] = cudaAddressModeClamp;
		GV_CUDA_SAFE_CALL( cudaBindTextureToArray(transFuncTex, d_transferFuncArray, channelDesc) );
	}

	/**
	 * ...
	 */
	static void transferFunctionUpdated()
	{
		cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc,
			transferFuncRes * sizeof(float4), cudaMemcpyHostToDevice);
	}

	/**
	 * ...
	 *
	 * @return ...
	 */
	static float4 *getTransferFunction()
	{
		return transferFunc;
	}
	
	/**
	 * ...
	 *
	 * @return ...
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

	//ShaderFractalKernel kernelObject;

	// Transfer function
	/**
	 * ...
	 */
	static float4* transferFunc;
	/**
	 * ...
	 */
	static uint transferFuncRes;
	/**
	 * ...
	 */
	static cudaArray* d_transferFuncArray;
	
	/******************************** METHODS *********************************/

};

/**
 * ...
 */
float4 *ShaderFractal::transferFunc = NULL;
/**
 * ...
 */
uint ShaderFractal::transferFuncRes = 0;
/**
 * ...
 */
cudaArray *ShaderFractal::d_transferFuncArray = NULL;

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

//#include "ShaderFractal.inl"

#endif // !_SHADERFRACTAL_H_
