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

#ifndef _KERNEL_H_
#define _KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// CUDA
#include <cuda_runtime.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Algorithm function
 */
__device__ unsigned char algorithmFunction( const unsigned int pIndex, const unsigned char* pInput );
__device__ unsigned char algorithmFunction_2( const unsigned int pIndex, const unsigned char* pInput );

/**
 * Define a function pointer type that will be used on host and device code
 */
typedef unsigned char (*FunctionPointerType)( const unsigned int pIndex, const unsigned char* pInput );

/**
 * Device-side function pointer
 */
//__device__ FunctionPointerType _d_algorithmFunction = NULL;
__device__ FunctionPointerType _d_algorithmFunction = algorithmFunction;
__device__ FunctionPointerType _d_algorithmFunction_2 = algorithmFunction_2;

/**
 * Host-side function pointer
 */
FunctionPointerType _h_algorithmFunction = NULL;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/**
 * ...
 *
 * @param pSize number of elements to process
 * @param pInput input array
 * @param pOutput output array
 */
__global__ void Kernel_StandardAlgorithm( const unsigned int pSize, const unsigned char* pInput, unsigned char* pOutput );
__global__ void Kernel_StandardAlgorithm_2( const unsigned int pSize, const unsigned char* pInput, unsigned char* pOutput );

/**
 * ...
 *
 * @param pSize number of elements to process
 * @param pInput input array
 * @param pOutput output array
 */
__global__ void Kernel_AlgorithmWithFunctionPointer( const unsigned int pSize, const unsigned char* pInput, unsigned char* pOutput );

/**
 * ...
 *
 * @param pSize number of elements to process
 * @param pInput input array
 * @param pOutput output array
 */
__global__ void Kernel_AlgorithmWithFunctionPointer_2Functions( const unsigned int pSize, const unsigned char* pInput, unsigned char* pOutput );

///**
// * Algorithm function
// *
// * @param pSize number of elements to process
// * @param pInput input array
// * @param pOutput output array
// */
//__device__ unsigned char algorithmFunction( const unsigned int pSize, const unsigned char* pInput );
//
///**
// * Algorithm function
// *
// * @param pSize number of elements to process
// * @param pInput input array
// * @param pOutput output array
// */
//__device__ unsigned char algorithmFunction_2( const unsigned int pSize, const unsigned char* pInput );

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "Kernel.inl"

#endif
