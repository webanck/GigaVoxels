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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * ...
 *
 * @param pSize number of elements to process
 * @param pInput input array
 * @param pOutput output array
 ******************************************************************************/
__global__
void Kernel_StandardAlgorithm( const unsigned int pSize, const unsigned char* pInput, unsigned char* pOutput )
{
	// Retrieve global data index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

	// Check bound
	if ( index < pSize )
	{
		// Compute value
		const unsigned char value = algorithmFunction( index, pInput );

		// Write value
		pOutput[ index ] = value;
	}
}

/******************************************************************************
 * ...
 *
 * @param pSize number of elements to process
 * @param pInput input array
 * @param pOutput output array
 ******************************************************************************/
__global__
void Kernel_StandardAlgorithm_2( const unsigned int pSize, const unsigned char* pInput, unsigned char* pOutput )
{
	// Retrieve global data index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

	// Check bound
	if ( index < pSize )
	{
		// Compute value
		const unsigned char value = algorithmFunction( index, pInput );

		// Compute value
		const unsigned char value_2 = algorithmFunction_2( index, pInput );

		// Write value
		pOutput[ index ] = value + value_2;
	}
}

/******************************************************************************
 * ...
 *
 * @param pSize number of elements to process
 * @param pInput input array
 * @param pOutput output array
 ******************************************************************************/
__global__
void Kernel_AlgorithmWithFunctionPointer( const unsigned int pSize, const unsigned char* pInput, unsigned char* pOutput )
{
	// Retrieve global data index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

	// Check bound
	if ( index < pSize )
	{
		// Compute value
		const unsigned char value = _d_algorithmFunction( pSize, pInput );

		// Write value
		pOutput[ index ] = value;
	}
}

/******************************************************************************
 * ...
 *
 * @param pSize number of elements to process
 * @param pInput input array
 * @param pOutput output array
 ******************************************************************************/
__global__
void Kernel_AlgorithmWithFunctionPointer_2Functions( const unsigned int pSize, const unsigned char* pInput, unsigned char* pOutput )
{
	// Retrieve global data index
	const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;

	// Check bound
	if ( index < pSize )
	{
		// Compute value
		const unsigned char value = _d_algorithmFunction( pSize, pInput );

		// Compute value
		const unsigned char value_2 = _d_algorithmFunction_2( pSize, pInput );

		// Write value
		pOutput[ index ] = value + value_2;
	}
}

/******************************************************************************
 * Algorithm function
 *
 * @param pIndex index of the element to process
 * @param pInput input array
 ******************************************************************************/
__device__
unsigned char algorithmFunction( const unsigned int pIndex, const unsigned char* pInput )
{
	unsigned char value = static_cast< unsigned char >( ( sinf( pInput[ pIndex ] * 3.141592f / 180.f ) * 0.5f + 0.5f ) * 255.f );

	return value;
}

/******************************************************************************
 * Algorithm function
 *
 * @param pIndex index of the element to process
 * @param pInput input array
 ******************************************************************************/
__device__
unsigned char algorithmFunction_2( const unsigned int pIndex, const unsigned char* pInput )
{
	unsigned char value = static_cast< unsigned char >( ( cosf( pInput[ pIndex ] * 3.141592f / 180.f ) * 0.5f + 0.5f ) * 255.f );

	return value;
}
