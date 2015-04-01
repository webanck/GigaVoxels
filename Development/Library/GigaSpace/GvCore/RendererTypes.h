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

#ifndef GVRENDERERTYPES_H
#define GVRENDERERTYPES_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Gigavoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/Array3D.h"
#include "GvCore/Array3DGPULinear.h"
#include "GvCore/Array3DGPUTex.h"
#include "GvCore/GPUPool.h"

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

namespace GvCore
{

	//Specialisations of GPUPool_getKernelPool (cf. GPUPool.h)
	/**
	 * GPUPool_KernelPoolFromHostPool struct specialization
	 */
	template< class TList >
	struct GPUPool_KernelPoolFromHostPool< Array3DGPUTex, TList >
	{
		typedef GPUPoolKernel< Array3DKernelTex, TList > Result;
	};

	/**
	 * GPUPool_KernelPoolFromHostPool struct specialization
	 */
	template< class TList >
	struct GPUPool_KernelPoolFromHostPool< Array3DGPULinear, TList >
	{
		typedef GPUPoolKernel< Array3DKernelLinear, TList > Result;
	};

	/**
	 * GPUPool_KernelPoolFromHostPool struct specialization
	 */
	template< class TList >
	struct GPUPool_KernelPoolFromHostPool< Array3D, TList >
	{
		typedef GPUPoolKernel< Array3DKernelLinear, TList > Result;
	};

} //namespace GvCore

//#include "RendererTypes.inl"

#endif
