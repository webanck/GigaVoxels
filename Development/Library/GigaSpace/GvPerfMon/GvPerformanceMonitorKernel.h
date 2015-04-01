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

#ifndef _GV_PERFORMANCE_MONITOR_KERNEL_H_
#define _GV_PERFORMANCE_MONITOR_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/gvTypes.h"

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

namespace GvPerfMon
{

/******************************************************************************
 * Get clock
 ******************************************************************************/
__device__
__forceinline__ GvCore::uint64 getClock();

__device__
__forceinline__ GvCore::uint64 getClock()
{
	GvCore::uint64 res;
	
	// Using inline PTX assembly in CUDA.
	// Target ISA Notes : %clock64 requires sm_20 or higher.
	//
	// The constraint on output is "=l" :
	//   - '=' modifier specified that the register is written to
	//   - 'l' modifier refers to the register type ".u64 reg"
	asm volatile ("mov.u64 %0,%clock64;" : "=l"(res) : : );		
	
	return res;
}

} // namespace GvPerfMon

#endif
