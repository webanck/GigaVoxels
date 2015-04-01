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

#ifndef _GV_UTILS_H_
#define _GV_UTILS_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * MACRO
 * 
 * Useful and required type definition for producer kernels
 * - it is used to access the DataStructure typedef passed in argument
 *
 * @param TDataStructureType a data structure type (should be the template parameter of a Producer Kernel)
 */
#define GV_MACRO_PRODUCER_KERNEL_REQUIRED_TYPE_DEFINITIONS( TDataStructureType )	\
	/**																				\
	 * Data Structure device-side associated object									\
	 */																				\
	typedef typename TDataStructureType::VolTreeKernelType DataStructureKernel;		\
																					\
	/**																				\
	 * Type definition of the node tile resolution									\
	 */																				\
	typedef typename TDataStructureType::NodeTileResolution NodeRes;				\
																					\
	/**																				\
	 * Type definition of the brick resolution										\
	 */																				\
	typedef typename TDataStructureType::BrickResolution BrickRes;					\
																					\
	/**																				\
	 * Enumeration to define the brick border size									\
	 */																				\
	enum																			\
	{																				\
		BorderSize = TDataStructureType::BrickBorderSize							\
	};																				\

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

#endif // !_GV_UTILS_H_
