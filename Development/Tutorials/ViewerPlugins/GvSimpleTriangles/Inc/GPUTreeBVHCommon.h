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

#ifndef _GPU_Tree_BVH_Common_h_
#define _GPU_Tree_BVH_Common_h_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

//////////////////////////////////
//BVH intersection
#define BVH_NUM_THREADS_PER_BLOCK_X 16
#define BVH_NUM_THREADS_PER_BLOCK_Y 12
///////////////////////////////////

/**
 * ...
 */
#define BVH_TRAVERSAL_STACK_SIZE 16

/**
 * ...
 */
#define BVH_DATA_PAGE_SIZE 32

/**
 * ...
 */
#define BVH_NODE_POOL_SIZE 4147426 //1048576  //NB real: 908460  //262144
/**
 * ...
 */
#define BVH_VERTEX_POOL_SIZE  4000000//983040 //7864320 //491520 //15728640 //30720 //52428800 //41943040 //2097152

/**
 * ...
 */
#define BVH_USE_COLOR 1
/**
 * ...
 */
#define BVH_TRAVERSAL_USE_MASKSTACK 0

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

#endif // !
