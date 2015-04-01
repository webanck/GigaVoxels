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

#ifndef _GV_CONFIG_H_
#define _GV_CONFIG_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

#define RENDER_USE_SWIZZLED_THREADS 1

#define USE_TESLA_OPTIMIZATIONS 0

#define USE_BRICK_USAGE_OPTIM 0

//#define GV_USE_BRICK_MINMAX 1

/**
 * CUDPP library
 *
 * By default, cudpp library is not used
 */
#define USE_CUDPP_LIBRARY 1

// ---------------- PERFORMANCE MONITOR ----------------

/**
 * Performance monitor
 *
 * By default, the performance monitor is not used
 */
//#define USE_CUDAPERFMON 1

// ---------------- RENDERING ----------------

/**
 * Flag to tell wheter or not to use a dedicated stream for renderer
 */
//#define _GS_RENDERER_USE_STREAM_

// ---------------- DATA PRODUCTION MANAGEMENT ----------------

//#define GV_USE_PRODUCTION_OPTIMIZATION
#define GV_USE_PRODUCTION_OPTIMIZATION_INTERNAL

// ---------------- PIPELINE MANAGEMENT ----------------

/**
 * Optimization to use non-blocking calls to CUDA functions in the default CUDA NULL stream
 * allowing better host/device concurrency.
 *
 * TO DO : use only one "#define"
 */
//#define GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_GVCACHEMANAGER
//#define GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_GVRENDERERCUDA
//#define GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_DATAPRODUCTIONMANAGER
//#define GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_SIMPLEPIPELINE
//#define GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_PRODUCER

/**
 * Reduce driver overhead and allow better host/device concurrency
 * by combining many calls to cudaMemcpyAsync() in a unique one.
 */
//#define GS_USE_OPTIMIZED_NON_BLOCKING_ASYNCHRONOUS_CALLS_PIPELINE_WITH_COMBINED_COPY

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/**
 * Dealing with "Warning: C4251"
 * http://www.unknownroad.com/rtfm/VisualStudio/warningC4251.html
 */
//#define GS_PRAGMA_WARNING_PUSH_DISABLE		\
//#if defined _MSC_VER							\
//	#pragma warning( push )						\
//	#pragma warning( disable:4251 )				\
//#endif
//
//#define GS_PRAGMA_WARNING_POP					\
//#if defined _MSC_VER							\
//	#pragma warning( pop )						\
//#endif

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

#endif // !_GV_CONFIG_H_
