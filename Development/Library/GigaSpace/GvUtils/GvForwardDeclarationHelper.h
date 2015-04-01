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

#ifndef _GV_FORWARD_DECLARATION_HELPER_H_
#define _GV_FORWARD_DECLARATION_HELPER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/vector_types_ext.h"

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
 * WARNING :
 *
 * This file is useful to define default parameter for GigaVoxels template classes.
 * Due to forward declaration of template classes requested by GigaVoxels,
 * it is not possible to define default parameters in other files.
 */

// GigaVoxels
namespace GvCore
{
	template< uint r >
	struct StaticRes1D;
}

namespace GvStructure
{
	// Data structure device-side
	template
	<
		class DataTList,
		class NodeTileRes, class BrickRes, uint BorderSize = 1U
	>
	struct VolumeTreeKernel;

	// Data structure host-side
	template
	<
		class DataTList,
		class NodeTileRes, class BrickRes, uint BorderSize = 1U,
		typename TDataStructureKernelType = VolumeTreeKernel< DataTList, NodeTileRes, BrickRes, BorderSize >
	>
	struct GvVolumeTree;
	
	// Cache host-side
	template
	<
		typename TDataStructureType
	>
	class GvDataProductionManager;
}

namespace GvRendering
{	
	// Renderer host-side
	template
	<
		typename TDataStructureType,
		typename VolumeTreeCacheType,
		typename SampleShader
	>
	class GvRendererCUDA;
}

namespace GvUtils
{	
	// Pass through host producer
	template
	<
		typename TKernelProducerType,
		typename TDataStructureType,
		typename TDataProductionManager = GvStructure::GvDataProductionManager< TDataStructureType >
	>
	class GvSimpleHostProducer;

	// Simple host shader
	template< typename TKernelShaderType >
	class GvSimpleHostShader;

	// Simple Pipeline
	template
	<
		typename TProducerType,
		typename TShaderType,
		typename TDataStructureType,
		typename TCacheType = GvStructure::GvDataProductionManager< TDataStructureType >
	>
	class GvSimplePipeline;
}

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#endif // !_GV_FORWARD_DECLARATION_HELPER_H_
