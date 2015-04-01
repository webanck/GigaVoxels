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

#include "GvUtils/GvPipeline.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaSpace
#include <GvStructure/GvIDataStructure.h>
#include <GvCore/GvIProvider.h>
#include <GvRendering/GvIRenderer.h>

// STL
#include <iostream>

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvVoxelizer
using namespace GvUtils;
using namespace GvCore;
using namespace GvStructure;
using namespace GvRendering;

// STL
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvPipeline::GvPipeline()
:	GvIPipeline()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvPipeline::~GvPipeline()
{
}

/******************************************************************************
 * Initialize
 *
 * @return pDataStructure the associated data structure
 * @return pProducer the associated producer
 * @return pRenderer the associated  renderer
 ******************************************************************************/
//void GvPipeline::initialize( GvStructure::GvIDataStructure* pDataStructure, GvCore::GvIProvider* pProducer, GvRendering::GvIRenderer* pRenderer )
//{
//}

/******************************************************************************
 * Finalize
 ******************************************************************************/
void GvPipeline::finalize()
{
}

/******************************************************************************
 * Get the data structure
 *
 * @return The data structure
******************************************************************************/
const GvIDataStructure* GvPipeline::getDataStructure() const
{
	return NULL;
}

/******************************************************************************
 * Get the data structure
 *
 * @return The data structure
******************************************************************************/
GvStructure::GvIDataStructure* GvPipeline::editDataStructure()
{
	return NULL;
}

/******************************************************************************
 * Get the data production manager
 *
 * @return The data production manager
******************************************************************************/
const GvIDataProductionManager* GvPipeline::getCache() const
{
	return NULL;
}

/******************************************************************************
 * Get the data production manager
 *
 * @return The data production manager
******************************************************************************/
GvStructure::GvIDataProductionManager* GvPipeline::editCache()
{
	return NULL;
}

/******************************************************************************
 * Get the producer
 *
 * @return The producer
******************************************************************************/
const GvIProvider* GvPipeline::getProducer() const
{
	return NULL;
}

/******************************************************************************
 * Get the producer
 *
 * @return The producer
******************************************************************************/
GvIProvider* GvPipeline::editProducer()
{
	return NULL;
}

/******************************************************************************
 * Get the renderer
 *
 * @return The renderer
******************************************************************************/
const GvIRenderer* GvPipeline::getRenderer( unsigned int pIndex ) const
{
	return NULL;
}

/******************************************************************************
 * Get the renderer
 *
 * @return The renderer
******************************************************************************/
GvIRenderer* GvPipeline::editRenderer( unsigned int pIndex )
{
	return NULL;
}
