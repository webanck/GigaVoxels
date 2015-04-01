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

#ifndef _GV_PIPELINE_H_
#define _GV_PIPELINE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaSpace
#include "GvCore/GvCoreConfig.h"
#include "GvCore/GvIPipeline.h"
#include "GvCore/vector_types_ext.h"

// Cuda
#include <vector_types.h>

// System
#include <cassert>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GigaSpace
namespace GvCore
{
	class GvIProvider;
}
namespace GvStructure
{
	class GvIDataStructure;
	class GvIDataProductionManager;
}
namespace GvRendering
{
	class GvIRenderer;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

 namespace GvUtils
 {
 
/** 
 * @class GvPipeline
 *
 * @brief The GvPipeline class provides the interface to manage GigaSpace pipelines
 * (i.e. data structure, cache, producers, renders, etc...)
 * 
 * @ingroup GvUtils
 *
 * This class is ...
 */
class GIGASPACE_EXPORT GvPipeline : public GvCore::GvIPipeline
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Destructor
	 */
	virtual ~GvPipeline();

	/**
	 * Initialize
	 *
	 * @return pDataStructure the associated data structure
	 * @return pProducer the associated producer
	 * @return pRenderer the associated  renderer
	 */
	//virtual void initialize( GvStructure::GvIDataStructure* pDataStructure, GvCore::GvIProvider* pProducer, GvRendering::GvIRenderer* pRenderer );

	/**
	 * Finalize
	 */
	virtual void finalize();
		
	/**
	 * Launch the main GigaSpace flow sequence
	 */
	//virtual void execute();
	//virtual void execute( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport );

	/**
	 * Get the data structure
	 *
	 * @return The data structure
	 */
	virtual const GvStructure::GvIDataStructure* getDataStructure() const;

	/**
	 * Get the data structure
	 *
	 * @return The data structure
	 */
	virtual GvStructure::GvIDataStructure* editDataStructure();

	/**
	 * Get the data production manager
	 *
	 * @return The data production manager
	 */
	virtual const GvStructure::GvIDataProductionManager* getCache() const;

	/**
	 * Get the data production manager
	 *
	 * @return The data production manager
	 */
	virtual GvStructure::GvIDataProductionManager* editCache();

	/**
	 * Get the producer
	 *
	 * @return The producer
	 */
	virtual const GvCore::GvIProvider* getProducer() const;

	/**
	 * Get the producer
	 *
	 * @return The producer
	 */
	virtual GvCore::GvIProvider* editProducer();

	/**
	 * Get the renderer
	 *
	 * @return The renderer
	 */
	virtual const GvRendering::GvIRenderer* getRenderer( unsigned int pIndex = 0 ) const;

	/**
	 * Get the renderer
	 *
	 * @return The renderer
	 */
	virtual GvRendering::GvIRenderer* editRenderer( unsigned int pIndex );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvPipeline();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvPipeline( const GvPipeline& );

	/**
	 * Copy operator forbidden.
	 */
	GvPipeline& operator=( const GvPipeline& );

	/******************************** METHODS *********************************/

};

} // namespace GvUtils

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvPipeline.inl"

#endif // !_GV_PIPELINE_H_
