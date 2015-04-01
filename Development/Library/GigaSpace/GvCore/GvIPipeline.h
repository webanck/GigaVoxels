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

#ifndef _GV_I_PIPELINE_H_
#define _GV_I_PIPELINE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/GvISerializable.h"

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

/** 
 * @class GvIPipeline
 *
 * @brief The GvIPipeline class provides the interface to manage GigaSpace pipelines
 * (i.e. data structure, cache, producers, renders, etc...)
 * 
 * @ingroup GvCore
 *
 * This class is the base class for all pipeline objects.
 */
class GIGASPACE_EXPORT GvIPipeline : public GvCore::GvISerializable
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Cache overflow policy
	 */
	enum ECacheOverflowPolicy
	{
		eCacheOverflow_Default = 0,
		eCacheOverflow_QualityFirst = 1,
		eCacheOverflow_FranckLag = 1 << 1,
		eCacheOverflow_DecreaseResolution = 1 << 2,
		eCacheOverflow_PriorityInProduction = 1 << 3,
		eCacheOverflow_All = ( 1 << 4 ) - 1
	};

	/**
	 * Cache full policy
	 */
	enum ECacheFullPolicy
	{
		eCacheFull_Default = 0,
		eCacheFull_QualityFirst = 1,
		eCacheFull_LockCache = 1 << 1,
		eCacheFull_LockLastFrameCache = 1 << 2,
		eCacheFull_PriorityInFreeing = 1 << 3,
		eCacheFull_All = ( 1 << 4 ) - 1
	};

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Destructor
	 */
	virtual ~GvIPipeline();

	/**
	 * Set the cache overflow policy
	 *
	 * @param pPolicy the cache overflow policy
	 */
	void setCacheOverflowPolicy( ECacheOverflowPolicy pPolicy );

	/**
	 * Get the cache overflow policy
	 *
	 * @return the cache overflow policy
	 */
	ECacheOverflowPolicy getCacheOverflowPolicy() const;

	/**
	 * Set the cache full policy
	 *
	 * @param pPolicy the cache full policy
	 */
	void setCacheFullPolicy( ECacheFullPolicy pPolicy );

	/**
	 * Get the cache full policy
	 *
	 * @return the cache full policy
	 */
	ECacheFullPolicy getCacheFullPolicy() const;

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Cache overflow policy
	 */
	ECacheOverflowPolicy _cacheOverflowPolicy;

	/**
	 * Cache full policy
	 */
	ECacheFullPolicy _cacheFullPolicy;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvIPipeline();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvIPipeline( const GvIPipeline& );

	/**
	 * Copy operator forbidden.
	 */
	GvIPipeline& operator=( const GvIPipeline& );

};

} // namespace GvCore

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvIPipeline.inl"

#endif // !_GV_I_PIPELINE_H_
