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

#ifndef _GV_RENDERER_H_
#define _GV_RENDERER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/vector_types_ext.h"
#include "GvRendering/GvIRenderer.h"
#include "GvPerfMon/GvPerformanceTimer.h"

// Cuda
#include <vector_types.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

// TO DO : faut-il, peut-on, mettre cette variable dans le namespace ?
/**
 * ...
 */
extern uint GvCacheManager_currentTime;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GigaVoxels
//namespace GvPerfMon
//{
//	class GvPerformanceTimer;
//}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvRendering
{
	
/** 
 * @class GvRenderer
 *
 * @brief The GvRenderer class provides the base interface to render a N-tree data structure.
 *
 * This class is used to render a data structure with the help of a cache and a producer.
 * While rendering (ray-tracing phase), requests are emitted to obtain missing data :
 * - node subdivisions,
 * - and brick loads.
 *
 * @param TDataStructureType The data stucture to render
 * @param VolumeTreeCacheType The cache used to store the data structure and handle produce data requests efficiently
 * (node subdivisions and brick loads)
 * @param ProducerType The producer used to provide data following requests emitted during rendering phase
 * (node subdivisions and brick loads)
 */
template< typename TDataStructureType, typename VolumeTreeCacheType/*, typename TRendererKernelType*/ >
class GvRenderer : public GvIRenderer
// TO DO
// - it seems the 2 templates can be removed ?
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
	virtual ~GvRenderer();

	/**
	 * Returns the current value of the general quality.
	 *
	 * @return the current value of the general quality
	 */
	float getGeneralQuality() const;

	/**
	 * Modify the current value of the general quality.
	 *
	 * @param pValue the current value of the general quality
	 */
	void setGeneralQuality( float pValue );

	/**
	 * Returns the current value of the update quality.
	 *
	 * @return the current value of the update quality
	 */
	float getUpdateQuality() const;

	/**
	 * Modify the current value of the update quality.
	 *
	 * @param pValue the current value of the update quality
	 */
	void setUpdateQuality( float pValue );

	/**
	 * Update the stored current time that represents the number of elapsed frames.
	 * Increment by one.
	 */
	inline void nextFrame();

	/**
	 * Tell if, during data structure traversal, priority of requests is set on brick
	 * loads or on node subdivisions first.
	 *
	 * @return the flag indicating the request strategy
	 */
	bool hasPriorityOnBricks() const;

	/**
	 * Set the request strategy indicating if, during data structure traversal,
	 * priority of requests is set on brick loads or on node subdivisions first.
	 *
	 * @param pFlag the flag indicating the request strategy
	 */
	void setPriorityOnBricks( bool pFlag );

	/**
	 * Specify clear values for the color buffers
	 */
	const uchar4& getClearColor() const;
	void setClearColor( const uchar4& pColor );

	/**
	 * Specify the clear value for the depth buffer
	 */
	float getClearDepth() const;
	void setClearDepth( float pDepth );

	/**
	 * Get the voxel size multiplier
	 *
	 * @return the voxel size multiplier
	 */
	float getVoxelSizeMultiplier() const;

	/**
	 * Set the voxel size multiplier
	 *
	 * @param the voxel size multiplier
	 */
	void setVoxelSizeMultiplier( float pValue );

	/**
	 * Projected 2D Bounding Box of the GigaVoxels 3D BBox.
	 * It holds its bottom left corner and its size.
	 * ( bottomLeftCorner.x, bottomLeftCorner.y, frameSize.x, frameSize.y )
	 *
	 * @return The projected 2D Bounding Box of the GigaVoxels 3D BBox
	 */
	const uint4& getProjectedBBox() const;

	/**
	 * Projected 2D Bounding Box of the GigaVoxels 3D BBox.
	 * It holds its bottom left corner and its size.
	 * ( bottomLeftCorner.x, bottomLeftCorner.y, frameSize.x, frameSize.y )
	 *
	 * @param pProjectedBBox The projected 2D Bounding Box of the GigaVoxels 3D BBox
	 */
	void setProjectedBBox( const uint4& pProjectedBBox );

	/**
	 * Get the time budget
	 *
	 * @return the time budget
	 */
	float getTimeBudget() const;

	/**
	 * Set the time budget
	 *
	 * @param pValue the time budget
	 */
	void setTimeBudget( float pValue );

	/**
	 * Start the timer
	 */
	void startTimer();

	/**
	 * Stop the timer
	 */
	void stopTimer();

	/**
	 * This method return the duration of the timer event between start and stop event
	 *
	 * @return the duration of the event in milliseconds
	 */
	float getElapsedTime();

	/**
	 * This method is called to serialize an object
	 *
	 * @param pStream the stream where to write
	 */
	virtual void write( std::ostream& pStream ) const;

	/**
	 * This method is called deserialize an object
	 *
	 * @param pStream the stream from which to read
	 */
	virtual void read( std::istream& pStream );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Current time.
	 * This represents the number of elapsed frames.
	 */
	uint _currentTime;

	/**
	 * Data stucture to render.
	 */
	TDataStructureType* _volumeTree;

	/**
	 * Cache used to store and data structure efficiently.
	 * It handles requests emitted during rendering phase
	 * (node subdivisions and brick loads).
	 */
	VolumeTreeCacheType* _volumeTreeCache;

	/**
	 * General quality value
	 *
	 * @todo explain
	 */
	float _generalQuality;

	/**
	 * Update quality value
	 *
	 * @todo explain
	 */
	float _updateQuality;

	/**
	 * Flag to tell if, during data structure traversal, priority is set on bricks or nodes.
	 * I.e request for a node subdivide strategy first or load bricks strategy first.
	 */
	bool _hasPriorityOnBricks;

	/**
	 * Specify clear values for the color buffers
	 */
	uchar4 _clearColor;

	/**
	 * Specify the clear value for the depth buffer
	 */
	float _clearDepth;

	/**
	 * Voxel size multiplier
	 */
	float _voxelSizeMultiplier;

	/**
	 * Projected 2D Bounding Box of the GigaVoxels 3D BBox.
	 * It holds its bottom left corner and its size.
	 * ( bottomLeftCorner.x, bottomLeftCorner.y, frameSize.x, frameSize.y )
	 */
	uint4 _projectedBBox;

	/**
	 * Time budget (in milliseconds)
	 */
	float _timeBudget;

	/**
	 * Performance timer
	 */
	GvPerfMon::GvPerformanceTimer* _performanceTimer;
	GvPerfMon::GvPerformanceTimer::Event _timerEvent;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * @param pDataStructure the data stucture to render.
	 * @param pCache the cache used to store the data structure and handle produce data requests efficiently.
	 * It handles requests emitted during rendering phase (node subdivisions and brick loads).
	 * @param pProducer the producer used to provide data following requests emitted during rendering phase
	 * (node subdivisions and brick loads).
	 */
	GvRenderer( TDataStructureType* pDataStructure, VolumeTreeCacheType* pCache );

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
	GvRenderer( const GvRenderer& );

	/**
	 * Copy operator forbidden.
	 */
	GvRenderer& operator=( const GvRenderer& );

};

} // namespace GvRendering

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvRenderer.inl"

#endif // !_GV_RENDERER_H_
