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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

//// GigaVoxels
//#include "GvPerfMon/GvPerformanceTimer.h"

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvRendering
{

/******************************************************************************
 * Constructor
 *
 * @param pDataStructure the data stucture to render.
 * @param pCache the cache used to store the data structure and handle produce data requests efficiently.
 * It handles requests emitted during rendering phase (node subdivisions and brick loads).
 * @param pProducer the producer used to provide data following requests emitted during rendering phase
 * (node subdivisions and brick loads).
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
GvRenderer< TDataStructureType, VolumeTreeCacheType >
::GvRenderer( TDataStructureType* pDataStructure, VolumeTreeCacheType* pCache )
:	GvIRenderer()
,	_timeBudget( 0.f )
,	_performanceTimer( NULL )
{
	assert( pDataStructure );
	assert( pCache );

	this->_volumeTree		= pDataStructure;
	this->_volumeTreeCache	= pCache;

	_updateQuality			= 0.3f;
	_generalQuality			= 1.0f;

	// This method update the associated "constant" in device memory
	setVoxelSizeMultiplier(	1.0f );

	_currentTime			= 10;
	
	// By default, during data structure traversal, request for load bricks strategy first
	// (not node subdivision first)
	_hasPriorityOnBricks	= true;

	// Specify clear values for the color and depth buffers
	_clearColor = make_uchar4( 0, 0, 0, 0 );
	_clearDepth = 1.f;

	// Projected 2D Bounding Box of the GigaVoxels 3D BBox.
	_projectedBBox = make_uint4( 0, 0, 0, 0 );

	// Time budget (in milliseconds)
	//
	// 60 fps
	_timeBudget = 1.f / 60.f;
	_performanceTimer = new GvPerfMon::GvPerformanceTimer();
	_timerEvent = _performanceTimer->createEvent();
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
GvRenderer< TDataStructureType, VolumeTreeCacheType >
::~GvRenderer()
{
	delete _performanceTimer;
}

/******************************************************************************
 * Returns the current value of the general quality.
 *
 * @return the current value of the general quality
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
float GvRenderer< TDataStructureType, VolumeTreeCacheType >
::getGeneralQuality() const
{
	return _generalQuality;
}

/******************************************************************************
 * Modify the current value of the general quality.
 *
 * @param pValue the current value of the general quality
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
void GvRenderer< TDataStructureType, VolumeTreeCacheType >
::setGeneralQuality( float pValue )
{
	_generalQuality = pValue;
}

/******************************************************************************
 * Returns the current value of the update quality.
 *
 * @return the current value of the update quality
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
float GvRenderer< TDataStructureType, VolumeTreeCacheType >
::getUpdateQuality() const
{
	return _updateQuality;
}

/******************************************************************************
 * Modify the current value of the update quality.
 *
 * @param pValue the current value of the update quality
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
void GvRenderer< TDataStructureType, VolumeTreeCacheType >
::setUpdateQuality( float pValue )
{
	_updateQuality = pValue;
}

/******************************************************************************
 * Update the stored current time that represents the number of elapsed frames.
 * Increment by one.
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
inline void GvRenderer< TDataStructureType, VolumeTreeCacheType >
::nextFrame()
{
	_currentTime++;
}

/******************************************************************************
 * Tell if, during data structure traversal, priority of requests is set on brick
 * loads or on node subdivisions first.
 *
 * @return the flag indicating the request strategy
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
bool GvRenderer< TDataStructureType, VolumeTreeCacheType >
::hasPriorityOnBricks() const
{
	return _hasPriorityOnBricks;
}

/******************************************************************************
 * Set the request strategy indicating if, during data structure traversal,
 * priority of requests is set on brick loads or on node subdivisions first.
 *
 * @param pFlag the flag indicating the request strategy
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
void GvRenderer< TDataStructureType, VolumeTreeCacheType >
::setPriorityOnBricks( bool pFlag )
{
	_hasPriorityOnBricks = pFlag;
}

/******************************************************************************
 * Specify clear values for the color buffers
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
const uchar4& GvRenderer< TDataStructureType, VolumeTreeCacheType >
::getClearColor() const
{
	return _clearColor;
}

/******************************************************************************
 * Specify clear values for the color buffers
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
void GvRenderer< TDataStructureType, VolumeTreeCacheType >
::setClearColor( const uchar4& pColor )
{
	_clearColor = pColor;
}

/******************************************************************************
 * Specify the clear value for the depth buffer
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
float GvRenderer< TDataStructureType, VolumeTreeCacheType >
::getClearDepth() const
{
	return _clearDepth;
}

/******************************************************************************
 * Specify the clear value for the depth buffer
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
void GvRenderer< TDataStructureType, VolumeTreeCacheType >
::setClearDepth( float pDepth )
{
	_clearDepth = pDepth;
}

/******************************************************************************
 * Get the voxel size multiplier
 *
 * @return the voxel size multiplier
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
float GvRenderer< TDataStructureType, VolumeTreeCacheType >
::getVoxelSizeMultiplier() const
{
	return _voxelSizeMultiplier;
}

/******************************************************************************
 * Set the voxel size multiplier
 *
 * @param pValue the voxel size multiplier
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
void GvRenderer< TDataStructureType, VolumeTreeCacheType >
::setVoxelSizeMultiplier( float pValue )
{
	_voxelSizeMultiplier = pValue;

	// Update CUDA memory with value
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_voxelSizeMultiplier, &_voxelSizeMultiplier, sizeof( _voxelSizeMultiplier ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Projected 2D Bounding Box of the GigaVoxels 3D BBox.
 * It holds its bottom left corner and its size.
 * ( bottomLeftCorner.x, bottomLeftCorner.y, frameSize.x, frameSize.y )
 *
 * @return The projected 2D Bounding Box of the GigaVoxels 3D BBox
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
const uint4& GvRenderer< TDataStructureType, VolumeTreeCacheType >
::getProjectedBBox() const
{
	return _projectedBBox;
}

/******************************************************************************
 * Projected 2D Bounding Box of the GigaVoxels 3D BBox.
 * It holds its bottom left corner and its size.
 * ( bottomLeftCorner.x, bottomLeftCorner.y, frameSize.x, frameSize.y )
 *
 * @param pProjectedBBox The projected 2D Bounding Box of the GigaVoxels 3D BBox
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
void GvRenderer< TDataStructureType, VolumeTreeCacheType >
::setProjectedBBox( const uint4& pProjectedBBox )
{
	_projectedBBox = pProjectedBBox;
}

/******************************************************************************
 * Get the time budget
 *
 * @return the time budget
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
inline float GvRenderer< TDataStructureType, VolumeTreeCacheType >
::getTimeBudget() const
{
	return _timeBudget;
}

/******************************************************************************
 * Set the time budget
 *
 * @param pValue the time budget
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
inline void GvRenderer< TDataStructureType, VolumeTreeCacheType >
::setTimeBudget( float pValue )
{
	_timeBudget = pValue;
}

/******************************************************************************
 * Start the timer
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
inline void GvRenderer< TDataStructureType, VolumeTreeCacheType >
::startTimer()
{
	return _performanceTimer->startEvent( _timerEvent );
}

/******************************************************************************
 * Stop the timer
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
inline void GvRenderer< TDataStructureType, VolumeTreeCacheType >
::stopTimer()
{
	return _performanceTimer->stopEvent( _timerEvent );
}

/******************************************************************************
 * This method return the duration of the timer event between start and stop event
 *
 * @return the duration of the event in milliseconds
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
float GvRenderer< TDataStructureType, VolumeTreeCacheType >
::getElapsedTime()
{
	return _performanceTimer->getEventDuration( _timerEvent );
}

/******************************************************************************
 * This method is called to serialize an object
 *
 * @param pStream the stream where to write
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
void GvRenderer< TDataStructureType, VolumeTreeCacheType >
::write( std::ostream& pStream ) const
{
}

/******************************************************************************
 * This method is called deserialize an object
 *
 * @param pStream the stream from which to read
 ******************************************************************************/
template< typename TDataStructureType, typename VolumeTreeCacheType >
void GvRenderer< TDataStructureType, VolumeTreeCacheType >
::read( std::istream& pStream )
{
}


} // namespace GvRendering
