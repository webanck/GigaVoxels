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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvError.h"

// System
#include <cstring>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvPerfMon
{

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvDevicePerformanceTimer::GvDevicePerformanceTimer()
{
}

/******************************************************************************
 * Desconstructor
 ******************************************************************************/
GvDevicePerformanceTimer::~GvDevicePerformanceTimer()
{
}

/******************************************************************************
 * This method create and initialize a new Event object.
 *
 * @return the created event
 ******************************************************************************/
inline GvDevicePerformanceTimer::Event GvDevicePerformanceTimer::createEvent() const
{
	GvDevicePerformanceTimer::Event evt;

	cudaEventCreate( &evt.cudaTimerStartEvt );
	cudaEventCreate( &evt.cudaTimerStopEvt );
	cudaDeviceSynchronize();
	
	GV_CHECK_CUDA_ERROR( "GvDevicePerformanceTimer::createEvent" );
	
	// Initialize values to zero
	::memset( evt.timersArray, 0, sizeof( evt.timersArray ) );

	return evt;
}

/******************************************************************************
 * This method set the start time of the given event to the current time.
 *
 * @param pEvent a reference to the event.
 ******************************************************************************/
inline void GvDevicePerformanceTimer::startEvent( GvDevicePerformanceTimer::Event& pEvent ) const
{
	cudaEventRecord( pEvent.cudaTimerStartEvt, 0 );
}

/******************************************************************************
 * This method set the stop time of the given event to the current time.
 *
 * @param pEvent a reference to the event.
 ******************************************************************************/
inline void GvDevicePerformanceTimer::stopEvent( GvDevicePerformanceTimer::Event& pEvent ) const
{
	cudaEventRecord( pEvent.cudaTimerStopEvt, 0 );
}

/******************************************************************************
 * This method return the duration of the given event
 *
 * @param pEvent a reference to the event.
 *
 * @return the duration of the event in milliseconds
 ******************************************************************************/
inline float GvDevicePerformanceTimer::getEventDuration( GvDevicePerformanceTimer::Event& pEvent ) const
{
	float time;
	cudaEventElapsedTime( &time, pEvent.cudaTimerStartEvt, pEvent.cudaTimerStopEvt );

	return time;
}

/******************************************************************************
 * This method return the difference between the begin of two events
 *
 * @param pEvent0 a reference to the first event
 * @param pEvent1 a reference to the second event
 *
 * @return the difference between the two events in milliseconds
 ******************************************************************************/
inline float GvDevicePerformanceTimer::getStartToStartTime( GvDevicePerformanceTimer::Event& pEvent0, GvDevicePerformanceTimer::Event& pEvent1 ) const
{
	float time;
	cudaEventElapsedTime( &time, pEvent0.cudaTimerStartEvt, pEvent1.cudaTimerStartEvt );

	return time;
}

} // namespace GvPerfMon
