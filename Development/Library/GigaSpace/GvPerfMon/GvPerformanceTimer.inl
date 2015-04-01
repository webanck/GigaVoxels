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

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvPerfMon
{

/******************************************************************************
 * This method create and initialize a new Event object.
 *
 * @return the created event
 ******************************************************************************/
inline GvPerformanceTimer::Event GvPerformanceTimer::createEvent() const
{
	GvPerformanceTimer::Event evt;

	getHighResolutionTime( &evt.timerStartEvt );
	getHighResolutionTime( &evt.timerStopEvt );

	return evt;
}

/******************************************************************************
 * This method set the start time of the given event to the current time.
 *
 * @param evt a reference to the event.
 ******************************************************************************/
inline void GvPerformanceTimer::startEvent( GvPerformanceTimer::Event& pEvent ) const
{
#if CUDAPERFTIMERCPU_GPUSYNC
	cudaDeviceSynchronize();
#endif
	getHighResolutionTime( &pEvent.timerStartEvt );
}

/******************************************************************************
 * This method set the stop time of the given event to the current time.
 *
 * @param evt a reference to the event.
 ******************************************************************************/
inline void GvPerformanceTimer::stopEvent( GvPerformanceTimer::Event& pEvent ) const
{
#if CUDAPERFTIMERCPU_GPUSYNC
	cudaDeviceSynchronize();
#endif

	getHighResolutionTime( &pEvent.timerStopEvt );
}

/******************************************************************************
 * This method return the duration of the given event
 *
 * @param evt a reference to the event.
 *
 * @return the duration of the event in milliseconds
 ******************************************************************************/
inline float GvPerformanceTimer::getEventDuration( GvPerformanceTimer::Event& pEvent ) const
{
	float tms;

	/*tms = ( pEvent.timerStopEvt.tv_sec - pEvent.timerStartEvt.tv_sec ) * 1000.0f // sec -> msec
	+ ( pEvent.timerStopEvt.tv_nsec - pEvent.timerStartEvt.tv_nsec ) * 1e-6f;  // nano -> milli*/
	tms = convertTimeDifferenceToSec( &pEvent.timerStopEvt, &pEvent.timerStartEvt ) * 1000.0f;

	return tms;
}

/******************************************************************************
 * This method return the difference between the begin of two events
 *
 * @param evt0 a reference to the first event
 * @param evt1 a reference to the second event
 *
 * @return the difference between the two events in milliseconds
 ******************************************************************************/
inline float GvPerformanceTimer::getStartToStartTime( GvPerformanceTimer::Event& pEvent0, GvPerformanceTimer::Event& pEvent1 ) const
{
	float tms;

	/*tms = ( pEvent1.timerStartEvt.tv_sec - pEvent0.timerStartEvt.tv_sec ) * 1000.0f // sec -> msec
	+ ( pEvent1.timerStartEvt.tv_nsec - pEvent0.timerStartEvt.tv_nsec ) * 1e-6f;  // nano -> milli*/

	tms = convertTimeDifferenceToSec( &pEvent1.timerStartEvt, &pEvent0.timerStartEvt ) * 1000.0f;

	return tms;
}

/******************************************************************************
 * Get high resolution time
 *
 * @param pPerformanceCount ...
 ******************************************************************************/
inline void GvPerformanceTimer::getHighResolutionTime( GvPerformanceTimer::timerStruct* pPerformanceCount ) const
{
#ifdef WIN32
	// Retrieves the current value of the high-resolution performance counter
	// - parameter :
	// ---- A pointer to a variable that receives the current performance-counter value, in counts.
	QueryPerformanceCounter( pPerformanceCount );
#else
	clock_gettime( CLOCK_REALTIME, pPerformanceCount );
#endif
}

/******************************************************************************
 * Convert time difference to sec
 *
 * @param end ...
 * @param begin ...
 *
 * @return ...
 ******************************************************************************/
inline float GvPerformanceTimer::convertTimeDifferenceToSec( GvPerformanceTimer::timerStruct* pEnd, GvPerformanceTimer::timerStruct* pBegin ) const
{
#ifdef WIN32
	timerStruct frequency;
	// Retrieves the frequency of the high-resolution performance counter, if one exists. The frequency cannot change while the system is running.
	// - parameter :
	// ---- A pointer to a variable that receives the current performance-counter frequency, in counts per second.
	// ---- If the installed hardware does not support a high-resolution performance counter, this parameter can be zero.
	QueryPerformanceFrequency( &frequency );

	return ( pEnd->QuadPart - pBegin->QuadPart ) / static_cast< float >( frequency.QuadPart );
#else
	return ( pEnd->tv_sec - pBegin->tv_sec ) + ( 1e-9 ) * ( pEnd->tv_nsec - pBegin->tv_nsec );
#endif
}

} // namespace GvPerfMon
