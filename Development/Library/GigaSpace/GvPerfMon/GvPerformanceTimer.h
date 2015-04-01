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

#ifndef _GV_PERFORMANCE_TIMER_H_
#define _GV_PERFORMANCE_TIMER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

// System
#ifdef WIN32
	#ifndef WIN32_LEAN_AND_MEAN
		#define WIN32_LEAN_AND_MEAN
	#endif
	#include <windows.h>
#else
	#include <time.h>
#endif

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

namespace GvPerfMon
{

/** 
 * @class GvPerformanceTimer
 *
 * @brief The GvPerformanceTimer class provides a host performance timer.
 *
 * Allows timing CPU events.
 */
class GIGASPACE_EXPORT GvPerformanceTimer
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Type definition of types required by timer functions (Operating System dependent)
	 */
#ifdef WIN32
		typedef LARGE_INTEGER timerStruct;
#else
		typedef struct timespec timerStruct;
#endif

	/**
	 * Structure used to store start end stop time of an event
	 */
	struct Event
	{
		timerStruct timerStartEvt;
		timerStruct timerStopEvt;
	};

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvPerformanceTimer();

	/**
	 * Destructor
	 */
	virtual ~GvPerformanceTimer();

	/**
	 * This method create and initialize a new Event object.
	 *
	 * @return the created event
	 */
	inline Event createEvent() const;

	/**
	 * This method set the start time of the given event to the current time.
	 *
	 * @param evt a reference to the event.
	 */
	inline void startEvent( Event& pEvent ) const;

	/**
	 * This method set the stop time of the given event to the current time.
	 *
	 * @param evt a reference to the event.
	 */
	inline void stopEvent( Event& pEvent ) const;

	/**
	 * This method return the duration of the given event
	 *
	 * @param evt a reference to the event.
	 *
	 * @return the duration of the event in milliseconds
	 */
	inline float getEventDuration( Event& pEvent )  const;

	/**
	 * This method return the difference between the begin of two events
	 *
	 * @param evt0 a reference to the first event
	 * @param evt1 a reference to the second event
	 *
	 * @return the difference between the two events in milliseconds
	 */
	inline float getStartToStartTime( Event& pEvent0, Event& pEvent1 ) const;
		
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Get high resolution time
	 *
	 * @param pPerformanceCount ...
	 */
	inline void getHighResolutionTime( timerStruct* pPerformanceCount ) const;

	/**
	 * Convert time difference to sec
	 *
	 * @param end ...
	 * @param begin ...
	 *
	 * @return ...
	 */
	inline float convertTimeDifferenceToSec( timerStruct* pEnd, timerStruct* pBegin ) const;

	/******************************** METHODS *********************************/

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
	//GvPerformanceTimer( const GvPerformanceTimer& );

	/**
	 * Copy operator forbidden.
	 */
	//GvPerformanceTimer& operator=( const GvPerformanceTimer& );

};

} // namespace GvPerfMon

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvPerformanceTimer.inl"

#endif
