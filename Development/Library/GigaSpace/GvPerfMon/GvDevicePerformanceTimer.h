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

#ifndef _GV_DEVICE_PERFORMANCE_TIMER_H_
#define _GV_DEVICE_PERFORMANCE_TIMER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvCore/gvTypes.h"

// System
#ifdef WIN32
	#ifndef WIN32_LEAN_AND_MEAN
		#define WIN32_LEAN_AND_MEAN
	#endif
	#include <windows.h>
#else
	#include <time.h>
#endif

// Cuda
#include <cuda_runtime.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Defines the number of timers available per-pixel in a kernel.
 */
#define CUDAPERFMON_KERNEL_TIMER_MAX 8

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
 * @class GvDevicePerformanceTimer
 *
 * @brief The GvDevicePerformanceTimer class provides a device performance timer.
 *
 * Allows timing GPU events.
 */
class GIGASPACE_EXPORT GvDevicePerformanceTimer
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * This structure contains informations we gather during the timing phase.
	 *
	 * @field cudaTimerStartEvent the time when we started the timer.
	 * @field cudaTimerStopEvt the time when we stopped the timer.
	 */
	struct Event
	{
		/**************************************************************************
		 ***************************** PUBLIC SECTION *****************************
		 **************************************************************************/

		/****************************** INNER TYPES *******************************/

		/******************************* ATTRIBUTES *******************************/

		/**
		 *  Cuda start event
		 */
		cudaEvent_t cudaTimerStartEvt;

		/**
		 *  Cuda stop event
		 */
		cudaEvent_t cudaTimerStopEvt;

		/**
		 *  Kernel timer min
		 */
		GvCore::uint64 kernelTimerMin[ CUDAPERFMON_KERNEL_TIMER_MAX ];
		
		/**
		 *  Kernel timer max
		 */
		GvCore::uint64 kernelTimerMax[ CUDAPERFMON_KERNEL_TIMER_MAX ];

		/**
		 * Timers array
		 */
		GvCore::uint64 timersArray[ CUDAPERFMON_KERNEL_TIMER_MAX ];

		/******************************** METHODS *********************************/

		/**************************************************************************
		 **************************** PROTECTED SECTION ***************************
		 **************************************************************************/

	protected:

		/****************************** INNER TYPES *******************************/

		/******************************* ATTRIBUTES *******************************/

		/******************************** METHODS *********************************/

		/**************************************************************************
		 ***************************** PRIVATE SECTION ****************************
		 **************************************************************************/

	private:

		/****************************** INNER TYPES *******************************/

		/******************************* ATTRIBUTES *******************************/

		/******************************** METHODS *********************************/

	};

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvDevicePerformanceTimer();

	/**
	 * Destructor
	 */
	virtual ~GvDevicePerformanceTimer();

	/**
	 * This method create and initialize a new Event object.
	 *
	 * @return the created event
	 */
	Event createEvent() const;

	/**
	 * This method set the start time of the given event to the current time.
	 *
	 * @param pEvent a reference to the event.
	 */
	void startEvent( Event& pEvent ) const;

	/**
	 * This method set the stop time of the given event to the current time.
	 *
	 * @param pEvent a reference to the event.
	 */
	void stopEvent( Event& pEvent ) const;

	/**
	 * This method return the duration of the given event
	 *
	 * @param pEvent a reference to the event.
	 *
	 * @return the duration of the event in milliseconds
	 */
	float getEventDuration( Event& pEvent ) const;

	/**
	 * This method return the difference between the begin of two events
	 *
	 * @param pEvent0 a reference to the first event
	 * @param pEvent1 a reference to the second event
	 *
	 * @return the difference between the two events in milliseconds
	 */
	float getStartToStartTime( Event& pEvent0, Event& pEvent1 ) const;

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

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
	//GvDevicePerformanceTimer( const GvDevicePerformanceTimer& );

	/**
	 * Copy operator forbidden.
	 */
	//GvDevicePerformanceTimer& operator=( const GvDevicePerformanceTimer& );

};

} // namespace GvPerfMon

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvDevicePerformanceTimer.inl"

#endif
