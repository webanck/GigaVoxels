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

#ifndef _GV_PERFORMANCE_MONITOR_H_
#define _GV_PERFORMANCE_MONITOR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvPerfMon/GvPerformanceTimer.h"
#include "GvPerfMon/GvDevicePerformanceTimer.h"
#include "GvPerfMon/GvPerformanceMonitorKernel.h"
#include "GvCore/vector_types_ext.h"
#include "GvCore/gvTypes.h"
#include "GvCore/Array3D.h"
#include "GvCore/Array3DGPULinear.h"
#include "GvCore/Array3DKernelLinear.h"
#include "GvCore/GvError.h"

//#include "NemoGraphics/ShaderProgramGLSL.h"

// STL
#include <vector>
#include <string>
#include <iostream>

// System
#include <cassert>

// Cuda
#include <cuda.h>
#include <vector_types.h>
#include <vector_functions.h>

// Cuda SDK
#include <helper_cuda.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * ...
 */
//#define CUDAPERFMON_CACHE_INFO 1
/**
 * ...
 */
#define CUDAPERFMON_GPU_TIMER_ENABLED 1
/**
 * ...
 */
#define GV_PERFMON_INTERNAL_DEVICE_TIMER_ENABLED 0
/**
 * ...
 */
#define CUDAPERFMON_GPU_TIMER_MAX_INSTANCES 32 // Prevent getting out of memory error
/**
 * ...
 */
#define CUDAPERFTIMERCPU_GPUSYNC 0

namespace GvPerfMon
{

	/**
	 * Timers array
	 */
	//__constant__ uint64 *k_timersArray;
	__constant__ GvCore::Array3DKernelLinear< GvCore::uint64 > k_timersArray;

	/**
	 * Timers mask
	 */
	__constant__ uchar* k_timersMask;

}

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
 * CUDA Performance monitoring class.
 */
class GIGASPACE_EXPORT CUDAPerfMon
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Flag to tell wherer or not to activate Monitoring
	 */
	static bool _isActivated;

	/**
	 * Define events list
	 */
#define CUDAPM_DEFINE_EVENT( evtName ) evtName,
	enum ApplicationEvent
	{
		// TO DO : enlever ce #include
		#include "GvPerfMon/GvPerformanceMonitorEvents.h"
		NumApplicationEvents
	};
#undef CUDAPM_DEFINE_EVENT

	/******************************* ATTRIBUTES *******************************/

	//-------------------------- TEST
	bool _requestResize;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	CUDAPerfMon();

	/**
	 * ...
	 *
	 * @return ...
	 */
	static CUDAPerfMon& get()
	{
		if ( ! _sInstance )
		{
			_sInstance = new CUDAPerfMon();
		}

		return (*_sInstance);
	}

	/**
	 * Initialize
	 */
	void init( /*int xres, int yres*/ );

	/**
	 * Start an event
	 *
	 * @param evtName event name's index
	 * @param hasKernelTimers flag to tell wheter or not to handle internal GPU timers
	 */
	void startEvent( ApplicationEvent evtName, bool hasKernelTimers = false );

	/**
	 * Stop an event
	 *
	 * @param evtName event name's index
	 * @param hasKernelTimers flag to tell wheter or not to handle internal GPU timers
	 */
	void stopEvent( ApplicationEvent evtName, bool hasKernelTimers = false );

	/**
	 * Start the main frame event
	 */
	void startFrame();

	/**
	 * Stop the main frame event
	 */
	void stopFrame();

	/**
	 * ...
	 *
	 * @param evtName ...
	 * @param n ...
	 */
	void setEventNumElems( ApplicationEvent evtName, uint n );

	/**
	 * ...
	 */
	void displayFrame();

	/**
	 * ...
	 *
	 * @param eventType ...
	 */
	void displayFrameGL( uint eventType = 0 );	// 0 GPU evts, 1 CPU evts

	/**
	 * ...
	 *
	 * @param overlayBuffer ...
	 */
	void displayOverlayGL( uchar* overlayBuffer );

	/**
	 * ...
	 */
	void displayCacheInfoGL();

	/**
	 * ...
	 *
	 * @param numNodePagesUsed ...
	 * @param numNodePagesWrited ...
	 * @param numBrickPagesUsed ...
	 * @param numBrickPagesWrited ...
	 */
	void saveFrameStats( uint numNodePagesUsed, uint numNodePagesWrited, uint numBrickPagesUsed, uint numBrickPagesWrited );

	/**
	 * ...
	 *
	 * @param frameRes ...
	 */
	void frameResized( uint2 frameRes );
	
	//--------------------- TEST -----------------------------------
	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< GvCore::uint64 >* getKernelTimerArray();
	//--------------------- TEST -----------------------------------
	
	/**
	 * ...
	 *
	 * @return ...
	 */
	uchar* getKernelTimerMask();

	/**
	 * ...
	 *
	 * @return ...
	 */
	GvCore::Array3DGPULinear< uchar4 >* getCacheStateArray() const;

	/**
	 * ...
	 *
	 * @param cacheStateArray ...
	 */
	void setCacheStateArray( GvCore::Array3DGPULinear< uchar4 >* cacheStateArray );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
		
	/**
	 * Singleton instance
	 */
	static CUDAPerfMon* _sInstance;

	/**
	 * Device timer manager
	 * - start / stop events
	 * - get elapsed time
	 */
	GvDevicePerformanceTimer _deviceTimer;

	/**
	 * Host timer manager
	 * - start / stop events
	 * - get elapsed time
	 */
	GvPerformanceTimer _hostTimer;

	/**
	 * List of all events name
	 */
	static const char* _eventNames[ NumApplicationEvents + 1 ];

	/**
	 * ...
	 */
	int frameCurrentInstance[ NumApplicationEvents ];

	/**
	 * List of device events
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::vector< GvDevicePerformanceTimer::Event > _deviceEvents[ NumApplicationEvents ];
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * List of host events
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::vector< GvPerformanceTimer::Event > _hostEvents[ NumApplicationEvents ];
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * ...
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::vector< uint > eventsNumElements[ NumApplicationEvents ];
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
	 * Flag to tell wheter or not the frame has started
	 */
	bool _frameStarted;

	/**
	 * Kernel timers
	 */
	int _deviceClockRate;

	/**
	 * Internal GPU timer's array
	 *
	 * - 3D array : (width, height) 2D array of window size + (depth) one by internal event's timer
	 */
	GvCore::Array3DGPULinear< GvCore::uint64 >* d_timersArray;
	
	/**
	 * ...
	 */
	uchar* d_timersMask;

	/**
	 * ...
	 */
	GLuint overlayTex;

	/**
	 * ...
	 */
	GLuint cacheStateTex;

	/**
	 * ...
	 */
	GvCore::Array3DGPULinear< uchar4 >* d_cacheStateArray;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} // namespace GvPerfMon

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

#ifdef USE_CUDAPERFMON

//#define CUDAPM_INIT( xres, yres ) ::GvPerfMon::CUDAPerfMon::get().init( xres, yres );
#define CUDAPM_INIT() ::GvPerfMon::CUDAPerfMon::get().init();
#define CUDAPM_RESIZE( frameSize ) ::GvPerfMon::CUDAPerfMon::get().frameResized( frameSize );
#define CUDAPM_END

/**
 * Start / Stop the main frame event
 */
#define CUDAPM_START_FRAME								\
	if ( GvPerfMon::CUDAPerfMon::_isActivated )			\
	{													\
		::GvPerfMon::CUDAPerfMon::get().startFrame();	\
	}
#define CUDAPM_STOP_FRAME								\
	if ( GvPerfMon::CUDAPerfMon::_isActivated )			\
	{													\
		::GvPerfMon::CUDAPerfMon::get().stopFrame();	\
	}

/**
 * Start / Stop an event
 */
#define CUDAPM_START_EVENT( eventName )														\
	if ( GvPerfMon::CUDAPerfMon::_isActivated )												\
	{																						\
		::GvPerfMon::CUDAPerfMon::get().startEvent( ::GvPerfMon::CUDAPerfMon::eventName );	\
	}
#define CUDAPM_STOP_EVENT( eventName )														\
	if ( GvPerfMon::CUDAPerfMon::_isActivated )												\
	{																						\
		::GvPerfMon::CUDAPerfMon::get().stopEvent( ::GvPerfMon::CUDAPerfMon::eventName );	\
	}

/**
 * Start / Stop an event with internal GPU timer
 */
#define CUDAPM_START_EVENT_GPU( eventName )															\
	if ( GvPerfMon::CUDAPerfMon::_isActivated )														\
	{																								\
		::GvPerfMon::CUDAPerfMon::get().startEvent( ::GvPerfMon::CUDAPerfMon::eventName, true );	\
	}
#define CUDAPM_STOP_EVENT_GPU( eventName )															\
	if ( GvPerfMon::CUDAPerfMon::_isActivated )														\
	{																								\
		::GvPerfMon::CUDAPerfMon::get().stopEvent( ::GvPerfMon::CUDAPerfMon::eventName, true );		\
	}

/**
 * Start / Stop an event given an identifier
 *
 * - based on the comparison of two identifiers
 */
#define CUDAPM_START_EVENT_CHANNEL( channelRef, channelNum, eventName )								\
	if ( GvPerfMon::CUDAPerfMon::_isActivated )														\
	{																								\
		if ( channelNum == channelRef )																\
		{																							\
			::GvPerfMon::CUDAPerfMon::get().startEvent( ::GvPerfMon::CUDAPerfMon::eventName );		\
		}																							\
	}
#define CUDAPM_STOP_EVENT_CHANNEL( channelRef, channelNum, eventName )								\
	if ( GvPerfMon::CUDAPerfMon::_isActivated )														\
	{																								\
		if ( channelNum == channelRef )																\
		{																							\
			::GvPerfMon::CUDAPerfMon::get().stopEvent( ::GvPerfMon::CUDAPerfMon::eventName );		\
		}																							\
	}

/**
 * ... given an identifier
 *
 * - based on the comparison of two identifiers
 */
#define CUDAPM_EVENT_NUMELEMS_CHANNEL( channelRef, channelNum, eventName, n )								\
	if ( GvPerfMon::CUDAPerfMon::_isActivated )																\
	{																										\
		if ( channelNum == channelRef )																		\
		{																									\
			::GvPerfMon::CUDAPerfMon::get().setEventNumElems( ::GvPerfMon::CUDAPerfMon::eventName, n );		\
		}																									\
	}

/**
 * ...
 */
#define CUDAPM_STAT_EVENT( stuff1, stuff2 ) {}

/**
 * ...
 */
#define CUDAPM_EVENT_NUMELEMS( eventName, n )																\
	if ( GvPerfMon::CUDAPerfMon::_isActivated )																\
	{																										\
		::GvPerfMon::CUDAPerfMon::get().setEventNumElems( ::GvPerfMon::CUDAPerfMon::eventName, n );			\
	}

/**
 * ...
 */
#define CUDAPM_GET_KERNEL_EVENT_MEMORY( stuff1, stuff2 ) {}

/**
 * Define an internal event on device
 */
#if GV_PERFMON_INTERNAL_DEVICE_TIMER_ENABLED==1
	#define CUDAPM_KERNEL_DEFINE_EVENT( evtSlot )		\
	GvCore::uint64 cudaPMKernelEvt##evtSlot##Clk = 0;	\
	GvCore::uint64 cudaPMKernelEvt##evtSlot##In;		
#else
	#define CUDAPM_KERNEL_DEFINE_EVENT( evtSlot ) {}
#endif

/**
 * Start / Stop an internal event on device
 */
#if GV_PERFMON_INTERNAL_DEVICE_TIMER_ENABLED==1
	#define CUDAPM_KERNEL_START_EVENT( pixelCoords, evtSlot )													\
	if ( GvPerfMon::k_timersMask[ pixelCoords.y * k_renderViewContext.frameSize.x + pixelCoords.x ] != 0 )		\
	{																											\
		cudaPMKernelEvt##evtSlot##In = GvPerfMon::getClock();													\
	}
#else
	#define CUDAPM_KERNEL_START_EVENT( pixelCoords, evtSlot ) {}
#endif
//#define CUDAPM_KERNEL_START_EVENT( pixelCoords, evtSlot ) {}
#if GV_PERFMON_INTERNAL_DEVICE_TIMER_ENABLED==1
	#define CUDAPM_KERNEL_STOP_EVENT( pixelCoords, evtSlot )																	\
	if ( GvPerfMon::k_timersMask[ pixelCoords.y * k_renderViewContext.frameSize.x + pixelCoords.x ] != 0 )						\
	{																															\
		cudaPMKernelEvt##evtSlot##Clk += ( GvPerfMon::getClock() - cudaPMKernelEvt##evtSlot##In );								\
		GvPerfMon::k_timersArray.set( make_uint3( pixelCoords.x, pixelCoords.y, evtSlot ), cudaPMKernelEvt##evtSlot##Clk );		\
	}
#else
	#define CUDAPM_KERNEL_STOP_EVENT( pixelCoords, evtSlot ) {}
#endif
//#define CUDAPM_KERNEL_STOP_EVENT( pixelCoords, evtSlot ) {}

// Not sure we need them
/*#define CUDAPM_KERNEL_START( pixelCoords ) { \
	uint64 kernelEvtIn = GvPerfMon::getClock(); \
}
#define CUDAPM_KERNEL_STOP( pixelCoords ) { \
	k_timersArray.get( make_uint3( pixelCoords.x, pixelCoords.y, 0 ) ) = GvPerfMon::getClock() - kernelEvtIn;\
}*/

# if CUDAPERFMON_CACHE_INFO == 1

#  define CUDAPM_RENDER_CACHE_INFO( xres, yres ) { \
	GvCore::Array3DGPULinear< uchar4 >* cacheStateArray = \
		::GvPerfMon::CUDAPerfMon::get().getCacheStateArray(); \
	if ( cacheStateArray ) \
	{ \
		const uint2 syntheticRenderSize = make_uint2( xres, yres ); \
		dim3 blockSize( 8, 8, 1 ); \
		dim3 gridSize( syntheticRenderSize.x / blockSize.x, syntheticRenderSize.y / blockSize.y, 1 ); \
		SyntheticInfo_Render<<< gridSize, blockSize, 0 >>>( cacheStateArray->getPointer(), cacheStateArray->getNumElements() ); \
		GV_CHECK_CUDA_ERROR( "SyntheticInfo_Render" ); \
	} \
}

# else
#  define CUDAPM_RENDER_CACHE_INFO( xres, yres ) {}
# endif

#else  // USE_CUDAPERFMON


#define CUDAPM_INIT()
#define CUDAPM_RESIZE( frameSize )
#define CUDAPM_END

#define CUDAPM_START_FRAME
#define CUDAPM_STOP_FRAME

#define CUDAPM_START_EVENT( stuff ) {}
#define CUDAPM_STOP_EVENT( stuff ) {}
#define CUDAPM_START_EVENT_GPU( stuff ) {}
#define CUDAPM_STOP_EVENT_GPU( stuff ) {}

#define CUDAPM_START_EVENT_CHANNEL( channelRef, channelNum, eventName ) {}
#define CUDAPM_STOP_EVENT_CHANNEL( channelRef, channelNum, eventName ) {}
#define CUDAPM_EVENT_NUMELEMS_CHANNEL( channelRef, channelNum, eventName, n ) {}

#define CUDAPM_STAT_EVENT( stuff1, stuff2 ) {}
#define CUDAPM_EVENT_NUMELEMS( eventName, n ) {}

#define CUDAPM_GET_KERNEL_EVENT_MEMORY( stuff1, stuff2 ){}


#define CUDAPM_KERNEL_DEFINE_EVENT( evtSlot ) {}
#define CUDAPM_KERNEL_START_EVENT( pixelCoords, evtSlot ) {}
#define CUDAPM_KERNEL_STOP_EVENT( pixelCoords, evtSlot ) {}

#define CUDAPM_RENDER_CACHE_INFO( xres, yres ) {}

#endif

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvPerformanceMonitor.inl"

#endif
