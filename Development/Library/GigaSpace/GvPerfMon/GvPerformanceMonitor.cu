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

#include "GvPerfMon/GvPerformanceMonitor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/functional_ext.h"

// STL
#include <iostream>
#include <fstream>

// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

// Thrust
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/functional.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvPerfMon;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/**
 * Instance of the singleton Parformance Monitor
 */
CUDAPerfMon* CUDAPerfMon::_sInstance = NULL;

/**
 * Flag to tell wheter or not to activate Monitoring
 */
bool CUDAPerfMon::_isActivated = true;

/**
 * List of events name
 *
 * - the MACRO is used to fill the array automatically from the file "GvPerformanceMonitorEvents.h"
 */
#define CUDAPM_DEFINE_EVENT( evtName ) #evtName,
	const char* CUDAPerfMon::_eventNames[] = {
	#include "GvPerfMon/GvPerformanceMonitorEvents.h"
	"" };
#undef CUDAPM_DEFINE_EVENT

/******************************************************************************
 * ...
 *
 * @param x ...
 * @param y ...
 * @param z ...
 * @param font ...
 * @param string ...
 ******************************************************************************/
inline void ccRenderBitmapString( float x, float y, float z, void* font, const char* string )
{
	char* c;
	glRasterPos3f( x, y, z );
	for ( c = const_cast< char* >( string ); *c != '\0'; c++ )
	{
		glutBitmapCharacter( font, *c );
	}
}

/******************************************************************************
 * ...
 *
 * @param startPos ...
 * @param endPos ...
 ******************************************************************************/
inline void ccDrawQuadGL( const float2& startPos, const float2& endPos )
{
	glBegin( GL_QUADS );
		glVertex3f( startPos.x, startPos.y,	0.f );
		glVertex3f( endPos.x,	startPos.y,	0.f );
		glVertex3f( endPos.x,	endPos.y,	0.f );
		glVertex3f( startPos.x,	endPos.y,	0.f );
	glEnd();
}

/******************************************************************************
 * Constructor
 ******************************************************************************/
CUDAPerfMon::CUDAPerfMon()
:	overlayTex( 0 )
,	cacheStateTex( 0 )
,	d_timersArray( 0 )
,	d_timersMask( 0 )
,	d_cacheStateArray( 0 )
{
	_frameStarted = false;
		
	//-------------------------- TEST
	_requestResize = false;
}

/******************************************************************************
 * Initialize
 ******************************************************************************/
void CUDAPerfMon::init( /*int xres, int yres*/ )
{
	cudaDeviceProp deviceProps;
	cudaGetDeviceProperties( &deviceProps, gpuGetMaxGflopsDeviceId() );		// TO DO : handle the case where user could want an other device
		
	_deviceClockRate = deviceProps.clockRate;
}

/******************************************************************************
 * Start an event
 *
 * @param evtName event name's index
 * @param hasKernelTimers flag to tell wheter or not to handle internal GPU timers
 ******************************************************************************/
void CUDAPerfMon::startEvent( ApplicationEvent evtName, bool hasKernelTimers )
{
	assert( evtName < NumApplicationEvents );

	if ( _frameStarted )
	{
		frameCurrentInstance[ evtName ]++;
		
		// Handle CUDA timers
#if CUDAPERFMON_GPU_TIMER_ENABLED

		if ( frameCurrentInstance[ evtName ] < CUDAPERFMON_GPU_TIMER_MAX_INSTANCES )
		{
			// GPU
			if ( _deviceEvents[ evtName ].size() <= frameCurrentInstance[ evtName ] )
			{
				_deviceEvents[ evtName ].push_back( _deviceTimer.createEvent() );
			}
			_deviceTimer.startEvent( _deviceEvents[ evtName ][ frameCurrentInstance[ evtName ] ] );
			
			// Clear GPU timers
			if ( hasKernelTimers )
			{
				d_timersArray->fill( 0 );
			}
		}

#endif

		// Handle Host timers

		// CPU
		if ( _hostEvents[ evtName ].size() <= frameCurrentInstance[ evtName ] )
		{
			_hostEvents[ evtName ].push_back( _hostTimer.createEvent() );
		}
		_hostTimer.startEvent( _hostEvents[ evtName ][ frameCurrentInstance[ evtName ] ] );

		if ( eventsNumElements[ evtName ].size() <= frameCurrentInstance[ evtName ] )
		{
			eventsNumElements[ evtName ].push_back( 0 );
		}
		eventsNumElements[ evtName ][ frameCurrentInstance[ evtName ] ] = 0;
	}
}

/******************************************************************************
 * Stop an event
 *
 * @param evtName event name's index
 * @param hasKernelTimers flag to tell wheter or not to handle internal GPU timers
 ******************************************************************************/
void CUDAPerfMon::stopEvent( ApplicationEvent evtName, bool hasKernelTimers )
{
	assert( evtName < NumApplicationEvents );

	if ( _frameStarted )
	{
		// Stop Host event
		_hostTimer.stopEvent( _hostEvents[ evtName ][ frameCurrentInstance[ evtName ] ] );

#if CUDAPERFMON_GPU_TIMER_ENABLED
		if ( frameCurrentInstance[ evtName ] < CUDAPERFMON_GPU_TIMER_MAX_INSTANCES )
		{
			// Stop CUDA event
			_deviceTimer.stopEvent( _deviceEvents[ evtName ][ frameCurrentInstance[ evtName ] ] );

			// Handle internal GPU timers
			if ( hasKernelTimers )
			{
	#if GV_PERFMON_INTERNAL_DEVICE_TIMER_ENABLED==1
				uint3 res = d_timersArray->getResolution();
				uint stride = res.x * res.y;

				thrust::host_vector< GvCore::uint64 > tempArray;
				tempArray.reserve( stride );

				thrust::device_ptr< uchar > d_timersMaskBeginPtr( d_timersMask );
				thrust::device_ptr< uchar > d_timersMaskEndPtr = d_timersMaskBeginPtr + stride;

				// Copy GPU timers back to CPU
				for ( int i = 0; i < CUDAPERFMON_KERNEL_TIMER_MAX; i++ )
				{
					thrust::device_ptr< GvCore::uint64 > d_timersArrayBeginPtr( d_timersArray->getPointer( i * stride ) );
					thrust::device_ptr< GvCore::uint64 > d_timersArrayEndPtr = d_timersArrayBeginPtr + stride;

					GvCore::uint64 result = thrust::count_if( d_timersMaskBeginPtr, d_timersMaskEndPtr, GvCore::not_equal_to_zero< GvCore::uint64 >() );
					if ( result > 0 )
					{
						thrust::device_vector< GvCore::uint64 > d_timersScale( result, result );
						thrust::device_vector< GvCore::uint64 > d_timersOutput( stride );
// TO DO : Before we had "#if 1" but it seems to generate crash at run-time... FIX this
#if 0
						thrust::device_ptr< uint64 > timerMinPtr = thrust::min_element( d_timersArrayBeginPtr, d_timersArrayEndPtr );
						thrust::device_ptr< uint64 > timerMaxPtr = thrust::max_element( d_timersArrayBeginPtr, d_timersArrayEndPtr );

						//printf( "max time at pixel %d, %d\n",
						//	( timerMaxPtr - d_timersArrayBeginPtr ) % res.x,
						//	( timerMaxPtr - d_timersArrayBeginPtr ) / res.x );

						_deviceEvents[ evtName ][ frameCurrentInstance[ evtName ] ].kernelTimerMin[ i ] = *timerMinPtr;
						_deviceEvents[ evtName ][ frameCurrentInstance[ evtName ] ].kernelTimerMax[ i ] = *timerMaxPtr;

						thrust::transform_if( d_timersArrayBeginPtr, d_timersArrayEndPtr, d_timersMaskBeginPtr,
							d_timersOutput.begin(), thrust::identity< uint64 >(), thrust::identity< uint64 >() );

						//thrust::transform( d_timersOutput.begin(), d_timersOutput.end(), d_timersScale.begin(), d_timersOutput.begin(), thrust::divides< uint64 >() );

						_deviceEvents[ evtName ][ frameCurrentInstance[ evtName ] ].timersArray[ i ] =
							thrust::reduce( d_timersOutput.begin(), d_timersOutput.end(), (uint64)0, thrust::plus< uint64 >() ) / result;

						// Wrong results (mostly due to an overflow)
						/*_deviceEvents[ evtName ][ frameCurrentInstance[ evtName ] ].timersArray[ i ] =
							thrust::reduce( d_timersArrayBeginPtr, d_timersArrayEndPtr, (uint64)0, thrust::plus< uint64 >() ) / result;*/
#else
						// CPU reference path

						thrust::copy( d_timersArrayBeginPtr, d_timersArrayEndPtr, tempArray.begin() );

						GvCore::uint64 sumc = 0;
						for ( uint j = 0; j < stride; ++j )
						{
							sumc += tempArray[ j ];
						}

						sumc = sumc / stride;
						_deviceEvents[ evtName ][ frameCurrentInstance[ evtName ] ].timersArray[ i ] = sumc;
#endif
					}
					else
					{
						_deviceEvents[ evtName ][ frameCurrentInstance[ evtName ] ].kernelTimerMin[ i ] = 0;
						_deviceEvents[ evtName ][ frameCurrentInstance[ evtName ] ].kernelTimerMax[ i ] = 0;
						_deviceEvents[ evtName ][ frameCurrentInstance[ evtName ] ].timersArray[ i ] = 0;
					}
				}
	#endif
			}
		}
#endif
	}
}

/******************************************************************************
 * ...
 *
 * @param evtName ...
 * @param n ...
 ******************************************************************************/
void CUDAPerfMon::setEventNumElems( ApplicationEvent evtName, uint n )
{
	assert( evtName < NumApplicationEvents );

	eventsNumElements[ evtName ][ frameCurrentInstance[ evtName ] ] = n;
}

/******************************************************************************
 * Start the main frame event
 ******************************************************************************/
void CUDAPerfMon::startFrame()
{
	_frameStarted = true;
	
	for ( uint i = 0; i < NumApplicationEvents; ++i )
	{
		frameCurrentInstance[ i ] = -1;
	}

	startEvent( cpmApplicationDefaultFrameEvent );
}

/******************************************************************************
 * Stop the main frame event
 ******************************************************************************/
void CUDAPerfMon::stopFrame()
{
	assert( _frameStarted );

	stopEvent( cpmApplicationDefaultFrameEvent );

	cudaDeviceSynchronize();

	_frameStarted = false;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void CUDAPerfMon::displayFrame()
{
	// Collect timings
	for ( uint evt = 0; evt < NumApplicationEvents; ++evt )
	{
		if ( frameCurrentInstance[ evt ] >= 0 )
		{
			std::cout << "Event " << _eventNames[ evt ] << " : ";

			for ( int ii = 0; ii <= frameCurrentInstance[ evt ]; ++ii )
			{
				if ( ii < CUDAPERFMON_GPU_TIMER_MAX_INSTANCES )
				{
					std::cout << _deviceTimer.getEventDuration( _deviceEvents[ evt ][ ii ] ) << " ";
				}
			}
			std::cout << "\n";
		}
	}
}

/******************************************************************************
 * Display information on screen
 *
 * - timeline by event
 *
 * @param eventType type of displayed information : 0 GPU evts, 1 CPU evts
 ******************************************************************************/
void CUDAPerfMon::displayFrameGL( uint eventType )
{
#ifdef USE_CUDAPERFMON

	// Push and pop the server attribute stack
	glPushAttrib( GL_ALL_ATTRIB_BITS );

	// Store Model View and Projection matrices
	// then configure transformations for 2D display (orthographic view)
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();
	
	// Remove texturing
	glActiveTexture( GL_TEXTURE0 );
	glDisable( GL_TEXTURE_2D );
	glDisable( GL_TEXTURE_RECTANGLE_ARB );
	glDisable( GL_TEXTURE_3D );
	glDisable( GL_DEPTH_TEST );

	// Set blending configuration
	glEnable( GL_BLEND );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

	// Draw fullscreen quad
	//
	// - translucent overlayed background
	glColor4f( 0.3f, 0.3f, 0.5f, 0.7f );
	glBegin( GL_QUADS );
		glVertex3f( -1.0, -1.0, 0.0 );
		glVertex3f( 1.0, -1.0, 0.0 );
		glVertex3f( 1.0, 1.0, 0.0 );
		glVertex3f( -1.0, 1.0, 0.0 );
	glEnd();

	// Display main title (in bottom left corner)
	if ( eventType == 0 )
	{
		glColor4f( 0.1f, 0.9f, 0.1f, 1.0f );
		ccRenderBitmapString( -1.0f, -1.0f, 0.0f, GLUT_BITMAP_HELVETICA_10, "CUDAPerformance monitor GPU. Timings in ms" );
	}
	else
	{
		glColor4f( 0.2f, 0.2f, 1.0f, 1.0f );
		ccRenderBitmapString( -1.0f, -1.0f, 0.0f, GLUT_BITMAP_HELVETICA_10, "CUDAPerformance monitor CPU. Timings in ms" );
	}

	// Retrieve main frame event duration
	float frameDuration = _hostTimer.getEventDuration( _hostEvents[ cpmApplicationDefaultFrameEvent ][ 0 ] );

#if CUDAPERFMON_GPU_TIMER_ENABLED
	frameDuration = max( frameDuration, _deviceTimer.getEventDuration( _deviceEvents[ cpmApplicationDefaultFrameEvent ][ 0 ] ) );
#endif

	float yOffset = 2.0f / 40.0f;
	float curPosOffset = 0.0f;

	// Iterate through all events
	for ( uint evt = 0; evt < NumApplicationEvents; ++evt )
	{
		// Draw current event information only if it is used in that frame
		if ( frameCurrentInstance[ evt ] >= 0 )
		{
			float2 barPos;
			barPos.x =- 1.0f;
			barPos.y = 0.9f - curPosOffset;

			// Draw quad
			// - lines (fullscreen width size)
			// - real %age timeline will be displayed inside after
			glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
			glColor4f( 0.0f, 0.0f, 0.3f, 0.9f );	// blue
			ccDrawQuadGL( barPos + make_float2( 0.0f, 0.0f ), barPos + make_float2( 2.0f, yOffset ) );
			glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );

			// Display event name
			glColor4f( 1.0f, 1.0f, 1.0f, 1.0f );	// white
			ccRenderBitmapString( barPos.x, barPos.y + 2.0f * yOffset / 10.0f, 0.0f, GLUT_BITMAP_HELVETICA_10, _eventNames[ evt ] );
			
			for ( int ii = 0; ii <= frameCurrentInstance[ evt ]; ++ii )
			{
				float2 subBarPosStart;
				float2 subBarPosEnd;
				float duration = 0.0f;

				// Handle event type
				if ( eventType == 0 )
				{
					// Handle device events

#if CUDAPERFMON_GPU_TIMER_ENABLED

					// Check bounds
					if ( ii < CUDAPERFMON_GPU_TIMER_MAX_INSTANCES )
					{
						// Retrieve current event's start time
						const float evtStartTime = _deviceTimer.getStartToStartTime( _deviceEvents[ cpmApplicationDefaultFrameEvent ][ 0 ], _deviceEvents[ evt ][ ii ] );
						const float evtStartOffset = ( evtStartTime / frameDuration ) * 2.0f;

						// Retrieve current event's duration
						duration = _deviceTimer.getEventDuration( _deviceEvents[ evt ][ ii ] );
						const float evtStopTime = evtStartTime + duration;
						const float evtStopOffset = ( evtStopTime / frameDuration ) * 2.0f;

						// Draw quad : current event %age timeline
						subBarPosStart = barPos + make_float2( evtStartOffset, 0.0f );
						subBarPosEnd = barPos + make_float2( evtStopOffset, yOffset );
						glColor4f( 0.1f, 0.8f, 0.2f, 0.7f );
						ccDrawQuadGL( subBarPosStart, subBarPosEnd );
					}
#endif
				}
				else
				{
					// Handle host events

					// Retrieve current event's start time
					float evtStartTime = _hostTimer.getStartToStartTime( _hostEvents[ cpmApplicationDefaultFrameEvent ][ 0 ], _hostEvents[ evt ][ ii ] );
					float evtStartOffset = ( evtStartTime / frameDuration ) * 2.0f;

					// Retrieve current event's duration
					duration = _hostTimer.getEventDuration( _hostEvents[ evt ][ ii ] );
					float evtStopTime = evtStartTime + duration;
					float evtStopOffset = ( evtStopTime / frameDuration ) * 2.0f;

					// Draw quad : current event %age timeline
					subBarPosStart = barPos + make_float2( evtStartOffset, 0.0f );
					subBarPosEnd = barPos + make_float2( evtStopOffset, yOffset );
					glColor4f( 0.1f, 0.2f, 0.8f, 0.5f );
					ccDrawQuadGL( subBarPosStart, subBarPosEnd );					
				}

				// Set current color
				glColor4f( 0.8f, 0.1f, 0.2f, 0.95f );

				// Write current event's duration in a temporary buffer
				// with customized formatting to be able to display it on screen
				char buff[ 255 ];	// TO DO : don't do allocation during rendering stage...
				const uint numElems = eventsNumElements[ evt ][ ii ];
				if ( numElems > 0 )
				{
					sprintf( buff, "%.4f (%d)", duration / (float)numElems, numElems );
					
					// Set current color
					glColor4f( 0.2f, 0.1f, 0.8f, 0.95f );
				}
				else
				{
					sprintf( buff, "%.4f", duration );	
				}

				// Display current event's duration
				if ( ( subBarPosEnd.x - subBarPosStart.x ) < 0.2f )
				{
					subBarPosEnd.x = subBarPosStart.x + 0.2f;
				}
				ccRenderBitmapString( ( subBarPosStart.x + subBarPosEnd.x ) / 2.0f - 0.05f, subBarPosStart.y + 2.0f * yOffset / 10.0f, 0.0f, GLUT_BITMAP_HELVETICA_10, buff );
			}

			// Update vertical offset for display
			curPosOffset += yOffset;

#if CUDAPERFMON_GPU_TIMER_ENABLED
	#if GV_PERFMON_INTERNAL_DEVICE_TIMER_ENABLED==1
			// Handle internal device events
			if ( eventType == 0 )
			{
				for ( int ii = 0; ii <= frameCurrentInstance[ evt ]; ++ii )
				{
					float2 subBarPosStart;
					float2 subBarPosEnd;
					//float duration = 0.0f;

					if ( ii < CUDAPERFMON_GPU_TIMER_MAX_INSTANCES )
					{
						// Retrieve current event's start time
						float evtStartTime = _deviceTimer.getStartToStartTime( _deviceEvents[ cpmApplicationDefaultFrameEvent ][ 0 ], _deviceEvents[ evt ][ ii ] );
						float evtStartOffset = ( evtStartTime / frameDuration ) * 2.0f;

						// Retrieve current event's stop time
						float evtStopTime = evtStartTime + _deviceTimer.getEventDuration( _deviceEvents[ evt ][ ii ] );
						float evtStopOffset = ( evtStopTime / frameDuration ) * 2.0f;

						/*double evtKrnDuration = 0.0;
						double evtKrnCycles = 0.0;*/

						// Iterate through internal device event
						for ( int jj = 0; jj < CUDAPERFMON_KERNEL_TIMER_MAX; ++jj )
						{
							// Get current device events
							GvDevicePerformanceTimer::Event& timerEvent = _deviceEvents[ evt ][ ii ];

							barPos.x = -1.0f;
							barPos.y = 0.9f - curPosOffset;

							subBarPosStart = barPos + make_float2( evtStartOffset, 0.0f );
							subBarPosEnd = barPos + make_float2( evtStopOffset, yOffset );

							if ( timerEvent.timersArray[ jj ] > 0 )
							{
								// kernel event duration in millisecond
								float evtKrnDuration = (float)timerEvent.timersArray[ jj ] / (float)_deviceClockRate;
								//float evtKrnDuration = ( (float)timerEvent.timersArray[ jj ] / 1000.0f ) / (float)_deviceClockRate;
								//evtKrnDuration *= 1000.0f;

								float evtKrnScale = (float)timerEvent.timersArray[ jj ] / (float)timerEvent.timersArray[ 0 ];

								//subBarPosStart = barPos + make_float2( evtKrnStartOffset * ( evtStopOffset - evtStartOffset ) + evtStartOffset, 0.0f );
								subBarPosEnd = barPos + make_float2( evtStartOffset + evtKrnScale * ( evtStopOffset - evtStartOffset ), yOffset );
								//subBarPosEnd = barPos + make_float2( evtKrnStopOffset, yOffset );

								// Draw quad : current event %age timeline
								glColor4f( 0.8f, 0.4f, 0.1f, 0.7f );
								ccDrawQuadGL( subBarPosStart, subBarPosEnd );

								//duration = _deviceTimer.getEventDuration( _deviceEvents[ evt ][ ii ] );

								// Display current event's duration
								char buff[ 255 ];
								sprintf( buff, "Event %d: %.4f (%llu. min %llu. max %llu)", jj, evtKrnDuration, timerEvent.timersArray[ jj ], timerEvent.kernelTimerMin[ jj ], timerEvent.kernelTimerMax[ jj ] );
								glColor4f( 0.0f, 0.0f, 0.0f, 0.95f );
								if ( ( subBarPosEnd.x - subBarPosStart.x ) < 0.2f )
								{
									subBarPosEnd.x = subBarPosStart.x + 0.2f;
								}
								ccRenderBitmapString( ( subBarPosStart.x + subBarPosEnd.x ) / 2.0f - 0.05f, subBarPosStart.y + 2.0f * yOffset / 10.0f, 0.0f, GLUT_BITMAP_HELVETICA_10, buff );

								// Update vertical offset for display
								curPosOffset += yOffset;
							}
						}
					}
				}
			}
	#endif
#endif
		}
	}

	// Disable blending
	glDisable( GL_BLEND );

	// Restore previous Model View and Projection matrices 
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();

	// Pop the server attribute stack
	glPopAttrib();

#endif
}

/******************************************************************************
 * ...
 *
 * @param overlayBuffer ...
 ******************************************************************************/
void CUDAPerfMon::displayOverlayGL( uchar* overlayBuffer )
{
	uint3 frameRes = d_timersArray->getResolution();

	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();

	glEnable( GL_TEXTURE_RECTANGLE_EXT );
	glEnable( GL_BLEND );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, overlayTex );
	glTexImage2D( GL_TEXTURE_RECTANGLE_EXT, 0, GL_R8, frameRes.x, frameRes.y, 0, GL_RED, GL_UNSIGNED_BYTE, overlayBuffer );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );

	//dispOverlayProg->setSamplerParam("inputOverlayTex", GL_TEXTURE_RECTANGLE_EXT, overlayTex);
	//dispOverlayProg->bind(true);
	glEnable( GL_TEXTURE_RECTANGLE_EXT );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, overlayTex );

	GLint u = frameRes.x;
	GLint v = frameRes.y;

	glBegin( GL_QUADS );
		glTexCoord2i( 0, v ); glVertex2i( -1, -1 );
		glTexCoord2i( u, v ); glVertex2i(  1, -1 );
		glTexCoord2i( u, 0 ); glVertex2i(  1,  1 );
		glTexCoord2i( 0, 0 ); glVertex2i( -1,  1 );
	glEnd();

	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
	//dispOverlayProg->unbind(true);
	
	glDisable( GL_TEXTURE_RECTANGLE_EXT );
	glDisable( GL_BLEND );

	glMatrixMode( GL_PROJECTION );
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void CUDAPerfMon::displayCacheInfoGL()
{
	// TODO
}

/******************************************************************************
 * ...
 *
 * @param numNodePagesUsed ...
 * @param numNodePagesWrited ...
 * @param numBrickPagesUsed ...
 * @param numBrickPagesWrited ...
 ******************************************************************************/
void CUDAPerfMon::saveFrameStats( uint numNodePagesUsed, uint numNodePagesWrited,
								 uint numBrickPagesUsed, uint numBrickPagesWrited )
{
	static uint frameNum = 0;

	std::ofstream ofs("perfMonStats.csv", (frameNum == 0) ? std::ios_base::trunc : std::ios_base::app);

	if ( ! ofs )
	{
		return;
	}

	if ( frameNum == 0 )
	{
		ofs << "frame";

		for ( uint evt = 0; evt < NumApplicationEvents; ++evt )
		{
			ofs << ";\"" << _eventNames[ evt ] << " (CPU)\"";
		}

#if CUDAPERFMON_GPU_TIMER_ENABLED
		for ( uint evt = 0; evt < NumApplicationEvents; ++evt )
		{
			ofs << ";\"" << _eventNames[ evt ] << " (GPU)\"";
			ofs << ";\"" << _eventNames[ evt ] << " (Count)\"";

			if ( frameCurrentInstance[ evt ] >= 0 )
			{
				for ( int jj = 0; jj < CUDAPERFMON_KERNEL_TIMER_MAX; ++jj )
				{
					if ( _deviceEvents[ evt ][ 0 ].timersArray[ jj ] > 0 )
					{
						ofs << ";\"" << _eventNames[ evt ] << "_Event" << jj << " (Kernel)\"";
					}
				}
			}
		}
#endif

#if CUDAPERFMON_CACHE_INFO==1
		ofs << ";\"Pages Used (Nodes)\"";
		ofs << ";\"Pages Writed (Nodes)\"";
		ofs << ";\"Pages Used (Bricks)\"";
		ofs << ";\"Pages Writed (Bricks)\"";
#endif
		ofs << std::endl;
	}

	ofs << frameNum;

	// Write CPU times
	for ( uint evt = 0; evt < NumApplicationEvents; ++evt )
	{
		float evtDuration = 0.0f;

		for ( int ii = 0; ii <= frameCurrentInstance[ evt ]; ++ii )
		{
			uint numElems = eventsNumElements[ evt ][ ii ];

			if ( numElems > 0 )
			{
				evtDuration += _hostTimer.getEventDuration( _hostEvents[ evt ][ ii ] ) / static_cast< float >( numElems );
			}
			else
			{
				evtDuration += _hostTimer.getEventDuration( _hostEvents[ evt ][ ii ] );
			}
		}

		ofs << ";" << evtDuration;
	}

	// Write GPU / Kernel times
#if CUDAPERFMON_GPU_TIMER_ENABLED
	for ( uint evt = 0; evt < NumApplicationEvents; ++evt )
	{
		float evtDuration = 0.0f;
		uint evtElements = 0;

		for ( int ii = 0; ii <= frameCurrentInstance[ evt ]; ++ii )
		{
			if ( ii < CUDAPERFMON_GPU_TIMER_MAX_INSTANCES )
			{
				uint numElems = eventsNumElements[ evt ][ ii ];

				if ( numElems > 0 )
				{
					evtDuration += _deviceTimer.getEventDuration( _deviceEvents[ evt ][ ii ] ) / static_cast< float >( numElems );
					evtElements += numElems;
				}
				else
				{
					evtDuration += _deviceTimer.getEventDuration( _deviceEvents[ evt ][ ii ] );
				}
			}
		}

		ofs << ";" << evtDuration;
		ofs << ";" << evtElements;

		for ( int ii = 0; ii <= frameCurrentInstance[ evt ]; ++ii )
		{
			if ( ii < CUDAPERFMON_GPU_TIMER_MAX_INSTANCES )
			{
				for ( int jj = 0; jj < CUDAPERFMON_KERNEL_TIMER_MAX; ++jj )
				{
					if ( _deviceEvents[ evt ][ ii ].timersArray[ jj ] > 0 )
					{
						ofs << ";" << ( static_cast< float >( _deviceEvents[ evt ][ ii ].timersArray[ jj ] ) / static_cast< float >( _deviceClockRate ) );
					}
				}
			}
		}
	}
#endif

#if CUDAPERFMON_CACHE_INFO==1
		ofs << ";" << numNodePagesUsed;
		ofs << ";" << numNodePagesWrited;
		ofs << ";" << numBrickPagesUsed;
		ofs << ";" << numBrickPagesWrited;
#endif

	ofs << std::endl;

	frameNum++;
}
