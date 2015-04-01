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

/**
 * @defgroup GsGraphics
 */
#ifndef _GS_GRAPHICS_CONFIG_H_
#define _GS_GRAPHICS_CONFIG_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

//*** GsGraphics Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GSGRAPHICS_MAKELIB	// Create a static library.
#		define GSGRAPHICS_EXPORT
#		define GSGRAPHICS_TEMPLATE_EXPORT
#	elif defined GSGRAPHICS_USELIB	// Use a static library.
#		define GSGRAPHICS_EXPORT
#		define GSGRAPHICS_TEMPLATE_EXPORT
#	elif defined GSGRAPHICS_MAKEDLL	// Create a DLL library.
#		define GSGRAPHICS_EXPORT	__declspec(dllexport)
#		define GSGRAPHICS_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GSGRAPHICS_EXPORT	__declspec(dllimport)
#		define GSGRAPHICS_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GSGRAPHICS_MAKEDLL) || defined(GSGRAPHICS_MAKELIB)
#		define GSGRAPHICS_EXPORT
#		define GSGRAPHICS_TEMPLATE_EXPORT
#	else
#		define GSGRAPHICS_EXPORT
#		define GSGRAPHICS_TEMPLATE_EXPORT	extern
#	endif
#endif

// ---------------- GLM library Management ----------------

/**
 * To remove warnings at compilation with GLM deprecated functions
 */
#define GLM_FORCE_RADIANS

#endif

