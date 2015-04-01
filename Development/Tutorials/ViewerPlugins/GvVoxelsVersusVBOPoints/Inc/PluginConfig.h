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
 * ...
 */
#ifndef GVVOXELSVERSUSVBOPOINTS_CONFIG_H
#define GVVOXELSVERSUSVBOPOINTS_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVVOXELSVERSUSVBOPOINTS_MAKELIB	// Create a static library.
#		define GVVOXELSVERSUSVBOPOINTS_EXPORT
#		define GVVOXELSVERSUSVBOPOINTS_TEMPLATE_EXPORT
#	elif defined GVVOXELSVERSUSVBOPOINTS_USELIB	// Use a static library.
#		define GVVOXELSVERSUSVBOPOINTS_EXPORT
#		define GVVOXELSVERSUSVBOPOINTS_TEMPLATE_EXPORT

#	elif defined GVVOXELSVERSUSVBOPOINTS_MAKEDLL	// Create a DLL library.
#		define GVVOXELSVERSUSVBOPOINTS_EXPORT	__declspec(dllexport)
#		define GVVOXELSVERSUSVBOPOINTS_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVVOXELSVERSUSVBOPOINTS_EXPORT	__declspec(dllimport)
#		define GVVOXELSVERSUSVBOPOINTS_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVVOXELSVERSUSVBOPOINTS_MAKEDLL) || defined(GVVOXELSVERSUSVBOPOINTS_MAKELIB)
#		define GVVOXELSVERSUSVBOPOINTS_EXPORT
#		define GVVOXELSVERSUSVBOPOINTS_TEMPLATE_EXPORT
#	else
#		define GVVOXELSVERSUSVBOPOINTS_EXPORT
#		define GVVOXELSVERSUSVBOPOINTS_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
