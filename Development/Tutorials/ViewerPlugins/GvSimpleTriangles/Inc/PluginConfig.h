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
#ifndef GVSIMPLETRIANGLES_CONFIG_H
#define GVSIMPLETRIANGLES_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVSIMPLETRIANGLES_MAKELIB	// Create a static library.
#		define GVSIMPLETRIANGLES_EXPORT
#		define GVSIMPLETRIANGLES_TEMPLATE_EXPORT
#	elif defined GVSIMPLETRIANGLES_USELIB	// Use a static library.
#		define GVSIMPLETRIANGLES_EXPORT
#		define GVSIMPLETRIANGLES_TEMPLATE_EXPORT

#	elif defined GVSIMPLETRIANGLES_MAKEDLL	// Create a DLL library.
#		define GVSIMPLETRIANGLES_EXPORT	__declspec(dllexport)
#		define GVSIMPLETRIANGLES_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVSIMPLETRIANGLES_EXPORT	__declspec(dllimport)
#		define GVSIMPLETRIANGLES_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVSIMPLETRIANGLES_MAKEDLL) || defined(GVSIMPLETRIANGLES_MAKELIB)
#		define GVSIMPLETRIANGLES_EXPORT
#		define GVSIMPLETRIANGLES_TEMPLATE_EXPORT
#	else
#		define GVSIMPLETRIANGLES_EXPORT
#		define GVSIMPLETRIANGLES_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
