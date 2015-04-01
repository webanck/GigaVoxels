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
#ifndef GVANIMATEDFAN_CONFIG_H
#define GVANIMATEDFAN_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVANIMATEDFAN_MAKELIB	// Create a static library.
#		define GVANIMATEDFAN_EXPORT
#		define GVANIMATEDFAN_TEMPLATE_EXPORT
#	elif defined GVANIMATEDFAN_USELIB	// Use a static library.
#		define GVANIMATEDFAN_EXPORT
#		define GVANIMATEDFAN_TEMPLATE_EXPORT
#	elif defined GVANIMATEDFAN_MAKEDLL	// Create a DLL library.
#		define GVANIMATEDFAN_EXPORT	__declspec(dllexport)
#		define GVANIMATEDFAN_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVANIMATEDFAN_EXPORT	__declspec(dllimport)
#		define GVANIMATEDFAN_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVANIMATEDFAN_MAKEDLL) || defined(GVANIMATEDFAN_MAKELIB)
#		define GVANIMATEDFAN_EXPORT
#		define GVANIMATEDFAN_TEMPLATE_EXPORT
#	else
#		define GVANIMATEDFAN_EXPORT
#		define GVANIMATEDFAN_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
