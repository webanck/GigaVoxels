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
#ifndef GVPROXYGEOMETRYMANAGER_CONFIG_H
#define GVPROXYGEOMETRYMANAGER_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVPROXYGEOMETRYMANAGER_MAKELIB	// Create a static library.
#		define GVPROXYGEOMETRYMANAGER_EXPORT
#		define GVPROXYGEOMETRYMANAGER_TEMPLATE_EXPORT
#	elif defined GVPROXYGEOMETRYMANAGER_USELIB	// Use a static library.
#		define GVPROXYGEOMETRYMANAGER_EXPORT
#		define GVPROXYGEOMETRYMANAGER_TEMPLATE_EXPORT

#	elif defined GVPROXYGEOMETRYMANAGER_MAKEDLL	// Create a DLL library.
#		define GVPROXYGEOMETRYMANAGER_EXPORT	__declspec(dllexport)
#		define GVPROXYGEOMETRYMANAGER_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVPROXYGEOMETRYMANAGER_EXPORT	__declspec(dllimport)
#		define GVPROXYGEOMETRYMANAGER_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVPROXYGEOMETRYMANAGER_MAKEDLL) || defined(GVPROXYGEOMETRYMANAGER_MAKELIB)
#		define GVPROXYGEOMETRYMANAGER_EXPORT
#		define GVPROXYGEOMETRYMANAGER_TEMPLATE_EXPORT
#	else
#		define GVPROXYGEOMETRYMANAGER_EXPORT
#		define GVPROXYGEOMETRYMANAGER_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
