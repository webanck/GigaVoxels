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

#ifndef _GV_COLLISION_CONFIG_H_
#define _GV_COLLISION_CONFIG_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/**
 * Plugin Library 
 */

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVCOLLISION_MAKELIB	// Create a static library.
#		define GVCOLLISION_EXPORT
#		define GVCOLLISION_TEMPLATE_EXPORT
#	elif defined GVCOLLISION_USELIB	// Use a static library.
#		define GVCOLLISION_EXPORT
#		define GVCOLLISION_TEMPLATE_EXPORT

#	elif defined GVCOLLISION_MAKEDLL	// Create a DLL library.
#		define GVCOLLISION_EXPORT	__declspec(dllexport)
#		define GVCOLLISION_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVCOLLISION_EXPORT	__declspec(dllimport)
#		define GVCOLLISION_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVCOLLISION_MAKEDLL) || defined(GVCOLLISION_MAKELIB)
#		define GVCOLLISION_EXPORT
#		define GVCOLLISION_TEMPLATE_EXPORT
#	else
#		define GVCOLLISION_EXPORT
#		define GVCOLLISION_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
