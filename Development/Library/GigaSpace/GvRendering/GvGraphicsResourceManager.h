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

#ifndef _GV_GRAPHICS_RESOURCE_MANAGER_H_
#define _GV_GRAPHICS_RESOURCE_MANAGER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

// STL
#include <vector>
#include <cstddef>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GigaVoxels
namespace GvRendering
{
	class GvGraphicsResource;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvRendering
{
	
	/** 
	 * @class GvGraphicsResourceManager
	 *
	 * @brief The GvGraphicsResourceManager class provides way to access graphics resource.
	 *
	 * @ingroup GvRenderer
	 *
	 * The GvGraphicsResourceManager class is the main accessor of all graphics resources.
	 */
	class GIGASPACE_EXPORT GvGraphicsResourceManager
	{

		/**************************************************************************
		 ***************************** PUBLIC SECTION *****************************
		 **************************************************************************/

	public:

		/******************************* ATTRIBUTES *******************************/

		/******************************** METHODS *********************************/

		/**
		 * Get the device manager
		 *
		 * @return the device manager
		 */
		static GvGraphicsResourceManager& get();

		/**
		 * Initialize the device manager
		 */
		bool initialize();

		/**
		 * Finalize the device manager
		 */
		void finalize();

		/**
		 * Get the number of devices
		 *
		 * @return the number of devices
		 */
		size_t getNbResources() const;

		/**
		 * Get the device given its index
		 *
		 * @param the index of the requested device
		 *
		 * @return the requested device
		 */
		const GvGraphicsResource* getResource( int pIndex ) const;

		/**
		 * Get the device given its index
		 *
		 * @param the index of the requested device
		 *
		 * @return the requested device
		 */
		GvGraphicsResource* editResource( int pIndex );
		
		/**************************************************************************
		 **************************** PROTECTED SECTION ***************************
		 **************************************************************************/

	protected:

		/******************************* ATTRIBUTES *******************************/

		/******************************** TYPEDEFS ********************************/

		/**
		 * The unique device manager
		 */
		static GvGraphicsResourceManager* msInstance;

		/**
		 * The container of devices
		 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
		std::vector< GvGraphicsResource* > _graphicsResources;
#if defined _MSC_VER
#pragma warning( pop )
#endif

		/**
		 * Flag to tell wheter or not the device manager is initialized
		 */
		bool _isInitialized;

		/******************************** METHODS *********************************/

		/**
		 * Constructor
		 */
		GvGraphicsResourceManager();

		/**
		 * Destructor
		 */
		~GvGraphicsResourceManager();

		/**************************************************************************
		 ***************************** PRIVATE SECTION ****************************
		 **************************************************************************/

	private:

		/******************************* ATTRIBUTES *******************************/

		/******************************** METHODS *********************************/

		/**
		 * Copy constructor forbidden.
		 */
		GvGraphicsResourceManager( const GvGraphicsResourceManager& );

		/**
		 * Copy operator forbidden.
		 */
		GvGraphicsResourceManager& operator=( const GvGraphicsResourceManager& );

	};

} // namespace GvRendering

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#endif // !_GV_GRAPHICS_RESOURCE_MANAGER_H_
