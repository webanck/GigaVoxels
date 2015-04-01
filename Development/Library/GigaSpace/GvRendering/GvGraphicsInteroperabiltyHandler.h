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

#ifndef _GV_GRAPHICS_INTEROPERABILTY_HANDLER_H_
#define _GV_GRAPHICS_INTEROPERABILTY_HANDLER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

// OpenGL
#include <GL/glew.h>

// Cuda
#include <driver_types.h>

// STL
#include <vector>
#include <utility>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

namespace GvRendering
{
	class GvGraphicsResource;
	struct GvRendererContext;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvRendering
{

/** 
 * @class GvGraphicsInteroperabiltyHandler
 *
 * @brief The GvGraphicsInteroperabiltyHandler class provides interface to
 *
 * Some resources from OpenGL may be mapped into the address space of CUDA,
 * either to enable CUDA to read data written by OpenGL, or to enable CUDA
 * to write data for consumption by OpenGL.
 */
class GIGASPACE_EXPORT GvGraphicsInteroperabiltyHandler
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Enumeration of the graphics IO slots
	 * connected to the GigaVoxles engine
	 * (i.e. color and depth inputs/outputs)
	 */
	enum GraphicsResourceSlot
	{
		eColorReadSlot,
		eColorWriteSlot,
		eColorReadWriteSlot,
		eDepthReadSlot,
		eDepthWriteSlot,
		eDepthReadWriteSlot,
		eNbGraphicsResourceSlots
	};

	/**
	 * Enumeration of the graphics IO slots
	 * connected to the GigaVoxles engine
	 * (i.e. color and depth inputs/outputs)
	 */
	enum GraphicsResourceDeviceSlot
	{
		eUndefinedGraphicsResourceDeviceSlot = -1,
		eColorInput,
		eColorOutput,
		eDepthInput,
		eDepthOutput,
		eNbGraphicsResourceDeviceSlots
	};

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvGraphicsInteroperabiltyHandler();

	/**
	 * Destructor
	 */
	 virtual ~GvGraphicsInteroperabiltyHandler();

	/**
	 * Initiliaze
	 *
	 * @param pWidth width
	 * @param pHeight height
	 */
	//void initialize( int pWidth, int pHeight );
	 void initialize();

	/**
	 * Finalize
	 */
	void finalize();

	/**
	 * Reset
	 */
	bool reset();

	/**
	 * Attach an OpenGL buffer object (i.e. a PBO, a VBO, etc...) to an internal graphics resource 
	 * that will be mapped to a color or depth slot used during rendering.
	 *
	 * @param pGraphicsResourceSlot the associated internal graphics resource (color or depth) and its type of access (read, write or read/write)
	 * @param pBuffer the OpenGL buffer
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool connect( GraphicsResourceSlot pSlot, GLuint pBuffer );

	/**
	 * Attach an OpenGL texture or renderbuffer object to an internal graphics resource 
	 * that will be mapped to a color or depth slot used during rendering.
	 *
	 * @param pGraphicsResourceSlot the associated internal graphics resource (color or depth) and its type of access (read, write or read/write)
	 * @param pImage the OpenGL texture or renderbuffer object
	 * @param pTarget the target of the OpenGL texture or renderbuffer object
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool connect( GraphicsResourceSlot pSlot, GLuint pImage, GLenum pTarget );

	/**
	 * Dettach an OpenGL buffer object (i.e. a PBO, a VBO, etc...), texture or renderbuffer object
	 * to its associated internal graphics resource mapped to a color or depth slot used during rendering.
	 *
	 * @param pGraphicsResourceSlot the internal graphics resource slot (color or depth)
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool disconnect( GraphicsResourceSlot pSlot );
		
	/**
	 * Map graphics resources into CUDA memory in order to be used during rendering.
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool mapResources();

	/**
	 * Unmap graphics resources from CUDA memory in order to be used by OpenGL.
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool unmapResources();

	bool setRendererContextInfo( GvRendererContext& pRendererContext );

	inline const std::vector< std::pair< GraphicsResourceSlot, GvGraphicsResource* > >& getGraphicsResources() const;
	inline std::vector< std::pair< GraphicsResourceSlot, GvGraphicsResource* > >& editGraphicsResources();

	//bool bindTo( const struct surfaceReference* surfref );
		
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	//GvGraphicsResource* _graphicsResources[ GraphicsResourceDeviceSlot ];
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::vector< std::pair< GraphicsResourceSlot, GvGraphicsResource* > > _graphicsResources;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	///**
	// * Width
	// */
	//int _width;

	///**
	// * Height
	// */
	//int _height;

	///**
	// * ...
	// */
	//bool _hasColorInput;
	//bool _hasColorOutput;
	//bool _hasDepthInput;
	//bool _hasDepthOutput;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} // namespace GvRendering

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvGraphicsInteroperabiltyHandler.inl"

#endif
