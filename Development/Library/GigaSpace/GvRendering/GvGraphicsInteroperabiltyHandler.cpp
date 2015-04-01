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

#include "GvRendering/GvGraphicsInteroperabiltyHandler.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvRendering/GvGraphicsResource.h"
#include "GvRendering/GvRendererContext.h"
//#include "GvRendering/GvRendererHelpersKernel.h"
#include "GvCore/GvError.h"

// Cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvRendering;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvGraphicsInteroperabiltyHandler::GvGraphicsInteroperabiltyHandler()
:	_graphicsResources()
{
}

/******************************************************************************
 * Desconstructor
 ******************************************************************************/
GvGraphicsInteroperabiltyHandler::~GvGraphicsInteroperabiltyHandler()
{
	//finalize();

	// Disconnect all registered graphics resources
	reset();
}

/******************************************************************************
 * Initiliaze
 *
 * @param pWidth width
 * @param pHeight height
 ******************************************************************************/
//void GvGraphicsInteroperabiltyHandler::initialize( int pWidth, int pHeight )
void GvGraphicsInteroperabiltyHandler::initialize()
{
}

/******************************************************************************
 * Finalize
 ******************************************************************************/
void GvGraphicsInteroperabiltyHandler::finalize()
{
	for ( int i = 0; i < _graphicsResources.size(); i++ )
	{
		delete _graphicsResources[ i ].second;
		_graphicsResources[ i ].second =  NULL;
	}
	_graphicsResources.clear();	// TO DO  : check if "clear()" call the destructor of elements
}

/******************************************************************************
 * Reset
 ******************************************************************************/
bool GvGraphicsInteroperabiltyHandler::reset()
{
	// Iterate through graphics resource info
	std::vector< std::pair< GraphicsResourceSlot, GvGraphicsResource* > >::iterator itGraphicsResourceInfo = _graphicsResources.begin();
	for ( ; itGraphicsResourceInfo != _graphicsResources.end(); ++itGraphicsResourceInfo )
	{
		// Retrieve current graphics resource info
		std::pair< GraphicsResourceSlot, GvGraphicsResource* >& graphicsResourceInfo = *itGraphicsResourceInfo;
		GvGraphicsResource* graphicsResource = graphicsResourceInfo.second;

		if ( graphicsResource != NULL )
		{
			graphicsResource->unregister();

			delete graphicsResource;
			graphicsResource = NULL;
		}
	}

	_graphicsResources.clear();

	return false;
}

/******************************************************************************
 * Attach an OpenGL buffer object (i.e. a PBO, a VBO, etc...) to an internal graphics resource 
 * that will be mapped to a color or depth slot used during rendering.
 *
 * @param pGraphicsResourceSlot the associated internal graphics resource (color or depth) and its type of access (read, write or read/write)
 * @param pBuffer the OpenGL buffer
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
bool GvGraphicsInteroperabiltyHandler::connect( GraphicsResourceSlot pSlot, GLuint pBuffer )
{
	std::pair< GraphicsResourceSlot, GvGraphicsResource* > graphicsResourceInfo;
	
	graphicsResourceInfo.first = pSlot;

	GvGraphicsResource* graphicsResource = new GvGraphicsResource();
	graphicsResourceInfo.second = graphicsResource;			// utiliser l'ID des graphics resource à la place ??

	// Retrieve flags
	unsigned int flags = cudaGraphicsRegisterFlagsNone;
	switch ( pSlot )
	{
		case eColorReadSlot:
		case eDepthReadSlot:
			flags = cudaGraphicsRegisterFlagsReadOnly;
			break;

		case eColorWriteSlot:
		case eDepthWriteSlot:
			flags = cudaGraphicsRegisterFlagsWriteDiscard;
			break;

		case eColorReadWriteSlot:
		case eDepthReadWriteSlot:
			flags = cudaGraphicsRegisterFlagsNone;
			break;

		default:
			assert( false );
			flags = cudaGraphicsRegisterFlagsNone;
			break;
	}

	// Register graphics resource
	graphicsResource->registerBuffer( pBuffer, flags );

	// Store the graphics resource info
	_graphicsResources.push_back( graphicsResourceInfo );

	return false;
}

/******************************************************************************
 * Attach an OpenGL texture or renderbuffer object to an internal graphics resource 
 * that will be mapped to a color or depth slot used during rendering.
 *
 * @param pGraphicsResourceSlot the associated internal graphics resource (color or depth) and its type of access (read, write or read/write)
 * @param pImage the OpenGL texture or renderbuffer object
 * @param pTarget the target of the OpenGL texture or renderbuffer object
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
bool GvGraphicsInteroperabiltyHandler::connect( GraphicsResourceSlot pSlot, GLuint pImage, GLenum pTarget )
{
	std::pair< GraphicsResourceSlot, GvGraphicsResource* > graphicsResourceInfo;
	
	graphicsResourceInfo.first = pSlot;

	GvGraphicsResource* graphicsResource = new GvGraphicsResource();
	graphicsResourceInfo.second = graphicsResource;			// utiliser l'ID des graphics resource à la place ??

	//_graphicsResources[ index ]->_isRegistered;
	//_graphicsResources[ index ]->unregister();
	
	// Retrieve flags
	// 
	// target must match the type of the object, and must be one of GL_TEXTURE_2D, GL_TEXTURE_RECTANGLE,
	// GL_TEXTURE_CUBE_MAP, GL_TEXTURE_3D, GL_TEXTURE_2D_ARRAY, or GL_RENDERBUFFER.
	unsigned int flags = cudaGraphicsRegisterFlagsNone;
	switch ( pSlot )				// TO DO : check valaidity with target !!!!!!!!!!!!!!!!!!!!
	{
		case eColorReadSlot:
		case eDepthReadSlot:
			flags = cudaGraphicsRegisterFlagsReadOnly;
			break;

		case eColorWriteSlot:
		case eDepthWriteSlot:
		case eColorReadWriteSlot:
		case eDepthReadWriteSlot:
			flags = cudaGraphicsRegisterFlagsSurfaceLoadStore;
			break;

		default:
			assert( false );
			flags = cudaGraphicsRegisterFlagsNone;
			break;
	}

	// Register graphics resource
	graphicsResource->registerImage( pImage, pTarget, flags );

	// Store the graphics resource info
	_graphicsResources.push_back( graphicsResourceInfo );

	return false;
}

/******************************************************************************
 * Dettach an OpenGL buffer object (i.e. a PBO, a VBO, etc...), texture or renderbuffer object
 * to its associated internal graphics resource mapped to a color or depth slot used during rendering.
 *
 * @param pGraphicsResourceSlot the internal graphics resource slot (color or depth)
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
bool GvGraphicsInteroperabiltyHandler::disconnect( GraphicsResourceSlot pSlot )
{
	std::vector< std::pair< GraphicsResourceSlot, GvGraphicsResource* > >::iterator itGraphicsResourceInfo = _graphicsResources.begin();

	for ( ; itGraphicsResourceInfo != _graphicsResources.end(); ++itGraphicsResourceInfo )
	{
		std::pair< GraphicsResourceSlot, GvGraphicsResource* >& graphicsResourceInfo = *itGraphicsResourceInfo;

		if ( graphicsResourceInfo.first == pSlot )
		{
			GvGraphicsResource* graphicsResource = graphicsResourceInfo.second;
			graphicsResource->unregister();

			delete graphicsResource;
			graphicsResource = NULL;

			_graphicsResources.erase( itGraphicsResourceInfo );

			break;
		}
	}

	return false;
}

/******************************************************************************
 * Map graphics resources into CUDA memory in order to be used during rendering.
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
bool GvGraphicsInteroperabiltyHandler::mapResources()
{
	for ( int i = 0; i < _graphicsResources.size(); i++ )
	{
		std::pair< GraphicsResourceSlot, GvGraphicsResource* >& graphicsResourceInfo = _graphicsResources[ i ];

		GraphicsResourceSlot graphicsResourceSlot = graphicsResourceInfo.first;
		GvGraphicsResource* graphicsResource  = graphicsResourceInfo.second;

		// [ 1 ] - Map resource
		graphicsResource->map();

		// [ 2 ] - Bind array to texture or surface if needed
		if ( graphicsResource->getMemoryType() == GvGraphicsResource::eCudaArray )
		{
			struct cudaArray* imageArray = static_cast< struct cudaArray* >( graphicsResource->getMappedAddress() );

			//cudaError_t error;
			switch ( graphicsResourceSlot )
			{
				case eColorReadSlot:
					//error = cudaBindTextureToArray( GvRendering::_inputColorTexture, imageArray );
					graphicsResource->_mappedAddressType = GvGraphicsResource::eTexture;
					break;

				case eColorWriteSlot:
				case eColorReadWriteSlot:
					//error = cudaBindSurfaceToArray( GvRendering::_colorSurface, imageArray );
					graphicsResource->_mappedAddressType = GvGraphicsResource::eSurface;
					break;

				case eDepthReadSlot:
					//error = cudaBindTextureToArray( GvRendering::_inputDepthTexture, imageArray );
					graphicsResource->_mappedAddressType = GvGraphicsResource::eTexture;
					break;

				case eDepthWriteSlot:
				case eDepthReadWriteSlot:
					//error = cudaBindSurfaceToArray( GvRendering::_depthSurface, imageArray );
					graphicsResource->_mappedAddressType = GvGraphicsResource::eSurface;
					break;

				default:
					assert( false );
					break;
			}
		}

		// [ 3 ] - store info into context
		// ...
		
		/*if ( _graphicsResources[ i ]->_mappedAddressType == GvGraphicsResource::eTexture )
		{
			_graphicsResources[ i ]->bind();
		}
		else if ( _graphicsResources[ i ]->_mappedAddressType == GvGraphicsResource::eSurface )
		{
			_graphicsResources[ i ]->bind();
		}*/
	}

	return false;
}

/******************************************************************************
 * Unmap graphics resources from CUDA memory in order to be used by OpenGL.
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
bool GvGraphicsInteroperabiltyHandler::unmapResources()
{
	for ( int i = 0; i < _graphicsResources.size(); i++ )
	{
		_graphicsResources[ i ].second->unmap();

		/*if ( _graphicsResources[ i ]->_mappedAddressType )
		{
			_graphicsResources[ i ]->unbind();
		}*/
	}

	return false;
}

/******************************************************************************
 * ...
 ******************************************************************************/
bool GvGraphicsInteroperabiltyHandler::setRendererContextInfo( GvRendererContext& pRendererContext )
{
	// Color info
	pRendererContext._graphicsResources[ eColorInput ] = NULL;
	pRendererContext._graphicsResources[ eColorOutput ] = NULL;

	pRendererContext._graphicsResourceAccess[ eColorInput ] = GvGraphicsResource::eUndefinedMappedAddressType;
	pRendererContext._graphicsResourceAccess[ eColorOutput ] = GvGraphicsResource::eUndefinedMappedAddressType;

	// Depth info
	pRendererContext._graphicsResources[ eDepthInput ] = NULL;
	pRendererContext._graphicsResources[ eDepthOutput ] = NULL;

	pRendererContext._graphicsResourceAccess[ eDepthInput ] = GvGraphicsResource::eUndefinedMappedAddressType;
	pRendererContext._graphicsResourceAccess[ eDepthOutput ] = GvGraphicsResource::eUndefinedMappedAddressType;

	// Iterate through graphics resources info
	for ( int i = 0; i < _graphicsResources.size(); i++ )
	{
		// Get current graphics resource info
		const std::pair< GraphicsResourceSlot, GvGraphicsResource* >& graphicsResourceInfo = _graphicsResources[ i ];
		GraphicsResourceSlot graphicsResourceSlot  = graphicsResourceInfo.first;
		GvGraphicsResource* graphicsResource  = graphicsResourceInfo.second;
		assert( graphicsResource != NULL );

		switch ( graphicsResourceSlot )
		{
			// Handle color

			case eColorReadSlot:
				// Color input
				pRendererContext._graphicsResources[ eColorInput ] = graphicsResource->getMappedAddress();
				pRendererContext._graphicsResourceAccess[ eColorInput ] = graphicsResource->getMappedAddressType();
				break;

			case eColorWriteSlot:
				// Color output
				pRendererContext._graphicsResources[ eColorOutput ] = graphicsResource->getMappedAddress();
				pRendererContext._graphicsResourceAccess[ eColorOutput ] = graphicsResource->getMappedAddressType();
				break;

			case eColorReadWriteSlot:
				// Color input and output mapped addresses
				pRendererContext._graphicsResources[ eColorInput ] = graphicsResource->getMappedAddress();
				pRendererContext._graphicsResources[ eColorOutput ] = pRendererContext._graphicsResources[ eColorInput ];
				// Color input and output mapped address types
				pRendererContext._graphicsResourceAccess[ eColorInput ] = graphicsResource->getMappedAddressType();
				pRendererContext._graphicsResourceAccess[ eColorOutput ] = pRendererContext._graphicsResourceAccess[ eColorInput ];
				break;

			// Handle depth

			case eDepthReadSlot:
				// Depth input
				pRendererContext._graphicsResources[ eDepthInput ] = graphicsResource->getMappedAddress();
				pRendererContext._graphicsResourceAccess[ eDepthInput ] = graphicsResource->getMappedAddressType();
				break;

			case eDepthWriteSlot:
				// Depth output
				pRendererContext._graphicsResources[ eDepthOutput ] = graphicsResource->getMappedAddress();
				pRendererContext._graphicsResourceAccess[ eDepthOutput ] = graphicsResource->getMappedAddressType();
				break;

			case eDepthReadWriteSlot:
				// Depth input and output mapped addresses
				pRendererContext._graphicsResources[ eDepthInput ] = graphicsResource->getMappedAddress();
				pRendererContext._graphicsResources[ eDepthOutput ] = pRendererContext._graphicsResources[ eDepthInput ];
				// Depth input and output mapped address types
				pRendererContext._graphicsResourceAccess[ eDepthInput ] = graphicsResource->getMappedAddressType();
				pRendererContext._graphicsResourceAccess[ eDepthOutput ] = pRendererContext._graphicsResourceAccess[ eDepthInput ];
				break;

			default:
				assert( false );
				// Color info
				pRendererContext._graphicsResources[ eColorInput ] = NULL;
				pRendererContext._graphicsResources[ eColorOutput ] = NULL;
				pRendererContext._graphicsResourceAccess[ eColorInput ] = GvGraphicsResource::eUndefinedMappedAddressType;
				pRendererContext._graphicsResourceAccess[ eColorOutput ] = GvGraphicsResource::eUndefinedMappedAddressType;
				// Depth info
				pRendererContext._graphicsResources[ eDepthInput ] = NULL;
				pRendererContext._graphicsResources[ eDepthOutput ] = NULL;
				pRendererContext._graphicsResourceAccess[ eDepthInput ] = GvGraphicsResource::eUndefinedMappedAddressType;
				pRendererContext._graphicsResourceAccess[ eDepthOutput ] = GvGraphicsResource::eUndefinedMappedAddressType;
				break;
		}
	}

	return false;
}

///******************************************************************************
// * Reset
// ******************************************************************************/
//void GvGraphicsInteroperabiltyHandler::reset()
//{
//}

///******************************************************************************
// * ...
// ******************************************************************************/
//bool GvGraphicsInteroperabiltyHandler::bindTo( const struct surfaceReference* surfref )
//{
//	for ( int i = 0; i < _graphicsResources.size(); i++ )
//	{
//		std::pair< GraphicsResourceSlot, GvGraphicsResource* >& graphicsResourceInfo = _graphicsResources[ i ];
//
//		GraphicsResourceSlot graphicsResourceSlot  = graphicsResourceInfo.first;
//		GvGraphicsResource* graphicsResource  = graphicsResourceInfo.second;
//
//		// [ 2 ] - Bind array to texture or surface if needed
//		if ( graphicsResource->getMemoryType() == GvGraphicsResource::eCudaArray )
//		{
//			//if ( graphicsResource->getMappedAddressType() == GvGraphicsResource::eTexture )
//			//{
//				struct cudaArray* imageArray = static_cast< struct cudaArray* >( graphicsResource->getMappedAddress() );
//
//				cudaError_t error;
//				switch ( graphicsResourceSlot )
//				{
//					case eColorReadSlot:
//						//error = cudaBindTextureToArray( GvRendering::_inputColorTexture, imageArray );
//
//						// ----------------------------------------
//						graphicsResource->_mappedAddressType = GvGraphicsResource::eTexture;
//						// ----------------------------------------
//
//						break;
//
//					case eColorWriteSlot:
//					case eColorReadWriteSlot:
//
//						//assert( false );
//
//						error = cudaBindSurfaceToArray( surfref, imageArray );
//
//						// ----------------------------------------
//						graphicsResource->_mappedAddressType = GvGraphicsResource::eSurface;
//						// ----------------------------------------
//
//						break;
//
//					case eDepthReadSlot:
//					//	error = cudaBindTextureToArray( GvRendering::_inputDepthTexture, imageArray );
//
//						// ----------------------------------------
//						graphicsResource->_mappedAddressType = GvGraphicsResource::eTexture;
//						// ----------------------------------------
//
//						break;
//
//					case eDepthWriteSlot:
//					case eDepthReadWriteSlot:
//
//						//assert( false );
//
//				//		error = cudaBindSurfaceToArray( GvRendering::_depthSurface, imageArray );
//
//						// ----------------------------------------
//						graphicsResource->_mappedAddressType = GvGraphicsResource::eSurface;
//						// ----------------------------------------
//
//						break;
//
//					default:
//						assert( false );
//						break;
//				}
//		
//	}
//
//	return false;
//}
