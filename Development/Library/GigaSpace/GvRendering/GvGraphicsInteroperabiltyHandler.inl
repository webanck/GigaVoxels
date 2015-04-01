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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvRendering
{
	
///******************************************************************************
// * ...
// ******************************************************************************/
//inline size_t GvGraphicsInteroperabiltyHandler::getInputColorTextureOffset() const
//{
//	return _inputColorTextureOffset;
//}
//
///******************************************************************************
// * ...
// ******************************************************************************/
//inline size_t GvGraphicsInteroperabiltyHandler::getInputDepthTextureOffset() const
//{
//	return _inputDepthTextureOffset;
//}
//
///******************************************************************************
// * ...
// ******************************************************************************/
//inline GvGraphicsInteroperabiltyHandler::GraphicsResourceMappedAdressType GvGraphicsInteroperabiltyHandler::getMappedAdressType( GraphicsResourceDeviceSlot pGraphicsResourceDeviceSlot )
//{
//	const unsigned int graphicsResourceIndex = _graphicResourceMappedAdressIndexes[ pGraphicsResourceDeviceSlot ];
//	if ( graphicsResourceIndex == eUndefinedGraphicsResourceDeviceSlot )
//	{
//		return eUndefinedGraphicsResourceMappedAdressType;
//	}
//
//	return _graphicResourceMappedAdressTypes[ graphicsResourceIndex ];
//}

/******************************************************************************
 * ...
 ******************************************************************************/
inline const std::vector< std::pair< GvGraphicsInteroperabiltyHandler::GraphicsResourceSlot, GvGraphicsResource* > >& GvGraphicsInteroperabiltyHandler::getGraphicsResources() const
{
	return _graphicsResources;
}

/******************************************************************************
 * ...
 ******************************************************************************/
inline std::vector< std::pair< GvGraphicsInteroperabiltyHandler::GraphicsResourceSlot, GvGraphicsResource* > >& GvGraphicsInteroperabiltyHandler::editGraphicsResources()
{
	return _graphicsResources;
}

} // namespace GvRendering
