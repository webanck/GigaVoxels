///*
// * GigaVoxels is a ray-guided streaming library used for efficient
// * 3D real-time rendering of highly detailed volumetric scenes.
// *
// * Copyright (C) 2011-2012 INRIA <http://www.inria.fr/>
// *
// * Authors : GigaVoxels Team
// *
// * This program is free software: you can redistribute it and/or modify
// * it under the terms of the GNU General Public License as published by
// * the Free Software Foundation, either version 3 of the License, or
// * (at your option) any later version.
// *
// * This program is distributed in the hope that it will be useful,
// * but WITHOUT ANY WARRANTY; without even the implied warranty of
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// * GNU General Public License for more details.
// *
// * You should have received a copy of the GNU General Public License
// * along with this program.  If not, see <http://www.gnu.org/licenses/>.
// */
//
///** 
// * @version 1.0
// */
//
///******************************************************************************
// ******************************* INCLUDE SECTION ******************************
// ******************************************************************************/
//
///******************************************************************************
// ****************************** INLINE DEFINITION *****************************
// ******************************************************************************/
//
//namespace GvRendering
//{
//
///******************************************************************************
// * Get the color at given pixel from input color buffer
// *
// * @param pPixel pixel coordinates
// *
// * @return the pixel color
// ******************************************************************************/
//__device__
//__forceinline__ uchar4 //GvGraphicsInteroperabiltyHandlerKernel
////::getInputColor( const uint2 pPixel )
//getInputColor( const uint2 pPixel )
//{
//	switch ( k_renderViewContext._graphicsResourceAccess[ GvGraphicsInteroperabiltyHandler::eColorInput ] )
//	{
//		case GvGraphicsResource::ePointer:
//			{
//				int offset = pPixel.x + pPixel.y * k_renderViewContext.frameSize.x;
//				return static_cast< uchar4* >( k_renderViewContext._graphicsResources[ GvGraphicsInteroperabiltyHandler::eColorInput ] )[ offset ];
//			}
//			break;
//
//		case GvGraphicsResource::eTexture:
//			return tex2D( GvRendering::_inputColorTexture, k_renderViewContext._inputColorTextureOffset + pPixel.x, pPixel.y );
//			break;
//
//		case GvGraphicsResource::eSurface:
//			// Note : cudaBoundaryModeTrap means that out-of-range accesses cause the kernel execution to fail.
//			// Possible values can be : cudaBoundaryModeTrap, cudaBoundaryModeClamp or cudaBoundaryModeZero.
//			return surf2Dread< uchar4 >( GvRendering::_colorSurface, pPixel.x * sizeof( uchar4 ), pPixel.y, cudaBoundaryModeTrap );
//			break;
//
//		default:
//			break;
//	}
//
//	return k_renderViewContext._clearColor;
//}
//
///******************************************************************************
// * Set the color at given pixel into output color buffer
// *
// * @param pPixel pixel coordinates
// * @param pColor color
// ******************************************************************************/
//__device__
//__forceinline__ void //GvGraphicsInteroperabiltyHandlerKernel
////::setOutputColor( const uint2 pPixel, uchar4 pColor )
//setOutputColor( const uint2 pPixel, uchar4 pColor )
//{
//	switch ( k_renderViewContext._graphicsResourceAccess[ GvGraphicsInteroperabiltyHandler::eColorOutput ] )
//	{
//		case GvGraphicsResource::ePointer:
//			{
//				int offset = pPixel.x + pPixel.y * k_renderViewContext.frameSize.x;
//				static_cast< uchar4* >( k_renderViewContext._graphicsResources[ GvGraphicsInteroperabiltyHandler::eColorOutput ] )[ offset ] = pColor;
//			}
//			break;
//
//		case GvGraphicsResource::eSurface:
//			// Note : cudaBoundaryModeTrap means that out-of-range accesses cause the kernel execution to fail.
//			// Possible values can be : cudaBoundaryModeTrap, cudaBoundaryModeClamp or cudaBoundaryModeZero.
//			surf2Dwrite( pColor, GvRendering::_colorSurface, pPixel.x * sizeof( uchar4 ), pPixel.y, cudaBoundaryModeTrap );
//			break;
//
//		default:
//			break;
//	}
//}
//
///******************************************************************************
// * Get the depth at given pixel from input depth buffer
// *
// * @param pPixel pixel coordinates
// *
// * @return the pixel depth
// ******************************************************************************/
//__device__
//__forceinline__ float //GvGraphicsInteroperabiltyHandlerKernel
////::getInputDepth( const uint2 pPixel )
//getInputDepth( const uint2 pPixel )
//{
//	float tmpfval = 1.0f;
//
//	// Read depth from Z-buffer
//	switch ( k_renderViewContext._graphicsResourceAccess[ GvGraphicsInteroperabiltyHandler::eDepthInput ] )
//	{
//		case GvGraphicsResource::ePointer:
//			{
//				int offset = pPixel.x + pPixel.y * k_renderViewContext.frameSize.x;
//				tmpfval = static_cast< float* >( k_renderViewContext._graphicsResources[ GvGraphicsInteroperabiltyHandler::eDepthInput ] )[ offset ];
//			}
//			break;
//
//		case GvGraphicsResource::eTexture:
//			tmpfval = tex2D( GvRendering::_inputDepthTexture, k_renderViewContext._inputDepthTextureOffset + pPixel.x, pPixel.y );
//			break;
//
//			//case GvGraphicsInteroperabiltyHandler::eSurface:
//			case GvGraphicsResource::eSurface:
//				// Note : cudaBoundaryModeTrap means that out-of-range accesses cause the kernel execution to fail.
//				// Possible values can be : cudaBoundaryModeTrap, cudaBoundaryModeClamp or cudaBoundaryModeZero.
//				surf2Dread< float >( &tmpfval, GvRendering::_depthSurface, pPixel.x * sizeof( float ), pPixel.y, cudaBoundaryModeTrap );
//				break;
//
//			default:
//				tmpfval = k_renderViewContext._clearDepth;
//				break;
//	}
//					
//	// Decode depth from Z-buffer
//	uint tmpival = __float_as_int( tmpfval );
//	tmpival = ( tmpival & 0xFFFFFF00 ) >> 8;
//
//	return __saturatef( static_cast< float >( tmpival ) / 16777215.0f );
//}
//
///******************************************************************************
// * Set the depth at given pixel into output depth buffer
// *
// * @param pPixel pixel coordinates
// * @param pDepth depth
// ******************************************************************************/
//__device__
//__forceinline__ void //GvGraphicsInteroperabiltyHandlerKernel
////::setOutputDepth( const uint2 pPixel, float pDepth )
//setOutputDepth( const uint2 pPixel, float pDepth )
//{
//	// Encode depth to Z-buffer
//	uint tmpival = static_cast< uint >( floorf( pDepth * 16777215.0f ) );
//	tmpival = tmpival << 8;
//	float Zdepth = __int_as_float( tmpival );
//
//	// Write depth to Z-buffer
//	switch ( k_renderViewContext._graphicsResourceAccess[ GvGraphicsInteroperabiltyHandler::eDepthOutput ] )
//	{
//		case GvGraphicsResource::ePointer:
//			{
//				int offset = pPixel.x + pPixel.y * k_renderViewContext.frameSize.x;
//				static_cast< float* >( k_renderViewContext._graphicsResources[ GvGraphicsInteroperabiltyHandler::eDepthOutput ] )[ offset ] = Zdepth;
//			}
//			break;
//
//		case GvGraphicsResource::eSurface:
//			// Note : cudaBoundaryModeTrap means that out-of-range accesses cause the kernel execution to fail.
//			// Possible values can be : cudaBoundaryModeTrap, cudaBoundaryModeClamp or cudaBoundaryModeZero.
//			surf2Dwrite( pDepth, GvRendering::_depthSurface, pPixel.x * sizeof( float ), pPixel.y, cudaBoundaryModeTrap );
//			break;
//
//		default:
//			break;
//	}
//}
//
//} // namespace GvRendering
