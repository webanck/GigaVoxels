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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace GvStructure
{

/******************************************************************************
 * Sample data in specified channel at a given position.
 * 3D texture are used with hardware tri-linear interpolation.
 *
 * @param pBrickPos Brick position in the pool of bricks
 * @param pPosInBrick Position in brick
 *
 * @return the sampled value
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize >
template< int TChannel >
__device__
__forceinline__ float4 VolumeTreeKernel< DataTList, NodeTileRes, BrickRes, BorderSize >
::getSampleValueTriLinear( float3 pBrickPos, float3 pPosInBrick ) const
{
	// Type definition of the data channel type
	typedef typename GvCore::DataChannelType< DataTList, TChannel >::Result ChannelType;

	float4 result = make_float4( 0.0f );
	float3 samplePos = pBrickPos + pPosInBrick;

	// Sample data in texture according to its type (float or not)
	if ( ! ( GvCore::IsFloatFormat< ChannelType >::value ) )
	{
		gpuPoolTexFetch( TEXDATAPOOL, TChannel, 3, ChannelType, cudaReadModeNormalizedFloat, samplePos, result );
	}
	else
	{
		gpuPoolTexFetch( TEXDATAPOOL, TChannel, 3, ChannelType, cudaReadModeElementType, samplePos, result );
	}

	return result;
}

/******************************************************************************
 * Sample data in specified channel at a given position.
 * 3D texture are used with hardware tri-linear interpolation.
 *
 * @param mipMapInterpCoef mipmap interpolation coefficient
 * @param brickChildPosInPool brick child position in pool
 * @param brickParentPosInPool brick parent position in pool
 * @param posInBrick position in brick
 * @param coneAperture cone aperture
 *
 * @return the sampled value
 ******************************************************************************/
// QUESTION : le paramètre "coneAperture" ne semble pas utilisé ? A quoi sert-il (servait ou servira) ?
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize >
template< int TChannel >
__device__
__forceinline__ float4 VolumeTreeKernel< DataTList, NodeTileRes, BrickRes, BorderSize >
::getSampleValueQuadriLinear( float mipMapInterpCoef, float3 brickChildPosInPool, float3 brickParentPosInPool, float3 posInBrick, float coneAperture ) const
{
	float3 samplePos0 = posInBrick;
	float3 samplePos1 = posInBrick / NodeTileRes::getFloat3();//* 0.5f;

	float4 vox0, vox1;

	// Sample data in brick
	vox1 = getSampleValueTriLinear< TChannel >( brickParentPosInPool, samplePos1 );

	if ( mipMapInterpCoef <= 1.0f )
	{
		// Sample data in child brick
		vox0 = getSampleValueTriLinear< TChannel >( brickChildPosInPool, samplePos0 );

		// Linear interpolation of results
		vox1 = lerp( vox0, vox1, mipMapInterpCoef );
	}

	return vox1;
}

/******************************************************************************
 * Sample data in specified channel at a given position.
 * 3D texture are used with hardware tri-linear interpolation.
 *
 * @param mipMapInterpCoef mipmap interpolation coefficient
 * @param brickChildPosInPool brick child position in pool
 * @param brickParentPosInPool brick parent position in pool
 * @param posInBrick position in brick
 * @param coneAperture cone aperture
 *
 * @return the sampled value
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize >
template< int TChannel >
__device__
__forceinline__ float4 VolumeTreeKernel< DataTList, NodeTileRes, BrickRes, BorderSize >
::getSampleValue( float3 brickChildPosInPool, float3 brickParentPosInPool, float3 sampleOffsetInBrick, float coneAperture, bool mipMapOn, float mipMapInterpCoef ) const
{
	float4 vox;

	// Sample data in texture
	if ( mipMapOn && mipMapInterpCoef > 0.0f )
	{
		vox = getSampleValueQuadriLinear< TChannel >( mipMapInterpCoef,
			brickChildPosInPool, brickParentPosInPool, sampleOffsetInBrick,
			coneAperture );
	}
	else
	{
		vox = getSampleValueTriLinear< TChannel >( brickChildPosInPool, sampleOffsetInBrick );
	}

	return vox;
}

/******************************************************************************
 * ...
 *
 * @param nodeTileAddress ...
 * @param nodeOffset ...
 *
 * @return ...
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize >
__device__
__forceinline__ uint VolumeTreeKernel< DataTList, NodeTileRes, BrickRes, BorderSize >
::computenodeAddress( uint3 nodeTileAddress, uint3 nodeOffset ) const
{
	return ( nodeTileAddress.x + NodeResolution::toFloat1( nodeOffset ) );
}

/******************************************************************************
 * ...
 *
 * @param nodeTileAddress ...
 * @param nodeOffset ...
 *
 * @return ...
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize >
__device__
__forceinline__ uint3 VolumeTreeKernel< DataTList, NodeTileRes, BrickRes, BorderSize >
::computeNodeAddress( uint3 nodeTileAddress, uint3 nodeOffset ) const
{
	return make_uint3( nodeTileAddress.x + NodeResolution::toFloat1( nodeOffset ), 0, 0 );
}

/******************************************************************************
 * Retrieve node information (address + flags) from data structure
 *
 * @param resnode ...
 * @param nodeTileAddress ...
 * @param nodeOffset ...
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize >
__device__
__forceinline__ void VolumeTreeKernel< DataTList, NodeTileRes, BrickRes, BorderSize >
::fetchNode( GvNode& resnode, uint3 nodeTileAddress, uint3 nodeOffset ) const
{
	uint nodeAddress = computenodeAddress(nodeTileAddress, nodeOffset);

#if USE_LINEAR_VOLTREE_TEX
	resnode.childAddress = tex1Dfetch( volumeTreeChildTexLinear, nodeAddress );
	resnode.brickAddress = tex1Dfetch( volumeTreeDataTexLinear, nodeAddress );
#else // USE_LINEAR_VOLTREE_TEX
	resnode.childAddress = k_volumeTreeChildArray.get( nodeAddress );
	resnode.brickAddress = k_volumeTreeDataArray.get( nodeAddress );
#endif // USE_LINEAR_VOLTREE_TEX
		
#ifdef GV_USE_BRICK_MINMAX
	resnode.metaDataAddress = nodeAddress;
#endif
}

/******************************************************************************
 * Retrieve node information (address + flags) from data structure
 *
 * @param resnode ...
 * @param nodeTileAddress ...
 * @param nodeOffset ...
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize >
__device__
__forceinline__ void VolumeTreeKernel< DataTList, NodeTileRes, BrickRes, BorderSize >
::fetchNode( GvNode& resnode, uint nodeTileAddress, uint nodeOffset ) const
{
	uint nodeAddress = nodeTileAddress + nodeOffset;

#if USE_LINEAR_VOLTREE_TEX
	resnode.childAddress = tex1Dfetch(volumeTreeChildTexLinear, nodeAddress);
	resnode.brickAddress = tex1Dfetch(volumeTreeDataTexLinear, nodeAddress);
#else //USE_LINEAR_VOLTREE_TEX
	resnode.childAddress = k_volumeTreeChildArray.get( nodeAddress );
	resnode.brickAddress = k_volumeTreeDataArray.get( nodeAddress );
#endif //USE_LINEAR_VOLTREE_TEX
		
#ifdef GV_USE_BRICK_MINMAX
	resnode.metaDataAddress = nodeAddress;
#endif
}

/******************************************************************************
 * Retrieve node information (address + flags) from data structure
 *
 * @param resnode ...
 * @param nodeAddress ...
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize >
__device__
__forceinline__ void VolumeTreeKernel< DataTList, NodeTileRes, BrickRes, BorderSize >
::fetchNode( GvNode& resnode, uint nodeAddress ) const
{
#if USE_LINEAR_VOLTREE_TEX
	resnode.childAddress = tex1Dfetch(volumeTreeChildTexLinear, nodeAddress);
	resnode.brickAddress = tex1Dfetch(volumeTreeDataTexLinear, nodeAddress);
#else //USE_LINEAR_VOLTREE_TEX
	resnode.childAddress = k_volumeTreeChildArray.get(nodeAddress);
	resnode.brickAddress = k_volumeTreeDataArray.get(nodeAddress);

#ifdef GV_USE_BRICK_MINMAX
	resnode.metaDataAddress = nodeAddress;
#endif

#endif //USE_LINEAR_VOLTREE_TEX
}

///******************************************************************************
// * ...
// *
// * @param resnode ...
// * @param nodeTileAddress ...
// * @param nodeOffset ...
// ******************************************************************************/
//template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize >
//__device__
//__forceinline__ void VolumeTreeKernel< DataTList, NodeTileRes, BrickRes, BorderSize >
//::fetchNodeChild( GvNode& resnode, uint nodeTileAddress, uint nodeOffset )
//{
//	uint address = nodeTileAddress + nodeOffset;
//
//#if USE_LINEAR_VOLTREE_TEX
//	resnode.childAddress = tex1Dfetch( volumeTreeChildTexLinear, address );
//#else //USE_LINEAR_VOLTREE_TEX
//	resnode.childAddress = k_volumeTreeChildArray.get( address );
//#endif //USE_LINEAR_VOLTREE_TEX
//}
//
///******************************************************************************
// * ...
// *
// * @param resnode ...
// * @param nodeTileAddress ...
// * @param nodeOffset ...
// ******************************************************************************/
//template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize >
//__device__
//__forceinline__ void VolumeTreeKernel< DataTList, NodeTileRes, BrickRes, BorderSize >
//::fetchNodeData( GvNode& resnode, uint nodeTileAddress, uint nodeOffset )
//{
//	uint address = nodeTileAddress + nodeOffset;
//
//#if USE_LINEAR_VOLTREE_TEX
//	resnode.brickAddress = tex1Dfetch( volumeTreeDataTexLinear, address );
//#else //USE_LINEAR_VOLTREE_TEX
//	resnode.brickAddress = k_volumeTreeDataArray.get( address );
//#endif //USE_LINEAR_VOLTREE_TEX
//}

/******************************************************************************
 * Write node information (address + flags) in data structure
 *
 * @param node ...
 * @param nodeTileAddress ...
 * @param nodeOffset ...
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize >
__device__
__forceinline__ void VolumeTreeKernel< DataTList, NodeTileRes, BrickRes, BorderSize >
::setNode( GvNode node, uint3 nodeTileAddress, uint3 nodeOffset )
{
	const uint3 nodeAddress = computeNodeAddress( nodeTileAddress, nodeOffset );

	k_volumeTreeChildArray.set( nodeAddress, node.childAddress );
	k_volumeTreeDataArray.set( nodeAddress, node.brickAddress );
}

/******************************************************************************
 * Write node information (address + flags) in data structure
 *
 * @param node ...
 * @param nodeTileAddress ...
 * @param nodeOffset ...
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize >
__device__
__forceinline__ void VolumeTreeKernel< DataTList, NodeTileRes, BrickRes, BorderSize >
::setNode( GvNode node, uint nodeTileAddress, uint nodeOffset )
{
	const uint nodeAddress = nodeTileAddress + nodeOffset;

	k_volumeTreeChildArray.set( nodeAddress, node.childAddress );
	k_volumeTreeDataArray.set( nodeAddress, node.brickAddress );
}

/******************************************************************************
 * Write node information (address + flags) in data structure
 *
 * @param node ...
 * @param nodeAddress ...
 ******************************************************************************/
template< class DataTList, class NodeTileRes, class BrickRes, uint BorderSize >
__device__
__forceinline__ void VolumeTreeKernel< DataTList, NodeTileRes, BrickRes, BorderSize >
::setNode( GvNode node, uint nodeAddress )
{
	k_volumeTreeChildArray.set( nodeAddress, node.childAddress );
	k_volumeTreeDataArray.set( nodeAddress, node.brickAddress );
}

} // namespace GvStructure
