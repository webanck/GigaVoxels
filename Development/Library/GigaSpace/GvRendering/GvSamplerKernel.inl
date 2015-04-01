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

namespace GvRendering
{

/******************************************************************************
 * Sample data at given cone aperture
 *
 * @param coneAperture the cone aperture
 *
 * @return the sampled value
 ******************************************************************************/
template< typename VolumeTreeKernelType >
template< int channel >
__device__
__forceinline__ float4 GvSamplerKernel< VolumeTreeKernelType >::getValue( const float coneAperture ) const
{
	return _volumeTree->template getSampleValue< channel >( _brickChildPosInPool, _brickParentPosInPool, _scaleTree2BrickPool * _sampleOffsetInNodeTree,
														   coneAperture,
														   _mipMapOn, _mipMapInterpCoef );
}

/******************************************************************************
 * Sample data at given cone aperture and offset in tree
 *
 * @param coneAperture the cone aperture
 * @param offsetTree the offset in the tree
 *
 * @return the sampled value
 ******************************************************************************/
template< typename VolumeTreeKernelType >
template< int channel >
__device__
__forceinline__ float4 GvSamplerKernel< VolumeTreeKernelType >::getValue( const float coneAperture, const float3 offsetTree ) const
{
	return _volumeTree->template getSampleValue< channel >( _brickChildPosInPool, _brickParentPosInPool, _scaleTree2BrickPool * ( _sampleOffsetInNodeTree + offsetTree ),
														   coneAperture,
														   _mipMapOn, _mipMapInterpCoef );
}

/******************************************************************************
 * Move sample offset in node tree
 *
 * @param offsetTree offset in tree
 ******************************************************************************/
template< typename VolumeTreeKernelType >
__device__
__forceinline__ void GvSamplerKernel< VolumeTreeKernelType >::moveSampleOffsetInNodeTree( const float3 offsetTree )
{
	_sampleOffsetInNodeTree = _sampleOffsetInNodeTree + offsetTree;
}

/******************************************************************************
 * Update MipMap parameters given cone aperture
 *
 * @param coneAperture the cone aperture
 *
 * @return It returns false if coneAperture > voxelSize in parent brick
 ******************************************************************************/
template< typename VolumeTreeKernelType >
__device__
__forceinline__ bool GvSamplerKernel< VolumeTreeKernelType >::updateMipMapParameters( const float pConeAperture )
{
	_mipMapInterpCoef = 0.0f;

	if ( _mipMapOn )
	{
		_mipMapInterpCoef = getMipMapInterpCoef< VolumeTreeKernelType::NodeResolution, VolumeTreeKernelType::BrickResolution >( pConeAperture, _nodeSizeTree );
		if ( _mipMapInterpCoef > 1.0f )
		{
			return false;
		}
	}

	return true;
}

} // namespace GvRendering
