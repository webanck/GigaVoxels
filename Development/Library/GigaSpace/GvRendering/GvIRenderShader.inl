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

/******************************************************************************
 * This method is called just before the cast of a ray. Use it to initialize any data
 *  you may need. You may also want to modify the initial distance along the ray (tTree).
 *
 * @param pRayStartTree the starting position of the ray in octree's space.
 * @param pRayDirTree the direction of the ray in octree's space.
 * @param pTTree the distance along the ray's direction we start from.
 ******************************************************************************/
template< typename TDerived >
__device__
inline void GvIRenderShader< TDerived >::preShade( const float3 pRayStartTree, const float3 pRayDirTree, float& pTTree )
{
	static_cast< TDerived* >( this )->preShadeImpl( pRayStartTree, pRayDirTree, pTTree );
}

/******************************************************************************
 * This method is called after the ray stopped or left the bounding
 * volume. You may want to do some post-treatment of the color.
 ******************************************************************************/
template< typename TDerived >
__device__
inline void GvIRenderShader< TDerived >::postShade()
{
	static_cast< TDerived* >( this )->postShadeImpl();
}

/******************************************************************************
 * This method returns the cone aperture for a given distance.
 *
 * @param pTTree the current distance along the ray's direction.
 *
 * @return the cone aperture
 ******************************************************************************/
template< typename TDerived >
__device__
inline float GvIRenderShader< TDerived >::getConeAperture( const float pTTree ) const
{
	return static_cast< const TDerived* >( this )->getConeApertureImpl( pTTree );
}

/******************************************************************************
 * This method returns the final rgba color that will be written to the color buffer.
 *
 * @return the final rgba color.
 ******************************************************************************/
template< typename TDerived >
__device__
inline float4 GvIRenderShader< TDerived >::getColor() const
{
	return static_cast< const TDerived* >( this )->getColorImpl();
}

/******************************************************************************
 * This method is called before each sampling to check whether or not the ray should stop.
 *
 * @param pRayPosInWorld the current ray's position in world space.
 *
 * @return true if you want to continue the ray. false otherwise.
 ******************************************************************************/
template< typename TDerived >
__device__
inline bool GvIRenderShader< TDerived >::stopCriterion( const float3 pRayPosInWorld ) const
{
	return static_cast< const TDerived* >( this )->stopCriterionImpl( pRayPosInWorld );
}

/******************************************************************************
 * This method is called to know if we should stop at the current octree's level.
 *
 * @param pVoxelSize the voxel's size in the current octree level.
 *
 * @return false if you want to stop at the current octree's level. true otherwise.
 ******************************************************************************/
template< typename TDerived >
__device__
inline bool GvIRenderShader< TDerived >::descentCriterion( const float pVoxelSize ) const
{
	return static_cast< const TDerived* >( this )->descentCriterionImpl( pVoxelSize );
}

/******************************************************************************
 * This method is called for each sample. For example, shading or secondary rays
 * should be done here.
 *
 * @param pBrickSampler brick sampler
 * @param pSamplePosScene position of the sample in the scene
 * @param pRayDir ray direction
 * @param pRayStep ray step
 * @param pConeAperture cone aperture
 ******************************************************************************/
template< typename TDerived >
template< typename TSamplerType, class TGPUCacheType  >
__device__
inline void GvIRenderShader< TDerived >::run(const TSamplerType& pBrickSampler, TGPUCacheType& pGpuCache, const float3 pSamplePosScene, const float3 pRayDir, float& pRayStep, const float pConeAperture) {
	static_cast< TDerived* >( this )->runImpl( pBrickSampler, pGpuCache, pSamplePosScene, pRayDir, pRayStep, pConeAperture );
}

} // namespace GvRendering
