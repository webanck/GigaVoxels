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

namespace GvPerfMon
{

/******************************************************************************
 * ...
 *
 * @param frameRes ...
 ******************************************************************************/
inline void CUDAPerfMon::frameResized( uint2 frameRes )
{
	if ( d_timersArray )
	{
		delete d_timersArray;
		d_timersArray = 0;
	}

	if ( d_timersMask )
	{
		GV_CUDA_SAFE_CALL( cudaFree( d_timersMask ) );
		d_timersMask = 0;
	}

	if ( overlayTex )
	{
		glDeleteTextures( 1, &overlayTex );
		overlayTex = 0;
	}

	if ( cacheStateTex )
	{
		glDeleteTextures( 1, &cacheStateTex );
		cacheStateTex = 0;
	}

	d_timersArray = new GvCore::Array3DGPULinear< GvCore::uint64 >( make_uint3( frameRes.x, frameRes.y, CUDAPERFMON_KERNEL_TIMER_MAX ) );

	// TEST --------------------------------- deplacer ça dans le renderer
	//GvCore::Array3DKernelLinear< GvCore::uint64 > h_timersArray = d_timersArray->getDeviceArray();
	//GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_timersArray, &h_timersArray, sizeof( h_timersArray ), 0, cudaMemcpyHostToDevice ) );

	GV_CUDA_SAFE_CALL( cudaMalloc( &d_timersMask, frameRes.x * frameRes.y ) );

	// TEST --------------------------------- deplacer ça dans le renderer
	//GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_timersMask, &d_timersMask, sizeof( d_timersMask ), 0, cudaMemcpyHostToDevice ) );

	// TEST
	_requestResize = true;

	// TEST -----------------------------------------------------------------------------------------------
	//	std::cout << "PERFORMANCE COUNTERS" << std::endl;
	//	std::cout << "k_timersArray = " << &k_timersArray << std::endl;
	//	std::cout << "k_timersMask = " << &k_timersMask << std::endl;
	//-----------------------------------------------------------------------------------------------------

	glGenTextures( 1, &overlayTex );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, overlayTex );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );

	glGenTextures( 1, &cacheStateTex );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, cacheStateTex );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
}

/******************************************************************************
 * ...
 *
 * @return ...
 ******************************************************************************/
inline GvCore::Array3DGPULinear< GvCore::uint64 >* CUDAPerfMon::getKernelTimerArray()
{
	return d_timersArray;
}

/******************************************************************************
 * ...
 *
 * @return ...
 ******************************************************************************/
inline uchar* CUDAPerfMon::getKernelTimerMask()
{
	return d_timersMask;
}

/******************************************************************************
 * ...
 *
 * @return ...
 ******************************************************************************/
inline GvCore::Array3DGPULinear< uchar4 >* CUDAPerfMon::getCacheStateArray() const
{
	return d_cacheStateArray;
}

/******************************************************************************
 * ...
 *
 * @param cacheStateArray ...
 ******************************************************************************/
inline void CUDAPerfMon::setCacheStateArray( GvCore::Array3DGPULinear< uchar4 >* cacheStateArray )
{
	d_cacheStateArray = cacheStateArray;
}

} // namespace GvPerfMon
