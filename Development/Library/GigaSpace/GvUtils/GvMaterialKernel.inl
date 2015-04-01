///*
// * GigaVoxels is a ray-guided streaming library used for efficient
// * 3D real-time rendering of highly detailed volumetric scenes.
// *
// * Copyright (C) 2011-2013 INRIA <http://www.inria.fr/>
// *
// * Authors : GigaVoxels Team
// *
// * GigaVoxels is distributed under a dual-license scheme.
// * You can obtain a specific license from Inria at gigavoxels-licensing@inria.fr.
// * Otherwise the default license is the GPL version 3.
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
//namespace GvUtils
//{
//
///******************************************************************************
// * Get the emissive color
// *
// * @return the emissive color
// ******************************************************************************/
//inline const float3& GvMaterial::getKe() const
//{
//	return _Ke;
//}
//
///******************************************************************************
// * Set the emissive color
// *
// * @param pValue the emissive color
// ******************************************************************************/
//inline void GvMaterial::setKe( const float3& pValue )
//{
//	_Ke = pValue;
//}
//
///******************************************************************************
// * Get the ambient color
// *
// * @return the ambient color
// ******************************************************************************/
//inline const float3& GvMaterial::getKa() const
//{
//	return _Ka;
//}
//
///******************************************************************************
// * Set the ambient color
// *
// * @param pValue the ambient color
// ******************************************************************************/
//inline void GvMaterial::setKa( const float3& pValue )
//{
//	_Ka = pValue;
//}
//
///******************************************************************************
// * Get the diffuse color
// *
// * @return the diffuse color
// ******************************************************************************/
//inline const float3& GvMaterial::getKd() const
//{
//	return _Kd;
//}
//
///******************************************************************************
// * Set the diffuse color
// *
// * @param pValue the diffuse color
// ******************************************************************************/
//inline void GvMaterial::setKd( const float3& pValue )
//{
//	_Kd = pValue;
//}
//
///******************************************************************************
// * Get the specular color
// *
// * @return the specular color
// ******************************************************************************/
//inline const float3& GvMaterial::getKs() const
//{
//	return _Ks;
//}
//
///******************************************************************************
// * Set the specular color
// *
// * @param pValue the specular color
// ******************************************************************************/
//inline void GvMaterial::setKs( const float3& pValue )
//{
//	_Ks = pValue;
//}
//
///******************************************************************************
// * Get the shininess color
// *
// * @return the shininess color
// ******************************************************************************/
//inline float GvMaterial::getShininess() const
//{
//	return _shininess;
//}
//
///******************************************************************************
// * Set the shininess color
// *
// * @param pValue the shininess color
// ******************************************************************************/
//inline void GvMaterial::setShininess( float pValue )
//{
//	_shininess = pValue;
//}
//
///******************************************************************************
// * Get the associated device-side object
// *
// * @return the associated device-side object
// ******************************************************************************/
//inline const GvMaterialKernel& GvMaterial::getKernelObject() const
//{
//	return _kernelObject;
//}
//
///******************************************************************************
// * Get the associated device-side object
// *
// * @return the associated device-side object
// ******************************************************************************/
//inline GvMaterialKernel& GvMaterial::editKernelObject()
//{
//	return _kernelObject;
//}
//
//} // namespace GvUtils
