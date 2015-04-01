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

#ifndef _GV_MATERIAL_H_
#define _GV_MATERIAL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"
#include "GvUtils/GvMaterialKernel.h"

// CUDA
#include <vector_types.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvUtils
{

/** 
 * @class GvMaterial
 *
 * @brief The GvMaterial class provides interface to materials.
 *
 * ...
 */
class GvMaterial
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Enumeration of different types of materials
	 */
	enum EMaterialType
	{
		eNbMaterialTypes
	};

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvMaterial();

	/**
	 * Destructor
	 */
	virtual ~GvMaterial();

	/**
	 * Get the emissive color
	 *
	 * @return the emissive color
	 */
	const float3& getKe() const;

	/**
	 * Set the emissive color
	 *
	 * @param pValue the emissive color
	 */
	void setKe( const float3& pValue );

	/**
	 * Get the ambient color
	 *
	 * @return the ambient color
	 */
	const float3& getKa() const;

	/**
	 * Set the ambient color
	 *
	 * @param pValue the ambient color
	 */
	void setKa( const float3& pValue );

	/**
	 * Get the diffuse color
	 *
	 * @return the diffuse color
	 */
	const float3& getKd() const;

	/**
	 * Set the diffuse color
	 *
	 * @param pValue the diffuse color
	 */
	void setKd( const float3& pValue );

	/**
	 * Get the specular color
	 *
	 * @return the specular color
	 */
	const float3& getKs() const;

	/**
	 * Set the specular color
	 *
	 * @param pValue the specular color
	 */
	void setKs( const float3& pValue );

	/**
	 * Get the shininess
	 *
	 * @return the shininess
	 */
	float getShininess() const;

	/**
	 * Set the shininess
	 *
	 * @param pValue the shininess
	 */
	void setShininess( float pValue );

	/**
	 * Get the associated device-side object
	 *
	 * @return the associated device-side object
	 */
	const GvMaterialKernel& getKernelObject() const;

	/**
	 * Get the associated device-side object
	 *
	 * @return the associated device-side object
	 */
	GvMaterialKernel& editKernelObject();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Material's emissive color
	 */
	float3 _Ke;

	/**
	 * Material's ambient color
	 */
	float3 _Ka;

	/**
	 * Material's diffuse color
	 */
	float3 _Kd;

	/**
	 * Material's specular color
	 */
	float3 _Ks;

	/**
	 * Shininess, associated to specular term, tells how shiny the surface is
	 */
	float _shininess;

	/**
	 * Associated device-side object
	 */
	GvMaterialKernel _kernelObject;
	
	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvMaterial( const GvMaterial& );

	/**
	 * Copy operator forbidden.
	 */
	GvMaterial& operator=( const GvMaterial& );

};

} // namespace GvUtils

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvMaterial.inl"

#endif
