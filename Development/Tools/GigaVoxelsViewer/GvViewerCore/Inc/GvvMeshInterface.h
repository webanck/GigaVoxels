/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2014 INRIA <http://www.inria.fr/>
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

#ifndef _GVV_MESH_INTERFACE_H_
#define _GVV_MESH_INTERFACE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvCoreConfig.h"
#include "GvvBrowsable.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GvViewer
namespace GvViewerCore
{
	class GvvProgrammableShaderInterface;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerCore
{

/** 
 * @class GvvDeviceInterface
 *
 * @brief The GvvDeviceInterface class provides info on a device.
 *
 * ...
 */
class GVVIEWERCORE_EXPORT GvvMeshInterface : public GvvBrowsable
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	static const char* cTypeName;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvvMeshInterface();

	/**
	 * Destructor
	 */
	virtual ~GvvMeshInterface();

	/**
	 * Returns the type of this browsable. The type is used for retrieving
	 * the context menu or when requested or assigning an icon to the
	 * corresponding item
	 *
	 * @return the type name of this browsable
	 */
	virtual const char* getTypeName() const;

	/**
     * Gets the name of this browsable
     *
     * @return the name of this browsable
     */
	virtual const char* getName() const;

	/**
	 * Get the flag telling wheter or not it has programmable shaders
	 *
	 * @return the flag telling wheter or not it has programmable shaders
	 */
	virtual bool hasProgrammableShader() const;

	/**
	 * Add a programmable shader
	 */
	virtual void addProgrammableShader( GvvProgrammableShaderInterface* pShader );

	/**
	 * Get the associated programmable shader
	 *
	 * @param pIndex shader index
	 *
	 * @return the associated programmable shader
	 */
	virtual void removeProgrammableShader( GvvProgrammableShaderInterface* pShader );

	/**
	 * Get the associated programmable shader
	 *
	 * @param pIndex shader index
	 *
	 * @return the associated programmable shader
	 */
	virtual const GvvProgrammableShaderInterface* getProgrammableShader( unsigned int pIndex ) const;

	/**
	 * Get the associated programmable shader
	 *
	 * @param pIndex shader index
	 *
	 * @return the associated programmable shader
	 */
	virtual GvvProgrammableShaderInterface* editProgrammableShader( unsigned int pIndex );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvvMeshInterface( const GvvMeshInterface& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvMeshInterface& operator=( const GvvMeshInterface& );

};

} // namespace GvViewerCore

#endif // !_GVV_MESH_INTERFACE_H_
