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

#ifndef _GVV_GL_SCENE_INTERFACE_H_
#define _GVV_GL_SCENE_INTERFACE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvCoreConfig.h"
#include "GvvBrowsable.h"

// Assimp
#include <assimp/scene.h>

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

namespace GvViewerCore
{

/** 
 * @class GvvGLSceneInterface
 *
 * @brief The GvvGLSceneInterface class provides...
 *
 * ...
 */
class GVVIEWERCORE_EXPORT GvvGLSceneInterface : public GvvBrowsable
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Type name
	 */
	static const char* cTypeName;

	// Mesh bounds
	//
	// @todo add accessors and move to protected section
	float _minX;
	float _minY;
	float _minZ;
	float _maxX;
	float _maxY;
	float _maxZ;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvvGLSceneInterface();

	/**
	 * Destructor
	 */
	virtual ~GvvGLSceneInterface();

	/**
	 * ...
	 *
	 * @param pScene ...
	 */
	void setScene( const aiScene* pScene );

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
	 * Initialize the scene
	 */
	virtual void initialize();

	/**
	 * Finalize the scene
	 */
	virtual void finalize();

	/**
	 * Draw the scene
	 */
	virtual void draw();
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * The root structure of the imported data. 
	 * 
	 *  Everything that was imported from the given file can be accessed from here.
	 *  Objects of this class are generally maintained and owned by Assimp, not
	 *  by the caller. You shouldn't want to instance it, nor should you ever try to
	 *  delete a given scene on your own.
	 */
	const aiScene* _scene;
	
	/******************************** METHODS *********************************/

	/**
	 * ...
	 */
	void recursive_render( const aiScene* pScene, const aiNode* pNode );

	/**
	 * ...
	 */
	void apply_material( const aiMaterial* mtl );

	/**
	 * ...
	 */
	static void color4_to_float4( const aiColor4D* c, float f[4] );

	/**
	 * ...
	 */
	static void set_float4( float f[4], float a, float b, float c, float d );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} // namespace GvViewerCore

#endif // !GVVPIPELINEINTERFACE_H
