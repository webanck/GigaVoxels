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

#ifndef _SKY_BOX_H_
#define _SKY_BOX_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// OpenGL
#include <GL/glew.h>

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

/**
 *  Classe permettant de donner un fond réaliste
 **/
class Skybox
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
    Skybox();

	/**
	 * Initialization
	 */
    void init();

	/**
	 * Draw
	 */
    void draw();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/****************************** INNER TYPES *******************************/

	/**
	 * Enumeration of the sky box faces
	 */
	enum SkyId
	{
        SKY_LEFT = 0,
        SKY_BACK,
        SKY_RIGHT,
        SKY_FRONT,
        SKY_TOP,
		SKY_BOTTOM 
    };

	/******************************* ATTRIBUTES *******************************/
	
	/**
	 * Taille de la skybox
	 */
    float size;
	
	/**
	 * Texture id of the sky box faces
	 */
	GLuint skybox[ 6 ];

	/******************************** METHODS *********************************/
	
	/**
	 * Fonction d'affichage de la skybox
	 */
    void drawSkybox();

    /**
	 * Fonction de creation de la Texture
	 *
	 * @param pTexId ...
	 * @param pFilename ...
	 */
    void CreateGLTexture( SkyId pTexId, const char* pFilename );

};

#endif // !_SKY_BOX_H_
