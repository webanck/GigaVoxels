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

#ifndef _GVG_MD2_MODEL_H_
#define _GVG_MD2_MODEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include <GvgObject.h>

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
 * File header of MD2 model files.
 *
 * Note : each frame holds a snapshot of the model in a particular position
 */
typedef struct
{
	int _ident;						// identifies the model as MD2 file "IDP2"
	int _version;					// should be equal to 8

	int _textureWidth;				// width of texture(s)
	int _textureHeight;				// height of texture(s)
	int _frameSize;					// number of bytes per frame

	int _nbTextures;				// number of textures
	int _nbPoints;					// number of points ( sum of the number of vertices in each frame)
	int _nbTextureCoordinates;		// number of texture coordinates
	int _nbTriangles;				// number of triangles
	int _nbGLCommands;				// number of OpenGL command types
	int _nbFrames;					// number of frames of animation (i.e. keyframe)

	int _texturesOffset;			// offset to texture names ( bytes each)
	int _textureCoordinatesOffset;	// offset to texture coodinates
	int _trianglesOffset;			// offset to triangle mesh
	int _framesOffset;				// offset to frame data (i.e. points)
	int _GLCommandsOffset;			// offset to type of OpenGL commands to use
	int _endOfFileOffset;			// end of file

} GvgMD2MjodelHeader;

/**
 * Vector
 */
typedef struct
{
	float _point[ 3 ];
} vector_t;

/**
 * Texture coodinate
 */
typedef struct
{
	float _s;
	float _t;
} texCoord_t;

/**
 * Texture coodinate index
 */
typedef struct
{
	short _s;
	short _t;
} stIndex_t;

/**
 * Frame point
 */
typedef struct
{
	unsigned char _v[ 3 ];
	unsigned char _normalIndex;
} framePoint_t;

/**
 * Frame
 */
typedef struct
{
	float _scale[ 3 ];
	float _translate[ 3 ];
	char _name[ 16 ];
	framePoint_t _fp[ 1 ];
} frame_t;

/**
 * Mesh
 */
typedef struct
{
	unsigned short _meshIndex[ 3 ];
	unsigned short _stIndex[ 3 ];
} mesh_t;

/**
 * Types of textures
 */
enum texTypes_t
{
	PCX, BMP, TGA
};

/**
 * Texture
 */
typedef struct
{
	texTypes_t _textureType;

	int _width;
	int _height;

	long int _scaledWidth;
	long int _scaledHeight;

	unsigned int _texID;

	unsigned char* _data;
	unsigned char* _palette;
} texture_t;

/** 
 * @class GvgMD2Model
 *
 * @brief The GvgMD2Model class provides ...
 *
 * @ingroup ...
 *
 * ...
 */
class GvgMD2Model : public GvgObject
{

	/**************************************************************************
	 ***************************** FRIEND SECTION *****************************
	 **************************************************************************/

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
	GvgMD2Model();

	/**
	 * Destructor
	 */
	virtual ~GvgMD2Model();

	/**
	 * Load a model
	 */
	bool load( const char* pFilename );

	/**
	 * Draw a keyframe
	 */
	void draw( int pKeyframe ) const;

	/**
	 * Number of keyframes
	 */
	int _nbKeyframes;

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
		
	/******************************** METHODS *********************************/

	void SetupSkin(texture_t *thisTexture);

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	///**
	// * Number of keyframes
	// */
	//int _nbKeyframes;

	/**
	 * Number of vertices
	 */
	int _nbVertices;					
	
	/**
	 * Number of triangles
	 */
	int _nbTriangles;

	/**
	 * Number of textures
	 */
	int _nbTextures;
	
	/**
	 * Number of texture coordinates
	 */
	int _nbTextureCoordinates;
		
	/**
	 * Number of bytes per keyframe
	 */
	int _frameSize;

	/**
	 * Mesh list
	 */
	vector_t* _vertices;
	mesh_t* _triangles;
	texCoord_t* _textureCoordinates;
	texture_t* _texture;

	/******************************** METHODS *********************************/

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#endif // !_GVG_MD2_MODEL_H_
