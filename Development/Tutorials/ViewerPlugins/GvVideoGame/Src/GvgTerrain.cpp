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

#include "GvgTerrain.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

#include <cstddef>
#include <cassert>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cmath>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Create method
 *
 * @return an instance of terrain
 ******************************************************************************/
GvgTerrain* GvgTerrain::create()
{
	return new GvgTerrain();
}

/******************************************************************************
 * Destroy method
 ******************************************************************************/
void GvgTerrain::destroy()
{
	delete this;
}

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvgTerrain::GvgTerrain()
:	GvgObject()
,	_heightmapHeight( 0 )
,	_heightmapWidth( 0 )
,	_heightmapHeights( NULL )
,	_terrainBuffer( 0 )
,	_terrainIndexBuffer( 0 )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvgTerrain::~GvgTerrain()
{
}

// CalculateNormal()
// desc: given 3 points, calculates the normal to the points
void CalculateNormal( float *p1, float *p2, float *p3, float* normal )
{
	float a[3], b[3], result[3];
	float length;

	a[0] = p2[0] - p1[0];
	a[1] = p2[1] - p1[1];
	a[2] = p2[2] - p1[2];

	b[0] = p3[0] - p1[0];;
	b[1] = p3[1] - p1[0];;
	b[2] = p3[2] - p1[0];;

	result[0] = a[1] * b[2] - b[1] * a[2];
	result[1] = b[0] * a[2] - a[0] * b[2];
	result[2] = a[0] * b[1] - b[0] * a[1];

	// calculate the length of the normal
	length = (float)sqrt(result[0]*result[0] + result[1]*result[1] + result[2]*result[2]);

	// normalize and specify the normal
	//glNormal3f(result[0]/length, result[1]/length, result[2]/length);
	normal[ 0 ] = result[ 0 ] / length;
	normal[ 1 ] = result[ 1 ] / length;
	normal[ 2 ] = result[ 2 ] / length;
}

#define BITMAP_ID 0x4D42		// the universal bitmap ID

// LoadBitmapFile
// desc: Returns a pointer to the bitmap image of the bitmap specified
//       by filename. Also returns the bitmap header information.
//		 No support for 8-bit bitmaps.
unsigned char* GvgTerrain::loadBitmapFile(char *filename, BITMAPINFOHEADER *bitmapInfoHeader)
{
	FILE *filePtr;							// the file pointer
	BITMAPFILEHEADER	bitmapFileHeader;		// bitmap file header
	unsigned char		*bitmapImage;			// bitmap image data
	unsigned int		imageIdx = 0;		// image index counter
	unsigned char		tempRGB;				// swap variable

	// open filename in "read binary" mode
	filePtr = fopen(filename, "rb");
	if (filePtr == NULL)
		return NULL;

	// read the bitmap file header
	fread(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, filePtr);
	
	// verify that this is a bitmap by checking for the universal bitmap id
	if (bitmapFileHeader.bfType != BITMAP_ID)
	{
		fclose(filePtr);
		return NULL;
	}

	// read the bitmap information header
	fread(bitmapInfoHeader, sizeof(BITMAPINFOHEADER), 1, filePtr);

	// move file pointer to beginning of bitmap data
	fseek(filePtr, bitmapFileHeader.bfOffBits, SEEK_SET);

	// allocate enough memory for the bitmap image data
	//bitmapImage = (unsigned char*)malloc( bitmapInfoHeader->biSizeImage );
	long bitsize;        /* Size of bitmap */
	bitsize = ( bitmapInfoHeader->biWidth * bitmapInfoHeader->biBitCount + 7 ) / 8 * abs( bitmapInfoHeader->biHeight );
	bitmapImage = (unsigned char*)malloc( bitsize );
	
	// verify memory allocation
	if (!bitmapImage)
	{
		free(bitmapImage);
		fclose(filePtr);
		return NULL;
	}

	// read in the bitmap image data
	//fread(bitmapImage, 1, bitmapInfoHeader->biSizeImage, filePtr);
	fread(bitmapImage, 1, bitsize, filePtr);

	// make sure bitmap image data was read
	if (bitmapImage == NULL)
	{
		fclose(filePtr);
		return NULL;
	}

	// swap the R and B values to get RGB since the bitmap color format is in BGR
	//for (imageIdx = 0; imageIdx < bitmapInfoHeader->biSizeImage; imageIdx+=3)
	for (imageIdx = 0; imageIdx < bitsize; imageIdx+=3)
	{
		tempRGB = bitmapImage[imageIdx];
		bitmapImage[imageIdx] = bitmapImage[imageIdx + 2];
		bitmapImage[imageIdx + 2] = tempRGB;
	}

	// close the file and return the bitmap image data
	fclose(filePtr);
	return bitmapImage;
}

/******************************************************************************
 * Initialize
 *
 * @return a flag to tell wheter or not it succeeds
 ******************************************************************************/
bool GvgTerrain::initialize()
{
	//-----------------------------------
	//-----------------------------------
	//glEnableClientState( GL_VERTEX_ARRAY );
	//-----------------------------------
	//-----------------------------------

	//-----------------------------------
	// TO DO
	// initiailize data
	// ...
	_heightmapWidth = 32;
	_heightmapHeight = 32;
	//-----------------------------------

	//// Allocate heightmap
	//_heightmapHeights = new float*[ _heightmapWidth ];
	//for ( unsigned int i = 0; i < _heightmapWidth; i++ )
	//{
	//	_heightmapHeights[ i ] = new float[ _heightmapHeight ];
	//}

	// Initialize heightmap heights
	// TO DO : initialize data bmp loading, random, etc...
	/* initialize random seed: */
	BITMAPINFOHEADER texInfo;		// BMP header
	/*int width;
	int height;*/
	unsigned char* data = NULL;//loadBitmapFile( "terrainHeightmap.bmp", &texInfo);
	if ( data == NULL )
	{
		//free(thisTexture);
		//return NULL;
	}
	//_heightmapWidth = texInfo.biWidth;
	//_heightmapHeight = texInfo.biHeight;
	// Allocate heightmap
	_heightmapHeights = new float*[ _heightmapWidth ];
	for ( unsigned int i = 0; i < _heightmapWidth; i++ )
	{
		_heightmapHeights[ i ] = new float[ _heightmapHeight ];
	}
	srand( time( NULL ) );
	float ground = -0.1f;
	for ( unsigned int z = 0; z < _heightmapHeight; z++ )
	{
		for ( unsigned int x = 0; x < _heightmapWidth; x++ )
		{
			float temp = ( ( static_cast< float >( rand() ) / static_cast< float >( RAND_MAX ) ) * 0.5f - 0.5f ) * 1.f;
			//float temp = ( sinf( 2.f * 3.14159265f * static_cast< float >( rand() ) / static_cast< float >( RAND_MAX ) ) ) * 0.1f;
			//float temp = ( static_cast< float >( data[ x + z * _heightmapWidth ] ) / static_cast< float >( 255 ) );
			_heightmapHeights[ z ][ x ] = ground + temp;
			
			//_heightmapHeights[ z ][ x ] = ( static_cast< float >( data[ x + z * _heightmapWidth ] ) / static_cast< float >( 255 ) );
		}
	}

	// [ Create the vertex array ]

	int normalOffset = _heightmapHeight * _heightmapWidth * 3;

	_terrainBuffer;
	glGenBuffers( 1, &_terrainBuffer );

	glBindBuffer( GL_ARRAY_BUFFER, _terrainBuffer );

	GLsizeiptr terrainBufferSize = sizeof( GLfloat ) * _heightmapHeight * _heightmapWidth * 6;
	glBufferData( GL_ARRAY_BUFFER, terrainBufferSize, NULL, GL_STATIC_DRAW );
	
	GLfloat* terrainBufferData = static_cast< GLfloat* >( glMapBuffer( GL_ARRAY_BUFFER, GL_WRITE_ONLY ) );
	int index = 0;
	//float mapScale = 5.f;
	float mapScale = 10.f;
	for ( unsigned int z = 0; z < _heightmapHeight; z++ )
	{
		for ( unsigned int x = 0; x < _heightmapWidth; x++ )
		{
			// Vertex
			terrainBufferData[ index++ ] = static_cast< GLfloat >( x ) / _heightmapWidth * mapScale;
			terrainBufferData[ index++ ] = _heightmapHeights[ z ][ x ];
			terrainBufferData[ index++ ] = static_cast< GLfloat >( z ) / _heightmapHeight * mapScale;

			//// Normal
			//CalculateNormal();
			//terrainBufferData[ index++ ] = static_cast< GLfloat >( z ) / _heightmapHeight * mapScale;
		}
	}
	glUnmapBuffer( GL_ARRAY_BUFFER );

	//----------------------------------------------------------------------------------------------------
	// NORMALS
	index = 0;
	terrainBufferData = static_cast< GLfloat* >( glMapBuffer( GL_ARRAY_BUFFER, GL_WRITE_ONLY ) );
	for ( unsigned int z = 0; z < _heightmapHeight; z++ )
	{
		for ( unsigned int x = 0; x < _heightmapWidth; x++ )
		{
			float a[ 3 ] = { 0.f, 0.f, 0.f };
			float b[ 3 ] = { 0.f, 0.f, 0.f };
			float c[ 3 ] = { 0.f, 0.f, 0.f };
			float normal[ 3 ] = { 0.f, 0.f, 0.f };
			float tmp[ 3 ] = { 0.f, 0.f, 0.f };

			a[ 0 ] = static_cast< GLfloat >( x ) / _heightmapWidth * mapScale;
			a[ 0 ] = _heightmapHeights[ x ][ z ];
			a[ 0 ] = static_cast< GLfloat >( z ) / _heightmapHeight * mapScale;

			if ( x > 0 )
			{
				if ( z > 0 )
				{
					b[ 0 ] = static_cast< GLfloat >( x ) / _heightmapWidth * mapScale;
					b[ 0 ] = _heightmapHeights[ x ][ z - 1 ];
					b[ 0 ] = static_cast< GLfloat >( z - 1 ) / _heightmapHeight * mapScale;
					c[ 0 ] = static_cast< GLfloat >( x - 1 ) / _heightmapWidth * mapScale;
					c[ 0 ] = _heightmapHeights[ x - 1 ][ z ];
					c[ 0 ] = static_cast< GLfloat >( z ) / _heightmapHeight * mapScale;
					CalculateNormal( a, b, c, normal );
					tmp[ 0 ] += normal[ 0 ];
					tmp[ 1 ] += normal[ 1 ];
					tmp[ 2 ] += normal[ 2 ];
				}

				if ( z < _heightmapHeight - 1 )
				{
					b[ 0 ] = static_cast< GLfloat >( x - 1 ) / _heightmapWidth * mapScale;
					b[ 0 ] = _heightmapHeights[ x - 1 ][ z ];
					b[ 0 ] = static_cast< GLfloat >( z ) / _heightmapHeight * mapScale;
					c[ 0 ] = static_cast< GLfloat >( x - 1 ) / _heightmapWidth * mapScale;
					c[ 0 ] = _heightmapHeights[ x - 1 ][ z + 1 ];
					c[ 0 ] = static_cast< GLfloat >( z + 1 ) / _heightmapHeight * mapScale;
					CalculateNormal( a, b, c, normal );
					tmp[ 0 ] += normal[ 0 ];
					tmp[ 1 ] += normal[ 1 ];
					tmp[ 2 ] += normal[ 2 ];

					b[ 0 ] = static_cast< GLfloat >( x - 1 ) / _heightmapWidth * mapScale;
					b[ 0 ] = _heightmapHeights[ x - 1 ][ z + 1 ];
					b[ 0 ] = static_cast< GLfloat >( z + 1 ) / _heightmapHeight * mapScale;
					c[ 0 ] = static_cast< GLfloat >( x ) / _heightmapWidth * mapScale;
					c[ 0 ] = _heightmapHeights[ x ][ z + 1 ];
					c[ 0 ] = static_cast< GLfloat >( z + 1 ) / _heightmapHeight * mapScale;
					CalculateNormal( a, b, c, normal );
					tmp[ 0 ] += normal[ 0 ];
					tmp[ 1 ] += normal[ 1 ];
					tmp[ 2 ] += normal[ 2 ];
				}
			}

			if ( x < _heightmapWidth - 1 )
			{
				if ( z < _heightmapHeight - 1 )
				{
					b[ 0 ] = static_cast< GLfloat >( x ) / _heightmapWidth * mapScale;
					b[ 0 ] = _heightmapHeights[ x ][ z + 1 ];
					b[ 0 ] = static_cast< GLfloat >( z + 1 ) / _heightmapHeight * mapScale;
					c[ 0 ] = static_cast< GLfloat >( x + 1 ) / _heightmapWidth * mapScale;
					c[ 0 ] = _heightmapHeights[ x + 1 ][ z ];
					c[ 0 ] = static_cast< GLfloat >( z ) / _heightmapHeight * mapScale;
					CalculateNormal( a, b, c, normal );
					tmp[ 0 ] += normal[ 0 ];
					tmp[ 1 ] += normal[ 1 ];
					tmp[ 2 ] += normal[ 2 ];
				}

				if ( z > 0 )
				{
					b[ 0 ] = static_cast< GLfloat >( x + 1 ) / _heightmapWidth * mapScale;
					b[ 0 ] = _heightmapHeights[ x + 1 ][ z ];
					b[ 0 ] = static_cast< GLfloat >( z ) / _heightmapHeight * mapScale;
					c[ 0 ] = static_cast< GLfloat >( x + 1 ) / _heightmapWidth * mapScale;
					c[ 0 ] = _heightmapHeights[ x + 1 ][ z - 1 ];
					c[ 0 ] = static_cast< GLfloat >( z - 1 ) / _heightmapHeight * mapScale;
					CalculateNormal( a, b, c, normal );
					tmp[ 0 ] += normal[ 0 ];
					tmp[ 1 ] += normal[ 1 ];
					tmp[ 2 ] += normal[ 2 ];

					b[ 0 ] = static_cast< GLfloat >( x + 1 ) / _heightmapWidth * mapScale;
					b[ 0 ] = _heightmapHeights[ x + 1 ][ z - 1 ];
					b[ 0 ] = static_cast< GLfloat >( z - 1 ) / _heightmapHeight * mapScale;
					c[ 0 ] = static_cast< GLfloat >( x ) / _heightmapWidth * mapScale;
					c[ 0 ] = _heightmapHeights[ x ][ z - 1 ];
					c[ 0 ] = static_cast< GLfloat >( z - 1 ) / _heightmapHeight * mapScale;
					CalculateNormal( a, b, c, normal );
					tmp[ 0 ] += normal[ 0 ];
					tmp[ 1 ] += normal[ 1 ];
					tmp[ 2 ] += normal[ 2 ];
				}
			}

			// calculate the length of the normal
			float length = (float)sqrt(tmp[0]*tmp[0] + tmp[1]*tmp[1] + tmp[2]*tmp[2]);

			// normalize and specify the normal
			//glNormal3f(result[0]/length, result[1]/length, result[2]/length);
			tmp[ 0 ] = tmp[ 0 ] / length;
			tmp[ 1 ] = tmp[ 1 ] / length;
			tmp[ 2 ] = tmp[ 2 ] / length;

			// Vertex
			terrainBufferData[ normalOffset ] = tmp[ 0 ];
			terrainBufferData[ normalOffset + 1 ] = tmp[ 1 ];
			terrainBufferData[ normalOffset + 2 ] = tmp[ 2 ];

			normalOffset += 3;
		}
	}
	glUnmapBuffer( GL_ARRAY_BUFFER );
	//----------------------------------------------------------------------------------------------------

	//----------------------------------------------------------------------------------------------------
	// Recalculate the offset in the array for the vertex, normal, and colors.
	normalOffset = _heightmapHeight * _heightmapWidth * 3;
	//----------------------------------------------------------------------------------------------------

	// [ Create the index array ]

	_terrainIndexBuffer;
	glGenBuffers( 1, &_terrainIndexBuffer );

	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _terrainIndexBuffer );

	GLsizeiptr terrainIndexBufferSize = sizeof( GLuint ) * ( ( ( _heightmapHeight - 1 ) * ( _heightmapWidth * 2 + 2 ) ) - 2 );
	glBufferData( GL_ELEMENT_ARRAY_BUFFER, terrainIndexBufferSize, NULL, GL_STATIC_DRAW );

	GLuint* terrainIndexBufferData = static_cast< GLuint* >( glMapBuffer( GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY ) );
	/*int */index = 0;
	for ( unsigned int z = 0; z < _heightmapHeight - 1; z++ )
	{
		for ( unsigned int x = 0; x < _heightmapWidth; x++ )
		{
			if ( x == 0 && z != 0 )
			{
				terrainIndexBufferData[ index++ ] = x + z * _heightmapWidth;
			}

			terrainIndexBufferData[ index++ ] = x + z * _heightmapWidth;
			terrainIndexBufferData[ index++ ] = x + ( z + 1 ) * _heightmapWidth;

			if ( x == ( _heightmapWidth - 1 ) && z != ( _heightmapHeight - 2 ) )
			{
				terrainIndexBufferData[ index++ ] = x + ( z + 1 ) * _heightmapWidth;
			}
		}
	}
	glUnmapBuffer( GL_ELEMENT_ARRAY_BUFFER );

	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	return true;
}

/******************************************************************************
 * Finalize
 *
 * @return a flag to tell wheter or not it succeeds
 ******************************************************************************/
bool GvgTerrain::finalize()
{
	// Free heightmap memory
	for ( unsigned int i = 0; i < _heightmapWidth; i++ )
	{
		delete[] _heightmapHeights[ i ];
	}
	delete _heightmapHeights;
	_heightmapHeights = NULL;

	glDeleteBuffers( 1, &_terrainBuffer );
	glDeleteBuffers( 1, &_terrainIndexBuffer );

	return true;
}

/******************************************************************************
 * Render the terrain
 ******************************************************************************/
void GvgTerrain::render()
{
	//--------------------------------------------------
	glColor3f( 0.7f, 0.7, 0.7f );
	//--------------------------------------------------

	glBindBuffer( GL_ARRAY_BUFFER, _terrainBuffer );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, _terrainIndexBuffer );

	glEnableClientState( GL_VERTEX_ARRAY );
	glEnableClientState( GL_NORMAL_ARRAY );

	glVertexPointer( 3, GL_FLOAT, 0, 0 );
	int normalOffset = _heightmapHeight * _heightmapWidth * 3;
	glNormalPointer( GL_FLOAT, 0, (GLfloat* )NULL + normalOffset );
		
	GLsizei terrainIndexBufferSize = ( ( _heightmapHeight - 1 ) * ( _heightmapWidth * 2 + 2 ) ) - 2;
	glDrawElements( GL_TRIANGLE_STRIP, terrainIndexBufferSize, GL_UNSIGNED_INT, 0 );

	glDisableClientState( GL_NORMAL_ARRAY );
	glDisableClientState( GL_VERTEX_ARRAY );

	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, 0 );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
}
