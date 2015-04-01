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

#include "GvgMD2Model.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

#include <cstdio>

// Cuda
#include <vector_types.h>

// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <iostream>
#include <cassert>

#include <windows.h>

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
 * Constructor
 ******************************************************************************/
GvgMD2Model::GvgMD2Model()
:	GvgObject()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvgMD2Model::~GvgMD2Model()
{
}

// only partial pcx file header
typedef struct
{
	unsigned char manufacturer;
	unsigned char version;
	unsigned char encoding;
	unsigned char bits;
	unsigned char xMin;
	unsigned char yMin;
	unsigned char xMax;
	unsigned char yMax;
	unsigned char *palette;
} PCXHEADER;

// LoadPCXFile()
// desc: loads a PCX file into memory
unsigned char *LoadPCXFile(char *filename, PCXHEADER *pcxHeader)
{
     int idx = 0;                  // counter index
     int c;                             // used to retrieve a char from the file
     int i;                             // counter index
     int numRepeat;      
     FILE *filePtr;                // file handle
     int width;                         // pcx width
     int height;                        // pcx height
     unsigned char *pixelData = NULL;     // pcx image data
     unsigned char *paletteData = NULL;   // pcx palette data

     // open PCX file
     filePtr = fopen(filename, "rb");
     if (filePtr == NULL)
          return NULL;

     // retrieve first character; should be equal to 10
     c = getc(filePtr);
     if (c != 10)
     {
          fclose(filePtr);
          return NULL;
     }

     // retrieve next character; should be equal to 5
     c = getc(filePtr);
     if (c != 5)
     {
          fclose(filePtr);
          return NULL;
     }

     // reposition file pointer to beginning of file
     rewind(filePtr);

     // read 4 characters of data to skip
     fgetc(filePtr);
     fgetc(filePtr);
     fgetc(filePtr);
     fgetc(filePtr);

     // retrieve leftmost x value of PCX
     pcxHeader->xMin = fgetc(filePtr);       // loword
     pcxHeader->xMin |= fgetc(filePtr) << 8; // hiword

     // retrieve bottom-most y value of PCX
     pcxHeader->yMin = fgetc(filePtr);       // loword
     pcxHeader->yMin |= fgetc(filePtr) << 8; // hiword

     // retrieve rightmost x value of PCX
     pcxHeader->xMax = fgetc(filePtr);       // loword
     pcxHeader->xMax |= fgetc(filePtr) << 8; // hiword

     // retrieve topmost y value of PCX
     pcxHeader->yMax = fgetc(filePtr);       // loword
     pcxHeader->yMax |= fgetc(filePtr) << 8; // hiword

     // calculate the width and height of the PCX
     width = pcxHeader->xMax - pcxHeader->xMin + 1;
     height = pcxHeader->yMax - pcxHeader->yMin + 1;

     // allocate memory for PCX image data
     pixelData = (unsigned char*)malloc(width*height);

     // set file pointer to 128th byte of file, where the PCX image data starts
     fseek(filePtr, 128, SEEK_SET);
     
     // decode the pixel data and store
     while (idx < (width*height))
     {
          c = getc(filePtr);
          if (c > 0xbf)
          {
               numRepeat = 0x3f & c;
               c = getc(filePtr);

               for (i = 0; i < numRepeat; i++)
               {
                    pixelData[idx++] = c;
               }
          }
          else
               pixelData[idx++] = c;

          fflush(stdout);
     }

     //// allocate memory for the PCX image palette
     //paletteData = (unsigned char*)malloc(768);

     //// palette is the last 769 bytes of the PCX file
     //fseek(filePtr, -769, SEEK_END);

     //// verify palette; first character should be 12
     //c = getc(filePtr);
     //if (c != 12)
     //{
     //     fclose(filePtr);
     //     return NULL;
     //}

     //// read and store all of palette
     //for (i = 0; i < 768; i++)
     //{
     //     c = getc(filePtr);
     //     paletteData[i] = c;
     //}

     // close file and store palette in header
     fclose(filePtr);
     pcxHeader->palette = paletteData;

     // return the pixel image data
     return pixelData;
}

// LoadPCXTexture()
// desc: loads a PCX image file as a texture
texture_t* LoadPCXTexture( char* filename )
{
     PCXHEADER texInfo;            // header of texture
     texture_t* thisTexture;       // the texture
     unsigned char *unscaledData;// used to calculate pcx
     int i;                             // index counter
     int j;                             // index counter
     int width;                         // width of texture
     int height;                        // height of texture

     // allocate memory for texture struct
     thisTexture = (texture_t*)malloc(sizeof(texture_t));
     if (thisTexture == NULL)
          return NULL;

     // load the PCX file into the texture struct
     thisTexture->_data = LoadPCXFile(filename, &texInfo);
     if (thisTexture->_data == NULL)
     {
          free(thisTexture->_data);
          return NULL;
     }

     // store the texture information
     thisTexture->_palette = texInfo.palette;
     thisTexture->_width = texInfo.xMax - texInfo.xMin + 1;
     thisTexture->_height = texInfo.yMax - texInfo.yMin + 1;
     thisTexture->_textureType = PCX;

     //// allocate memory for the unscaled data
     //unscaledData = (unsigned char*)malloc(thisTexture->_width*thisTexture->_height*4);

     //// store the unscaled data via the palette
     //for (j = 0; j < thisTexture->_height; j++) 
     //{
     //     for (i = 0; i < thisTexture->_width; i++) 
     //     {
     //          unscaledData[4*(j*thisTexture->_width+i)+0] = (unsigned char)thisTexture->_palette[3*thisTexture->_data[j*thisTexture->_width+i]+0];
     //          unscaledData[4*(j*thisTexture->_width+i)+1] = (unsigned char)thisTexture->_palette[3*thisTexture->_data[j*thisTexture->_width+i]+1];
     //          unscaledData[4*(j*thisTexture->_width+i)+2] = (unsigned char)thisTexture->_palette[3*thisTexture->_data[j*thisTexture->_width+i]+2];
     //          unscaledData[4*(j*thisTexture->_width+i)+3] = (unsigned char)255;
     //     }
     //}

     //// find width and height's nearest greater power of 2
     //width = thisTexture->_width;
     //height = thisTexture->_height;

     //// find width's
     //i = 0;
     //while (width)
     //{
     //     width /= 2;
     //     i++;
     //}
     //thisTexture->_scaledHeight = (long)pow((float)2, i-1);

     //// find height's
     //i = 0;
     //while (height)
     //{
     //     height /= 2;
     //     i++;
     //}
     //thisTexture->_scaledWidth = (long)pow((float)2, i-1);

     //// clear the texture data
     //if (thisTexture->_data != NULL)
     //{
     //     free(thisTexture->_data);
     //     thisTexture->_data = NULL;
     //}

     //// reallocate memory for the texture data
     //thisTexture->_data = (unsigned char*)malloc(thisTexture->_scaledWidth*thisTexture->_scaledHeight*4);
     //
     //// use the GL utility library to scale the texture to the unscaled dimensions
     //gluScaleImage( GL_RGBA, thisTexture->_width, thisTexture->_height, GL_UNSIGNED_BYTE, unscaledData, thisTexture->_scaledWidth, thisTexture->_scaledHeight, GL_UNSIGNED_BYTE, thisTexture->_data );

     return thisTexture;
}

#define BITMAP_ID 0x4D42		// the universal bitmap ID

// LoadBitmapFile
// desc: Returns a pointer to the bitmap image of the bitmap specified
//       by filename. Also returns the bitmap header information.
//		 No support for 8-bit bitmaps.
unsigned char *LoadBitmapFile(char *filename, BITMAPINFOHEADER *bitmapInfoHeader)
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

// LoadBMPTexture()
// desc: loads a texture of the BMP format
texture_t *LoadBMPTexture(char *filename)
{
	BITMAPINFOHEADER texInfo;		// BMP header
	texture_t *thisTexture;			// the texture

	// allocate memory for the texture
	thisTexture = (texture_t*)malloc(sizeof(texture_t));
	if (thisTexture == NULL)
		return NULL;

	// store BMP data in texture
	thisTexture->_data = LoadBitmapFile(filename, &texInfo);
	if (thisTexture->_data == NULL)
	{
		free(thisTexture);
		return NULL;
	}
	
	// store texture information
	thisTexture->_width = texInfo.biWidth;
	thisTexture->_height = texInfo.biHeight;
	thisTexture->_palette = NULL;
	thisTexture->_scaledHeight = 0;
	thisTexture->_scaledWidth = 0;
	thisTexture->_textureType = BMP;

	return thisTexture;
}

// CMD2Model::SetupSkin()
// access: private
// desc: sets up the model skin/texture for OpenGL
void GvgMD2Model::SetupSkin(texture_t *thisTexture)
{
     // set the proper parameters for an MD2 texture
     glGenTextures(1, &thisTexture->_texID);
     glBindTexture(GL_TEXTURE_2D, thisTexture->_texID);
     glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP);
     glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP);

	// glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S, GL_REPEAT);
	// glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T, GL_REPEAT);

     glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
     glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
     
     switch (thisTexture->_textureType)
     {
     case BMP:
          gluBuild2DMipmaps( GL_TEXTURE_2D, GL_RGB, thisTexture->_width, thisTexture->_height, 
		            GL_RGB, GL_UNSIGNED_BYTE, thisTexture->_data );
		 //glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, thisTexture->_width, thisTexture->_height, 0, GL_RGB, GL_UNSIGNED_BYTE, thisTexture->_data );
		 break;
     case PCX:
          gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, thisTexture->_width, thisTexture->_height,
               GL_RGBA, GL_UNSIGNED_BYTE, thisTexture->_data);
     case TGA:
          break;
     default:
          break;
     }
}

/******************************************************************************
 * Load a model
 ******************************************************************************/
bool GvgMD2Model::load( const char* pFilename )
{
	FILE* file = NULL;
	long int fileSize = 0;
	char* buffer = NULL;
	GvgMD2MjodelHeader* modelHeader = NULL;
	frame_t* frame;
	vector_t* verticesPtr = NULL;
	mesh_t* trianglesPtr = NULL;
	stIndex_t* textureCoordinatesPtr = NULL;
	texture_t* texture = NULL;

	_vertices = NULL;
	_triangles = NULL;

	// Open file
	file = fopen( pFilename, "rb" );
	if ( file == NULL )
	{
		return false;
	}

	// Retrieve file length
	fseek( file, 0, SEEK_END );
	fileSize = ftell( file );
	fseek( file, 0, SEEK_SET );

	// Read file
	buffer = new char[ fileSize + 1 ];
	size_t temp = fread( buffer, sizeof( char ), fileSize, file );

	// Retrieve file header
	modelHeader = reinterpret_cast< GvgMD2MjodelHeader* >( buffer );
	
	// -------- [ Vertices ] --------
	// Allocate vertices
	_vertices = new vector_t[ modelHeader->_nbPoints * modelHeader->_nbFrames ];
	_nbVertices = modelHeader->_nbPoints;
	_nbKeyframes = modelHeader->_nbFrames;
	_frameSize = modelHeader->_frameSize;
	// Iterate through keyframes
	for ( int i = 0; i < modelHeader->_nbFrames; i++ )
	{
		// Retrieve current frame data
		frame = reinterpret_cast< frame_t* >( &buffer[ modelHeader->_framesOffset + i * modelHeader->_frameSize ] );

		// Iterate through vertices
		verticesPtr = static_cast< vector_t* >( &_vertices[ i * modelHeader->_nbPoints ] );
		for ( int j = 0; j < modelHeader->_nbPoints; j++ )
		{
			verticesPtr[ j ]._point[ 0 ] = frame->_scale[ 0 ] * frame->_fp[ j ]._v[ 0 ] + frame->_translate[ 0 ];
			verticesPtr[ j ]._point[ 1 ] = frame->_scale[ 1 ] * frame->_fp[ j ]._v[ 1 ] + frame->_translate[ 1 ];
			verticesPtr[ j ]._point[ 2 ] = frame->_scale[ 2 ] * frame->_fp[ j ]._v[ 2 ] + frame->_translate[ 2 ];
		}
	}
	// -------- [ Vertices ] --------

	//texture = LoadPCXTexture( "D:\\Projects\\GigaVoxelsTrunk\\Media\\MD2Models\\yoshi.pcx" );
	//texture = LoadBMPTexture( "D:\\Projects\\GigaVoxelsTrunk\\Media\\MD2Models\\yoshi.bmp" );
	texture = LoadPCXTexture( "cyan_yoshi.pcx" );
	//texture = LoadBMPTexture( "cyan_yoshi.bmp" );
	//texture = LoadBMPTexture( "D:\\Projects\\GigaVoxelsTrunk\\Media\\MD2Models\\yoshi_i.bmp" );
	if ( texture != NULL )
	{
	//	SetupSkin( texture );
		_texture = texture;
	}
	else
	{
		assert( false );
	}

	// -------- [ Texture Coordinates ] --------
	// Allocate texture coordinates
	_textureCoordinates = new texCoord_t[ modelHeader->_nbTextureCoordinates ];
	_nbTextureCoordinates = modelHeader->_nbTextureCoordinates;
	textureCoordinatesPtr = reinterpret_cast< stIndex_t* >( &buffer[ modelHeader->_textureCoordinatesOffset ] );
	for ( int i = 0; i < modelHeader->_nbTextureCoordinates; i++ )
	{
		_textureCoordinates[ i ]._s = static_cast< float >( textureCoordinatesPtr[ i ]._s ) / static_cast< float >( modelHeader->_textureWidth );
		_textureCoordinates[ i ]._t = static_cast< float >( textureCoordinatesPtr[ i ]._t ) / static_cast< float >( modelHeader->_textureHeight );
	}
	// -------- [ Texture Coordinates ] --------

	// -------- [ Triangles ] --------
	// Allocate triangles
	_nbTriangles = modelHeader->_nbTriangles;
	_triangles = new mesh_t[ modelHeader->_nbTriangles ];
	// Iterate through triangles
	trianglesPtr = reinterpret_cast< mesh_t* >( &buffer[ modelHeader->_trianglesOffset ] );
	for ( int i = 0; i < modelHeader->_nbTriangles; i++ )
	{
		_triangles[ i ]._meshIndex[ 0 ] = trianglesPtr[ i ]._meshIndex[ 0 ];
		_triangles[ i ]._meshIndex[ 1 ] = trianglesPtr[ i ]._meshIndex[ 1 ];
		_triangles[ i ]._meshIndex[ 2 ] = trianglesPtr[ i ]._meshIndex[ 2 ];

		_triangles[ i ]._stIndex[ 0 ] = trianglesPtr[ i ]._stIndex[ 0 ];
		_triangles[ i ]._stIndex[ 1 ] = trianglesPtr[ i ]._stIndex[ 1 ];
		_triangles[ i ]._stIndex[ 2 ] = trianglesPtr[ i ]._stIndex[ 2 ];
	}
	// -------- [ Triangles ] --------

	// Close file and free memory
	fclose( file );
	delete [] buffer;

	return false;
}

// CalculateNormal()
// desc: given 3 points, calculates the normal to the points
void CalculateNormal( float *p1, float *p2, float *p3 )
{
   float a[3], b[3], result[3];
   float length;

   a[0] = p1[0] - p2[0];
   a[1] = p1[1] - p2[1];
   a[2] = p1[2] - p2[2];

   b[0] = p1[0] - p3[0];
   b[1] = p1[1] - p3[1];
   b[2] = p1[2] - p3[2];

   result[0] = a[1] * b[2] - b[1] * a[2];
   result[1] = b[0] * a[2] - a[0] * b[2];
   result[2] = a[0] * b[1] - b[0] * a[1];

   // calculate the length of the normal
   length = (float)sqrt(result[0]*result[0] + result[1]*result[1] + result[2]*result[2]);

   // normalize and specify the normal
   glNormal3f(result[0]/length, result[1]/length, result[2]/length);
}

/******************************************************************************
 * Draw a keyframe
 ******************************************************************************/
void GvgMD2Model::draw( int pKeyframe ) const
{
	vector_t* verticesPtr = NULL;

	verticesPtr = static_cast< vector_t* >( &_vertices[ _nbVertices * pKeyframe ] );

//	float x = 0.f;
//	float y = 0.f;
//	float z = 0.f;

//	float* tmp;

	//glColor3f( 1.f, 0.f, 0.f );



	GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat mat_shininess[] = { 50.0 };
	GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };
	//glClearColor (0.0, 0.0, 0.0, 0.0);
	glShadeModel (GL_SMOOTH);

	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);



	//glBindTexture( GL_TEXTURE_2D, _texture->_texID );

	glBegin( GL_TRIANGLES );
	for ( int i = 0; i < _nbTriangles; i++ )
	{
		CalculateNormal(verticesPtr[_triangles[i]._meshIndex[0]]._point,
			verticesPtr[_triangles[i]._meshIndex[2]]._point,
			verticesPtr[_triangles[i]._meshIndex[1]]._point);

	//	glTexCoord2f( _textureCoordinates[ _triangles[ i ]._stIndex[ 0 ] ]._s, _textureCoordinates[ _triangles[ i ]._stIndex[ 0 ] ]._t );
		glVertex3fv( verticesPtr[ _triangles[ i ]._meshIndex[ 0 ] ]._point );
		
	//	glTexCoord2f( _textureCoordinates[ _triangles[ i ]._stIndex[ 2 ] ]._s, _textureCoordinates[ _triangles[ i ]._stIndex[ 2 ] ]._t );
		glVertex3fv( verticesPtr[ _triangles[ i ]._meshIndex[ 2 ] ]._point );
		
	//	glTexCoord2f( _textureCoordinates[ _triangles[ i ]._stIndex[ 1 ] ]._s, _textureCoordinates[ _triangles[ i ]._stIndex[ 1 ] ]._t );
		glVertex3fv( verticesPtr[ _triangles[ i ]._meshIndex[ 1 ] ]._point );

		/*CalculateNormal(verticesPtr[_triangles[i]._meshIndex[0]]._point,
			verticesPtr[_triangles[i]._meshIndex[1]]._point,
			verticesPtr[_triangles[i]._meshIndex[2]]._point);

		glTexCoord2f( _textureCoordinates[ _triangles[ i ]._stIndex[ 0 ] ]._s, _textureCoordinates[ _triangles[ i ]._stIndex[ 0 ] ]._t );
		glVertex3fv( verticesPtr[ _triangles[ i ]._meshIndex[ 0 ] ]._point );
		
		glTexCoord2f( _textureCoordinates[ _triangles[ i ]._stIndex[ 1 ] ]._s, _textureCoordinates[ _triangles[ i ]._stIndex[ 1 ] ]._t );
		glVertex3fv( verticesPtr[ _triangles[ i ]._meshIndex[ 1 ] ]._point );
		
		glTexCoord2f( _textureCoordinates[ _triangles[ i ]._stIndex[ 2 ] ]._s, _textureCoordinates[ _triangles[ i ]._stIndex[ 2 ] ]._t );
		glVertex3fv( verticesPtr[ _triangles[ i ]._meshIndex[ 2 ] ]._point );*/
	}
	glEnd();
}
