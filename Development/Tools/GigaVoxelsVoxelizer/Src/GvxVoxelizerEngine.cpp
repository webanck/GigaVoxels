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

#include "GvxVoxelizerEngine.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// STL
#include <iostream>
#include <cmath>
#include <cstring>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvVoxelizer
using namespace Gvx;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * This flag tell wheter or not to generate normals
 */
// #define NORMALS 1

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvxVoxelizerEngine::GvxVoxelizerEngine()
:	_texture( cimg_library::CImg< float >() )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvxVoxelizerEngine::~GvxVoxelizerEngine()
{
	// TO DO : check _dataStructureIOHandler deletion
	// ...
	// delete _dataStructureIOHandler;
}

/******************************************************************************
 * Initialize the voxelizer
 *
 * Call before voxelization
 *
 * @param pLevel Max level of resolution
 * @param pBrickWidth Width a brick
 * @param pName Filename to be processed
 * @param pDataType Data type that will be processed
 ******************************************************************************/
void GvxVoxelizerEngine::init( int pLevel, int pBrickWidth, const std::string& pName, GvxDataTypeHandler::VoxelDataType pDataType )
{
	// Store initialization values
	_level = pLevel;
	_brickWidth = pBrickWidth;
	_fileName = pName;

	// Handle data types to process
	_dataTypes.push_back( pDataType );

	if (_normals)
	{
		_dataTypes.push_back( Gvx::GvxDataTypeHandler::gvHALF4 );
	}

	// Create a file/streamer handler to read/write GigaVoxels data
	_dataStructureIOHandler = new GvxDataStructureIOHandler( _fileName, _level, _brickWidth, _dataTypes, true );
}

/******************************************************************************
 * Set The number of times we apply the filter
 ******************************************************************************/
void GvxVoxelizerEngine::setNbFilterApplications(int pValue)
{
	_nbFilterApplications = pValue;
}

/******************************************************************************
 * Set the type of the filter
 * 0 = mean
 * 1 = gaussian
 * 2 = laplacian
 ******************************************************************************/
void GvxVoxelizerEngine::setFilterType (int pValue)
{
	_filterType = pValue;
}

/******************************************************************************
 * Set the _normals value
 ******************************************************************************/
void GvxVoxelizerEngine::setNormals( bool value)
{
	_normals=value;
}

/******************************************************************************
 * Finalize the voxelizer
 *
 * Call after voxelization
 ******************************************************************************/
void GvxVoxelizerEngine::end()
{
	// Normalize normals (if activated)
	// normalize();

	// Compute borders (this will be done for the last level of resolution)
	updateBorders();

	// delete file/streamer handler
	delete _dataStructureIOHandler;
	_dataStructureIOHandler = NULL;

	// apply a smoothing filter
	applyFilter();

	// Mipmap data
	mipmap();
}

/******************************************************************************
 * Store a 3D position in the vertex buffer.
 * During voxelization, each triangle attribute is stored.
 * Due to kind of circular buffer technique, calling setVertex() method on each vertex
 * of a triangle, register each position internally.
 *
 * @param pX x coordinate
 * @param pY y coordinate
 * @param pZ z coordinate
 ******************************************************************************/
void GvxVoxelizerEngine::setVertex( float pX, float pY, float pZ )
{
	// Update circular vertex buffer
	memcpy( _v1, _v2, 3 * sizeof( float ) );
	memcpy( _v2, _v3, 3 * sizeof( float ) );

	// Store 3D position
	_v3[ 0 ] = pX;
	_v3[ 1 ] = pY;
	_v3[ 2 ] = pZ;
}

/******************************************************************************
 * Store a color in the color buffer.
 * During voxelization, each triangle attribute is stored.
 * Due to kind of circular buffer technique, calling setColor() method on each vertex
 * of a triangle, register each color internally.
 *
 * @param pR red color component
 * @param pG green color component
 * @param pB blue color component
 ******************************************************************************/
void GvxVoxelizerEngine::setColor( float pR, float pG, float pB )
{
	// Update circular color buffer
	memcpy( _c1, _c2, 3 * sizeof( float ) );
	memcpy( _c2, _c3, 3 * sizeof( float ) );

	// Store color
	_c3[ 0 ] = pR;
	_c3[ 1 ] = pG;
	_c3[ 2 ] = pB;
}

/******************************************************************************
 * Store a texture coordinates in the texture coordinates buffer.
 * During voxelization, each triangle attribute is stored.
 * Due to kind of circular buffer technique, calling setTexCoord() method on each vertex
 * of a triangle, register each texture coordinate internally.
 *
 * @param pR r texture coordinate
 * @param pS s texture coordinate
 ******************************************************************************/
void GvxVoxelizerEngine::setTexCoord( float pR, float pS )
{
	// Update circular texture coordinates buffer
	memcpy( _t1, _t2, 2 * sizeof( float ) );
	memcpy( _t2, _t3, 2 * sizeof( float ) );

	// Store texture coordinates
	_t3[ 0 ] = pR;
	_t3[ 1 ] = pS;
}

/******************************************************************************
 * Store a normal in the buffer of normals.
 * During voxelization, each triangle attribute is stored.
 * Due to kind of circular buffer technique, calling setNormal() method on each vertex
 * of a triangle, register each normal internally.
 *
 * @param pX x normal component
 * @param pY y normal component
 * @param pZ z normal component
 ******************************************************************************/
void GvxVoxelizerEngine::setNormal( float pX, float pY, float pZ )
{
	// Update circular buffer of normals
	memcpy( _n1, _n2, 3 * sizeof( float ) );
	memcpy( _n2, _n3, 3 * sizeof( float ) );

	// Store normal
	_n3[ 0 ] = pX;
	_n3[ 1 ] = pY;
	_n3[ 2 ] = pZ;

	// float length = sqrtf(pX * pX + pY * pY + pZ * pZ);
	// float normalized[3] = {abs(pX)/length, abs(pY)/length, abs(pZ)/length};
	// _n3[ 0 ] = normalized[0];
	// _n3[ 1 ] = normalized[1];
	// _n3[ 2 ] = normalized[2];
}

/******************************************************************************
 * Construct image from reading an image file.
 *
 * @param pFilename the image filename
 ******************************************************************************/
void GvxVoxelizerEngine::setTexture( const std::string& pFilename )
{
	// WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// TO DO : important !!!!!!!!!!
	// Mettre un break point et v�rifier s'il n'y a pas de memory leak
	// ...

	// Construct image from reading an image file.
	// Construct a new image instance with pixels of type T, and initialize pixel values with the data read from an image file.
	_texture = cimg_library::CImg< float >( pFilename.data() );
}

unsigned short float2HalfInUshort (float f) {

	// if (abs(f)<0.001)
	// 	return (unsigned short)(0);
	unsigned int a = *((unsigned int *)(&f)); // reinterpret cast
	unsigned int s = (a & 0x80000000); // isolating sign
	s= s>>16; // moving it to match a 16 bits number
	unsigned short ss = (unsigned short)(s); // casting

	unsigned int exp = (a & 0x7FFFFFFF) >> 23; // masking the bit and removing mantissa

	exp = (exp+15)-127; // Decentring the exponent of the 8 bit space, recentering it on a 5 bit space
	exp  = exp<<10;
	unsigned short exps = (unsigned short)(exp); // casting
	unsigned int mantissa = (a & 0x007FFFFF ) >> 13 ; // isolating matissa and removing 13 lights bits
	unsigned short mantissas = (unsigned short)(mantissa); // casting

	return ss | exps | mantissas; // Concatenation


}

float halfInUshort2Float (unsigned short u) {

	if (u==0)
		return 0.f;
	unsigned short s = (u & 0x8000); // isolating sign
	unsigned int si = (unsigned int)(s); // casting
	si = si<<16; // shifting to its proper place
	unsigned short exp = (u & 0x7FFF) >> 10;   // isolating exposant
	unsigned int expi = (unsigned int)(exp);   // casting
	expi = (expi+127)-15;   // decentring on the 5 bits scale, recentring on a 8 bit scale
	expi = expi<<23;      // shifting to its position, after mantissa

	unsigned short mantissa = (u & 0x03FF );     // isolating mantissa
	unsigned int mantissai = (unsigned int)(mantissa);   // casting
	mantissai = mantissai<<13;    // replacing by zero padding

	unsigned int ret = si | expi | mantissai;  // concatenation
	return * ((float *)(&ret));     // reinterpret cast

}


/******************************************************************************
 * Voxelize a triangle.
 *
 * Given vertex attributes previously set for a triangle (positions, normals,
 * colors and texture coordinates), it voxelizes triangle (by writing data).
 ******************************************************************************/
void GvxVoxelizerEngine::voxelizeTriangle()
{
	// Compute length of each border of the current triangle
	float length1 = sqrtf( ( _v1[0] - _v2[0] ) * ( _v1[0] - _v2[0] ) + ( _v1[1] - _v2[1] ) * ( _v1[1] - _v2[1] ) + ( _v1[2] - _v2[2] ) * ( _v1[2] - _v2[2] ) );
	float length2 = sqrtf( ( _v1[0] - _v3[0] ) * ( _v1[0] - _v3[0] ) + ( _v1[1] - _v3[1] ) * ( _v1[1] - _v3[1] ) + ( _v1[2] - _v3[2] ) * ( _v1[2] - _v3[2] ) );
	float length3 = sqrtf( ( _v2[0] - _v3[0] ) * ( _v2[0] - _v3[0] ) + ( _v2[1] - _v3[1] ) * ( _v2[1] - _v3[1] ) + ( _v2[2] - _v3[2] ) * ( _v2[2] - _v3[2] ) );

	// Compute the tesselation value
	// (.i.e how many voxels can be put in the largest length of the triangle borders)
	float length = std::max< float >( length1, std::max< float >( length2, length3 ) );
	int tesselation = static_cast< int >( length / _dataStructureIOHandler->getVoxelSize() ) + 1;

	// Iterate through voxels
	for ( int i = 0; i < tesselation; ++i )
	for ( int j = 0; j < tesselation; ++j )
	{
		// Vertex, color and normal variables
		float v[ 3 ];
		float c[ 3 ];
		float n[ 3 ];


		// Compute weights associated to vertices (barycentric coordinates)
		float w1 = i / static_cast< float >( tesselation );
		float w2 = (1.0f - w1) * j / static_cast< float >( tesselation );
		float w3 = 1.0f - w1 - w2;

		// Compute weighted position (barycentric coordinates)
		v[0] = w1 * _v1[0] + w2 * _v2[0] + w3 * _v3[0];
		v[1] = w1 * _v1[1] + w2 * _v2[1] + w3 * _v3[1];
		v[2] = w1 * _v1[2] + w2 * _v2[2] + w3 * _v3[2];

		// Compute color
		if ( _useTexture )
		{
			// Compute weighted texture coordinates
			float t[2];
			t[0] = w1 * _t1[0] + w2 * _t2[0] + w3 * _t3[0];
			t[1] = w1 * _t1[1] + w2 * _t2[1] + w3 * _t3[1];

			// Handle negative texture coordinates
			t[0] = ( t[0] >= 0.0f ) ? t[0] : t[0] - floor( t[0] );
			t[1] = ( t[1] >= 0.0f ) ? t[1] : t[1] - floor( t[1] );

			// Retrieve indexed pixel coordinates from original image
			unsigned int tx = static_cast< unsigned int >( t[0] * ( _texture.width() - 1 ) );
			tx = tx % _texture.width();
			unsigned int ty = static_cast< unsigned int >( t[1] * ( _texture.height() - 1 ) );
			ty = ty % _texture.height();

			// Sample texture and normalize value
			c[0] = _texture( tx, ty, 0, 0 ) / 255.f;
			c[1] = _texture( tx, ty, 0, 1 ) / 255.f;
			c[2] = _texture( tx, ty, 0, 2 ) / 255.f;
		}
		else
		{
			// Compute weighted colors
			c[0] = w1 * _c1[0] + w2 * _c2[0] + w3 * _c3[0];
			c[1] = w1 * _c1[1] + w2 * _c2[1] + w3 * _c3[1];
			c[2] = w1 * _c1[2] + w2 * _c2[2] + w3 * _c3[2];
			// c[0] = 0.5f;
			// c[1] = 0.5f;
			// c[2] = 0.5f;
		}

		// TO DO
		// WARNING : question ==> Why "unsigned char" ? This should be the type of GvxDataTypeHandler specified by the user ?
		// ...
		// color data uchar4
		unsigned char voxelData[4];
		voxelData[0] = static_cast< unsigned char >( c[0] * 255.f ); // Red
		voxelData[1] = static_cast< unsigned char >( c[1] * 255.f ); // Green
		voxelData[2] = static_cast< unsigned char >( c[2] * 255.f ); // Blue
		voxelData[3] = 255; // Alpha

		unsigned short normalData[4];
		if (_normals)
		{
			// Compute weighted normals
			n[0] = w1 * _n1[0] + w2 * _n2[0] + w3 * _n3[0];
			n[1] = w1 * _n1[1] + w2 * _n2[1] + w3 * _n3[1];
			n[2] = w1 * _n1[2] + w2 * _n2[2] + w3 * _n3[2];
			//Compute normal as cross product manually...
			// const float e1[3] = {_v2[0] - _v1[0], _v2[1] - _v1[1], _v2[2] - _v1[2]};
			// const float e2[3] = {_v3[0] - _v2[0], _v3[1] - _v2[1], _v3[2] - _v2[2]};
			// n[0] = e1[1]*e2[2] - e1[2]*e2[1];
			// n[1] = e1[2]*e2[0] - e1[0]*e2[2];
			// n[2] = e1[0]*e2[1] - e1[1]*e2[0];
			// const float length = sqrtf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
			// n[0] /= length;
			// n[1] /= length;
			// n[2] /= length;
			// n[0] = (n[0] < 0 ? -n[0] : n[0]);
			// n[1] = (n[1] < 0 ? -n[1] : n[1]);
			// n[2] = (n[2] < 0 ? -n[2] : n[2]);

			// std::cout << n[0] << ";" << n[1] << ";" << n[2] << ";" << std::endl;
			// normal data uchar4

			normalData[0] = float2HalfInUshort( n[0] ); // normal x
			// normalData[0] = float2HalfInUshort( n[0] ); // normal x
			/*if (abs(n[0] - halfInUshort2Float( float2HalfInUshort( n[0] )) ) >1.f)
				printf ("%f != %f\n",n[0],halfInUshort2Float( float2HalfInUshort( n[0] )));*/

			normalData[1] = float2HalfInUshort( n[1] );
			// normalData[1] = float2HalfInUshort( n[1] ); // normal y
			/*if (abs(n[1] - halfInUshort2Float( float2HalfInUshort( n[1] )) ) >1.f)
				printf ("%f != %f\n",n[1],halfInUshort2Float( float2HalfInUshort( n[1] )));*/

			normalData[2] = float2HalfInUshort( n[2] );
			// normalData[2] = float2HalfInUshort( n[2] ); // normal z
			/*if (abs(n[2] - halfInUshort2Float( float2HalfInUshort( n[2] )) ) >1.f)
				printf ("%f != %f\n",n[2],halfInUshort2Float( float2HalfInUshort( n[2] )));*/

			normalData[3] = 1; // flag as non-empty
		}

		// Retrieve voxel positions in octree
		unsigned int voxelPos[3];
		unsigned int voxelPosInBrick[3];
		_dataStructureIOHandler->getVoxelPosition( v, voxelPos );
		_dataStructureIOHandler->getVoxelPositionInBrick( v, voxelPosInBrick );

		// Splat in octree with 2 voxels width
		for ( int z = 0; z <= 1; ++z )
		for ( int y = 0; y <= 1; ++y )
		for ( int x = 0; x <= 1; ++x )
		{
			unsigned int voxelPos2[3];

			//
			voxelPos2[0] = ( voxelPosInBrick[0] == 1 ) ? voxelPos[0] + x : voxelPos[0] - x;
			voxelPos2[1] = ( voxelPosInBrick[1] == 1 ) ? voxelPos[1] + y : voxelPos[1] - y;
			voxelPos2[2] = ( voxelPosInBrick[2] == 1 ) ? voxelPos[2] + z : voxelPos[2] - z;

			// Set voxel data
			_dataStructureIOHandler->setVoxel( voxelPos2, voxelData, 0 );

			if (_normals)
			{
				// Set voxel normal
				_dataStructureIOHandler->setVoxel( voxelPos2, normalData, 1 );
			}
		}
	}
}

/******************************************************************************
 * Apply the update borders algorithmn.
 * Fill borders with data.
 ******************************************************************************/
void GvxVoxelizerEngine::updateBorders()
{
	std::cout << "GvxVoxelizerEngine::updateBorders : level : " << _dataStructureIOHandler->_level << std::endl;
	_dataStructureIOHandler->computeBorders();
}

/******************************************************************************
 * Apply the normalize algorithmn
 ******************************************************************************/
void GvxVoxelizerEngine::normalize()
{
	if (_normals)
	{
		unsigned short* brick1 = new unsigned short[ 4 * _dataStructureIOHandler->_brickSize ];

		// Iterate through nodes
		unsigned int nodePos[ 3 ];
		for ( nodePos[2] = 0; nodePos[2] < _dataStructureIOHandler->_nodeGridSize; ++nodePos[2] )
		for ( nodePos[1] = 0; nodePos[1] < _dataStructureIOHandler->_nodeGridSize; ++nodePos[1] )
		for ( nodePos[0] = 0; nodePos[0] < _dataStructureIOHandler->_nodeGridSize; ++nodePos[0] )
		{
			// If node is empty, continue
			unsigned int node = _dataStructureIOHandler->getNode( nodePos );
			if ( GvxDataStructureIOHandler::isEmpty( node ) )
			{
				continue;
			}

			// Get brick data for channel normals (stored in the second channel)
			_dataStructureIOHandler->getBrick( nodePos, brick1, 1 );

			// Iterate through brick voxels and process them
			for ( unsigned int voxel = 0; voxel < _dataStructureIOHandler->_brickSize; voxel++ )
			{
				// If flag empty
				if ( brick1[ 4 * voxel + 3 ] == 0 )
				{
					// Set normal components to 0		// TO DO => QUESTION : what's that ??
					brick1[ 4 * voxel + 0 ] = 128;
					brick1[ 4 * voxel + 1 ] = 128;
					brick1[ 4 * voxel + 2 ] = 128;
				}
				// Set flag empty to 0
				brick1[ 4 * voxel + 3 ] = 0;
			}

			// Set brick data
			_dataStructureIOHandler->setBrick( nodePos, brick1, 1 );
		}

		delete [] brick1;
	}
}

// Helper function to have more accurate type casts (because static_cast<unsigned char>(f) always return the floor value of f)
unsigned char float2uchar(float a) {
	return  static_cast<unsigned char>(a+0.5f);
}


#define FILTER 1          // the half-tchickness of the 3D kernel
#define SIGMA FILTER      // the sigma of the gaussian
#define DIM (FILTER*2+1)  // the dimension of the full kernel
#define SIZE (DIM*DIM*DIM)// the total num of elements of the kernel

// Different functions for filtering
float identity(float x)
{
	return 1;
}
float gaussian(float x)
{
	return exp(-(x*x)/(2*SIGMA*SIGMA)); // gaussian evaluation;
}
float laplacian(float x)
{
	return 1;
}

/******************************************************************************
 * Apply the filtering algorithm
 ******************************************************************************/
void GvxVoxelizerEngine::applyFilter()
{
	//Retrieving the IO Handler
	GvxDataStructureIOHandler* dataStructureIOHandlerUP = new GvxDataStructureIOHandler( _fileName, _level, _brickWidth, _dataTypes, false );

	// Defining the blurring kernel
	float kernel[SIZE] ;  // The gaussian/laplacian/mean kernel
	float sum = 0;		  // The sum of all the coefficients of the gaussian kernel

	// Choosing the appropriate function for the kernel
	printf ("Applying a ");
	float (*filter)(float);
	switch (_filterType)
	{
	case 0:
		printf ("mean ");
		filter = &identity;
		break;
	case 1:
		printf ("gaussian ");
		filter = &gaussian;
		break;
	case 2:
		printf ("laplacian ");
		filter = &laplacian;
		break;
	default:
		printf ("default (mean) but how did you get in here ? ");
		filter = &identity;
		break;
	}
	printf ("Filter %d times...\n",_nbFilterApplications);

	// filling the filter with the function evaluated at each point of the kernel
	for (int i = -FILTER;i<=FILTER;i++)
	for (int j = -FILTER;j<=FILTER;j++)
	for (int k = -FILTER;k<=FILTER;k++)
	{
		// x is the disance to the center of the kernel
		float x = sqrtf(i*i+j*j+k*k); // distance to center of the kernel
		// y is f(x) , the value to assign in the kernel (f = 1D function)
		float y = filter(x);
		// linearization of the kernel (no matter the order, it's symetric)
		kernel[(i+FILTER)*DIM*DIM+(j+FILTER)*DIM+(k+FILTER)] = y ;
		// builfing the local sum to be able to normalize later on
		sum+=y;
	}

	// The data of the grid with a neighborhood corresponding to the kernel size
	unsigned char voxelDataTotal[SIZE][4] ;
	unsigned short voxelNormalTotal[SIZE][4] ;

	unsigned char voxelData[ 4 ];
	unsigned char voxelDataTemp[ 4 ];

	unsigned short voxelNormal[ 4 ];
	unsigned short voxelNormalTemp[ 4 ];

	unsigned int nodePos[ 3 ];

	// We load the data brick per brick for now
	// WARNING, this forces the kernel to be 3*3*3 only because we can anly use the borders to make the convolution
	unsigned char  * voxelDataBrick;// [4000];        // WARNING, 4000 HARDCODED and unsigned char : (_brickWidth + bordersize*2) * dataTypeSize (8 + 1*2)*4
	// the backup array is the array in which we store the convolution results (at the end we do a kind of swapArrays).
	unsigned char  * voxelDataBrickBackup;// [4000];  // WARNING, 4000 HARDCODED and unsigned char : (_brickWidth + bordersize*2) * dataTypeSize (8 + 1*2)*4

	unsigned short  * voxelNormalBrick;// [4000];      // WARNING, 4000 HARDCODED and unsigned char : (_brickWidth + bordersize*2) * dataTypeSize (8 + 1*2)*4
	unsigned short *  voxelNormalBrickBackup;// [4000];// WARNING, 4000 HARDCODED and unsigned char : (_brickWidth + bordersize*2) * dataTypeSize (8 + 1*2)*4

	voxelDataBrick = new unsigned char[4 * (_brickWidth+2)*(_brickWidth+2)*(_brickWidth+2)] ;
	voxelDataBrickBackup = new unsigned char[4 * (_brickWidth+2)*(_brickWidth+2)*(_brickWidth+2)] ;

	if (_normals)
	{
		voxelNormalBrick = new unsigned short[4 * (_brickWidth+2)*(_brickWidth+2)*(_brickWidth+2)] ;
		voxelNormalBrickBackup = new unsigned short[4 * (_brickWidth+2)*(_brickWidth+2)*(_brickWidth+2)] ;
	}

	// We may have to apply successively many times the filter (to emulate a bigger blur)
	for (int var = 0; var<_nbFilterApplications ; var++)
	{

		// for all the nodes of the grid
		for ( nodePos[2] = 0; nodePos[2] < dataStructureIOHandlerUP->_nodeGridSize; nodePos[2]++ )
		for ( nodePos[1] = 0; nodePos[1] < dataStructureIOHandlerUP->_nodeGridSize; nodePos[1]++ )
		for ( nodePos[0] = 0; nodePos[0] < dataStructureIOHandlerUP->_nodeGridSize; nodePos[0]++ )
		{
			//// Retrieve the current node info
			unsigned int node = dataStructureIOHandlerUP->getNode( nodePos );

			//// If node is empty no need to blur, go to next node
			if ( GvxDataStructureIOHandler::isEmpty( node ) )
			{
				continue;
			}

			// If not empty, get the brick from file and store it in memory
			dataStructureIOHandlerUP->getBrick(nodePos,voxelDataBrick,0);
			if (_normals)
			{
				dataStructureIOHandlerUP->getBrick(nodePos,voxelNormalBrick,1);
			}

			// We will now iterate through voxels of the current node and apply the kernel on their neighborhood
			unsigned int voxelPos[ 3 ];
			unsigned int voxelPos2[ 3 ];

			// For all the voxels of the brick WITHOUT THE BORDER !
			for ( voxelPos[ 2 ] =1; voxelPos[ 2 ] < (_brickWidth)+2-1; voxelPos[ 2 ] +=1 )
			for ( voxelPos[ 1 ] =1; voxelPos[ 1 ] < (_brickWidth)+2-1; voxelPos[ 1 ] +=1 )
			for ( voxelPos[ 0 ] =1; voxelPos[ 0 ] < (_brickWidth)+2-1; voxelPos[ 0 ] +=1 )
			{

				// Retrieve the data of the neighborhood of the voxel ( 27 values for a kernel of 3*3*3)
				for (int i = -FILTER;i<=FILTER;i++)
				for (int j = -FILTER;j<=FILTER;j++)
				for (int k = -FILTER;k<=FILTER;k++)
				{
					// Compute the position of the neighbor
					voxelPos2[0] = voxelPos[0]+i;voxelPos2[1] = voxelPos[1]+j;voxelPos2[2] = voxelPos[2]+k;

					// Linearize it to get an index in the brickData pool
					unsigned int deb =  voxelPos2[0]*10*10*4 + voxelPos2[1]*10*4 + voxelPos2[2]*4;    // Warning HARDCODED 10 = _brickWidth + 2*borderSize
					// retrieve the data from the brick
					voxelDataTemp[0] = voxelDataBrick[deb];
					voxelDataTemp[1] = voxelDataBrick[deb+1];
					voxelDataTemp[2] = voxelDataBrick[deb+2];
					voxelDataTemp[3] = voxelDataBrick[deb+3];

					// Store it in the neighborhood matrix
					voxelDataTotal[(i+FILTER)*DIM*DIM+(j+FILTER)*DIM+(k+FILTER)][0] = voxelDataTemp[0];
					voxelDataTotal[(i+FILTER)*DIM*DIM+(j+FILTER)*DIM+(k+FILTER)][1] = voxelDataTemp[1];
					voxelDataTotal[(i+FILTER)*DIM*DIM+(j+FILTER)*DIM+(k+FILTER)][2] = voxelDataTemp[2];
					voxelDataTotal[(i+FILTER)*DIM*DIM+(j+FILTER)*DIM+(k+FILTER)][3] = voxelDataTemp[3];

					if (_normals)
					{
						// Same for normals
						voxelNormalTemp[0] = voxelNormalBrick[deb];
						voxelNormalTemp[1] = voxelNormalBrick[deb+1];
						voxelNormalTemp[2] = voxelNormalBrick[deb+2];
						voxelNormalTemp[3] = voxelNormalBrick[deb+3];
						voxelNormalTotal[(i+FILTER)*DIM*DIM+(j+FILTER)*DIM+(k+FILTER)][0] = voxelNormalTemp[0];
						voxelNormalTotal[(i+FILTER)*DIM*DIM+(j+FILTER)*DIM+(k+FILTER)][1] = voxelNormalTemp[1];
						voxelNormalTotal[(i+FILTER)*DIM*DIM+(j+FILTER)*DIM+(k+FILTER)][2] = voxelNormalTemp[2];
						voxelNormalTotal[(i+FILTER)*DIM*DIM+(j+FILTER)*DIM+(k+FILTER)][3] = voxelNormalTemp[3];
					}

				} // from now on voxelNormalTotal contains the 27 values of the neighboors of the voxel we are considering (the value of the voxel itself being in the center)


				// Computing the convolution of the filter and the neighborhood
				float temp = 0.f;
				float sumNormal =0.f;
				float sumColor =0.f;
				float tempColor[3] = {0.f,0.f,0.f};
				float tempNormal[3] = {0.f,0.f,0.f};

				// For all the values of the 3*3*3 kernel (linearized)
				for (int k = 0;k<SIZE;k++)
				{
					// make the convolution of the opacity (in float)
					temp+= (static_cast<float>(voxelDataTotal[k][3])/255.f) * kernel[k];

					// also make the convolution of normals and color , but only when opacity is not null (void has no color and no normal)
					if (voxelDataTotal[k][3] !=0)
					{
						// Don't forget that the colors are premultiplied by the opacity so we have to demultiply them
						tempColor[0]+= ((static_cast<float>(voxelDataTotal[k][0])/255.f)/(static_cast<float>(voxelDataTotal[k][3])/255.f))*kernel[k];
						tempColor[1]+= ((static_cast<float>(voxelDataTotal[k][1])/255.f)/(static_cast<float>(voxelDataTotal[k][3])/255.f))*kernel[k];
						tempColor[2]+= ((static_cast<float>(voxelDataTotal[k][2])/255.f)/(static_cast<float>(voxelDataTotal[k][3])/255.f))*kernel[k];
						//building the local sum for color (can be different from the local sum of the kernel, because we skip the void elements)
						sumColor +=   kernel[k];
						if (_normals)
						{
							// converting the normals back in float
							tempNormal[0]+= (halfInUshort2Float(voxelNormalTotal[k][0]))*kernel[k];
							tempNormal[1]+= (halfInUshort2Float(voxelNormalTotal[k][1]))*kernel[k];
							tempNormal[2]+= (halfInUshort2Float(voxelNormalTotal[k][2]))*kernel[k];
							//building the local sum for normals (can be different from the local sum of the kernel, because we skip the void elements)
							sumNormal +=  kernel[k];
						}
					}


				} // end of the convolution

				// we normalize the sums and transform everyone to uchar
				float opacity = temp/sum;
				voxelData[3] = float2uchar (opacity*255.f);

				if (sumColor> 0)
				{
					// normalization + apha premultiplication + cast to uchar
					voxelData[0] = float2uchar ((tempColor[0]*opacity/sumColor)*255.f);
					voxelData[1] = float2uchar ((tempColor[1]*opacity/sumColor)*255.f);
					voxelData[2] = float2uchar ((tempColor[2]*opacity/sumColor)*255.f);
				}
				else
				{
					voxelData[0] = float2uchar (0.f);
					voxelData[1] = float2uchar (0.f);
					voxelData[2] = float2uchar (0.f);
				}

				if (_normals)
				{

					if (sumNormal>0 )
					{
						// normalization
						tempNormal[0] /= sumNormal;
						tempNormal[1] /= sumNormal;
						tempNormal[2] /= sumNormal;
						float length= sqrtf(tempNormal[0]*tempNormal[0]+tempNormal[1]*tempNormal[1]+tempNormal[2]*tempNormal[2]);
						if ( length> 0)
						{
							// normalization (in the sense of the normal) + apha premultiplication + cast to uchar
							voxelNormal[0] = float2HalfInUshort ((tempNormal[0])/length);
							voxelNormal[1] = float2HalfInUshort ((tempNormal[1])/length);
							voxelNormal[2] = float2HalfInUshort ((tempNormal[2])/length);
							voxelNormal[3] = 1;
						}
						else
						{
							voxelNormal[0] = float2HalfInUshort (0.f);
							voxelNormal[1] = float2HalfInUshort (0.f);
							voxelNormal[2] = float2HalfInUshort (0.f);
							voxelNormal[3] = 0;
						}
					}
					else
					{
						voxelNormal[0] = float2HalfInUshort (0.f);
						voxelNormal[1] = float2HalfInUshort (0.f);
						voxelNormal[2] = float2HalfInUshort (0.f);
						voxelNormal[3] = 0;
					}


				}

				// Storing the new data in the Brick data
				unsigned int deb =  voxelPos[0]*10*10*4 + voxelPos[1]*10*4 + voxelPos[2]*4;  // Warning HARDCODED 10 = _brickWidth + 2*borderSize
				voxelDataBrickBackup[deb] = voxelData[0];
				voxelDataBrickBackup[deb+1] = voxelData[1];
				voxelDataBrickBackup[deb+2] = voxelData[2];
				voxelDataBrickBackup[deb+3] = voxelData[3];

				if (_normals)
				{
					voxelNormalBrickBackup[deb] = voxelNormal[0];
					voxelNormalBrickBackup[deb+1] = voxelNormal[1];
					voxelNormalBrickBackup[deb+2] = voxelNormal[2];
					voxelNormalBrickBackup[deb+3] = voxelNormal[3];

				}

			}

			// Post pass to recopy the values from the backup array to the final array
			for ( voxelPos[ 2 ] =1; voxelPos[ 2 ] < (_brickWidth)+2-1; voxelPos[ 2 ] +=1 )
			for ( voxelPos[ 1 ] =1; voxelPos[ 1 ] < (_brickWidth)+2-1; voxelPos[ 1 ] +=1 )
			for ( voxelPos[ 0 ] =1; voxelPos[ 0 ] < (_brickWidth)+2-1; voxelPos[ 0 ] +=1 )
			{
				// linearization of the index
				unsigned int deb =  voxelPos[0]*10*10*4 + voxelPos[1]*10*4 + voxelPos[2]*4;  // Warning HARDCODED 10 = _brickWidth + 2*borderSize

				voxelDataBrick[deb+3] = voxelDataBrickBackup[deb+3];
				voxelDataBrick[deb] =  voxelDataBrickBackup[deb];
				voxelDataBrick[deb+1] =  voxelDataBrickBackup[deb+1];
				voxelDataBrick[deb+2] =  voxelDataBrickBackup[deb+2];


				if (_normals)
				{
					voxelNormalBrick[deb] = voxelNormalBrickBackup[deb];
					voxelNormalBrick[deb+1] = voxelNormalBrickBackup[deb+1];
					voxelNormalBrick[deb+2] = voxelNormalBrickBackup[deb+2];
					voxelNormalBrick[deb+3] = voxelNormalBrickBackup[deb+3];

				}
			}

			// storing the modified brick
			dataStructureIOHandlerUP->setBrick(nodePos,voxelDataBrick,0);
			if (_normals)
			{
				dataStructureIOHandlerUP->setBrick(nodePos,voxelNormalBrick,1);
			}
		}

		// Recompute the borders to make them accord the new changes
		dataStructureIOHandlerUP->computeBorders();

	}

	delete[] voxelDataBrick;
	delete[] voxelDataBrickBackup;
	if (_normals)
	{
		delete[] voxelNormalBrick;
		delete[] voxelNormalBrickBackup;
	}


}


/******************************************************************************
 * Apply the mip-mapping algorithmn.
 * Given a pre-filtered voxel scene at a given level of resolution,
 * it generates a mip-map pyramid hierarchy of coarser levels (until 0).
 ******************************************************************************/
void GvxVoxelizerEngine::mipmap()
{
	// The mip-map pyramid hierarchy is built recursively from adjacent levels.
	// Two files/streamers are used :
	// UP is an already pre-filtered scene at resolution [ N ]
	// DOWN is the coarser version to generate at resolution [ N - 1 ]

	GvxDataStructureIOHandler* dataStructureIOHandlerUP = new GvxDataStructureIOHandler( _fileName, _level, _brickWidth, _dataTypes, false );
	GvxDataStructureIOHandler* dataStructureIOHandlerDOWN = NULL;


	// Iterate through levels of resolution
	for ( int level = _level - 1; level >= 0; level-- )
	{
		// LOG info
		std::cout << "GvxVoxelizerEngine::mipmap : level : " << level << std::endl;

		// The coarser data handler is allocated dynamically due to memory consumption considerations.
		dataStructureIOHandlerDOWN = new GvxDataStructureIOHandler( _fileName, level, _brickWidth, _dataTypes, true );

		// Iterate through nodes of the structure
		unsigned int nodePos[ 3 ];
		for ( nodePos[2] = 0; nodePos[2] < dataStructureIOHandlerUP->_nodeGridSize; nodePos[2]++ )
		for ( nodePos[1] = 0; nodePos[1] < dataStructureIOHandlerUP->_nodeGridSize; nodePos[1]++ )
		{
			// LOG info
			std::cout << "mipmap - LEVEL [ " << level << " ] - Node [ " << "x" << " / " << nodePos[1] << " / " << nodePos[2] << " ] - " << dataStructureIOHandlerUP->_nodeGridSize << std::endl;

		for ( nodePos[ 0 ] = 0; nodePos[ 0 ] < dataStructureIOHandlerUP->_nodeGridSize; nodePos[ 0 ]++ )
		{
			// Retrieve the current node info
			unsigned int node = dataStructureIOHandlerUP->getNode( nodePos );

			// If node is empty, go to next node
			if ( GvxDataStructureIOHandler::isEmpty( node ) )
			{
				continue;
			}

			// Iterate through voxels of the current node
			unsigned int voxelPos[ 3 ];
			for ( voxelPos[ 2 ] = _brickWidth * nodePos[ 2 ]; voxelPos[ 2 ] < dataStructureIOHandlerUP->_brickWidth * ( nodePos[ 2 ] + 1 ); voxelPos[ 2 ] +=2 )
			for ( voxelPos[ 1 ] = _brickWidth * nodePos[ 1 ]; voxelPos[ 1 ] < dataStructureIOHandlerUP->_brickWidth * ( nodePos[ 1 ] + 1 ); voxelPos[ 1 ] +=2 )
			for ( voxelPos[ 0 ] = _brickWidth * nodePos[ 0 ]; voxelPos[ 0 ] < dataStructureIOHandlerUP->_brickWidth * ( nodePos[ 0 ] + 1 ); voxelPos[ 0 ] +=2 )
			{
				float voxelDataDOWNf[ 4 ] = { 0.f, 0.f, 0.f, 0.f };
				float voxelDataDOWNf2[ 4 ] = { 0.f, 0.f, 0.f, 0.f };

				// As the underlying structure is an octree,
				// to compute data at coaser level,
				// we need to iterate through 8 voxels and take the mean value.
				for ( unsigned int z = 0; z < 2; z++ )
				for ( unsigned int y = 0; y < 2; y++ )
				for ( unsigned int x = 0; x < 2; x++ )
				{
					// Retrieve position of voxel in the UP resolution version
					unsigned int voxelPosUP[ 3 ];
					voxelPosUP[ 0 ] = voxelPos[ 0 ] + x;
					voxelPosUP[ 1 ] = voxelPos[ 1 ] + y;
					voxelPosUP[ 2 ] = voxelPos[ 2 ] + z;

					// Get associated data (in the UP resolution version)
					unsigned char voxelDataUP[ 4 ];
					dataStructureIOHandlerUP->getVoxel( voxelPosUP, voxelDataUP, 0 );
					voxelDataDOWNf[ 0 ] += voxelDataUP[ 0 ];
					voxelDataDOWNf[ 1 ] += voxelDataUP[ 1 ];
					voxelDataDOWNf[ 2 ] += voxelDataUP[ 2 ];
					voxelDataDOWNf[ 3 ] += voxelDataUP[ 3 ];

					unsigned short voxelNormalUP[ 4 ];
					if (_normals)
					{
						// Get associated normal (in the UP resolution version)
						dataStructureIOHandlerUP->getVoxel( voxelPosUP, voxelNormalUP, 1 );
						voxelDataDOWNf2[ 0 ] += halfInUshort2Float(voxelNormalUP[ 0 ]) ;
						voxelDataDOWNf2[ 1 ] += halfInUshort2Float(voxelNormalUP[ 1 ]) ;
						voxelDataDOWNf2[ 2 ] += halfInUshort2Float(voxelNormalUP[ 2 ]) ;
						//voxelDataDOWNf2[ 3 ] += 0.f;
					}
				}

				// Coarser voxel is scaled from current UP voxel (2 times smaller for octree)
				unsigned int voxelPosDOWN[3];
				voxelPosDOWN[ 0 ] = voxelPos[ 0 ] / 2;
				voxelPosDOWN[ 1 ] = voxelPos[ 1 ] / 2;
				voxelPosDOWN[ 2 ] = voxelPos[ 2 ] / 2;

				// Set data in coarser voxel
				unsigned char vd[4];		// "vd" stands for "voxel data"
				vd[ 0 ] = float2uchar ( voxelDataDOWNf[ 0 ] / 8.f );
				vd[ 1 ] = float2uchar ( voxelDataDOWNf[ 1 ] / 8.f );
				vd[ 2 ] = float2uchar ( voxelDataDOWNf[ 2 ] / 8.f );
				vd[ 3 ] = float2uchar ( voxelDataDOWNf[ 3 ] / 8.f );
				dataStructureIOHandlerDOWN->setVoxel( voxelPosDOWN, vd, 0 );


				if (_normals)
				{
					unsigned short vd2[4];
					// Set normal in coarser voxel
					float norm = sqrtf( voxelDataDOWNf2[ 0 ] * voxelDataDOWNf2[ 0 ] + voxelDataDOWNf2[ 1 ] * voxelDataDOWNf2[ 1 ] + voxelDataDOWNf2[ 2 ] * voxelDataDOWNf2[ 2 ] );
					if ( norm < 0.00001 ) // check EPSILLON value to avoid "div by 0"
					{
						vd2[ 0 ] = 0;
						vd2[ 1 ] = 0;
						vd2[ 2 ] = 0;
						vd2[ 3 ] = 0; // => not sure about that one ?
					}
					else
					{
						vd2[ 0 ] = float2HalfInUshort( voxelDataDOWNf2[ 0 ] / norm );
						vd2[ 1 ] = float2HalfInUshort( voxelDataDOWNf2[ 1 ] / norm );
						vd2[ 2 ] = float2HalfInUshort( voxelDataDOWNf2[ 2 ] / norm );
						vd2[ 3 ] = 1;
					}
					dataStructureIOHandlerDOWN->setVoxel( voxelPosDOWN, vd2, 1 );
				}
			}
		}
		}



		// Generate the border data of the coarser scene
		dataStructureIOHandlerDOWN->computeBorders();




		// Destroy the coarser data handler (due to memory consumption considerations)
		delete dataStructureIOHandlerUP;

		// The mip-map pyramid hierarchy is built recursively from adjacent levels.
		// Now that the coarser version has been generated, a coarser one need to be generated from it.
		// So, the coarser one is the UP version.
		dataStructureIOHandlerUP = dataStructureIOHandlerDOWN;
	}

	// Free memory
	delete dataStructureIOHandlerDOWN;
}
