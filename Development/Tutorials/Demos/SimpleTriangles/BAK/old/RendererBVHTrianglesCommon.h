/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2012 INRIA <http://www.inria.fr/>
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

#ifndef _VOXEL_SCENE_RENDERER_COMMON_H_
#define _VOXEL_SCENE_RENDERER_COMMON_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

//#include <GL/glew.h>
//#include <GL/glut.h>

//#include <vector_types.h>
//#include <vector_functions.h>
//
//#include <cutil.h>
//#include <cutil_math.h>

//#include <nvMatrix.h> //crash cudrt !

//#include <stdlib.h>
//#include <stdio.h>
//#include <string.h>
//#include <vector>
//#include <inttypes.h>

//#include "vector_types_ext.h"
//#include "Array3D.h"
//#include "Array3DKernel.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

using namespace GvCore;

//-maxrregcount=64
//-arch compute_12 -code sm_12


#define USE_SURFACE 0
#define USE_NEW_VOLTREE_INTERF 0 //encapsulation within classes
#define USE_NEW_RENDERER 1 //New rendering tests


#define VOXEL_TYPE_FLOAT 0
#define VOXEL_PREMULTALPHA 0

//Enable storing TAU
#define VOXEL_STORE_TAU 0
#define VOXEL_DEFAULT_RAYSTEP 0.0004f


#define USE_SHADOWS 0
#define USE_DOF 0
#define USE_REFLECTIONS 0		//Not compatible with deffered shading
#define RENDER_USE_FREE_RAYS 0
#define RENDER_USE_HIGHER_BRICKS  1
#define RENDER_USE_MIPMAPS  0
#define RENDER_USE_TRANSFER_FUNCTION  1
#define GPUPRODUCER_EVALUATE_TRANSFER_FUNCTION  0 //BETA

#define USE_DEBUGDISPLAY 1
#define DEBUG_USE_CUPRINTF 0

//New AO system (Mandelbulb)
#define RENDER_COMPUTE_AO  0
#define USE_PRECOMPUTED_AO  0

#define SAMPLESTEP_FACTOR 0.3333333f

#define USE_SCENE_TREE 0				//Forced false in NEW PATH
#define USE_SEPARATE_BVH 0

#define USE_GPUCACHE 1					//Forced true in NEW PATH
#define USE_GPUCACHE_GPULOCALIZATIONINFO 1

#define USE_GPUCACHE_SUBDIVISION_AS_LOAD 1
#define USE_GPUCACHE_DYNAMICDATA 1			//New flag for dynamic load producer

#define USE_GPUFETCHDATA 1

#define USE_BRICKS_TRANSFER 0			//Forced false in NEW PATH
#define MAX_NUM_BRICKS_TRANSFER 5000

#define USE_GPUPOOLS 1

#define SYNTHETIC_REDUC_FACTOR 2
#define USE_SYNTHETIC_INFO 1


// -maxrregcount=32
#define PRODUCER_CACHE_DATA 0
#define USE_GRADIENT_CACHE 0

#define VT_NODE_RES 2
#define VT_NODE_RES_POT 1 //power of two of the node res
#define VT_BRICKS_RES 16 //14 //8 for surfaces
#define VT_BRICKS_BORDER_SIZE 1
#define VT_BRICKS_RES_WITH_BORDER ( VT_BRICKS_RES + 2 * VT_BRICKS_BORDER_SIZE )
#define VT_NODE_RES_SIZE ( VT_NODE_RES * VT_NODE_RES * VT_NODE_RES )

#define VOL_TREE_POOL_RES (128)
#define BRICKS_POOL_RES (512)
//#define BRICKS_POOL_RES (512, 256, 256)



////VOLTREE////
#define USE_LINEAR_VOLTREE_ADDRESS 1	 //new path  //Forced true in producer dynamic
#define USE_LINEAR_VOLTREE 1			 //Forced true in NEW PATH
#define USE_LINEAR_VOLTREE_TEX 1
#define USE_LINEAR_VOLTREE_TILE 0		//Forced false in NEW PATH

//#define USE_SEPARATE_VOLTREE 1			//Forced true in NEW PATH

////BRICKS////
#define BRICKS_INTERP 1

#define USE_LINEAR_BRICKPOOL 0
#define USE_LINEAR_BRICKPOOL_TEX 0
#define USE_LINEAR_BRICKPOOL_TILE 0


////VOLTREE RENDER////
#define NUM_RAYS_PER_BLOCK_X 8
#define NUM_RAYS_PER_BLOCK_Y 4

#define OCTREE_TRAVERSAL_STACK_SIZE 32
#define OCTREE_PACKET_USE_BLOCKFETCH 1
//#define OCTREE_PACKET_USE_TEXFETCH 1

#define OCTREE_DISABLE_BRICK_SAMPLING 0
#define OCTREE_DISABLE_BRICK_SAMPLING_ALPHA 1.00f

#define OCTREE_PACKETSTACK_POINTLOC 0
#define OCTREE_PACKETRESTART_PACKET 0

#define OCTREE_TRAVERSAL_USE_LOCCODE 1

////Deferred Shading////
#define DEFERRED_NUM_THREADS_PER_BLOCK_X 8
#define DEFERRED_NUM_THREADS_PER_BLOCK_Y 8

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

//Global current time
//extern uint currentTime;


#if VOXEL_TYPE_FLOAT
__host__ __device__
inline voxeltype make_voxeltype(float x){
	voxeltype res;
	res.x = x;
	res.y = x;
	res.z = x;
	res.w = x;

	return res;
}

inline voxeltype make_voxeltype(const float4 &v){
	return v;
}
#else //uchar
/*__host__ __device__
inline voxeltype make_voxeltype(float v){
	voxeltype res;
	res.x = (uchar)(v*255.0f);
	res.y = (uchar)(v*255.0f);
	res.z = (uchar)(v*255.0f);
	res.w = (uchar)(v*255.0f);

	return res;
}

inline voxeltype make_voxeltype(const float4 &v){
	voxeltype res;
	res.x = (uchar)(v.x*255.0f);
	res.y = (uchar)(v.y*255.0f);
	res.z = (uchar)(v.z*255.0f);
	res.w = (uchar)(v.w*255.0f);

	return res;
}*/
#endif

inline bool VoxelDiscriminator( const uchar4& vox )
{
	return vox.w > 0;
}

inline bool VoxelDiscriminator( const float4& vox )
{
	return vox.w > 0.0f;
}

inline bool VoxelDiscriminator( const uchar& vox )
{
	return vox > 100; //140;
}

/**
 * Load raw data from disk
 *
 * @param filename ...
 * @param size ...
 * @param data ...
 *
 * @return ...
 */
inline uchar* loadRawFile( char* filename, size_t size, uchar* data = NULL )
{
	FILE* fp = fopen( filename, "rb" );
	if ( !fp )
	{
		fprintf( stderr, "Error opening file '%s'\n", filename );
		return 0;
	}

	if ( !data )
	{
		data = (uchar *) malloc( size );
	}
	size_t read = fread( data, 1, size, fp );
	fclose( fp );

	printf( "Read '%s', %lu bytes\n", filename, (long unsigned)read );

	return data;
}

/**
 * Write raw data to disk
 *
 * @param filename ...
 * @param data ...
 * @param size ...
 */
inline void saveRawFile( char* filename, uchar* data, size_t size )
{
	FILE* fp = fopen( filename, "wb" );
	if ( !fp )
	{
		fprintf( stderr, "Error opening file '%s'\n", filename );
		return;
	}
	
	size_t write = fwrite( data, 1, size, fp );
	fclose( fp );

	printf( "Write '%s', %lu bytes\n", filename, (long unsigned)write );
}

/**
 * ...
 *
 * @param mat ...
 *
 * @return ...
 */
inline float matrixDet( float (&mat)[16] )
{
	float det;
	det = mat[0] * mat[5] * mat[10];
	det += mat[4] * mat[9] * mat[2];
	det += mat[8] * mat[1] * mat[6];
	det -= mat[8] * mat[5] * mat[2];
	det -= mat[4] * mat[1] * mat[10];
	det -= mat[0] * mat[9] * mat[6];

	return det;
}

/**
 * ...
 *
 * @param mat ...
 * @param ret ...
 */
inline void matrixInverse( float (&mat)[16], float (&ret)[16] )
{
	float idet = 1.0f / matrixDet(mat);
	ret[0] =  (mat[5] * mat[10] - mat[9] * mat[6]) * idet;
	ret[1] = -(mat[1] * mat[10] - mat[9] * mat[2]) * idet;
	ret[2] =  (mat[1] * mat[6] - mat[5] * mat[2]) * idet;
	ret[3] = 0.0;
	ret[4] = -(mat[4] * mat[10] - mat[8] * mat[6]) * idet;
	ret[5] =  (mat[0] * mat[10] - mat[8] * mat[2]) * idet;
	ret[6] = -(mat[0] * mat[6] - mat[4] * mat[2]) * idet;
	ret[7] = 0.0;
	ret[8] =  (mat[4] * mat[9] - mat[8] * mat[5]) * idet;
	ret[9] = -(mat[0] * mat[9] - mat[8] * mat[1]) * idet;
	ret[10] =  (mat[0] * mat[5] - mat[4] * mat[1]) * idet;
	ret[11] = 0.0;
	ret[12] = -(mat[12] * ret[0] + mat[13] * ret[4] + mat[14] * ret[8]);
	ret[13] = -(mat[12] * ret[1] + mat[13] * ret[5] + mat[14] * ret[9]);
	ret[14] = -(mat[12] * ret[2] + mat[13] * ret[6] + mat[14] * ret[10]);
	ret[15] = 1.0;
}

/**
 * ...
 *
 * @param mat ...
 * @param v ...
 *
 * @return ...
 */
inline float4 matrixMul( float (&mat)[16], float4 v )
{
	float4 ret;
	ret.x = mat[0] * v.x + mat[4] * v.y + mat[8] * v.z + mat[12] * v.w;
	ret.y = mat[1] * v.x + mat[5] * v.y + mat[9] * v.z + mat[13] * v.w;
	ret.z = mat[2] * v.x + mat[6] * v.y + mat[10] * v.z + mat[14] * v.w;
	ret.w = mat[3] * v.x + mat[7] * v.y + mat[11] * v.z + mat[15] * v.w;

	return ret;
}

/**
 * ...
 */
#define GET_GLERROR()										  \
{																 \
	GLenum err = glGetError();									\
	if ( err != GL_NO_ERROR ) {									 \
	fprintf( stderr, "[line %d] GL Error: %s\n",				\
	__LINE__, gluErrorString( err ) );					 \
	fflush( stderr );											   \
	}															 \
}

/**
 * Draw cube
 *
 * @param p1 ...
 * @param p2 ...
 */
inline void drawCube( const float3& p1, const float3& p2 )
{
	glBegin( GL_QUADS );
		// Front Face
		glVertex3f( p1.x, p1.y, p2.z);	// Bottom Left Of The Texture and Quad
		glVertex3f( p1.x, p2.y, p2.z);	// Top Left Of The Texture and Quad
		glVertex3f( p2.x, p2.y, p2.z);	// Top Right Of The Texture and Quad
		glVertex3f( p2.x, p1.y, p2.z);	// Bottom Right Of The Texture and Quad

		// Back Face
		glVertex3f( p1.x, p1.y, p1.z);	// Bottom Right Of The Texture and Quad
		glVertex3f( p2.x, p1.y, p1.z);	// Bottom Left Of The Texture and Quad
		glVertex3f( p2.x, p2.y, p1.z);	// Top Left Of The Texture and Quad
		glVertex3f( p1.x, p2.y, p1.z);	// Top Right Of The Texture and Quad

		// Top Face
		glVertex3f( p1.x, p2.y, p1.z);	// Top Left Of The Texture and Quad
		glVertex3f( p2.x, p2.y, p1.z);	// Top Right Of The Texture and Quad
		glVertex3f( p2.x, p2.y, p2.z);	// Bottom Right Of The Texture and Quad
		glVertex3f( p1.x, p2.y, p2.z);	// Bottom Left Of The Texture and Quad

		// Bottom Face
		glVertex3f( p1.x, p1.y, p1.z);	// Top Right Of The Texture and Quad
		glVertex3f( p1.x, p1.y, p2.z);	// Bottom Right Of The Texture and Quad
		glVertex3f( p2.x, p1.y, p2.z);	// Bottom Left Of The Texture and Quad
		glVertex3f( p2.x, p1.y, p1.z);	// Top Left Of The Texture and Quad

		// Right face
		glVertex3f( p2.x, p1.y, p1.z);	// Bottom Right Of The Texture and Quad
		glVertex3f( p2.x, p1.y, p2.z);	// Bottom Left Of The Texture and Quad
		glVertex3f( p2.x, p2.y, p2.z);	// Top Left Of The Texture and Quad
		glVertex3f( p2.x, p2.y, p1.z);	// Top Right Of The Texture and Quad

		// Left Face
		glVertex3f( p1.x, p1.y, p1.z);	// Bottom Left Of The Texture and Quad
		glVertex3f( p1.x, p2.y, p1.z);	// Top Left Of The Texture and Quad
		glVertex3f( p1.x, p2.y, p2.z);	// Top Right Of The Texture and Quad
		glVertex3f( p1.x, p1.y, p2.z);	// Bottom Right Of The Texture and Quad

	glEnd();
}

/**
 * Draw cube
 *
 * @param p1 ...
 * @param p2 ...
 */
inline void drawCubeQuadPrimStarted( const float3& p1, const float3& p2 )
{
		// Front Face
		glVertex3f(p1.x, p1.y,  p2.z);	// Bottom Left Of The Texture and Quad
		glVertex3f(p1.x,  p2.y,  p2.z);	// Top Left Of The Texture and Quad
		glVertex3f( p2.x,  p2.y,  p2.z);	// Top Right Of The Texture and Quad
		glVertex3f( p2.x, p1.y,  p2.z);	// Bottom Right Of The Texture and Quad

		// Back Face
		glVertex3f(p1.x, p1.y, p1.z);	// Bottom Right Of The Texture and Quad
		glVertex3f( p2.x, p1.y, p1.z);	// Bottom Left Of The Texture and Quad
		glVertex3f( p2.x,  p2.y, p1.z);	// Top Left Of The Texture and Quad
		glVertex3f(p1.x,  p2.y, p1.z);	// Top Right Of The Texture and Quad

		// Top Face
		glVertex3f(p1.x,  p2.y, p1.z);	// Top Left Of The Texture and Quad
		glVertex3f( p2.x,  p2.y, p1.z);	// Top Right Of The Texture and Quad
		glVertex3f( p2.x,  p2.y,  p2.z);	// Bottom Right Of The Texture and Quad
		glVertex3f(p1.x,  p2.y,  p2.z);	// Bottom Left Of The Texture and Quad

		// Bottom Face
		glVertex3f(p1.x, p1.y, p1.z);	// Top Right Of The Texture and Quad
		glVertex3f(p1.x, p1.y,  p2.z);	// Bottom Right Of The Texture and Quad
		glVertex3f( p2.x, p1.y,  p2.z);	// Bottom Left Of The Texture and Quad
		glVertex3f( p2.x, p1.y, p1.z);	// Top Left Of The Texture and Quad

		// Right face
		glVertex3f( p2.x, p1.y, p1.z);	// Bottom Right Of The Texture and Quad
		glVertex3f( p2.x, p1.y,  p2.z);	// Bottom Left Of The Texture and Quad
		glVertex3f( p2.x,  p2.y,  p2.z);	// Top Left Of The Texture and Quad
		glVertex3f( p2.x,  p2.y, p1.z);	// Top Right Of The Texture and Quad

		// Left Face
		glVertex3f(p1.x, p1.y, p1.z);	// Bottom Left Of The Texture and Quad
		glVertex3f(p1.x,  p2.y, p1.z);	// Top Left Of The Texture and Quad
		glVertex3f(p1.x,  p2.y,  p2.z);	// Top Right Of The Texture and Quad
		glVertex3f(p1.x, p1.y,  p2.z);	// Bottom Right Of The Texture and Quad
}

////Shared memory utility////
#if 0
template< typename T, int BSX, int BSY >
class KernelVarInSharedMemPtr
{

public:
	
	T val[ BSX ][ BSY ];

	__device__
	T& operator*()
	{
		return val[ threadIdx.x ][ threadIdx.y ];
	}
	__device__
	T* operator->()
	{
		return &val[ threadIdx.x ][ threadIdx.y ];
	}
	__device__
	T* operator()()
	{
		return &val[ threadIdx.x ][ threadIdx.y ];
	}

};
#endif

#if 0
__device__ __host__
inline void compZCurve2D(uint val, uint2 &res){ //max val: 31x31

	res.x=0;
	res.y=0;

	res.x|=val&1;
	res.y|=(val&2)>>1;

	res.x|=(val&4)>>1;
	res.y|=(val&8)>>2;

	res.x|=(val&16)>>2;
	res.y|=(val&32)>>3;

	res.x|=(val&64)>>3;
	res.y|=(val&128)>>4;

	res.x|=(val&256)>>4;
	res.y|=(val&512)>>5;

	//for blocks
	res.x|=(val&1024)>>5;
	res.y|=(val&2048)>>6;

	res.x|=(val&4096)>>6;
	res.y|=(val&8192)>>7;

	res.x|=(val&16384)>>7;
	res.y|=(val&32768)>>8;
}
#endif

/**
 * ...
 */
__device__ __host__
inline void decodeZCurve2D( uint2& pos, uint& val )
{ // max val: 31x31
	val = 0;
	val |= pos.x&1;
	val |= (pos.y&1)<<1;

	val |= (pos.x&2)<<1;
	val |= (pos.y&2)<<2;

	val |= (pos.x&4)<<2;
	val |= (pos.y&4)<<3;

	val |= (pos.x&8)<<3;
	val |= (pos.y&8)<<4;

	val |= (pos.x&16)<<4;
	val |= (pos.y&16)<<5;

	val |= (pos.x&32)<<5;
	val |= (pos.y&32)<<6;

	val |= (pos.x&64)<<6;
	val |= (pos.y&64)<<7;
}

/**
 * ...
 */
__device__ __host__
inline uint interleaveBits( uint3 input )
{	//New version !
	uint res;

	input.x = (input.x | (input.x << 16)) & 0x030000FF;
	input.x = (input.x | (input.x <<  8)) & 0x0300F00F;
	input.x = (input.x | (input.x <<  4)) & 0x030C30C3;
	input.x = (input.x | (input.x <<  2)) & 0x09249249;

	input.y = (input.y | (input.y << 16)) & 0x030000FF;
	input.y = (input.y | (input.y <<  8)) & 0x0300F00F;
	input.y = (input.y | (input.y <<  4)) & 0x030C30C3;
	input.y = (input.y | (input.y <<  2)) & 0x09249249;

	input.z = (input.z | (input.z << 16)) & 0x030000FF;
	input.z = (input.z | (input.z <<  8)) & 0x0300F00F;
	input.z = (input.z | (input.z <<  4)) & 0x030C30C3;
	input.z = (input.z | (input.z <<  2)) & 0x09249249;

	res= input.x | (input.y << 1) | (input.z << 2);

	return res;
}

/**
 * ...
 */
#ifndef WIN32
	#define _fseeki64 fseek
	#define _ftelli64 ftell
	typedef long long long64;
#else
	typedef __int64 long64;
#endif

/**
 * MACRO used to copy any symbol to constant memory on device
 */
#define CUDAUploadConstant( constName, hostVariable ) \
	{ CUDA_SAFE_CALL( cudaMemcpyToSymbol( constName, &hostVariable, sizeof( hostVariable ), 0, cudaMemcpyHostToDevice ) ); }

#endif // !_VOXEL_SCENE_RENDERER_COMMON_H_
