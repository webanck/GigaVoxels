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

#ifndef _GV_VECTOR_TYPES_EXT_H_
#define _GV_VECTOR_TYPES_EXT_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include "GvCore/GvCoreConfig.h"

#include <cuda.h>	// without this include, Linux has problem to compile. TO DO : resolve this.
#include <device_functions.h> // float <-> half
#include <vector_types.h>
#include <vector_functions.h>
//#include <driver_types.h>

// CUDA SDK
#include <helper_math.h>

// System
#include <cmath>
//#include <stdint.h>
#include <iostream>

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

typedef unsigned int uint;
typedef unsigned char uchar;

struct /*__align__(4)*/ half2
{
	unsigned short x, y;
};

struct /*__align__(8)*/ half4
{
	unsigned short x, y, z, w;
};

#ifndef maxcc
#define maxcc(a,b)			(((a) > (b)) ? (a) : (b))
#endif

#ifndef mincc
#define mincc(a,b)			(((a) < (b)) ? (a) : (b))
#endif

#if 0
__host__
static inline uint32_t cclog2(const uint32_t x) {
  uint32_t y;
  asm ( "\tbsr %1, %0\n"
	  : "=r"(y)
	  : "r" (x)
  );
  return y;
}

#else

inline unsigned int cclog2(unsigned int value)
{
	unsigned int l = 0;
	while( (value >> l) > 1 ) ++l;
	return l;
}
#endif

//Kernel functions !
#if USE_TESLA_OPTIMIZATIONS == 1
# define __imul(a, b)	__mul24(a, b)
# define __uimul(a, b)	__umul24(a, b)
#else
# define __imul(a, b)	((a) * (b))
# define __uimul(a, b)	((a) * (b))
#endif

/**
 * ...
 */
struct NullType
{
};

/**
 * ...
 */
__device__ __host__
inline int iDivUp( int a, int b )
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

////////////////////////////////////////////////////////////////////////////////
// constructors
////////////////////////////////////////////////////////////////////////////////

/**
 * ...
 */
template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<half2>(void)
{
	return cudaCreateChannelDescHalf2();
}

template<> __inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<half4>(void)
{
	return cudaCreateChannelDescHalf4();
}

#ifdef __CUDA_ARCH__

__device__
__forceinline__ half2 make_half2(float x, float y)
{
	half2 t;

	t.x = __float2half_rn(x);
	t.y = __float2half_rn(y);

	return t;
}

__device__
__forceinline__ half4 make_half4(float x, float y, float z, float w)
{
	half4 t;

	t.x = __float2half_rn(x);
	t.y = __float2half_rn(y);
	t.z = __float2half_rn(z);
	t.w = __float2half_rn(w);

	return t;
}

#else

// FIXME: is there something more reliable ?

inline unsigned short __float2half_rn_host(float f)
{
	unsigned int x = *((unsigned int *)&f);
	unsigned int u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
	unsigned int sign, exponent, mantissa;

	if (u > 0x7f800000)
		return 0x7fff;

	sign = ((x >> 16) & 0x8000);

	if (u > 0x477fefff)
		return sign | 0x7c00;
	
	if (u < 0x33000001)
		return sign | 0x0000;

	exponent = ((u >> 23) & 0xff);
	mantissa = (u & 0x7fffff);

	if (exponent > 0x70) {
		shift = 13;
		exponent -= 0x70;
	} else {
		shift = 0x7e - exponent;
		exponent = 0;
		mantissa |= 0x800000;
	}
	lsb = (1 << shift);
	lsb_s1 = (lsb >> 1);
	lsb_m1 = (lsb - 1);

	remainder = (mantissa & lsb_m1);
	mantissa >>= shift;
	if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
		++mantissa;
		if (!(mantissa & 0x3ff)) {
			++exponent;
			mantissa = 0;
		}
	}

	return sign | (exponent << 10) | mantissa;
}

inline half2 make_half2(float x, float y)
{
	half2 t;

	t.x = __float2half_rn_host(x);
	t.y = __float2half_rn_host(y);

	return t;
}

inline half4 make_half4(float x, float y, float z, float w)
{
	half4 t;

	t.x = __float2half_rn_host(x);
	t.y = __float2half_rn_host(y);
	t.z = __float2half_rn_host(z);
	t.w = __float2half_rn_host(w);

	return t;
}

#endif

__host__ __device__
inline dim3 make_dim3(cudaExtent v){
	dim3 res;
	res.x = static_cast<unsigned int>(v.width);
	res.y = static_cast<unsigned int>(v.height);
	res.z = static_cast<unsigned int>(v.depth);

	return res;
}
__host__ __device__
inline dim3 make_dim3(const uint3 &v){
	dim3 res;
	res.x = (unsigned int)v.x;
	res.y = (unsigned int)v.y;
	res.z = (unsigned int)v.z;

	return res;
}

__host__ __device__
inline uint3 make_uint3(const dim3 &v){
	uint3 res;
	res.x = (unsigned int)v.x;
	res.y = (unsigned int)v.y;
	res.z = (unsigned int)v.z;

	return res;
}

//__host__ __device__
//inline uint3 make_uint3(const int3 &v){
//	uint3 res;
//	res.x = (unsigned int)v.x;
//	res.y = (unsigned int)v.y;
//	res.z = (unsigned int)v.z;
//
//	return res;
//}

__host__ __device__
inline uint3 make_uint3(const cudaExtent &v){
	uint3 res;
	res.x = (unsigned int)v.width;
	res.y = (unsigned int)v.height;
	res.z = (unsigned int)v.depth;

	return res;
}

__host__ __device__
inline uint3 make_uint3(float3 a)
{
	return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

//__host__ __device__
//inline int3 make_int3(const uint3 &v){
//	int3 res;
//	res.x = (int)v.x;
//	res.y = (int)v.y;
//	res.z = (int)v.z;
//
//	return res;
//}

__host__ __device__
inline uchar3 make_uchar3(const float3 &v){
	uchar3 res;
	res.x = (uchar)v.x;
	res.y = (uchar)v.y;
	res.z = (uchar)v.z;

	return res;
}
__host__ __device__
inline uchar3 make_uchar3(const uint3 &v){
	uchar3 res;
	res.x = (uchar)v.x;
	res.y = (uchar)v.y;
	res.z = (uchar)v.z;

	return res;
}


__host__ __device__
inline cudaExtent make_cudaExtent(const dim3 &v){
	cudaExtent res;
	res.width	= (unsigned int)v.x;
	res.height	= (unsigned int)v.y;
	res.depth	= (unsigned int)v.z;

	return res;
}


/*__host__ __device__
inline float3 make_float3(const int3 &v){
	float3 res;
	res.x=(float)v.x;
	res.y=(float)v.y;
	res.z=(float)v.z;

	return res;
}*/

//__host__ __device__
//inline float3 make_float3(const uint3 &v){
//	float3 res;
//	res.x=(float)v.x;
//	res.y=(float)v.y;
//	res.z=(float)v.z;
//
//	return res;
//}

__host__ __device__
inline float3 make_float3(const uchar3 &v){
	float3 res;
	res.x=(float)v.x;
	res.y=(float)v.y;
	res.z=(float)v.z;

	return res;
}

__host__ __device__
inline float4 make_float4(const float4 &v){
	return v;
}

__host__ __device__
inline float4 make_float4(const char4 &v){
	float4 res;
	res.x=(float)v.x;
	res.y=(float)v.y;
	res.z=(float)v.z;
	res.w=(float)v.w;

	return res;
}

__host__ __device__
inline float4 make_float4(const uchar4 &v){
	float4 res;
	res.x=(float)v.x;
	res.y=(float)v.y;
	res.z=(float)v.z;
	res.w=(float)v.w;

	return res;
}

__host__ __device__
inline float4 make_float4(const short4 &v)
{
	return make_float4((float)v.x, (float)v.y, (float)v.z, (float)v.w);
}

__host__ __device__
inline float4 make_float4(const ushort4 &v)
{
	return make_float4((float)v.x, (float)v.y, (float)v.z, (float)v.w);
}

////////////////////////////////////////////////////////////////////////////////
// operators
////////////////////////////////////////////////////////////////////////////////

inline dim3 operator/(const dim3 &d1, const dim3 &d2){
	dim3 res;
	res.x=d1.x/d2.x;
	res.y=d1.y/d2.y;
	res.z=d1.z/d2.z;

	return res;
}

inline bool operator==(const dim3 &d1, const dim3 &d2){

	return d1.x==d2.x && d1.y==d2.y && d1.z==d2.z ;
}

__host__ __device__
inline uint2 operator/(const uint2 &d1, const uint2 &d2){
	uint2 res;
	res.x=d1.x/d2.x;
	res.y=d1.y/d2.y;

	return res;
}

__host__ __device__
inline uint2 operator/(const uint2 &a, const uint &b)
{
	return make_uint2(a.x / b, a.y / b);
}

__host__ __device__
inline uint3 operator/(uint3 a, uint b)
{
	return make_uint3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__
inline uint3 operator/(uint3 a, uint3 b)
{
	return make_uint3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__ __device__
inline uint3 operator>>(const uint3 &v, const uint &d){
	uint3 res;
	res.x=v.x>>d;
	res.y=v.y>>d;
	res.z=v.z>>d;

	return res;
}
__host__ __device__
inline uint3 operator<<(const uint3 &v, const uint &d){
	uint3 res;
	res.x=v.x<<d;
	res.y=v.y<<d;
	res.z=v.z<<d;

	return res;
}

__host__ /*__device__*/
inline std::ostream &operator<<(std::ostream &output, const uint3 &d){
	output<<"("<<d.x<<", "<<d.y<<", "<<d.z<<")";

	return output;
}

__host__ /*__device__*/
inline std::ostream &operator<<(std::ostream &output, const float4 &d){
	output<<"("<<d.x<<", "<<d.y<<", "<<d.z<<", "<<d.w<<")";

	return output;
}
__host__ /*__device__*/
inline std::ostream &operator<<(std::ostream &output, const float3 &d){
	output<<"("<<d.x<<", "<<d.y<<", "<<d.z<<")";

	return output;
}
__host__ /*__device__*/
inline std::ostream &operator<<(std::ostream &output, const uchar4 &d){
	output<<"("<<(int)d.x<<", "<<(int)d.y<<", "<<(int)d.z<<", "<<(int)d.w<<")";

	return output;
}


__host__ __device__
inline uint3 operator&(const uint3 &v1, const uint3 &v2){
	uint3 res;
	res.x=v1.x & v2.x;
	res.y=v1.y & v2.y;
	res.z=v1.z & v2.z;

	return res;
}

__host__ __device__
inline uint3 operator&(const uint3 &v1, const uint &d){
	uint3 res;
	res.x=v1.x & d;
	res.y=v1.y & d;
	res.z=v1.z & d;

	return res;
}

__host__ __device__
inline uint3 operator|(const uint3 &v1, const uint3 &v2){
	uint3 res;
	res.x=v1.x | v2.x;
	res.y=v1.y | v2.y;
	res.z=v1.z | v2.z;

	return res;
}
__host__ __device__
inline uint3 operator|(const uint3 &v1, const uint &d){
	uint3 res;
	res.x=v1.x | d;
	res.y=v1.y | d;
	res.z=v1.z | d;

	return res;
}


//__host__ __device__
//inline float4 operator*(const float4 &d1, const float4 &d2){
//	float4 res;
//	res.x=d1.x*d2.x;
//	res.y=d1.y*d2.y;
//	res.z=d1.z*d2.z;
//	res.w=d1.z*d2.w;
//
//	return res;
//}


__host__ __device__
inline float3 min(const float3 &d1, const float3 &d2){
	float3 res;
	res.x=mincc(d1.x, d2.x);
	res.y=mincc(d1.y, d2.y);
	res.z=mincc(d1.z, d2.z);

	return res;
}
__host__ __device__
inline float3 max(const float3 &d1, const float3 &d2){
	float3 res;
	res.x=maxcc(d1.x, d2.x);
	res.y=maxcc(d1.y, d2.y);
	res.z=maxcc(d1.z, d2.z);

	return res;
}

////////////////////////////////////////////////////////////////////////////////
// conversions
////////////////////////////////////////////////////////////////////////////////

template<class T>
__host__ __device__
inline NullType &convert_type(T v, NullType &res){
	return res;
}
template<class T>
__host__ __device__
inline T &convert_type(NullType v, T &res){
	return res;
}


template<class T>
__host__ __device__
inline T &convert_type(T in, T &res){
	res=in;
	return res;
}

__host__ __device__
inline char4 &convert_type(float4 v, char4 &res){
	res.x = (char)floorf((v.x < 0.0f ? v.x * 128.0f : v.x * 127.0f) + 0.5f);
	res.y = (char)floorf((v.y < 0.0f ? v.y * 128.0f : v.y * 127.0f) + 0.5f);
	res.z = (char)floorf((v.z < 0.0f ? v.z * 128.0f : v.z * 127.0f) + 0.5f);
	res.w = (char)floorf((v.w < 0.0f ? v.w * 128.0f : v.w * 127.0f) + 0.5f);
	return res;
}

__host__ __device__
inline uchar4 &convert_type(float4 v, uchar4 &res){
	res.x = (uchar)ceil(v.x*255.0f);
	res.y = (uchar)ceil(v.y*255.0f);
	res.z = (uchar)ceil(v.z*255.0f);
	res.w = (uchar)ceil(v.w*255.0f);

	return res;
}

__host__ __device__
inline uchar4 &convert_type(float3 v, uchar4 &res){
	res.x = (uchar)ceil(v.x*255.0f);
	res.y = (uchar)ceil(v.y*255.0f);
	res.z = (uchar)ceil(v.z*255.0f);
	res.w = (uchar)ceil( ((v.x+v.y+v.z)/3.0f)*255.0f);

	return res;
}

__host__ __device__
inline uchar &convert_type(float3 v, uchar &res){
	res = (uchar)ceil(v.x*255.0f);

	return res;
}


__host__ __device__
inline uchar &convert_type(float4 v, uchar &res){
	res = (uchar)ceil(v.w*255.0f);

	return res;
}
__host__ __device__
inline float4 &convert_type(float4 v, float4 &res){
	res=v;
	return res;
}
__host__ __device__
inline float &convert_type(uchar v, float &res){
	res = (float)(v/255.0f);

	return res;
}
__host__ __device__
inline float4 &convert_type(char4 v, float4 &res){
	res.x = (v.x < 0 ? (float)v.x / 128.0f : (float)v.x / 127.0f);
	res.y = (v.y < 0 ? (float)v.y / 128.0f : (float)v.y / 127.0f);
	res.z = (v.z < 0 ? (float)v.z / 128.0f : (float)v.z / 127.0f);
	res.w = (v.w < 0 ? (float)v.w / 128.0f : (float)v.w / 127.0f);
	return res;
}
__host__ __device__
inline float4 &convert_type(uchar4 v, float4 &res){
	res.x = (float)(v.x/255.0f);
	res.y = (float)(v.y/255.0f);
	res.z = (float)(v.z/255.0f);
	res.w = (float)(v.w/255.0f);

	return res;
}

// TO DO
// - check what types of data we need to handle
/*__host__ __device__
inline float4 &convert_type(ushort v, float4 &res){
	res.x = (float)(v/65535.f);
	res.y = 0.f;
	res.z = 0.f;
	res.w = 0.f;

	return res;
}
*/

__host__ __device__
inline float &convert_type(uchar4 v, float &res){
	res = (float)(v.w/255.0f);

	return res;
}

__host__ __device__
inline float &convert_type(float4 v, float &res){
	res = (v.w);

	return res;
}

__host__ __device__
inline uchar &convert_type(float v, uchar &res){
	res = (uchar)ceil(v*255.0f);

	return res;
}
__host__ __device__
inline uchar4 &convert_type(float v, uchar4 &res){
	res.x = res.y =res.w =res.z =(uchar)ceil(v*255.0f);

	return res;
}

__host__ __device__
inline float4 &convert_type(float v, float4 &res){
	res.x = res.y =res.w =res.z =v;

	return res;
}


__host__ __device__
inline uchar &convert_type(uchar v, uchar &res){
	res = v;

	return res;
}
__host__ __device__
inline uchar4 &convert_type(uchar v, uchar4 &res){
	res.x = v; res.y = v; res.z = v; res.w = v;

	return res;
}
__host__ __device__
inline uchar &convert_type(int v, uchar &res){
	res = v;

	return res;
}
__host__ __device__
inline uchar4 &convert_type(int v, uchar4 &res){
	res.x = v; res.y = v; res.z = v; res.w = v;

	return res;
}


__host__ __device__
inline float4 convert_float4(const uchar4 &v){
	float4 res;
	res.x = ((float)v.x/255.0f);
	res.y = ((float)v.y/255.0f);
	res.z = ((float)v.z/255.0f);
	res.w = ((float)v.w/255.0f);

	return res;
}
__host__ __device__
inline float4 convert_float4(const uchar &v){
	float4 res;
	res.x = ((float)v/255.0f);
	res.y = ((float)v/255.0f);
	res.z = ((float)v/255.0f);
	res.w = ((float)v/255.0f);

	return res;
}

__host__ __device__
inline float4 convert_float4(const float4 &v){
	float4 res;
	res=v;

	return res;
}

//#ifdef __CUDACC__
__host__ __device__
inline half4 &convert_type(float4 v, half4 &res)
{
	res = make_half4(v.x, v.y, v.z, v.w);
	return res;
}

#ifdef __CUDA_ARCH__
__device__
__forceinline__ float4 &convert_type(half4 v, float4 &res)
{
	res = make_float4(__half2float(v.x), __half2float(v.y), __half2float(v.z), __half2float(v.w));
	return res;
}
#endif

__device__
__forceinline__ float3 step(float edge, float3 in){
	float3 res;
	res.x = in.x < edge ? 0.0f : 1.0f;
	res.y = in.y < edge ? 0.0f : 1.0f;
	res.z = in.z < edge ? 0.0f : 1.0f;

	return res;
}

/*#ifdef __CUDACC__
__device__
__forceinline__ float3 stepZero(float3 in){
	float3 res;
	res.x = ceil(__saturatef(in.x));
	res.y = ceil(__saturatef(in.y));
	res.z = ceil(__saturatef(in.z));

	return res;
}
#endif*/

/******************************************************************************
 * Step zero ...
 *
 * @param in ...
 *
 * @return ...
 ******************************************************************************/
__device__
__forceinline__ float3 stepZero( float3 in )
{
	float3 res;
	res.x = in.x < 0.0f ? 0.0f : 1.0f;
	res.y = in.y < 0.0f ? 0.0f : 1.0f;
	res.z = in.z < 0.0f ? 0.0f : 1.0f;

	return res;
}

/******************************************************************************
 * Step ...
 *
 * @param edge ...
 * @param in ...
 *
 * @return ...
 ******************************************************************************/
__device__
__forceinline__ float3 step( float3 edge, float3 in )
{
	float3 res;
	res.x = in.x < edge.x ? 0.0f : 1.0f;
	res.y = in.y < edge.y ? 0.0f : 1.0f;
	res.z = in.z < edge.z ? 0.0f : 1.0f;

	return res;
}

/******************************************************************************
 * ...
 ******************************************************************************/
//__device__ __host__
//inline float3 floorf(float3 in){
//	return make_float3( floorf(in.x), floorf(in.y) , floorf(in.z) );
//}

/******************************************************************************
 * ...
 ******************************************************************************/
__device__
__forceinline__ float3 abs(float3 in)
{
	float3 res;
	res.x = fabs(in.x);
	res.y = fabs(in.y);
	res.z = fabs(in.z);

	return res;
}

/******************************************************************************
 * ...
 ******************************************************************************/
__device__
__forceinline__ float squaredLength( float3 v )
{
	return v.x*v.x + v.y*v.y + v.z*v.z;
}

/******************************************************************************
 * ...
 ******************************************************************************/
__device__
__forceinline__ float distSqr( float3 p0, float3 p1 )
{
	float3 v = p1 - p0;

	return v.x*v.x + v.y*v.y + v.z*v.z;
}

/******************************************************************************
 * ...
 ******************************************************************************/
inline float3 rotate( float3 src, const float3& v, float ang )
{
	   // First we calculate [w,x,y,z], the rotation quaternion
	   float w,x,y,z;
	   float3 V=v;
	   V=normalize(V);
	   w=cos(-ang/2.0f);  // The formula rotates counterclockwise, and I
					   // prefer clockwise, so I change 'ang' sign
	   float s=sin(-ang/2.0f);
	   x=V.x*s;
	   y=V.y*s;
	   z=V.z*s;
	   // now we calculate [w^2, x^2, y^2, z^2]; we need it
	   float w2=w*w;
	   float x2=x*x;
	   float y2=y*y;
	   float z2=z*z;

	   // And apply the formula
	   float3 res=make_float3((src).x*(w2+x2-y2-z2) + (src).y*2*(x*y+w*z)   + (src).z*2*(x*z-w*y),
			   (src).x*2*(x*y-w*z)   + (src).y*(w2-x2+y2-z2) + (src).z*2*(y*z+w*x),
			   (src).x*2*(x*z+w*y)   + (src).y*2*(y*z-w*x)   + (src).z*(w2-x2-y2+z2));


	   return (res);
}

////////// MATRIX TYPE ///////////

/**
 * ...
 *
 * 0 4  8 12    ==> m[ 0 ]
 * 1 5  9 13    ==> m[ 1 ]
 * 2 6 10 14    ==> m[ 2 ]
 * x x  x  x
 */
typedef struct
{
	float4 m[ 3 ];

} float3x4;

/******************************************************************************
 * Transform vector by matrix (no translation)
 *
 * @param M matrix
 * @param v vector
 *
 * @return ...
 ******************************************************************************/
__device__
__forceinline__ float3 mulRot( const float3x4& M, const float3& v )
{
	float3 r;

	r.x = dot( v, make_float3( M.m[ 0 ] ) );
	r.y = dot( v, make_float3( M.m[ 1 ] ) );
	r.z = dot( v, make_float3( M.m[ 2 ] ) );
	
	return r;
}

/******************************************************************************
 * Transform vector by matrix with translation
 *
 * @param M matrix
 * @param v vector
 *
 * @return ...
 ******************************************************************************/
__device__ __host__
inline float4 mul( const float3x4& M, const float4& v )
{
	float4 r;

	r.x = dot( v, M.m[ 0 ] );
	r.y = dot( v, M.m[ 1 ] );
	r.z = dot( v, M.m[ 2 ] );
	r.w = 1.0f;
	
	return r;
}

/**
 * Matrix
 *
 * Array of elements stored in colum-major order (for OpenGL interoperability) :
 *
 * 0 4  8 12
 * 1 5  9 13
 * 2 6 10 14
 * 3 7 11 15
 *
 */
typedef struct
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

	/****************************** INNER TYPES *******************************/

	/**
	 * ...
	 */
	union
	{
		/**
		 * Array of lines
		 */
		float4 m[ 4 ];

		/**
		 * Array of elements stored in colum-major order
		 */
		float _array[ 16 ];
	};

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * ...
	 *
	 * @param row ...
	 * @param col ...
	 *
	 * @return ...
	 */
	float& element( int row, int col )
	{
		return _array[ row | ( col << 2 ) ];
	}

	/**
	 * ...
	 *
	 * @param row ...
	 * @param col ...
	 *
	 * @return ...
	 */
	float element( int row, int col ) const
	{
		return _array[ row | ( col << 2 ) ];
	}

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

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

} float4x4;

///******************************************************************************
// * Helper function : Multiply a point/vector by a matrix
// *
// * @param M1 matrix
// * @param M2 matrix
// *
// * @return ...
// ******************************************************************************/
///*__device__ __host__*/
//inline float4x4 mul( const float4x4& M1, const float4x4& M2 )
//{
//	float4x4 res;
//
//	float4 r1 = mul( M1, make_float4( M2._array[ 0 ], M2._array[ 1 ], M2._array[ 2 ], M2._array[ 3 ] ) );
//	float4 r2 = mul( M1, make_float4( M2._array[ 4 ], M2._array[ 5 ], M2._array[ 6 ], M2._array[ 7 ] ) );
//	float4 r3 = mul( M1, make_float4( M2._array[ 8 ], M2._array[ 9 ], M2._array[ 10 ], M2._array[ 11 ] ) );
//	float4 r4 = mul( M1, make_float4( M2._array[ 12 ], M2._array[ 13 ], M2._array[ 14 ], M2._array[ 15 ] ) );
//
//	res._array[ 0 ] = r1.x;
//	res._array[ 1 ] = r1.y;
//	res._array[ 2 ] = r1.z;
//	res._array[ 3 ] = r1.w;
//
//	res._array[ 4 ] = r2.x;
//	res._array[ 5 ] = r2.y;
//	res._array[ 6 ] = r2.z;
//	res._array[ 7 ] = r2.w;
//
//	res._array[ 8 ] = r3.x;
//	res._array[ 9 ] = r3.y;
//	res._array[ 10 ] = r3.z;
//	res._array[ 11 ] = r3.w;
//
//	res._array[ 12 ] = r4.x;
//	res._array[ 13 ] = r4.y;
//	res._array[ 14 ] = r4.z;
//	res._array[ 15 ] = r4.w;
//
//	return res;
//}

/******************************************************************************
 * Helper function : Multiply a point/vector by a matrix
 *
 * @param M matrix
 * @param v0 point/vector
 *
 * @return ...
 ******************************************************************************/
__device__ __host__
inline float4 mul( const float4x4& M, const float4& v )
{
	float4 r;

	r.x = dot( v, M.m[ 0 ] );
	r.y = dot( v, M.m[ 1 ] );
	r.z = dot( v, M.m[ 2 ] );
	r.w = dot( v, M.m[ 3 ] );

	return r;
}

/******************************************************************************
 * Helper function : Multiply a point/vector by a matrix
 *
 * @param M matrix
 * @param v0 point/vector
 *
 * @return ...
 ******************************************************************************/
__device__ __host__
inline float3 mul( const float4x4& M, const float3& v0 )
{
	float4 r = mul( M, make_float4( v0.x, v0.y, v0.z, 1.0f ) );

	return make_float3( r.x / r.w, r.y / r.w, r.z / r.w );
}

/******************************************************************************
 * Helper function : Multiply a point/vector by the 3x3 upper-left part of a 4x4 matrix
 *
 * @param M matrix
 * @param v point/vector
 ******************************************************************************/
//__device__ __host__
//inline float3 mulRot(const float4x4 &M, const float3 &v) {
//	float3 r;
//	r.x = dot(v, make_float3(M.m[0]));
//	r.y = dot(v, make_float3(M.m[1]));
//	r.z = dot(v, make_float3(M.m[2]));
//
//	return r;
//}

/******************************************************************************
 * Helper function : Multiply a point/vector by the 3x3 upper-left part of a 4x4 matrix
 *
 * @param M matrix
 * @param v point/vector
 ******************************************************************************/
__device__ __host__
inline float3 mulRot( const float4x4& M, const float3& v )
{
	float3 r;

	r.x = dot( v, make_float3( M._array[ 0 ], M._array[ 4 ], M._array[ 8 ] ) );
	r.y = dot( v, make_float3( M._array[ 1 ], M._array[ 5 ], M._array[ 9 ] ) );
	r.z = dot( v, make_float3( M._array[ 2 ], M._array[ 6 ], M._array[ 10 ] ) );

	return r;
}

/******************************************************************************
 * Helper function : Multiply a point/vector by the 3x3 upper-left part of a 4x4 matrix
 *
 * @param M matrix
 * @param v0 point/vector
 ******************************************************************************/
__device__ __host__
inline float det( const float4x4& mat )
{
	float det;

	det = mat._array[0] * mat._array[5] * mat._array[10];
	det += mat._array[4] * mat._array[9] * mat._array[2];
	det += mat._array[8] * mat._array[1] * mat._array[6];
	det -= mat._array[8] * mat._array[5] * mat._array[2];
	det -= mat._array[4] * mat._array[1] * mat._array[10];
	det -= mat._array[0] * mat._array[9] * mat._array[6];

	return det;
}

/******************************************************************************
 * ...
 ******************************************************************************/
__host__
inline float4x4 transpose( const float4x4& mat )
{
	float4x4 ret;

	ret._array[0] = mat._array[0]; ret._array[1] = mat._array[4]; ret._array[2] = mat._array[8]; ret._array[3] = mat._array[12];
	ret._array[4] = mat._array[1]; ret._array[5] = mat._array[5]; ret._array[6] = mat._array[9]; ret._array[7] = mat._array[13];
	ret._array[8] = mat._array[2]; ret._array[9] = mat._array[6]; ret._array[10] = mat._array[10]; ret._array[11] = mat._array[14];
	ret._array[12] = mat._array[3]; ret._array[13] = mat._array[7]; ret._array[14] = mat._array[11]; ret._array[15] = mat._array[15];

	return ret;
}

/******************************************************************************
 * ...
 ******************************************************************************/
__host__
inline float4x4 inverse( const float4x4& mat )
{
#if 0
	float4x4 ret;

	float idet = 1.0f / det(mat);
	ret._array[0] =  (mat._array[5] * mat._array[10] - mat._array[9] * mat._array[6]) * idet;
	ret._array[1] = -(mat._array[1] * mat._array[10] - mat._array[9] * mat._array[2]) * idet;
	ret._array[2] =  (mat._array[1] * mat._array[6] - mat._array[5] * mat._array[2]) * idet;
	ret._array[3] = 0.0;
	ret._array[4] = -(mat._array[4] * mat._array[10] - mat._array[8] * mat._array[6]) * idet;
	ret._array[5] =  (mat._array[0] * mat._array[10] - mat._array[8] * mat._array[2]) * idet;
	ret._array[6] = -(mat._array[0] * mat._array[6] - mat._array[4] * mat._array[2]) * idet;
	ret._array[7] = 0.0;
	ret._array[8] =  (mat._array[4] * mat._array[9] - mat._array[8] * mat._array[5]) * idet;
	ret._array[9] = -(mat._array[0] * mat._array[9] - mat._array[8] * mat._array[1]) * idet;
	ret._array[10] =  (mat._array[0] * mat._array[5] - mat._array[4] * mat._array[1]) * idet;
	ret._array[11] = 0.0;
	ret._array[12] = -(mat._array[12] * ret._array[0] + mat._array[13] * ret._array[4] + mat._array[14] * ret._array[8]);
	ret._array[13] = -(mat._array[12] * ret._array[1] + mat._array[13] * ret._array[5] + mat._array[14] * ret._array[9]);
	ret._array[14] = -(mat._array[12] * ret._array[2] + mat._array[13] * ret._array[6] + mat._array[14] * ret._array[10]);
	ret._array[15] = 1.0;

	return ret;
#else
	float4x4 minv;
	minv._array[0]=minv._array[1]=minv._array[2]=minv._array[3]=
		minv._array[4]=minv._array[5]=minv._array[6]=minv._array[7]=
		minv._array[8]=minv._array[9]=minv._array[10]=minv._array[11]=
		minv._array[12]=minv._array[13]=minv._array[14]=minv._array[15]=0;

		float r1[8], r2[8], r3[8], r4[8];
		float *s[4], *tmprow;

		s[0] = &r1[0];
		s[1] = &r2[0];
		s[2] = &r3[0];
		s[3] = &r4[0];

		register int i,j,p,jj;
		for(i=0;i<4;i++) {
			for(j=0;j<4;j++) {
				s[i][j] = mat.element(i,j);
				if(i==j) s[i][j+4] = 1.0;
				else	 s[i][j+4] = 0.0;
			}
		}
		float scp[4];
		for(i=0;i<4;i++) {
			scp[i] = float(fabs(s[i][0]));
			for(j=1;j<4;j++)
				if(float(fabs(s[i][j])) > scp[i]) scp[i] = float(fabs(s[i][j]));
			if(scp[i] == 0.0) return minv; // singular matrix!
		}

		int pivot_to;
		float scp_max;
		for(i=0;i<4;i++) {
			// select pivot row
			pivot_to = i;
			scp_max = float(fabs(s[i][i]/scp[i]));
			// find out which row should be on top
			for(p=i+1;p<4;p++)
				if (float(fabs(s[p][i]/scp[p])) > scp_max) {
					scp_max = float(fabs(s[p][i]/scp[p]));
					pivot_to = p;
				}
			// Pivot if necessary
			if(pivot_to != i) {
				tmprow = s[i];
				s[i] = s[pivot_to];
				s[pivot_to] = tmprow;
				float tmpscp;
				tmpscp = scp[i];
				scp[i] = scp[pivot_to];
				scp[pivot_to] = tmpscp;
			}

			float mji;
			// perform gaussian elimination
			for(j=i+1;j<4;j++) {
				mji = s[j][i]/s[i][i];
				s[j][i] = 0.0;
				for(jj=i+1;jj<8;jj++)
					s[j][jj] -= mji*s[i][jj];
			}
		}
		if(s[3][3] == 0.0) return minv; // singular matrix!

		float mij;
		for(i=3;i>0;i--) {
			for(j=i-1;j > -1; j--) {
				mij = s[j][i]/s[i][i];
				for(jj=j+1;jj<8;jj++)
					s[j][jj] -= mij*s[i][jj];
			}
		}

		for(i=0;i<4;i++)
			for(j=0;j<4;j++)
				minv.element(i,j) = s[i][j+4] / s[i][i];


		return minv;
#endif
}

#endif
