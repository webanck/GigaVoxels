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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvStructure/GvNode.h>
//#include <GvStructure/GvVolumeTree.h>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/**
 * Eye position
 */
__device__ static const float3 cEyePos = { 0.5f, 0.0f, 1.0f };

/******************************************************************************
 * ...
 ******************************************************************************/
__device__
int icoolfFunc3d2( int n )
{
	n = ( n << 13 )^n;
	return ( n * ( n * n * 15731 + 789221 ) + 1376312589 ) & 0x7fffffff;
}

/******************************************************************************
 * ...
 ******************************************************************************/
__device__
float coolfFunc3d2( int n )
{
	return static_cast< float >( icoolfFunc3d2( n ) );
}

/******************************************************************************
 * ...
 ******************************************************************************/
__device__
float noise3f( float3 p )
{
	int3 ip = make_int3(floorf(p));
	float3 u = fracf(p);
	u = u * u * (3.0f - 2.0f * u);

	int n = ip.x + ip.y * 57 + ip.z * 113;

	float res = lerp(lerp(lerp(coolfFunc3d2(n+(0+57*0+113*0)),
		coolfFunc3d2(n+(1+57*0+113*0)),u.x),
		lerp(coolfFunc3d2(n+(0+57*1+113*0)),
		coolfFunc3d2(n+(1+57*1+113*0)),u.x),u.y),
		lerp(lerp(coolfFunc3d2(n+(0+57*0+113*1)),
		coolfFunc3d2(n+(1+57*0+113*1)),u.x),
		lerp(coolfFunc3d2(n+(0+57*1+113*1)),
		coolfFunc3d2(n+(1+57*1+113*1)),u.x),u.y),u.z);

	return 1.0f - res*(1.0f / 1073741824.0f);
}

/******************************************************************************
 * Sum of Perlin noise functions
 ******************************************************************************/
__device__
float fbm( float3 p )
{
	return 0.5000f * noise3f( p * 1.0f ) + 0.2500f * noise3f( p * 2.0f ) + 0.1250f * noise3f( p * 4.0f ) + 0.0625f * noise3f( p * 8.0f );
}

/******************************************************************************
 * Ceiling
 ******************************************************************************/
__device__
float techo( float x, float y )
{
	y = 1.0f - y;

	if ( x < 0.1f || x > 0.9f )
	{
		return y;
	}

	x = x - 0.5f;

	return -( sqrtf( x * x + y * y ) - 0.4f );
}

/******************************************************************************
 * Distance function for basic primitive
 ******************************************************************************/
__device__
float distToBox( float3 p, float3 abc )
{
	float3 di = fmaxf( fabs( p ) - abc, make_float3( 0.0f ) );

	return dot( di, di );
}

/******************************************************************************
 * ...
 ******************************************************************************/
__device__
float columna( float x, float y, float z, float mindist, float offx )
{
	float3 p = make_float3( x, y, z );
	float di0 = distToBox( p, make_float3( 0.14f, 1.0f, 0.14f ) );
	if ( di0 > ( mindist * mindist ) )
	{
		return mindist + 1.0f;
	}

	float y2 = y - 0.40f;
	float y3 = y - 0.35f;
	float y4 = y - 1.00f;

	float di1 = distToBox( p, make_float3( 0.10f, 1.00f, 0.10f ) );
	float di2 = distToBox( p, make_float3( 0.12f, 0.40f, 0.12f ) );
	float di3 = distToBox( p, make_float3( 0.05f, 0.35f, 0.14f ) );
	float di4 = distToBox( p, make_float3( 0.14f, 0.35f, 0.05f ) );
	float di9 = distToBox( make_float3( x, y4, z ), make_float3( 0.14f, 0.02f, 0.14f ) );

	float di5 = distToBox( make_float3( ( x - y2 ) * 0.7071f, ( y2 + x ) * 0.7071f, z ), make_float3( 0.10f * 0.7071f, 0.10f * 0.7071f, 0.12f ) );
	float di6 = distToBox( make_float3( x, ( y2 + z ) * 0.7071f, ( z - y2 ) * 0.7071f ), make_float3( 0.12f,  0.10f * 0.7071f, 0.10f * 0.7071f ) );
	float di7 = distToBox( make_float3( (x - y3 ) * 0.7071f, ( y3 + x ) * 0.7071f, z ), make_float3( 0.10f * 0.7071f,  0.10f * 0.7071f, 0.14f ) );
	float di8 = distToBox( make_float3( x, ( y3 + z ) * 0.7071f, ( z - y3 ) * 0.7071f ), make_float3( 0.14f,  0.10f * 0.7071f, 0.10f * 0.7071f ) );

	float di = min( min( min( di1, di2 ), min( di3, di4 ) ), min( min( di5, di6 ), min( di7, di8 ) ) );
	di = min( di, di9 );

	//  di += 0.00000003 * max( fbm( 10.1 * p ), 0.0 );

	return di;
}

/******************************************************************************
 * ...
 ******************************************************************************/
__device__
float bicho( float3 x, float mindist )
{
	//    float ramo = noise3f( vec3( 2.0  *time, 2.3  *time, 0.0 ) );

	x -= make_float3( 0.64f, 0.5f, 1.5f );

	float r2 = dot( x, x );

	float sa = smoothstep( 0.0f, 0.5f, r2 );
	float fax = 0.75f + 0.25f * sa;
	float fay = 0.80f + 0.20f * sa;

	x.x *= fax;
	x.y *= fay;
	x.z *= fax;

	r2 = dot( x, x );

	float r = sqrtf( r2 );

	float a1 = 1.0f - smoothstep( 0.0f, 0.75f, r );
	a1 *= 0.40f;
	float si1 = sinf( a1 );
	float co1 = cosf( a1 );
	//x.xy = mat2( co1, si1, -si1, co1 ) * x.xy;
	float2 xy;
	xy.x = co1 * x.x - si1 * x.y;
	xy.y = si1 * x.x + co1 * x.y;
	x.x = xy.x;
	x.y = xy.y;

	float mindist2 = 100000.0f;

	float rr = 0.05f+sqrt(dot(make_float2(x.x, x.z), make_float2(x.x, x.z)));
	float ca = (0.5f-0.045f*0.75f) -6.0f*rr*exp2f(-10.0f*rr);
	for( int j = 1; j < 7; j++ )
	{
		float an = (6.2831f/7.0f) * float(j);
		float aa = an + 0.40f*rr*noise3f( make_float3(4.0f*rr, 2.5f, an) ) + 0.29f;
		float rc = cosf(aa);
		float rs = sinf(aa);
		float3 q = make_float3( x.x*rc-x.z*rs, x.y+ca, x.x*rs+x.z*rc );
		float dd = dot(make_float2(q.y, q.z), make_float2(q.y, q.z));
		if( q.x>0.0f && q.x<1.5f && dd<mindist2 ) mindist2=dd;
	}

	float c = sqrtf( mindist2 ) - 0.045f;
	float d = r - 0.30f;
	float a = clamp( r * 3.0f, 0.0f, 1.0f );
	return c * a + d * ( 1.0f - a );
}

/******************************************************************************
 * Distance field function
 *
 * @param pos 3D position
 ******************************************************************************/
__device__
float map( float3 pos, int& sid, int& submat )
{
	submat = 0;
	float dis, mindist;

	//-----------------------
	// Floor (suelo)
	//-----------------------
	dis = pos.y;
	float2 axz = make_float2( 128.0f ) + 6.0f * make_float2( pos.x + pos.z, pos.x - pos.z );
	int2 ixz = make_int2( floorf( axz ) );
	submat = icoolfFunc3d2( ixz.x + 53 * ixz.y );
	float2 peldxz = fracf( axz );
	float peld = smoothstep( 0.975f, 1.0f, fmaxf( peldxz.x, peldxz.y ) );
	if ( ( ( ( submat >> 10 ) & 7 ) > 6 ) )
	{
		peld = 1.0f;
	}
	dis += 0.005f * peld;
	mindist = dis;
	sid = 0;
	if ( peld > 0.0000001f )
	{
		sid = 2;
	}

	//-----------------------
	// Ceiling (techo)
	//-----------------------
	float fx = fracf( pos.x + 128.0f );
	float fz = fracf( pos.z + 128.0f );
	if ( pos.y > 1.0f )
	{
		dis = max( techo( fx, pos.y ), techo( fz, pos.y ) );
		if ( dis < mindist )
		{
			mindist = dis;
			sid = 5;
		}
	}
	fx = fracf( pos.x + 128.0f + .5f );
	fz = fracf( pos.z + 128.0f + .5f );

	//-----------------------
	// columnas
	//-----------------------
	dis = columna( fx - .5f, pos.y, fz - .5f, mindist, 13.1f * floorf( pos.x ) + 17.7f * floorf( pos.z ) );
	if ( dis < ( mindist * mindist ) )
	{
		mindist = sqrtf( dis );
		sid = 1;
	}

	//-----------------------
	// bicho
	//-----------------------
	dis = bicho( pos, mindist );
	if ( dis < mindist )
	{
		mindist = dis;
		sid = 4;
	}

	return mindist;
}

/******************************************************************************
 * Initialize the producer
 * 
 * @param volumeTreeKernel Reference on a volume tree data structure
 ******************************************************************************/
template< typename TDataStructureType >
inline void ProducerKernel< TDataStructureType >
::initialize( DataStructureKernel& pDataStructure )
{
	//_dataStructureKernel = pDataStructure;
}

/******************************************************************************
 * Produce data on device.
 * Implement the produceData method for the channel 0 (nodes)
 *
 * Producing data mecanism works element by element (node tile or brick) depending on the channel.
 *
 * In the function, user has to produce data for a node tile or a brick of voxels :
 * - for a node tile, user has to defined regions (i.e nodes) where lies data, constant values,
 * etc...
 * - for a brick, user has to produce data (i.e voxels) at for each of the channels
 * user had previously defined (color, normal, density, etc...)
 *
 * @param pGpuPool The device side pool (nodes or bricks)
 * @param pRequestID The current processed element coming from the data requests list (a node tile or a brick)
 * @param pProcessID Index of one of the elements inside a node tile or a voxel bricks
 * @param pNewElemAddress The address at which to write the produced data in the pool
 * @param pParentLocInfo The localization info used to locate an element in the pool
 * @param Loki::Int2Type< 0 > corresponds to the index of the node pool
 *
 * @return A feedback value that the user can return.
 ******************************************************************************/
template < typename TDataStructureType >
template < typename GPUPoolKernelType >
__device__
inline uint ProducerKernel< TDataStructureType >
::produceData( GPUPoolKernelType &nodePool, uint requestID, uint processID, uint3 newElemAddress,
			  const GvCore::GvLocalizationInfo &parentLocInfo, Loki::Int2Type< 0 > )
{
	const GvCore::GvLocalizationInfo::CodeType *parentLocCode = &parentLocInfo.locCode;
	const GvCore::GvLocalizationInfo::DepthType *parentLocDepth = &parentLocInfo.locDepth;

	if ( processID < NodeRes::getNumElements() )
	{
		uint3 subOffset = NodeRes::toFloat3( processID );

		uint3 regionCoords = parentLocCode->addLevel<NodeRes>(subOffset).get();
		uint regionDepth = parentLocDepth->addLevel().get();

		GvStructure::GvNode newnode;
		newnode.childAddress=0;

		GPUVoxelProducer::GPUVPRegionInfo nodeinfo = getRegionInfo( regionCoords, regionDepth );

		if ( nodeinfo == GPUVoxelProducer::GPUVP_CONSTANT )
		{
			//newnode.data.setValue(0.0f);
			//newnode.setStoreValue();
			newnode.setTerminal( true );
		}
		else if ( nodeinfo == GPUVoxelProducer::GPUVP_DATA )
		{
			//newnode.data.brickAddress = 0;
			newnode.setStoreBrick();
			newnode.setTerminal( false );
		}
		else if ( nodeinfo == GPUVoxelProducer::GPUVP_DATA_MAXRES )
		{
			//newnode.data.brickAddress = 0;
			newnode.setStoreBrick();
			newnode.setTerminal( true );
		}

		// Write node info into the node pool
		nodePool.getChannel( Loki::Int2Type< 0 >() ).set( newElemAddress.x + processID, newnode.childAddress );
		//nodePool.getChannel( Loki::Int2Type< 1 >() ).set( newElemAddress.x + processID, newnode.data.brickAddress );
		nodePool.getChannel( Loki::Int2Type< 1 >() ).set( newElemAddress.x + processID, newnode.brickAddress );
	}

	return 0;
}

/******************************************************************************
 * Produce data on device.
 * Implement the produceData method for the channel 1 (bricks)
 *
 * Producing data mecanism works element by element (node tile or brick) depending on the channel.
 *
 * In the function, user has to produce data for a node tile or a brick of voxels :
 * - for a node tile, user has to defined regions (i.e nodes) where lies data, constant values,
 * etc...
 * - for a brick, user has to produce data (i.e voxels) at for each of the channels
 * user had previously defined (color, normal, density, etc...)
 *
 * @param pGpuPool The device side pool (nodes or bricks)
 * @param pRequestID The current processed element coming from the data requests list (a node tile or a brick)
 * @param pProcessID Index of one of the elements inside a node tile or a voxel bricks
 * @param pNewElemAddress The address at which to write the produced data in the pool
 * @param pParentLocInfo The localization info used to locate an element in the pool
 * @param Loki::Int2Type< 1 > corresponds to the index of the brick pool
 *
 * @return A feedback value that the user can return.
 ******************************************************************************/
template < typename TDataStructureType >
template < typename GPUPoolKernelType >
__device__
inline uint ProducerKernel< TDataStructureType >
::produceData( GPUPoolKernelType &dataPool, uint requestID, uint processID, uint3 newElemAddress,
			  const GvCore::GvLocalizationInfo &parentLocInfo, Loki::Int2Type< 1 > )
{
	// Retrieve current brick localization information code and depth
	const GvCore::GvLocalizationInfo::CodeType parentLocCode = parentLocInfo.locCode;
	const GvCore::GvLocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;

	// Shared memory declaration
	//
	// Each threads of a block process one and unique brick of voxels.
	// We store in shared memory common useful variables that will be used by each thread.
	__shared__ uint3 brickRes;
	__shared__ uint3 levelRes; 
	__shared__ float3 levelResInv; 
	__shared__ int3 brickPos;
	__shared__ float3 brickPosF;

	brickRes = BrickRes::get();
	levelRes = make_uint3(1 << parentLocDepth.get()) * brickRes;
	levelResInv = make_float3(1.0f) / make_float3(levelRes);

	brickPos = make_int3(parentLocCode.get() * brickRes) - BorderSize;
	brickPosF = make_float3(brickPos) * levelResInv;

	uint3 elemSize = BrickRes::get() + make_uint3(2 * BorderSize);
	uint3 elemOffset;

	// The original KERNEL execution configuration on the HOST has a 2D block size :
	// dim3 blockSize( 16, 8, 1 );
	//
	// Each block process one brick of voxels.
	//
	// One thread iterate in 3D space given a pattern defined by the 2D block size
	// and the following "for" loops. Loops take into account borders.
	// In fact, each thread of the current 2D block compute elements layer by layer
	// on the z axis.
	//
	// One thread process only a subset of the voxels of the brick.
	//
	// Iterate through z axis step by step as blockDim.z is equal to 1
	for ( elemOffset.z = 0; elemOffset.z < elemSize.z; elemOffset.z += blockDim.z )
	{
		for ( elemOffset.y = 0; elemOffset.y < elemSize.y; elemOffset.y += blockDim.y )
		{
			for ( elemOffset.x = 0; elemOffset.x < elemSize.x; elemOffset.x += blockDim.x )
			{
				uint3 locOffset = elemOffset + make_uint3( threadIdx.x, threadIdx.y, threadIdx.z );

				if ( locOffset.x < elemSize.x && locOffset.y < elemSize.y && locOffset.z < elemSize.z )
				{
					// Position of the current voxel's center (relative to the brick)
					float3 voxelPosInBrickF = ( make_float3( locOffset ) + 0.5f ) * levelResInv;
					// Position of the current voxel's center (absolute, in [0.0;1.0] range)
					float3 voxelPosF = brickPosF + voxelPosInBrickF;
					// Position of the current voxel's center (scaled to the range [-1.0;1.0])
					float3 posF = voxelPosF * 2.0f - 1.0f;

					posF += cEyePos;
					posF *= 4.f;

					int matId = 666;
					int subMatId;

					// Distance field function
					float h = map( posF, matId, subMatId );

					// Initial data
					float4 voxelColor = make_float4( 0.f );
					float4 voxelNormal = make_float4( 0.f );

					float3 rgb = make_float3( 0.f );

					if ( matId != 666 )
					{
						float eps = 0.5f * levelResInv.x;

						// unused vars
						int m1, m2;

						// Normal
						// - computed by central differences on the distance field at the shading point (gradient approximation)
						float3 normal;
						normal.x = map( make_float3( posF.x + eps, posF.y, posF.z ), m1, m2 ) - map( make_float3( posF.x - eps, posF.y, posF.z ), m1, m2 );
						normal.y = map( make_float3( posF.x, posF.y + eps, posF.z ), m1, m2 ) - map( make_float3( posF.x, posF.y - eps, posF.z ), m1, m2 );
						normal.z = map( make_float3( posF.x, posF.y, posF.z + eps ), m1, m2 ) - map( make_float3( posF.x, posF.y, posF.z - eps ), m1, m2 );
						normal = normalize( normal );

						// Bump mapping
						// - computed by adding the gradient of a fractal sum of Perlin noise functions to the surface normal
						// - bump is small and depend on the material
						float kke = 0.0001f;
						float bumpa = 0.0075f;
						if ( matId != 5 )
						{
							bumpa *= 0.75f;
						}
						if ( matId == 4 )
						{
							bumpa *= 0.50f;
						}
						bumpa /= kke;
						float kk = fbm( 32.0f * posF );
						normal.x += bumpa * ( fbm( 32.0f * make_float3( posF.x + kke, posF.y, posF.z ) ) - kk );
						normal.y += bumpa * ( fbm( 32.0f * make_float3( posF.x, posF.y + kke, posF.z ) ) - kk );
						normal.z += bumpa * ( fbm( 32.0f * make_float3( posF.x, posF.y, posF.z + kke ) ) - kk );
						normal = normalize( normal );

						// light
						float spe = 0.0f;
						float3 lig = make_float3( 0.5f - posF.x, 0.8f - posF.y, 1.5f - posF.z );
						float llig = dot( lig, lig );
						float im = rsqrtf( llig );
						lig = lig * im;
						float dif = dot( normal, lig );
						if ( matId == 4 )
						{
							dif = 0.5f + 0.5f * dif;
						}
						else
						{
							dif = 0.1f + 0.9f * dif;
						}
						//if( dif < 0.0f ) dif = 0.0f;
						//dif = max( dif, 0.0f );
						dif = clamp( dif, 0.0f, 1.0f );
						dif *= 2.5f * exp2f( -1.75f * llig );
						float dif2 = ( normal.x + normal.y ) * 0.075f;

						// Materials
						if ( matId == 0 )
						{
							float xoff = 13.1f * float( subMatId & 255 );
							float fb = fbm( 16.0f * make_float3( posF.x + xoff, posF.y, posF.z ) );
							rgb = make_float3( 0.7f ) + fb * make_float3( 0.20f, 0.22f, 0.25f );

							float baldscale = float( ( subMatId >> 9 ) & 15 ) / 14.0f;
							baldscale = 0.51f + 0.34f * baldscale;
							rgb *= baldscale;
							float fx = 1.0f;
							if ( ( subMatId & 256 ) != 0 )
							{
								fx = -1.0f;
							}
							float m = sin( 64.0f * posF.z * fx + 64.0f * posF.x + 4.0f * fb );
							m = smoothstep( 0.25f, 0.5f, m ) - smoothstep( 0.5f, 0.75f, m );
							rgb += m * make_float3( 0.15f );
						}
						else if ( matId == 2 ) // floor
						{
							rgb = make_float3( 0.0f );
						}
						else if ( matId == 1 ) // columns
						{
							float fb = fbm( 16.0f * posF );
							float m = sin( 64.0f * posF.z + 64.0f * posF.x + 4.0f * fb );
							m = smoothstep( 0.30f, 0.5f, m ) - smoothstep( 0.5f, 0.70f, m );
							rgb = make_float3( 0.59f ) + fb * make_float3( 0.17f, 0.18f, 0.21f ) + m * make_float3( 0.15 ) + make_float3( dif2 );
						}
						else if ( matId == 4 ) // monster
						{
							float ft = fbm( 16.0f * posF );
							rgb = make_float3( 0.82f, 0.73f, 0.65f ) + ft * make_float3( 0.1f );

							float fs = 0.9f + 0.1f * fbm( 32.0f * posF );
							rgb *= fs;

							float fre = 0.9f;//max( -dot( normal, rd ), 0.0f);
							rgb -= make_float3( fre * fre * 0.45f );
							spe = clamp( ( normal.y - normal.z ) * 0.707f, 0.0f, 1.0f );
							spe = 0.20f * __powf( spe, 32.0f );
						}
						// techo
						else //if( matID==5 )
						{
							float fb = fbm( 16.0f * posF );
							rgb = make_float3( 0.64f, 0.61f, 0.59f ) + fb * make_float3( 0.21f, 0.19f, 0.19f ) + dif2;
						}

						// Ambient occlusion
						//
						// Fake and fast Ambient Occlusion.
						// VERY CHEAP, even cheaper than primary rays! Only 5 distance evaluations
						// instead of casting thousand of rays/evaluations.
						//
						//In a regular raytracer, primary rays/AO cost is 1:2000. Here, it’s 3:1 (that’s
						// almost four orders of magnitude speedup!).
						// It’s NOT the screen space trick (SSAO), but 3D.
						// The basic technique was invented by Alex Evans, aka Statix (“Fast
						// Approximation for Global Illumnation on Dynamic Scenes”, 2006). Greets to him!
						//
						// The idea: let p be the point to shade. Sample the distance field at a few (5)
						// points around p and compare the result to the actual distance to p. That
						// gives surface proximity information that can easily be interpreted as an
						// (ambient) occlusion factor.
						float ao;
						float totao = 0.0f;
						float sca = 10.0f;
						for ( int aoi = 0; aoi < 5; aoi++ )
						{
							float hr = 0.01f + 0.015f * float( aoi * aoi );
							float3 aopos =  normal * hr + posF;
							int kk, kk2;
							float dd = map( aopos, kk, kk2 );
							ao = -( dd - hr );
							totao += ao * sca;
							sca *= 0.5f;
						}
						ao = 1.0f - clamp( totao, 0.0f, 1.0f );

						// Soft shadows
						//
						// Fake and fast soft shadows.
						// Only 6 distance evaluations used instead of casting hundrends of rays.
						// Pure geometry-based, not bluring.
						// Recipe: take n points on the line from the surface to the light and evaluate
						// the distance to the closest geometry. Find a magic formula to blend the n
						// distances to obtain a shadow factor.
						float so = 0.0f;
						for ( int i = 0; i < 6; i++ )
						{
							float h = float( i ) / 6.0f;
							float hr = 0.01f + h;
							float3 aopos = lig * hr + posF;
							int kk, kk2;
							float dd = map( aopos, kk, kk2 );
							so += ( 1.0f - h ) * dd * 2.0f * ( 10.0f / 6.0f );
						}
						dif *= clamp( ( so - 0.40f ) * 1.5f, 0.0f, 1.0f );

						// Lighting
						rgb = make_float3( spe ) + rgb * ( ao * make_float3( 0.25f, 0.30f, 0.35f ) + dif * make_float3( 1.95f, 1.65f, 1.05f ) );
						// fog
						//rgb = rgb * exp2(-0.4f * t);
					}

					// Color correct
					rgb = ( make_float3( sqrtf( rgb.x ), sqrtf( rgb.y ), sqrtf( rgb.z ) ) * 0.7f + 0.3f * rgb ) * make_float3( 0.83f, 1.0f, 0.83f ) * 1.2f;

					// Vigneting
					//rgb *= 0.25f + 0.75f * clamp( 0.60f * fabsf( pixel.x - 1.0f ) * fabsf( pixel.x + 1.0f ), 0.0f, 1.0f );
					rgb *= 0.25f + 0.75f * 0.6f;

					voxelColor = make_float4( rgb, 1.f - clamp( .5f * levelRes.x * h, 0.f, 1.f ) );

					// Alpha pre-multiplication used to avoid "color bleeding" effect
					voxelColor.x *= voxelColor.w;
					voxelColor.y *= voxelColor.w;
					voxelColor.z *= voxelColor.w;

					// Compute the new element's address
					uint3 destAddress = newElemAddress + locOffset;
					// Write the voxel's color in the first field
					dataPool.template setValue< 0 >( destAddress, voxelColor );
					// Write the voxel's normal in the second field
					//dataPool.template setValue< 1 >( destAddress, voxelNormal );
				}
			}
		}
	} 

	return 0;
}

/******************************************************************************
 * Helper function used to determine the type of zones in the data structure.
 *
 * The data structure is made of regions containing data, empty or constant regions.
 * Besides, this function can tell if the maximum resolution is reached in a region.
 *
 * @param regionCoords region coordinates
 * @param regionDepth region depth
 *
 * @return the type of the region
 ******************************************************************************/
template < typename TDataStructureType >
__device__
inline GPUVoxelProducer::GPUVPRegionInfo ProducerKernel< TDataStructureType >
::getRegionInfo( uint3 regionCoords, uint regionDepth )
{
	//if (regionDepth <= 4)
	//return GPUVoxelProducer::GPUVP_DATA;

	// Limit the depth
	if (regionDepth >= 32)
	{
		return GPUVoxelProducer::GPUVP_DATA_MAXRES;
	}

	//const GvCore::GvLocalizationInfo::CodeType parentLocCode = parentLocInfo.locCode;
	//const GvCore::GvLocalizationInfo::DepthType parentLocDepth = parentLocInfo.locDepth;

	__shared__ uint3 brickRes;
	__shared__ uint3 levelRes; 
	__shared__ float3 levelResInv; 

	brickRes = BrickRes::get();
	levelRes = make_uint3(1 << regionDepth) * brickRes;
	levelResInv = make_float3(1.0f) / make_float3(levelRes);

	int3 brickPos = make_int3(regionCoords * brickRes) - BorderSize;
	float3 brickPosF = make_float3(brickPos) * levelResInv;

	uint3 elemSize = BrickRes::get() + make_uint3(2 * BorderSize);
	uint3 elemOffset;

	bool isEmpty = true;

	//float brickSize = 1.0f / (float)(1 << regionDepth);

	for ( elemOffset.z = 0; elemOffset.z < elemSize.z && isEmpty; elemOffset.z++ )
	{
		for ( elemOffset.y = 0; elemOffset.y < elemSize.y  && isEmpty; elemOffset.y++ )
		{
			for ( elemOffset.x = 0; elemOffset.x < elemSize.x && isEmpty; elemOffset.x++ )
			{
				uint3 locOffset = elemOffset;// + make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);

				if ( locOffset.x < elemSize.x && locOffset.y < elemSize.y && locOffset.z < elemSize.z )
				{
					// Position of the current voxel's center (relative to the brick)
					float3 voxelPosInBrickF = ( make_float3( locOffset ) + 0.5f ) * levelResInv;
					// Position of the current voxel's center (absolute, in [0.0;1.0] range)
					float3 voxelPosF = brickPosF + voxelPosInBrickF;
					// Position of the current voxel's center (scaled to the range [-1.0;1.0])
					float3 posF = voxelPosF * 2.0f - 1.0f;

					posF += cEyePos;
					posF *= 4.f;

					// If the distance at the position is less than the size of one voxel, the brick is ont empty
					int matId = 0;
					int subMatId = 0;

					if ( map( posF, matId, subMatId ) <= levelResInv.x )
					{
						isEmpty = false;
					}
				}
			}
		}
	}

	if ( isEmpty )
	{
		return GPUVoxelProducer::GPUVP_CONSTANT;
	}

	return GPUVoxelProducer::GPUVP_DATA;
}
