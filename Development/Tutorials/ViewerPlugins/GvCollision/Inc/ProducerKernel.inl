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
#include <GvUtils/GvNoiseKernel.h>

//#include "SampleCore.h"

/******************************************************************************
 ****************************** CONSTANTS DEFINITION **************************
 ******************************************************************************/

/******************************************************************************
 ****************************** SIMPLEX NOISE *********************************
 ******************************************************************************/

namespace GvSimplexNoise {
	__device__
	const int perm[ 512 ] =
	{
		151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233, 7,225,
		140, 36,103, 30, 69,142, 8, 99, 37,240, 21, 10, 23,190, 6,148,
		247,120,234, 75, 0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
		57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
		74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
		60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
		65, 25, 63,161, 1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
		200,196,135,130,116,188,159, 86,164,100,109,198,173,186, 3, 64,
		52,217,226,250,124,123, 5,202, 38,147,118,126,255, 82, 85,212,
		207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
		119,248,152, 2, 44,154,163, 70,221,153,101,155,167, 43,172, 9,
		129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
		218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
		81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
		184, 84,204,176,115,121, 50, 45,127, 4,150,254,138,236,205, 93,
		222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180,
		// repeat
		151,160,137, 91, 90, 15,131, 13,201, 95, 96, 53,194,233, 7,225,
		140, 36,103, 30, 69,142, 8, 99, 37,240, 21, 10, 23,190, 6,148,
		247,120,234, 75, 0, 26,197, 62, 94,252,219,203,117, 35, 11, 32,
		57,177, 33, 88,237,149, 56, 87,174, 20,125,136,171,168, 68,175,
		74,165, 71,134,139, 48, 27,166, 77,146,158,231, 83,111,229,122,
		60,211,133,230,220,105, 92, 41, 55, 46,245, 40,244,102,143, 54,
		65, 25, 63,161, 1,216, 80, 73,209, 76,132,187,208, 89, 18,169,
		200,196,135,130,116,188,159, 86,164,100,109,198,173,186, 3, 64,
		52,217,226,250,124,123, 5,202, 38,147,118,126,255, 82, 85,212,
		207,206, 59,227, 47, 16, 58, 17,182,189, 28, 42,223,183,170,213,
		119,248,152, 2, 44,154,163, 70,221,153,101,155,167, 43,172, 9,
		129, 22, 39,253, 19, 98,108,110, 79,113,224,232,178,185,112,104,
		218,246, 97,228,251, 34,242,193,238,210,144, 12,191,179,162,241,
		81, 51,145,235,249, 14,239,107, 49,192,214, 31,181,199,106,157,
		184, 84,204,176,115,121, 50, 45,127, 4,150,254,138,236,205, 93,
		222,114, 67, 29, 24, 72,243,141,128,195, 78, 66,215, 61,156,180,
	};
	__device__
	const int3 grad3[] = {{1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
		 {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
		 {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}};

	__device__
	__forceinline__ float dotN(int3 g, float x, float y, float z) {
		return g.x*x + g.y*y + g.z*z;
	}

	__device__
	__forceinline__ float getValue( const float3 pPoint )
	{
		float n0, n1, n2, n3; // Noise contributions from the four corners
		// Skew the input space to determine which simplex cell we're in
		const float F3 = 1.0/3.0;
		float s = (pPoint.x+pPoint.y+pPoint.z)*F3;
		int i = floorf(pPoint.x+s);
		int j = floorf(pPoint.y+s);
		int k = floorf(pPoint.z+s);

		// Unskew
		const float G3 = 1.0/6.0;
		float t = (i+j+k)*G3;
		// Unskew the cell origin back to (x,y,z) space
		float X0 = i-t;
		float Y0 = j-t;
		float Z0 = k-t;

		// The x,y,z distances from the cell origin
		float x0 = pPoint.x-X0;
		float y0 = pPoint.y-Y0;
		float z0 = pPoint.z-Z0;

		// For the 3D case, the simplex shape is a slightly irregular tetrahedron.
		// Determine which simplex we are in.
		int i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
		int i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords
		if(x0>=y0) {
			if(y0>=z0)
			{ i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; } // X Y Z order
			else if(x0>=z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; } // X Z Y order
			else { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; } // Z X Y order
		}
		else { // x0<y0
			if(y0<z0) { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; } // Z Y X order
			else if(x0<z0) { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; } // Y Z X order
			else { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; } // Y X Z order
		}
		// A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
		// a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
		// a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
		// c = 1/6.
		float x1 = x0 - i1 + G3; // Offsets for second corner in (x,y,z) coords
		float y1 = y0 - j1 + G3;
		float z1 = z0 - k1 + G3;
		float x2 = x0 - i2 + 2.0*G3; // Offsets for third corner in (x,y,z) coords
		float y2 = y0 - j2 + 2.0*G3;
		float z2 = z0 - k2 + 2.0*G3;
		float x3 = x0 - 1.0 + 3.0*G3; // Offsets for last corner in (x,y,z) coords
		float y3 = y0 - 1.0 + 3.0*G3;
		float z3 = z0 - 1.0 + 3.0*G3;

		// Work out the hashed gradient indices of the four simplex corners
		int ii = i & 255;
		int jj = j & 255;
		int kk = k & 255;
		int gi0 = perm[ii+perm[jj+perm[kk]]] % 12;
		int gi1 = perm[ii+i1+perm[jj+j1+perm[kk+k1]]] % 12;
		int gi2 = perm[ii+i2+perm[jj+j2+perm[kk+k2]]] % 12;
		int gi3 = perm[ii+1+perm[jj+1+perm[kk+1]]] % 12;

		// Calculate the contribution from the four corners
		float t0 = 0.5 - x0*x0 - y0*y0 - z0*z0;
		if(t0<0) n0 = 0.0;
		else {
			t0 *= t0;
			n0 = t0 * t0 * dotN(grad3[gi0], x0, y0, z0);
		}
		float t1 = 0.5 - x1*x1 - y1*y1 - z1*z1;
		if(t1<0) n1 = 0.0;
		else {
			t1 *= t1;
			n1 = t1 * t1 * dotN(grad3[gi1], x1, y1, z1);
		}
		float t2 = 0.5 - x2*x2 - y2*y2 - z2*z2;
		if(t2<0) n2 = 0.0;
		else {
			t2 *= t2;
			n2 = t2 * t2 * dotN(grad3[gi2], x2, y2, z2);
		}
		float t3 = 0.5 - x3*x3 - y3*y3 - z3*z3;
		if(t3<0) n3 = 0.0;
		else {
			t3 *= t3;
			n3 = t3 * t3 * dotN(grad3[gi3], x3, y3, z3);
		}
		// Add contributions from each corner to get the final noise value.
		// The result is scaled to stay just inside [-1,1]
		return 32.f*(n0 + n1 + n2 + n3);
	}
};


/******************************************************************************
 ****************************** KERNEL DEFINITION *****************************
 ******************************************************************************/

/**
 * Transfer function texture
 */
texture< float4, cudaTextureType1D, cudaReadModeElementType > transferFunctionTexture;

/******************************************************************************
 * Map a distance to a color (with a transfer function)
 *
 * @param pDistance the distance to map
 *
 * @return The color corresponding to the distance
 ******************************************************************************/
__device__
inline float4 distToColor( float pDistance )
{
	// Fetch data from transfer function
	return tex1D( transferFunctionTexture, 1 - pDistance );
}

/******************************************************************************
 * Helper class, compute the distance and the normal of a point.
 *
 * @param pPoint the point to test
 *
 * @return normal + distance.
 ******************************************************************************/
__device__
inline float4 computeDistance( const float3 pPoint )
{
	float4 result;

	float radius = rsqrt( pPoint.x * pPoint.x + pPoint.y * pPoint.y );
	float3 circleCenter = make_float3( pPoint.x * cTorusRadius * radius,
			                           pPoint.y * cTorusRadius * radius,
			                           0.f );

	float l = length( pPoint - circleCenter );
	
	result = make_float4(( pPoint - circleCenter ) / l, //normal
						   l - cTubeRadius ); //length

	return result;
}

/******************************************************************************
 * Get the alpha data of distance field + noise.
 *
 * @param voxelPosF 3D position in the current brick
 * @param levelRes number of voxels at the current brick resolution
 *
 * @return computed alpha 
 ******************************************************************************/
template< typename TDataStructureType >
__device__
inline float ProducerKernel< TDataStructureType >::getAlpha( float3 voxelPosF, uint3 levelRes )
{
	// Type definition for the noise
	typedef GvUtils::GvNoiseKernel Noise;

	// Compute normal
	float4 voxelNormalAndDist = computeDistance(voxelPosF);
	float distance = voxelNormalAndDist.w;
	float3 voxelNormal = make_float3( voxelNormalAndDist.x, voxelNormalAndDist.y, voxelNormalAndDist.z );
	float alpha;

	// Compute color by mapping a distance to a color (with a transfer function)
	float4 color = distToColor( clamp( 0.5f - 0.5f * cNoiseFirstFrequency * distance, 0.f, 1.f ) );

	// Compute noise
	float dist_noise = 0.0f;
	for ( float frequency = cNoiseFirstFrequency; 
			frequency <=  cNoiseMaxFrequency ;
		   	frequency *= 2.f ) {
		if ( cNoiseType == SIMPLEX ) {
			dist_noise += cNoiseStrength / frequency * GvSimplexNoise::getValue( frequency * ( voxelPosF - distance * voxelNormal ) );
		}
		else if ( cNoiseType == PERLIN ) {
			dist_noise += cNoiseStrength / frequency * Noise::getValue( frequency * ( voxelPosF - distance * voxelNormal ) );
		}
	}

	// Compute alpha
	alpha = clamp( 0.5f - 0.5f * ( distance + dist_noise ) * static_cast< float >( levelRes.x ), 0.f, 1.f );

	return alpha;
}


/******************************************************************************
 * Get the RGBA data of distance field + noise.
 * Note : color is alpha pre-multiplied to avoid color bleeding effect.
 *
 * @param voxelPosF 3D position in the current brick
 * @param levelRes number of voxels at the current brick resolution
 *
 * @return computed RGBA color
 ******************************************************************************/
template< typename TDataStructureType >
__device__
inline float4 ProducerKernel< TDataStructureType >::getRGBA( float3 voxelPosF, uint3 levelRes )
{
	// Type definition for the noise
	typedef GvUtils::GvNoiseKernel Noise;

	// Compute normal
	float4 voxelNormalAndDist = computeDistance(voxelPosF);
	float distance = voxelNormalAndDist.w;
	float3 voxelNormal = make_float3( voxelNormalAndDist.x, voxelNormalAndDist.y, voxelNormalAndDist.z );
	float4 voxelRGBA;

	// Compute color by mapping a distance to a color (with a transfer function)
	float4 color = distToColor( clamp( 0.5f - 0.5f * distance * cNoiseFirstFrequency, 0.f, 1.f ) );
	if ( color.w > 0.f ) {
		// De multiply color with alpha because transfer function data has been pre-multiplied when generated
		color.x /= color.w;
		color.y /= color.w;
		color.z /= color.w;
	}

	// Compute noise
	float dist_noise = 0.0f;
	for ( float frequency = cNoiseFirstFrequency; 
			 frequency <= cNoiseMaxFrequency ; 
			frequency *= 2.f ) {
		if ( cNoiseType == SIMPLEX ) {
			dist_noise += cNoiseStrength / frequency * GvSimplexNoise::getValue( frequency * ( voxelPosF - distance * voxelNormal ) );
		}
		else if ( cNoiseType == PERLIN ) {
			dist_noise += cNoiseStrength / frequency * Noise::getValue( frequency * ( voxelPosF - distance * voxelNormal ) );
		}
	}

	// Compute alpha
	voxelRGBA.w = clamp( 0.5f - 0.5f * ( distance + dist_noise ) * static_cast< float >( levelRes.x ), 0.f, 1.f );

	// Pre-multiply color with alpha
	voxelRGBA.x = color.x * voxelRGBA.w;
	voxelRGBA.y = color.y * voxelRGBA.w;
	voxelRGBA.z = color.z * voxelRGBA.w;

	return voxelRGBA;
}

/******************************************************************************
 * Get the normal of distance field + noise
 *
 * @param voxelPosF 3D position in the current brick
 * @param levelRes number of voxels at the current brick resolution
 *
 * @return ...
 ******************************************************************************/
template< typename TDataStructureType >
__device__
inline float3 ProducerKernel< TDataStructureType >::getNormal( float3 voxelPosF, uint3 levelRes )
{
	// Type definition for the noise
	typedef GvUtils::GvNoiseKernel Noise;

	// Get the normal and the distance without noise.
	float4 voxelNormalAndDist = computeDistance( voxelPosF );
	float distance = voxelNormalAndDist.w;
	float3 voxelNormal= make_float3( voxelNormalAndDist.x, voxelNormalAndDist.y, voxelNormalAndDist.z );

	float eps = 0.5f / static_cast< float >( levelRes.x );

	// Compute symetric gradient noise
	float3 grad_noise = make_float3( 0.0f );
	for ( float frequency = cNoiseFirstFrequency;
		   	 frequency <= cNoiseMaxFrequency ;
		   	frequency *= 2.f ) {
		if ( cNoiseType == SIMPLEX ) {
			grad_noise.x += cNoiseStrength / frequency * GvSimplexNoise::getValue( frequency * ( voxelPosF + make_float3( eps, 0.0f, 0.0f ) - distance * voxelNormal ) )
							-cNoiseStrength / frequency * GvSimplexNoise::getValue( frequency * ( voxelPosF - make_float3( eps, 0.0f, 0.0f ) - distance * voxelNormal ) );

			grad_noise.y += cNoiseStrength / frequency * GvSimplexNoise::getValue( frequency * ( voxelPosF + make_float3( 0.0f, eps, 0.0f ) - distance * voxelNormal ) )
							-cNoiseStrength / frequency * GvSimplexNoise::getValue( frequency * ( voxelPosF - make_float3( 0.0f, eps, 0.0f ) - distance * voxelNormal ) );

			grad_noise.z += cNoiseStrength / frequency * GvSimplexNoise::getValue( frequency * ( voxelPosF + make_float3( 0.0f, 0.0f, eps ) - distance * voxelNormal ) )
							-cNoiseStrength / frequency * GvSimplexNoise::getValue( frequency * ( voxelPosF - make_float3( 0.0f, 0.0f, eps ) - distance * voxelNormal ) );
		} else if ( cNoiseType == PERLIN ) {
			grad_noise.x += cNoiseStrength / frequency * Noise::getValue( frequency * ( voxelPosF + make_float3( eps, 0.0f, 0.0f ) - distance * voxelNormal ) )
							-cNoiseStrength / frequency * Noise::getValue( frequency * ( voxelPosF - make_float3( eps, 0.0f, 0.0f ) - distance * voxelNormal ) );

			grad_noise.y += cNoiseStrength / frequency * Noise::getValue( frequency * ( voxelPosF + make_float3( 0.0f, eps, 0.0f ) - distance * voxelNormal ) )
							-cNoiseStrength / frequency * Noise::getValue( frequency * ( voxelPosF - make_float3( 0.0f, eps, 0.0f ) - distance * voxelNormal ) );

			grad_noise.z += cNoiseStrength / frequency * Noise::getValue( frequency * ( voxelPosF + make_float3( 0.0f, 0.0f, eps ) - distance * voxelNormal ) )
							-cNoiseStrength / frequency * Noise::getValue( frequency * ( voxelPosF - make_float3( 0.0f, 0.0f, eps ) - distance * voxelNormal ) );

		}
	}

	grad_noise *= 0.5f / eps;

	voxelNormal = normalize( voxelNormal + grad_noise - dot( grad_noise, voxelNormal ) * voxelNormal );

	return voxelNormal;
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
template< typename TDataStructureType >
template< typename GPUPoolKernelType >
__device__
inline uint ProducerKernel< TDataStructureType >
::produceData( GPUPoolKernelType& nodePool, uint pRequestID, uint pProcessID, uint3 newElemAddress,
			 const GvCore::GvLocalizationInfo& pParentLocInfo, Loki::Int2Type< 0 > )
{
	// NOTE :
	// In this method, you are inside a node tile.
	// The goal is to determine, for each node of the node tile, which type of data it holds.
	// Data type can be :
	// - a constant region,
	// - a region with data,
	// - a region where max resolution is reached.
	// So, one thread is responsible of the production of one node of a node tile.

	// Process ID gives the 1D index of a node in the current node tile
	if ( pProcessID < NodeRes::getNumElements() ) {

		// First, compute the 3D offset of the node in the node tile
		uint3 subOffset = NodeRes::toFloat3( pProcessID );

		// Node production corresponds to subdivide a node tile.
		// So, based on the index of the node, find the node child.
		// As we want to sudbivide the curent node, we retrieve localization information
		// at the next level
		uint3 regionCoords = pParentLocInfo.locCode.addLevel< NodeRes >( subOffset ).get();
		uint regionDepth = pParentLocInfo.locDepth.addLevel().get();

		// Create a new node for which you will have to fill its information.
		GvStructure::GvNode newnode;
		newnode.childAddress = 0;
		newnode.brickAddress = 0;

		// Call what we call an oracle that will determine the type of the region of the node accordingly
		GPUVoxelProducer::GPUVPRegionInfo nodeinfo = getRegionInfo( regionCoords, regionDepth );

		// Now that the type of the region is found, fill the new node information
		if ( nodeinfo == GPUVoxelProducer::GPUVP_CONSTANT ) {
			newnode.setTerminal( true );
		} else if ( nodeinfo == GPUVoxelProducer::GPUVP_DATA ) {
			newnode.setStoreBrick();
			newnode.setTerminal( false );
		} else if ( nodeinfo == GPUVoxelProducer::GPUVP_DATA_MAXRES ) {
			newnode.setStoreBrick();
			newnode.setTerminal( true );
		}

		// Finally, write the new node information into the node pool by selecting channels :
		// - Loki::Int2Type< 0 >() points to node information
		// - Loki::Int2Type< 1 >() points to brick information
		//
		// newElemAddress.x + pProcessID : is the adress of the new node in the node pool
		nodePool.getChannel( Loki::Int2Type< 0 >() ).set( newElemAddress.x + pProcessID, newnode.childAddress );
		nodePool.getChannel( Loki::Int2Type< 1 >() ).set( newElemAddress.x + pProcessID, newnode.brickAddress );
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
template< typename TDataStructureType >
template< typename GPUPoolKernelType >
__device__
inline uint ProducerKernel< TDataStructureType >
::produceData( GPUPoolKernelType& dataPool, uint pRequestID, uint pProcessID, uint3 newElemAddress,
			 const GvCore::GvLocalizationInfo& pParentLocInfo, Loki::Int2Type< 1 > )
{
	// NOTE :
	// In this method, you are inside a brick of voxels.
	// The goal is to determine, for each voxel of the brick, the value of each of its channels.
	//
	// Data type has previously been defined by the user, as a
	// typedef Loki::TL::MakeTypelist< uchar4, half4 >::Result DataType;
	//
	// In this tutorial, we have choosen two channels containing color at channel 0 and normal at channel 1.

	// Retrieve current brick localization information code and depth
	const GvCore::GvLocalizationInfo::CodeType parentLocCode = pParentLocInfo.locCode;
	const GvCore::GvLocalizationInfo::DepthType parentLocDepth = pParentLocInfo.locDepth;

	// Shared memory declaration
	//
	// Each threads of a block process one and unique brick of voxels.
	// We store in shared memory common useful variables that will be used by each thread.
	__shared__ uint3 smLevelRes;
	__shared__ float3 smLevelResInv;
	__shared__ int3 smBrickPos;
	__shared__ float3 smBrickPosF;
	__shared__ uint3 smElemSize;

	// Shared Memory initialization
	if ( pProcessID == 0) {
		// Compute useful variables used for retrieving positions in 3D space
		smLevelRes = make_uint3( 1 << parentLocDepth.get() ) * BrickRes::get(); // number of voxels (in each dimension)
		smLevelResInv = make_float3( 1.0f ) / make_float3( smLevelRes ); // size of a voxel (in each dimension)
		smBrickPos = make_int3( parentLocCode.get() * BrickRes::get() ) - BorderSize;
		smBrickPosF = make_float3( smBrickPos ) * smLevelResInv;
		smElemSize = BrickRes::get() + make_uint3( 2 * BorderSize ); // Real brick size (with borders)
	}

	// Thread Synchronization
	__syncthreads();

	// The original KERNEL execution configuration on the HOST has a 3D block size :
	// dim3 blockSize( 16, 8, 1 );
	//
	// Each block process one brick of voxels.
	//
	// One thread process only a subset of the voxels of the brick.
	uint3 elemOffset;
	for ( elemOffset.z = 0; elemOffset.z < smElemSize.z; elemOffset.z += blockDim.z ) {
		for ( elemOffset.y = 0; elemOffset.y < smElemSize.y; elemOffset.y += blockDim.y ) {
			for ( elemOffset.x = 0; elemOffset.x < smElemSize.x; elemOffset.x += blockDim.x ) {
				// Compute position index
				const uint3 locOffset = elemOffset + make_uint3( threadIdx.x, threadIdx.y, threadIdx.z );

				// Test if the computed position index is inside the brick (with borders)
				if ( locOffset.x < smElemSize.x && locOffset.y < smElemSize.y && locOffset.z < smElemSize.z ) {
					// Position of the current voxel's center (relative to the brick)
					//
					// In order to make the mip-mapping mecanism OK,
					// data values must be set at the center of voxels.
					const float3 voxelPosInBrickF = ( make_float3( locOffset ) + 0.5f ) * smLevelResInv;
					// Position of the current voxel's center (absolute, in [0.0;1.0] range)
					const float3 voxelPosF = smBrickPosF + voxelPosInBrickF;
					// Position of the current voxel's center (scaled to the range [-1.0;1.0])
					const float3 posF = voxelPosF * 2.0f - 1.0f;

					// Generate normal
					float4 voxelNormal = make_float4 (getNormal( posF, smLevelRes ));
					// Generate color
					float4 voxelColor = getRGBA( posF, smLevelRes );

					// Compute the new element's address where to write in cache (i.e data pool)
					const uint3 destAddress = newElemAddress + locOffset;
					// Write the voxel's color in the first channel
					dataPool.template setValue< 0 >( destAddress, voxelColor );
					// Write the voxel's normal in the second channel
					dataPool.template setValue< 1 >( destAddress, voxelNormal );
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
template< typename TDataStructureType >
__device__
inline GPUVoxelProducer::GPUVPRegionInfo ProducerKernel< TDataStructureType >
::getRegionInfo( uint3 regionCoords, uint regionDepth )
{
	// Limit the depth.
	// Currently, 32 is the max depth of the GigaVoxels engine.
	if ( regionDepth >= 32 ) {
		return GPUVoxelProducer::GPUVP_DATA_MAXRES;
	}

	uint3 brickRes;
	uint3 levelRes;
	float3 levelResInv;
	int3 brickPos;
	float3 brickPosF;

	brickRes = BrickRes::get();
	levelRes = make_uint3( 1 << regionDepth ) * brickRes;
	levelResInv = make_float3( 1.f ) / make_float3( levelRes );

	brickPos = make_int3( regionCoords * brickRes ) - BorderSize;
	brickPosF = make_float3( brickPos ) * levelResInv;

	uint3 elemSize = BrickRes::get() + make_uint3( 2 * BorderSize );
	uint3 elemOffset;

	bool isEmpty = true;

	// Iterate through voxels
	for ( elemOffset.z = 0; elemOffset.z < elemSize.z && isEmpty; elemOffset.z++ ) {
		for ( elemOffset.y = 0; elemOffset.y < elemSize.y && isEmpty; elemOffset.y++ ) {
			for ( elemOffset.x = 0; elemOffset.x < elemSize.x && isEmpty; elemOffset.x++ ) {
				uint3 locOffset = elemOffset;// + make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);

				if ( locOffset.x < elemSize.x && locOffset.y < elemSize.y && locOffset.z < elemSize.z ) {
					// Position of the current voxel's center (relative to the brick)
					float3 voxelPosInBrickF = ( make_float3( locOffset ) + 0.5f ) * levelResInv;
					// Position of the current voxel's center (absolute, in [0.0;1.0] range)
					float3 voxelPosF = brickPosF + voxelPosInBrickF;
					// Position of the current voxel's center (scaled to the range [-1.0;1.0])
					const float3 posF = voxelPosF * 2.0f - 1.0f;

					// Test opacity to determine if there is data
					float alpha = getAlpha( posF, levelRes );
					if ( alpha > 0.0f ) {
						isEmpty = false;
					}
				}
			}
		}
	}

	if ( isEmpty ) {
		return GPUVoxelProducer::GPUVP_CONSTANT;
	}

	return GPUVoxelProducer::GPUVP_DATA;
}
