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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvStructure/GvNode.h>
#include <GvRendering/GvNodeVisitorKernel.h>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

//__device__
//inline float fastInvSqrt(float x) {
//   float xhalf = 0.5f * x;
//   int i = __float_as_int(x); // store floating-point bits in integer
//   i = 0x5f3759d5 - (i >> 1); // initial guess for Newton's method
//   x = __int_as_float(i); // convert new bits into float
//   x = x*(1.5f - xhalf*x*x); // One round of Newton's method
//   return x;
//}
//
//__device__
//inline float invSqrt(float x) {
//  return rsqrt(x);
//}

/******************************************************************************
 * HELPER function
 *
 * @param depth ...
 *
 * @return ...
 ******************************************************************************/
__device__
inline uint getNumIter( uint depth )
{
	if ( cHasFactalAdaptativeIterations )
	{
		return static_cast< uint >( ceilf( ( log2f( static_cast< float >( depth ) ) + 1.0f ) * 1.5f ) );
	}
	else
	{
		return cFractalNbIterations;
	}
}

///******************************************************************************
// * ...
// *
// * @param z ...
// * @param grad ...
// * @param maxIter ...
// *
// * @return ...
// ******************************************************************************/
//__device__
//inline float compMandelbrotNewOptim( float3 z, float3& pGrad, int maxIter )
//{
//	// 
//	// http://en.wikipedia.org/wiki/Orbit_trap
//	// http://www.iquilezles.org/www/articles/ftrapsgeometric/ftrapsgeometric.htm
//	// http://www.iquilezles.org/www/articles/orbittraps3d/orbittraps3d.htm
//	
//	// Algorithmn
//	//
//	// We can estimate the distance to the fractal surface as :
//	// d = |Zn| log( |Zn| ) / |Z'n|
//	//
//	// see http://www.iquilezles.org/www/articles/distancefractals/distancefractals.htm
//
//	// Mandelbrot
//	// Z[n+1] = Z[n]� + C
//
//	// Z
//
//	// power of fractal
//	const float cPower = 8.0f;
//
//	const float cThreshold = 4.0f;
//	const float cMinDist = 9999.0f;
//	const int cMaxIterations = maxIter;
//	const float cPhase1 = 0.0f;
//	const float cPhase = 0.0f;
//
//	const float3 c = z;			// constant term in iterative formula
//
//	// divergence threshold nearly infinite for negative powers
//	const float d_threshold = ( cPower < 0.0f ) ? 1E9 : min( 2.0f, __powf( cThreshold, 1.0f / cPower ) );
//
//	// orbit trapping continues at existing cMinDist
//	float min_dst = cMinDist;
//
//	// point z polar coordinates
//	float z_rho  = length( z );
//	float z_theta = atan2f( z.y, z.x );
//	float z_phi = asinf( z.z / z_rho ) + cPhase1;  // initial phase offset
//
//	// orbit trapping relative to point (0,0,0)
//	if ( z_rho < min_dst )
//	{
//		min_dst = z_rho;
//	}
//
//	float3 dz;
//	float dz_phi = 0.0f;
//	float dz_theta = 0.0f;
//	float dz_rho  = 1.0f;
//
//	// Iterate to compute the distance estimator.
//	int i = cMaxIterations;
//
//	while ( i-- )
//	{
//		if ( cPower > 0.0f )
//		{
//			// positive powers cPower
//
//			// purely scalar dz iteration (thanks to Enforcer for the great tip)
//			float zr = __powf( z_rho, cPower - 1 );
//			dz_rho = zr * dz_rho * cPower + 1;
//
//			// z iteration
//			float P_ = zr * z_rho; // equivalent to __powf( z_rho, cPower );
//			float s1, c1; __sincosf( cPower * z_phi, &s1, &c1 );
//			float s2, c2; __sincosf( cPower * z_theta, &s2, &c2 );
//			z.x = P_ * c1 * c2 + c.x;
//			z.y = P_ * c1 * s2 + c.y;
//			z.z = P_ * s1 + c.z;
//		}
//		else
//		{
//			// negative powers cPower
//
//			// derivative dz iteration
//			float pP_ = cPower * __powf( z_rho, cPower - 1.0f );
//			float c1, s1;   __sincosf( dz_phi + ( cPower - 1.0f ) * z_phi, &s1, &c1 );
//			float s1_, c1_; __sincosf( dz_theta + ( cPower - 1.0f ) * z_theta, &s1_, &c1_ );
//			dz.x = pP_ * dz_rho * c1 * c1_ + 1.0f;
//			dz.y = pP_ * dz_rho * c1 * s1_;
//			dz.z = pP_ * dz_rho * s1;
//
//			// polar coordinates of derivative dz
//			dz_rho  = sqrtf( dz.x * dz.x + dz.y * dz.y + dz.z * dz.z );
//			dz_theta = atan2f( dz.y, dz.x );
//			dz_phi = asinf( dz.z / dz_rho );
//
//			// z iteration
//			float P_ = __powf( z_rho, cPower );
//			float s2, c2; __sincosf( cPower * z_phi, &s2, &c2 );
//			float s2_, c2_; __sincosf( cPower * z_theta, &s2_, &c2_ );
//			z.x = P_ * c2 * c2_ + c.x;
//			z.y = P_ * c2 * s2_ + c.y;
//			z.z = P_ * s2 + c.z;
//		}
//		// compute new length of z
//		z_rho  = length( z );
//
//		// results are not stored for the "extra" iteration at i == 0
//		// orbit trapping relative to point (0,0,0)
//		if ( z_rho < min_dst )
//		{
//			min_dst = z_rho;
//		}
//
//		// Stop when we know the point diverges and return the result.
//		if( z_rho > d_threshold )
//		{
//			break;
//		}
//
//		// compute remaining polar coordinates of z and iterate further
//		z_theta = atan2f( z.y, z.x );
//		z_phi = asinf( z.z / z_rho ) + cPhase;  // iterative phase offset
//	}
//
//	// return the result if we reached convergence
//	///cMinDist = min_dst;
//	pGrad.x = min_dst;
//
//	return 0.5f * z_rho * __logf( z_rho ) / dz_rho;
//}

/******************************************************************************
 * ...
 *
 * @param z ...
 * @param grad ...
 * @param maxIter ...
 *
 * @return ...
 ******************************************************************************/
__device__
inline float compMandelbrotNewOptim( float3 z, float3& pGrad, int maxIter )
{
	// 
	// http://en.wikipedia.org/wiki/Orbit_trap
	// http://www.iquilezles.org/www/articles/ftrapsgeometric/ftrapsgeometric.htm
	// http://www.iquilezles.org/www/articles/orbittraps3d/orbittraps3d.htm

	// http://blog.hvidtfeldts.net/index.php/category/mandelbulb/
	
	// Algorithmn
	//
	// We can estimate the distance to the fractal surface as :
	// d = |Zn| log( |Zn| ) / |Z'n|
	//
	// see http://www.iquilezles.org/www/articles/distancefractals/distancefractals.htm

	// Mandelbrot
	// Z[n+1] = Z[n]� + C

	// Z

	// http://www.iquilezles.org/www/articles/mandelbulb/mandelbulb.htm
	// http://www.skytopia.com/project/fractal/mandelbulb.html
	// http://www.skytopia.com/project/fractal/2mandelbulb.html
	// http://blog.hvidtfeldts.net/index.php/category/mandelbulb/
	// http://www.subblue.com/blog/2009/12/13/mandelbulb

	const float cThreshold = 4.0f;
	const float cMinDist = 9999.0f;
	const int cMaxIterations = maxIter;
	const float cPhase1 = 0.0f;
	const float cPhase = 0.0f;

	const float3 c = z;			// constant term in iterative formula

	// divergence threshold nearly infinite for negative powers
	const float d_threshold = ( cFractalPower < 0.0f ) ? 1E9 : min( 2.0f, __powf( cThreshold, 1.0f / cFractalPower ) );

	// orbit trapping continues at existing cMinDist
	float min_dst = cMinDist;

	// Polar coordinates
	float z_rho = length( z );
	float z_theta = atan2f( z.y, z.x );
	float z_phi = asinf( z.z / z_rho ) + cPhase1;  // initial phase offset

	// orbit trapping relative to point (0,0,0)
	if ( z_rho < min_dst )
	{
		min_dst = z_rho;
	}

	//float3 dz;
	//float dz_phi = 0.0f;
	//float dz_theta = 0.0f;
	float dz_rho = 1.0f;

	// Iterate to compute the distance estimator
	int i = cMaxIterations;
	while ( i-- )
	{
		// purely scalar dz iteration (thanks to Enforcer for the great tip)
		float zr = __powf( z_rho, cFractalPower - 1.0f );
		dz_rho = cFractalPower * zr * dz_rho + 1.0f;

		// Z[n+1] = Z[n]� + C

		// Z = Zn + C
		// Z' = n . Z . Z' + 1

		// Z = Rn . < ( cos( n.theta ) . cos( n.Phi ) ), >
		// Z' = n . Rn-1 . < ( cos( n.theta ) . cos( n.Phi ) ), >

		// z iteration
		float P_ = zr * z_rho; // equivalent to __powf( z_rho, cFractalPower );
		float s1, c1; __sincosf( cFractalPower * z_phi, &s1, &c1 );
		float s2, c2; __sincosf( cFractalPower * z_theta, &s2, &c2 );
		z.x = P_ * c1 * c2 + c.x;
		z.y = P_ * c1 * s2 + c.y;
		z.z = P_ * s1 + c.z;

		// Z[n] = pow(R,8) * ( cos_n_phi * cos_n_tetha, cos_n_phi * sin_n_tetha, sin_n_phi ) + C

		// compute new length of z
		z_rho = length( z );

		// results are not stored for the "extra" iteration at i == 0
		// orbit trapping relative to point (0,0,0)
		if ( z_rho < min_dst )
		{
			min_dst = z_rho;
		}

		// Stop when we know the point diverges and return the result.
		if( z_rho > d_threshold )
		{
			break;
		}

		// compute remaining polar coordinates of z and iterate further
		z_theta = atan2f( z.y, z.x );
		z_phi = asinf( z.z / z_rho ) + cPhase;  // iterative phase offset
	}

	// return the result if we reached convergence
	///cMinDist = min_dst;
	pGrad.x = min_dst;

	//return 0.5f * z_rho * __logf( z_rho ) / dz_rho;		// => why 1/2 ? It seems to be an error, IQuilez used z�, that's why...
	return z_rho * __logf( z_rho ) / dz_rho;
}

/******************************************************************************
 * ...
 *
 * @param x ...
 * @param grad ...
 * @param maxIter ...
 *
 * @return ...
 ******************************************************************************/
__device__
inline float compMandelbrot( float3 x, float3& grad, int maxIter )
{
	return compMandelbrotNewOptim( x, grad, maxIter );
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
	_dataStructureKernel = pDataStructure;
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
::produceData( GPUPoolKernelType& nodePool, uint requestID, uint processID, uint3 newElemAddress,
			  const GvCore::GvLocalizationInfo& parentLocInfo, Loki::Int2Type< 0 > )
{
	// NOTE :
	// In this method, you are inside a node tile.
	// The goal is to determine, for each node of the node tile, which type of data it holds.
	// Data type can be :
	// - a constant region,
	// - a region with data,
	// - a region where max resolution is reached.
	// So, one thread is responsible of the production of one node of a node tile.
	
	// Retrieve current node tile localization information code and depth
	const GvCore::GvLocalizationInfo::CodeType *parentLocCode = &parentLocInfo.locCode;
	const GvCore::GvLocalizationInfo::DepthType *parentLocDepth = &parentLocInfo.locDepth;

	// Process ID gives the 1D index of a node in the current node tile
	if ( processID < NodeRes::getNumElements() )
	{
		// First, compute the 3D offset of the node in the node tile
		uint3 subOffset = NodeRes::toFloat3( processID );

		// Node production corresponds to subdivide a node tile.
		// So, based on the index of the node, find the node child.
		// As we want to sudbivide the curent node, we retrieve localization information
		// at the next level
		uint3 regionCoords = parentLocCode->addLevel< NodeRes >( subOffset ).get();
		uint regionDepth = parentLocDepth->addLevel().get();

		// Create a new node for which you will have to fill its information.
		GvStructure::GvNode newnode;
		newnode.childAddress = 0;
		newnode.brickAddress = 0;

		// Call what we call an oracle that will determine the type of the region of the node accordingly
		GPUVoxelProducer::GPUVPRegionInfo nodeinfo = getRegionInfo( regionCoords, regionDepth );

		// Now that the type of the region is found, fill the new node information
		if ( nodeinfo == GPUVoxelProducer::GPUVP_CONSTANT )
		{
			newnode.setTerminal( true );
		}
		else if ( nodeinfo == GPUVoxelProducer::GPUVP_DATA )
		{
			newnode.setStoreBrick();
			newnode.setTerminal( false );
		}
		else if ( nodeinfo == GPUVoxelProducer::GPUVP_DATA_MAXRES )
		{
			newnode.setStoreBrick();
			newnode.setTerminal( true );
		}

		// Finally, write the new node information into the node pool by selecting channels :
		// - Loki::Int2Type< 0 >() points to node information
		// - Loki::Int2Type< 1 >() points to brick information
		//
		// newElemAddress.x + processID : is the adress of the new node in the node pool
		nodePool.getChannel( Loki::Int2Type< 0 >() ).set( newElemAddress.x + processID, newnode.childAddress );
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
template< typename TDataStructureType >
template< typename GPUPoolKernelType >
__device__
inline uint ProducerKernel< TDataStructureType >
::produceData( GPUPoolKernelType& dataPool, uint requestID, uint processID, uint3 newElemAddress,
			  const GvCore::GvLocalizationInfo& parentLocInfo, Loki::Int2Type< 1 > )
{
	// NOTE :
	// In this method, you are inside a brick of voxels.
	// The goal is to determine, for each voxel of the brick, the value of each of its channels.
	//
	// Data type has previously been defined by the user, as a
	// typedef Loki::TL::MakeTypelist< uchar4 >::Result DataType;
	//
	// In this tutorial, we have choosen one channel containing normal and opacity at channel 0.

	// Retrieve current brick localization information code and depth
	uint3 parentLocCode = parentLocInfo.locCode.get();
	uint parentLocDepth = parentLocInfo.locDepth.get();

	// Shared memory declaration
	//
	// Each threads of a block process one and unique brick of voxels.
	// We store in shared memory common useful variables that will be used by each thread.
	__shared__ uint level;
	__shared__ uint3 blockRes;
	__shared__ uint3 levelRes; 
	__shared__ float3 levelResInv; 
	__shared__ int3 brickPos;
	__shared__ float3 brickPosF;
	
	__shared__ uint maxIter;
	__shared__ uint maxIterGrad;

	// Compute useful variables used for retrieving positions in 3D space
    //uint3 elemSize = BrickRes::get() + make_uint3( 2 * BorderSize );

	//level=(uint)parentLocDepth+1;
	level = parentLocDepth;
	blockRes = BrickRes::get();
	levelRes = make_uint3( 1 << parentLocDepth ) * blockRes;
	levelResInv = make_float3( 1.0f ) / make_float3( levelRes );
	brickPos = make_int3( parentLocCode * blockRes ) - BorderSize;
	brickPosF = make_float3( brickPos ) * levelResInv;

	maxIter = getNumIter( level ) * evalIterCoef;
	maxIterGrad = getNumIter( level ) * gradIterCoef;

	// Shared Memory declaration
	//
	// - optimization to be able to modify the "content" of the node
	//   => if it is "empty", it returns 2 to modify node "state"
    __shared__ bool smNonNull;
	if ( threadIdx.x == 0 && threadIdx.y == 0 )
	{
        smNonNull = false;
	}
	// Thread Synchronization
	__syncthreads();

	//// Shared Memory declaration
	////
	//__shared__ bool b0_OK;
	//__shared__ float b0_nodeSize;
	//__shared__ float3 b0_nodePos;
	//__shared__ float3 b0_brickPosInPool;
	//__shared__ float b0_brickScaleInPool;

	//float3 samplePos = brickPosF + ( make_float3( elemSize ) * 0.5f + 1.0f ) * levelResInv;

	//GvStructure::GvNode pnode;
	//float pnodeSize = 1.0f;
	//float3 pnodePos = make_float3( 0.0f );
	//uint pnodeDepth = 0;
	//uint pbrickAddressEnc = 0;
	//float3 pbrickPos = make_float3( 0.0f );
	//float pbrickScale = 1.0f;

	// TO DO
	//
	// - this method seems to make the program crash in Debug mode
	//GvRendering::GvNodeVisitorKernel::visit( _dataStructureKernel, parentLocDepth - 3, samplePos, _dataStructureKernel._rootAddress,
	//										 pnode, pnodeSize, pnodePos, pnodeDepth, pbrickAddressEnc, pbrickPos, pbrickScale );

	//b0_OK = pbrickAddressEnc;
	//b0_nodeSize = pnodeSize;
	//b0_nodePos = pnodePos; 

	//b0_brickScaleInPool = pbrickScale;

	//b0_brickPosInPool = make_float3( GvStructure::GvNode::unpackBrickAddress( pbrickAddressEnc ) ) *
	//	_dataStructureKernel.brickCacheResINV + pbrickPos * _dataStructureKernel.brickSizeInCacheNormalized.x;

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

    // - number of voxels
    const uint3 elemSize = BrickRes::get() + make_uint3( 2 * BorderSize );  // real brick size (with borders)
    const int nbVoxels = elemSize.x * elemSize.y * elemSize.z;
    // - number of threads
    //const int nbThreads = blockDim.x * blockDim.y * blockDim.z;
    const int nbThreads = blockDim.x;
    // - global thread index in the block (linearized)
    //const int threadIndex1D = threadIdx.z * ( blockDim.x * blockDim.y ) + ( threadIdx.y * blockDim.x + threadIdx.x ); // written in FMAD-style
    const int threadIndex1D = threadIdx.x;
    uint3 locOffset;
    for ( int index = threadIndex1D; index < nbVoxels; index += nbThreads )
    {
       // Transform 1D per block?s global thread index to associated thread?s 3D voxel position
       locOffset.x = index % elemSize.x;
       locOffset.y = ( index / elemSize.x ) % elemSize.y;
       locOffset.z = index / ( elemSize.x * elemSize.y );

                    //int3 voxelPos= brickPos + locOffset ;
					//float3 voxelPosF=(make_float3(voxelPos)+0.5f)*levelResInv;

                    float3 voxelPosInBrickF = ( make_float3( locOffset ) + 0.5f ) * levelResInv;
					float3 voxelPosF = brickPosF + voxelPosInBrickF;

					// Transform coordinates from [ 0.0, 1.0 ] to [ -1.0, 1.0 ]
					float3 pos = voxelPosF * 2.0f - 1.0f;

					// Estimate distance from point to fractal (shortest distance)
					float3 dz = make_float3( 0.0f, 0.0f, 0.0f );
					float dist = compMandelbrot( pos, dz, maxIter );

					float val = dist * 0.5f;

					// Compute derivative (i.e. gradient)
					float step = levelResInv.x; // vodel size
					float3 ndz;
					float3 grad;
					grad.x = compMandelbrot( pos + make_float3( +step, 0.0f, 0.0f ), ndz, maxIterGrad )
						   - compMandelbrot( pos + make_float3( -step, 0.0f, 0.0f ), ndz, maxIterGrad );
					grad.y = compMandelbrot( pos + make_float3( 0.0f, +step, 0.0f ), ndz, maxIterGrad )
						   - compMandelbrot( pos + make_float3( 0.0f, -step, 0.0f ), ndz, maxIterGrad );
					grad.z = compMandelbrot( pos + make_float3( 0.0f, 0.0f, +step ), ndz, maxIterGrad )
						   - compMandelbrot( pos + make_float3( 0.0f, 0.0f, -step ), ndz, maxIterGrad );

					float vis = 1.0f;

					// FIXME: Broken, need to figure out why.
					//if ( b0_OK )
					//{
					//	float3 voxelPosInNode = voxelPosF - b0_nodePos;

					//	//float dpscale;
					//	//float3 voxelPosInBrick = _dataStructureKernel.getPosInBrick( b0_nodeSize, voxelPosInNode, dpscale );
					//	float3 voxelPosInBrick = voxelPosInNode * _dataStructureKernel.brickSizeInCacheNormalized.x / b0_nodeSize;
					//	voxelPosInBrick = voxelPosInBrick * b0_brickScaleInPool;

					//	float4 pVox = _dataStructureKernel.template getSampleValueTriLinear< 0 >( b0_brickPosInPool, voxelPosInBrick );

					//	float d = pVox.w / float( distanceMultiplier );
					
					//	float curD = val;

					//	vis = d / curD;
					//	vis = vis * vis;

					//	vis = clamp( vis, 0.0f, 1.0f );
					//}	

					grad = normalize( grad ) * vis;
					grad = ( grad + 1.0f ) * 0.5f;

					val = val * float( distanceMultiplier );
					if ( val < 1.0f )
					{
                        smNonNull = true;
					}

					// set opacity in [ 0.0, 1.0 ]
					val = max( min( val, 1.0f ), 0.0f );

					// Voxel data
					//
					// - normal and opacity
					float4 finalValue;
					finalValue.x = grad.x;
					finalValue.y = grad.y;
					finalValue.z = grad.z;
					finalValue.w = val;

					// Compute the new element's address in cache (i.e. data pool)
                    uint3 destAddress = newElemAddress + make_uint3( locOffset );
					// Write the voxel's data in the first field
					dataPool.template setValue< 0 >( destAddress, finalValue );
	}

	// Thread Synchronization
	__syncthreads();
	// Optimization to be able to modify the "content" of the node
	//   => if it is "empty", it returns 2 to modify node "state"
    if ( ! smNonNull )
	{
		return 2;
	}
	
	// Default value
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
::getRegionInfo( uint3 pRegionCoords, uint pRegionDepth )
{
	// GigaVoxels work in Model space,
	// i.e. a BBox lying in [ 0.0, 1.0 ] x [ 0.0, 1.0 ] x [ 0.0, 1.0 ]

	// Number of nodes in each dimension (at current depth)
	float3 levelRes = make_float3( static_cast< float >( 1 << pRegionDepth ) );
	// Size of node
	float3 nodeSize = make_float3( 1.0f ) / levelRes;
	// Bottom left corner of node
	float3 nodePos = make_float3( pRegionCoords ) * nodeSize;
	// Node center
	float3 nodeCenter = nodePos + nodeSize * 0.5f;

	// Transform coordinates from [ 0.0, 0.0 ] to [ -1.0, 1.0 ]
	// - node center
	float3 nodeCenterMandel = ( nodeCenter * 2.0f - 1.0f );

	// Derivative
	float3 dz = make_float3( 1.0f, 0.0f, 0.0f );

	float distMandel = 1000.0f;

	uint maxIter = getNumIter( pRegionDepth ) * regionInfoIterCoef;

	float3 offset;
	for ( offset.z =- 1.0f; offset.z <= 1.0f; offset.z += 1.0f )
	{
		for ( offset.y =- 1.0f; offset.y <= 1.0f; offset.y += 1.0f )
		{
			for ( offset.x =- 1.0f; offset.x <= 1.0f; offset.x += 1.0f )
			{
				// Compute distance estimation
				float distMandel0 = compMandelbrot( nodeCenterMandel + offset * nodeSize, dz, maxIter );

				distMandel = min( distMandel, distMandel0 );
			}
		}
	}

	// Check criteria
	if ( distMandel <= 0.0f )
	{
		if ( pRegionDepth == GvCore::GvLocalizationInfo::maxDepth )
		{
			// Region with max level of detail
			return GPUVoxelProducer::GPUVP_DATA_MAXRES;
		}
		else
		{
			// Region with data inside
			return GPUVoxelProducer::GPUVP_DATA; 
		}
	}
	else
	{
		// Empty region
		return GPUVoxelProducer::GPUVP_CONSTANT;
	}
}

