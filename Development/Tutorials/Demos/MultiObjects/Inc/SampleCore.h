/*
 * Copyright (C) 2011 Fabrice Neyret <Fabrice.Neyret@imag.fr>
 * Copyright (C) 2011 Cyril Crassin <Cyril.Crassin@icare3d.org>
 * Copyright (C) 2011 Morgan Armand <morgan.armand@gmail.com>
 *
 * This file is part of Gigavoxels.
 *
 * Gigavoxels is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Gigavoxels is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Gigavoxels.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _SAMPLECORE_H_
#define _SAMPLECORE_H_

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_types_ext.h>

#include <cutil_math.h>

#include <loki/Typelist.h>

// Forward references
namespace gigavoxels
{
	template<
		typename NodePoolType, typename BrickPoolType,
		class NodeTileRes, class BrickRes, uint BorderSize >
	struct VolumeTree;

	template<uint r>
	struct StaticRes1D;

	template<typename T>
	class Array3DGPULinear;

	template<typename T>
	class Array3DGPUTex;

	template <template<typename> class HostArray, class TList >
	class GPUPoolHost;
}

// Producers
template <typename NodeRes, typename BrickRes, uint BorderSize, typename VolumeTreeType>
class SphereProducer;

// Shaders
class SphereShader;

// Renderers
template<
	class VolTreeType, class NodeResolution,
	class BrickResolution, uint BorderSize,
	class GPUProducer, class SampleShader >
class RendererVolTreeCUDA;

// BEGIN: EXPERIMENTAL
template <typename NodeRes, typename BrickRes, uint BorderSize, typename VolumeTreeType>
class CubeProducer;

// Shaders
class CubeShader;
// END: EXPERIMENTAL

// Defines the type list representing the content of one node
typedef Loki::TL::MakeTypelist<uint, uint>::Result NodeType;

// Defines the type list representing the content of one voxel
typedef Loki::TL::MakeTypelist<uchar4, half4>::Result DataType;

// Defines the size of a node tile
typedef gigavoxels::StaticRes1D<2> NodeRes;

// Defines the size of a brick
typedef gigavoxels::StaticRes1D<8> BrickRes;

// Defines the size of the border around a brick
enum { BrickBorderSize = 1 };

// Defines the total size of a brick
typedef gigavoxels::StaticRes1D<8 + 2 * BrickBorderSize> RealBrickRes;

// Defines the type of the node pool
typedef gigavoxels::GPUPoolHost<gigavoxels::Array3DGPULinear, NodeType> NodePoolType;

// Defines the type of the brick pool
typedef gigavoxels::GPUPoolHost<gigavoxels::Array3DGPUTex, DataType> BrickPoolType;

// Defines the type of structure we want to use. Array3DGPUTex is the type of array used 
// to store the bricks.
typedef gigavoxels::VolumeTree<
	NodePoolType, BrickPoolType,
	NodeRes, BrickRes, BrickBorderSize >		VolumeTreeType;

// Defines the type of the producer
typedef SphereShader ShaderType;

// Defines the type of the shader
typedef SphereProducer< NodeRes, BrickRes,
	BrickBorderSize, VolumeTreeType >			ProducerType;

// Defines the type of the renderer we want to use.
typedef RendererVolTreeCUDA< VolumeTreeType,
	NodeRes, BrickRes, BrickBorderSize,
	ProducerType, ShaderType >					RendererType;

// BEGIN: EXPERIMENTAL
// Defines the type of the producer
typedef CubeShader CubeShaderType;

// Defines the type of the shader
typedef CubeProducer< NodeRes, BrickRes,
	BrickBorderSize, VolumeTreeType >			CubeProducerType;

// Defines the type of the renderer we want to use.
typedef RendererVolTreeCUDA< VolumeTreeType,
	NodeRes, BrickRes, BrickBorderSize,
	CubeProducerType, CubeShaderType >			CubeRendererType;
// END: EXPERIMENTAL

class SampleCore
{
public:
	SampleCore();
	~SampleCore();

	void init();
	void draw();
	void resize(int width, int height);

	void clearCache();

	void toggleDisplayOctree();
	void toggleDynamicUpdate();
	void togglePerfmonDisplay(uint mode);

	void incMaxVolTreeDepth();
	void decMaxVolTreeDepth();

private:
	VolumeTreeType	*mVolumeTree;
	RendererType	*mRenderer;
	ProducerType	*mProducer;

	GLuint			mColorBuffer;
	GLuint			mDepthBuffer;

	GLuint			mColorTex;
	GLuint			mDepthTex;

	GLuint			mFrameBuffer;

	struct cudaGraphicsResource	*mColorResource;
	struct cudaGraphicsResource	*mDepthResource;

	int				mWidth;
	int				mHeight;

	bool			mDisplayOctree;
	uint			mDisplayPerfmon;
	uint			mMaxVolTreeDepth;

	// BEGIN: Experimental
	VolumeTreeType		*mCubeVolumeTree;
	CubeRendererType	*mCubeRenderer;
	CubeProducerType	*mCubeProducer;
	// END: Experimental
};

#endif // !_SAMPLECORE_H_