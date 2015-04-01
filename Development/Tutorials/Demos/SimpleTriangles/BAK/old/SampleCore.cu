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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/StaticRes3D.h>
#include <GvPerfMon/GvPerformanceMonitor.h>

// Cutil
#include <cutil_inline.h>
#include <cutil_math.h>
#include <cutil_gl_error.h>

// Simple Triangles
#include "SampleCore.h"
#include "BvhTree.h"
#include "BvhTreeCache.h"
#include "BvhTreeRenderer.h"
#include "GPUTriangleProducerBVH.h"

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

// Defines the size allowed for each type of pool
#define NODEPOOL_MEMSIZE	( 8U * 1024U * 1024U )		// 8 Mo
#define BRICKPOOL_MEMSIZE	( 256U * 1024U * 1024U )	// 256 Mo

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
SampleCore::SampleCore()
	: mColorTex(0)
	, mDepthTex(0)
	, mFrameBuffer(0)
	, mColorBuffer(0)
	, mDepthBuffer(0)
	, mColorResource(0)
	, mDepthResource(0)
	, mDisplayOctree(false)
	, mDisplayPerfmon(0)
	, mMaxVolTreeDepth( 5 )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
	delete mBvhTreeRenderer;
	delete mBvhTreeCache;
	delete mBvhTree;
	delete mProducer;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::init()
{
	CUDAPM_INIT();

	// Initialize CUDA with OpenGL Interoperability
	cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
	CUT_CHECK_ERROR( "cudaGLSetGLDevice" );

	// FIXME: what is that ?
	uint3 volTreePoolRes = make_uint3( BVH_NODE_POOL_SIZE, 1, 1 );
	uint3 vertexPoolRes = make_uint3( BVH_VERTEX_POOL_SIZE, 1, 1 );
	uint3 nodeTileRes = make_uint3( 2, 1, 1 );
	uint3 vertexTileRes = make_uint3( BVH_DATA_PAGE_SIZE, 1, 1 );

	// Instanciate our objects
	mBvhTree = new BvhTreeType( BVH_NODE_POOL_SIZE, BVH_VERTEX_POOL_SIZE );

	//mProducer = new ProducerType( mBvhTree, "../../media/meshes/sponza.obj" );
	mProducer = new ProducerType( mBvhTree, "sponza.obj" );	// ajouter un test sur le fichier d'entrée (warning, error, etc..)

	mBvhTreeCache = new BvhTreeCacheType( mBvhTree, mProducer, volTreePoolRes, nodeTileRes, vertexPoolRes, vertexTileRes );
	mBvhTreeRenderer = new RendererType( mBvhTree, mBvhTreeCache, mProducer );

	// Configure the renderer
	mBvhTreeRenderer->setMaxVolTreeDepth( mMaxVolTreeDepth );
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::draw()
{
	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT(frame);
	CUDAPM_START_EVENT(app_init_frame);

	glBindFramebuffer(GL_FRAMEBUFFER, mFrameBuffer);

	glClearColor(0.0f, 0.1f, 0.3f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);

	// draw the octree where the sphere will be
	if (mDisplayOctree)
	{
		glPushMatrix();
		glTranslatef(-0.5f, -0.5f, -0.5f);
		//mBvhTreeRenderer->renderFullGL();
		//mProducer->renderFullGL();
		//mProducer->renderGL();
		mProducer->renderDebugGL();
		glPopMatrix();
	}
	
	// copy the current scene into PBO
	glBindBuffer(GL_PIXEL_PACK_BUFFER, mColorBuffer);
	glReadPixels(0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	CUT_CHECK_ERROR_GL();

	glBindBuffer(GL_PIXEL_PACK_BUFFER, mDepthBuffer);
	glReadPixels(0, 0, mWidth, mHeight, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	CUT_CHECK_ERROR_GL();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// extract view transformations
	float4x4 viewMatrix;
	float4x4 projectionMatrix;
	glGetFloatv(GL_MODELVIEW_MATRIX, viewMatrix._array);
	glGetFloatv(GL_PROJECTION_MATRIX, projectionMatrix._array);

	// extract viewport
	GLint params[4];
	glGetIntegerv(GL_VIEWPORT, params);
	int4 viewport = make_int4(params[0], params[1], params[2], params[3]);

	// render the scene into textures
	CUDAPM_STOP_EVENT(app_init_frame);

	// build the world transformation matrix
	float4x4 modelMatrix;

	glPushMatrix();
	glLoadIdentity();
	//glTranslatef(-0.5f, -0.5f, -0.5f);
	glGetFloatv(GL_MODELVIEW_MATRIX, modelMatrix._array);
	glPopMatrix();

	// render
	mBvhTreeRenderer->renderImpl(modelMatrix, viewMatrix, projectionMatrix, viewport);

	// upload changes into the textures
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mColorBuffer);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, mColorTex);
	glTexSubImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, 0, 0, mWidth, mHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	/*glBindBuffer(GL_PIXEL_UNPACK_BUFFER, mDepthBuffer);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, mDepthTex);
	glTexSubImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, 0, 0, mWidth, mHeight, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);*/

	// render the result to the screen
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	glEnable(GL_TEXTURE_RECTANGLE_EXT);
	glDisable(GL_DEPTH_TEST);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, mColorTex);

	GLint sMin = 0;
	GLint tMin = 0;
	GLint sMax = mWidth;
	GLint tMax = mHeight;

	glBegin(GL_QUADS);
		glColor3f(1.0f, 1.0f, 1.0f);
		glTexCoord2i(sMin, tMin); glVertex2i(-1, -1);
		glTexCoord2i(sMax, tMin); glVertex2i( 1, -1);
		glTexCoord2i(sMax, tMax); glVertex2i( 1,  1);
		glTexCoord2i(sMin, tMax); glVertex2i(-1,  1);
	glEnd();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);

	glDisable(GL_TEXTURE_RECTANGLE_EXT);

	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	mBvhTreeRenderer->nextFrame();

	CUDAPM_STOP_EVENT(frame);
	CUDAPM_STOP_FRAME;

	if (mDisplayPerfmon)
		GvPerfMon::CUDAPerfMon::getApplicationPerfMon().displayFrameGL(mDisplayPerfmon - 1);
}

/******************************************************************************
 * ...
 *
 * @param width ...
 * @param height ...
 ******************************************************************************/
void SampleCore::resize( int width, int height )
{
	mWidth = width;
	mHeight = height;

	// Re-init Perfmon subsystem
	CUDAPM_RESIZE(make_uint2(mWidth, mHeight));

	// Create frame-dependent objects
	if (mColorResource)
		CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(mColorResource));
	if (mDepthResource)
		CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(mDepthResource));

	if (mColorBuffer)
		glDeleteBuffers(1, &mColorBuffer);
	if (mDepthBuffer)
		glDeleteBuffers(1, &mDepthBuffer);

	if (mColorTex)
		glDeleteTextures(1, &mColorTex);
	if (mDepthTex)
		glDeleteTextures(1, &mDepthTex);

	if (mFrameBuffer)
		glDeleteFramebuffers(1, &mFrameBuffer);

	glGenBuffers(1, &mColorBuffer);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, mColorBuffer);
	glBufferData(GL_PIXEL_PACK_BUFFER, width * height * sizeof(GLubyte) * 4, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	CUT_CHECK_ERROR_GL();

	glGenTextures(1, &mColorTex);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, mColorTex);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);
	CUT_CHECK_ERROR_GL();

	glGenBuffers(1, &mDepthBuffer);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, mDepthBuffer);
	glBufferData(GL_PIXEL_PACK_BUFFER, width * height * sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	CUT_CHECK_ERROR_GL();

	glGenTextures(1, &mDepthTex);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, mDepthTex);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, GL_DEPTH24_STENCIL8_EXT, width, height, 0, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, NULL);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);
	CUT_CHECK_ERROR_GL();

	glGenFramebuffers(1, &mFrameBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, mFrameBuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_EXT, mColorTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, mDepthTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, mDepthTex, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	CUT_CHECK_ERROR_GL();

	// Create CUDA resources from OpenGL objects
	CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&mColorResource, mColorBuffer, cudaGraphicsRegisterFlagsNone));
	CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&mDepthResource, mDepthBuffer, cudaGraphicsRegisterFlagsNone));

	//// Pass resources to the renderer
	mBvhTreeRenderer->setColorResource(mColorResource);
	mBvhTreeRenderer->setDepthResource(mDepthResource);
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::clearCache()
{
	//mBvhTreeRenderer->clearCache();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::toggleDisplayOctree()
{
	mDisplayOctree = !mDisplayOctree;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::toggleDynamicUpdate()
{
	//mBvhTreeRenderer->dynamicUpdateState() = !mBvhTreeRenderer->dynamicUpdateState();
}

/******************************************************************************
 * ...
 *
 * @param mode ...
 ******************************************************************************/
void SampleCore::togglePerfmonDisplay( uint mode )
{
	if (mDisplayPerfmon)
		mDisplayPerfmon = 0;
	else
		mDisplayPerfmon = mode;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::incMaxVolTreeDepth()
{
	if (mMaxVolTreeDepth < 32)
		mMaxVolTreeDepth++;

	//mBvhTreeRenderer->setMaxVolTreeDepth( mMaxVolTreeDepth );
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::decMaxVolTreeDepth()
{
	if (mMaxVolTreeDepth > 0)
		mMaxVolTreeDepth--;

	//mBvhTreeRenderer->setMaxVolTreeDepth( mMaxVolTreeDepth );
}
