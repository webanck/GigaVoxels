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

#include "SampleCore.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/StaticRes3D.h>
#include <GvStructure/GvVolumeTree.h>
#include <GvRenderer/GvVolumeTreeRendererCUDA.h>
#include <GvRenderer/GvGraphicsInteroperabiltyHandler.h>
#include <GvPerfMon/CUDAPerfMon.h>
#include <GvCore/GvError.h>

// Project
#include "SphereProducer.h"
#include "SphereShader.h"

// GvViewer
#include <GvvApplication.h>
#include <GvvMainWindow.h>

// Cuda GPU Computing SDK
#include <helper_math.h>

// QGLViewer
#include <QGLViewer/manipulatedFrame.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvRenderer;

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
:	GvvPipelineInterface()
,	_clippingPlaneFrame( NULL )
,	_colorTex( 0 )
,	_depthTex( 0 )
,	_frameBuffer( 0 )
,	_depthBuffer( 0 )
,	_displayOctree( false )
,	_displayPerfmon( 0 )
,	_maxVolTreeDepth( 5 )
{
	// Translation used to position the GigaVoxels data structure
	_translation[ 0 ] = -0.5f;
	_translation[ 1 ] = -0.5f;
	_translation[ 2 ] = -0.5f;

	// Light position
	_lightPosition = make_float3( 1.f, 1.f, 1.f );

	// clipping plane
	_clippingPlane = make_float4( 0.f, 0.f, 0.f, 0.f );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
	delete _volumeTreeRenderer;
	delete _volumeTreeCache;
	delete _volumeTree;
	delete _producer;
}

/******************************************************************************
 * Gets the name of this browsable
 *
 * @return the name of this browsable
 ******************************************************************************/
const char* SampleCore::getName() const
{
	return "ClippingPlane";
}

/******************************************************************************
 * Initialize the GigaVoxels pipeline
 ******************************************************************************/
void SampleCore::init()
{
	CUDAPM_INIT();

	// Initialize CUDA with OpenGL Interoperability
	if ( ! GvViewerGui::GvvApplication::get().isGPUComputingInitialized() )
	{
		//cudaGLSetGLDevice( gpuGetMaxGflopsDeviceId() );	// to do : deprecated, use cudaSetDevice()
		//GV_CHECK_CUDA_ERROR( "cudaGLSetGLDevice" );
		cudaSetDevice( gpuGetMaxGflopsDeviceId() );
		GV_CHECK_CUDA_ERROR( "cudaSetDevice" );
		
		GvViewerGui::GvvApplication::get().setGPUComputingInitialized( true );
	}

	// Compute the size of one element in the cache for nodes and bricks
	size_t nodeElemSize = NodeRes::numElements * sizeof( GvStructure::OctreeNode );
	size_t brickElemSize = RealBrickRes::numElements * GvCore::DataTotalChannelSize< DataType >::value;

	// Compute how many we can fit into the given memory size
	size_t nodePoolNumElems = NODEPOOL_MEMSIZE / nodeElemSize;
	size_t brickPoolNumElems = BRICKPOOL_MEMSIZE / brickElemSize;

	// Compute the resolution of the pools
	uint3 nodePoolRes = make_uint3( static_cast< uint >( floorf( powf( static_cast< float >( nodePoolNumElems ), 1.0f / 3.0f ) ) ) ) * NodeRes::get();
	uint3 brickPoolRes = make_uint3( static_cast< uint >( floorf( powf( static_cast< float >( brickPoolNumElems ), 1.0f / 3.0f ) ) ) ) * RealBrickRes::get();
	
	std::cout << "\nnodePoolRes: " << nodePoolRes << std::endl;
	std::cout << "brickPoolRes: " << brickPoolRes << std::endl;

	// Producer initialization
	_producer = new ProducerType();

	// Data structure initialization
	_volumeTree = new VolumeTreeType( nodePoolRes, brickPoolRes, 0 );
	_volumeTree->setMaxDepth( _maxVolTreeDepth );

	// Cache initialization
	_volumeTreeCache = new VolumeTreeCacheType( _volumeTree, _producer, nodePoolRes, brickPoolRes );

	// Renderer initialization
	_volumeTreeRenderer = new VolumeTreeRendererType( _volumeTree, _volumeTreeCache, _producer );

	// Clipping Plane
	_clippingPlaneFrame = new qglviewer::ManipulatedFrame();
	// GigaVoxels model matrix : no optimal...
	//_clippingPlaneFrame->setPosition( -0.5f, -0.5f, -0.5f );	// GigaVoxels
	//_clippingPlaneFrame->translate( 0.5f, 0.5f, 0.5f );	// center of GigaVoxels BBox
	//_clippingPlaneFrame->setTranslation( -0.5f, -0.5f, -0.5f );	// GigaVoxels
	_clippingPlaneFrame->setTranslation( -0.5f + 0.5f, -0.5f + 0.5f, -0.5f + 0.5f );	// GigaVoxels
	//-------------------------- TEST
	qglviewer::LocalConstraint* localConstraint = new qglviewer::LocalConstraint();
	localConstraint->setTranslationConstraintType( qglviewer::AxisPlaneConstraint::AXIS );
	qglviewer::Vec dir( 0.0, 0.0, 1.0 );
	localConstraint->setTranslationConstraintDirection( dir );
	_clippingPlaneFrame->setConstraint( localConstraint );
	//--------------------------

	// Custom initialization
	// Note : this could be done via an XML settings file loaded at initialization
	setClippingPlane( 0.f, 0.f, 1.0f, 0.f );
	//setClippingPlane( 0.f, 0.f, 1.0f, -0.5f );

	//** Setups connection
//	connect( _clippingPlaneFrame, SIGNAL( manipulated() ), this, SLOT( onClippingFrameManipulated() ) );
}

//---------------------------------------------------------------------------------

// OutVD > 0 means ray is back-facing the plane
// returns false if there is no intersection because ray is perpedicular to plane
bool ray_to_plane( const float3& pRayOrig, const float3& pRayDir, const float4& pPlane, float* pOutT, float* pOutVD )
{
	*pOutVD = pPlane.x * pRayDir.x + pPlane.y * pRayDir.y + pPlane.z * pRayDir.z;
	
	if ( *pOutVD == 0.0f )
	{
		return false;
	}
	
	*pOutT = - ( pPlane.x * pRayOrig.x + pPlane.y * pRayOrig.y + pPlane.z * pRayOrig.z + pPlane.w ) / *pOutVD;

	return true;
}

// Maximum out_point_count == 6, so out_points must point to 6-element array.
// out_point_count == 0 mean no intersection.
// out_points are not sorted.
void calc_plane_aabb_intersection_points( const float4 &plane,
    const float3 &aabb_min, const float3 &aabb_max,
    float3 *out_points, unsigned &out_point_count )
{
    out_point_count = 0;
    float vd, t;

    // Test edges along X axis, pointing right.
    float3 dir = make_float3(aabb_max.x - aabb_min.x, 0.f, 0.f);
    float3 orig = aabb_min;
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;
    orig = make_float3(aabb_min.x, aabb_max.y, aabb_min.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;
    orig = make_float3(aabb_min.x, aabb_min.y, aabb_max.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;
    orig = make_float3(aabb_min.x, aabb_max.y, aabb_max.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;

    // Test edges along Y axis, pointing up.
    dir = make_float3(0.f, aabb_max.y - aabb_min.y, 0.f);
    orig = make_float3(aabb_min.x, aabb_min.y, aabb_min.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;
    orig = make_float3(aabb_max.x, aabb_min.y, aabb_min.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;
    orig = make_float3(aabb_min.x, aabb_min.y, aabb_max.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;
    orig = make_float3(aabb_max.x, aabb_min.y, aabb_max.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;

    // Test edges along Z axis, pointing forward.
    dir = make_float3(0.f, 0.f, aabb_max.z - aabb_min.z);
    orig = make_float3(aabb_min.x, aabb_min.y, aabb_min.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;
    orig = make_float3(aabb_max.x, aabb_min.y, aabb_min.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;
    orig = make_float3(aabb_min.x, aabb_max.y, aabb_min.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;
    orig = make_float3(aabb_max.x, aabb_max.y, aabb_min.z);
    if (ray_to_plane(orig, dir, plane, &t, &vd) && t >= 0.f && t <= 1.f)
        out_points[out_point_count++] = orig + dir * t;

	assert( out_point_count <= 6 );
}

struct Comp3DVectorfunctor
{
	bool operator()( const float3& pPointA, const float3& pPointB )
	{
		float3 v = cross( pPointA - _origin, pPointB - _origin );

		return dot( v, _planeNormal ) < 0.0f;
	}

	float3 _planeNormal;
	float3 _origin;
};

/******************************************************************************
 * ...
 ******************************************************************************/
void sort_points( float3* pPoints, unsigned pPoint_count, const float4& pPlane )
{
    if ( pPoint_count == 0 )
	{
		return;
	}

    const float3 plane_normal = make_float3( pPlane.x, pPlane.y, pPlane.z );
    const float3 origin = pPoints[ 0 ];

	Comp3DVectorfunctor comp3DVector;
	comp3DVector._planeNormal = plane_normal;
	comp3DVector._origin = origin;

    std::sort( pPoints, pPoints + pPoint_count, comp3DVector );
}

/******************************************************************************
 * Used to display the N-tree
 *
 * @param p1 Position
 * @param p2 Position
 ******************************************************************************/
void drawCube( const float3& p1, const float3& p2 )
{
	glBegin(GL_QUADS);
	// Front Face
	glVertex3f(p1.x, p1.y, p2.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p2.z);	// Top Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p2.z);	// Top Right Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p2.z);	// Bottom Right Of The Texture and Quad

	// Back Face
	glVertex3f(p1.x, p1.y, p1.z);	// Bottom Right Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p1.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p1.z);	// Top Left Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p1.z);	// Top Right Of The Texture and Quad

	// Top Face
	glVertex3f(p1.x, p2.y, p1.z);	// Top Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p1.z);	// Top Right Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p2.z);	// Bottom Right Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p2.z);	// Bottom Left Of The Texture and Quad

	// Bottom Face
	glVertex3f(p1.x, p1.y, p1.z);	// Top Right Of The Texture and Quad
	glVertex3f(p1.x, p1.y, p2.z);	// Bottom Right Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p2.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p1.z);	// Top Left Of The Texture and Quad

	// Right face
	glVertex3f(p2.x, p1.y, p1.z);	// Bottom Right Of The Texture and Quad
	glVertex3f(p2.x, p1.y, p2.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p2.z);	// Top Left Of The Texture and Quad
	glVertex3f(p2.x, p2.y, p1.z);	// Top Right Of The Texture and Quad

	// Left Face
	glVertex3f(p1.x, p1.y, p1.z);	// Bottom Left Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p1.z);	// Top Left Of The Texture and Quad
	glVertex3f(p1.x, p2.y, p2.z);	// Top Right Of The Texture and Quad
	glVertex3f(p1.x, p1.y, p2.z);	// Bottom Right Of The Texture and Quad

	glEnd();
}
//---------------------------------------------------------------------------------

/******************************************************************************
 * Draw function called of frame
 ******************************************************************************/
void SampleCore::draw()
{
	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT( frame );
	CUDAPM_START_EVENT( app_init_frame );

	glMatrixMode( GL_MODELVIEW );
	
	glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	glEnable( GL_DEPTH_TEST );

	// draw the octree where the sphere will be
	glPushMatrix();
	// Translation used to position the GigaVoxels data structure
	glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );
	if ( _displayOctree )
	{
		_volumeTree->displayDebugOctree();
	}
	glPopMatrix();

	//----------------------------------------
	// Compute in World Space
	const GLdouble* clippingPlaneMatrix = _clippingPlaneFrame->worldMatrix();
	float4x4 myClippingPlaneMatrix;
	for ( int i = 0; i < 16; i++ )
	{
		myClippingPlaneMatrix._array[ i ] = static_cast< float >( clippingPlaneMatrix[ i ] );
	}
	float3 planeNormal = mulRot( myClippingPlaneMatrix, make_float3( 0.0f, 0.0f, 1.0f ) );
	float3 planeOrigin = mul( transpose( myClippingPlaneMatrix ), make_float3( 0.0f, 0.0f, 0.0f ) );
	float d = -dot( planeNormal, planeOrigin );
	float4 plane = make_float4( planeNormal.x, planeNormal.y, planeNormal.z, d );

	//--------------------------------
	// TEST
	setClippingPlane( planeNormal.x, planeNormal.y, planeNormal.z, -d );
	//--------------------------------
	
	////--------------------------
	//std::cout << "\nCLIP_FRAME [ " << myClippingPlaneMatrix._array[ 12 ] << " ; " << myClippingPlaneMatrix._array[ 13 ] << " ; " << myClippingPlaneMatrix._array[ 14 ] << " ]" << std::endl;
	//std::cout << "NORMAL [ " << planeNormal.x << " ; " << planeNormal.y << " ; " << planeNormal.z << " ]" << std::endl;
	//std::cout << "ORIGIN [ " << planeOrigin.x << " ; " << planeOrigin.y << " ; " << planeOrigin.z << " ]" << std::endl;
	//std::cout << "PLANE [ " << plane.x << " ; " << plane.y << " ; " << plane.z << " ; " << plane.w << " ]" << std::endl;
	////--------------------------
	float3 aabb_min = make_float3( 0.0f, 0.0f, 0.0f );	// no world transform for GigaVoxels => it has been commented...
	float3 aabb_max = make_float3( 1.0f, 1.0f, 1.0f );
	//--------------------
	aabb_min += make_float3( -0.5f, -0.5f, -0.5f );	// no world transform for GigaVoxels => it has been commented...
	aabb_max += make_float3( -0.5f, -0.5f, -0.5f );
	//--------------------
	float3 out_points[ 6 ];
	unsigned out_point_count = 0;
	calc_plane_aabb_intersection_points( plane, aabb_min, aabb_max, out_points, out_point_count );
	sort_points( out_points, out_point_count, plane );
	
	glLineWidth( 3.f );

	glPushMatrix();
	//glTranslatef( -0.5f, -0.5f, -0.5f );
		glBegin( GL_LINE_LOOP );
			glColor3f( 1.0f, 0.0f, 0.0f );
			for ( unsigned int i = 0; i < out_point_count; i++ )
			{
				glVertex3f( out_points[ i ].x, out_points[ i ].y, out_points[ i ].z );
			}
		glEnd();
	glPopMatrix();

	glLineWidth( 1.f );

	//----------------------------------------------------------------------------
	glPushMatrix();
	glMultMatrixd( _clippingPlaneFrame->matrix() );

	bool clippingPlaneUsed = false;
	if ( _clippingPlaneFrame->grabsMouse() )
	{
		clippingPlaneUsed = true;
	}
	else
	{
		//
	}
	// Draws a 3D arrow along the positive Z axis.
	glColor3f( 0.2f, 0.8f, 0.5f );
	//glPushMatrix();
	//glTranslatef( -0.5f, -0.5f, -0.5f );	// center of bbox
	//	drawArrow( 0.4f, 0.015f );
	//glPopMatrix();
	
	glPopMatrix();

	glDepthMask( GL_FALSE );
	// Draw a plane representation: Its normal...
	// ...and a quad (with a slightly shifted z so that it is not clipped).
	if ( clippingPlaneUsed )
	{
		//glColor3f( 0.0f, 1.0f, 0.0f );
		glColor4f( 0.0f, 1.0f, 0.0f, 0.2f );
	}
	else
	{
		//glColor3f( 0.8f, 0.8f, 0.8f );
		glColor4f( 0.8f, 0.8f, 0.8f, 0.2f );
	}
	glEnable( GL_BLEND );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
	glPushMatrix();
		glBegin( GL_POLYGON );
			//glColor4f( 0.0f, 1.0f, 0.0f, 0.2f );
			for ( unsigned int i = 0; i < out_point_count; i++ )
			{
				glVertex3f( out_points[ i ].x, out_points[ i ].y, out_points[ i ].z );
			}
		glEnd();
	glPopMatrix();
	glDisable( GL_BLEND );
	glDepthMask( GL_TRUE );
	//----------------------------------------------------------------------------

	// ------------------------------------------
	glPushMatrix();
	glTranslatef( -0.5f, -0.5f, -0.5f );
		glPolygonMode( GL_FRONT_AND_BACK , GL_LINE );
		glColor3f( 1.f, 1.f, 1.f );
		drawCube( make_float3( 0.f, 0.f, 0.f ), make_float3( 1.f, 1.f, 1.f ) );
		glPolygonMode( GL_FRONT_AND_BACK , GL_FILL);
	glPopMatrix();
	// ------------------------------------------

	glDisable( GL_DEPTH_TEST );

	// Clear the depth PBO (pixel buffer object) by reading from the previously cleared FBO (frame buffer object)
	glBindBuffer( GL_PIXEL_PACK_BUFFER, _depthBuffer );
	glReadPixels( 0, 0, _width, _height, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0 );
	glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
	GV_CHECK_GL_ERROR();

	glBindFramebuffer( GL_FRAMEBUFFER, 0 );

	// extract view transformations
	float4x4 viewMatrix;
	float4x4 projectionMatrix;
	glGetFloatv( GL_MODELVIEW_MATRIX, viewMatrix._array );
	glGetFloatv( GL_PROJECTION_MATRIX, projectionMatrix._array );

	// extract viewport
	GLint params[4];
	glGetIntegerv( GL_VIEWPORT, params );
	int4 viewport = make_int4(params[0], params[1], params[2], params[3]);

	// render the scene into textures
	CUDAPM_STOP_EVENT( app_init_frame );

	// Build the world transformation matrix
	float4x4 modelMatrix;
	glPushMatrix();
	glLoadIdentity();
	glTranslatef( -0.5f, -0.5f, -0.5f );
	glGetFloatv( GL_MODELVIEW_MATRIX, modelMatrix._array );
	glPopMatrix();

	// Render
	_volumeTreeRenderer->render( modelMatrix, viewMatrix, projectionMatrix, viewport );

	// Render the result to the screen
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();

	glDisable( GL_DEPTH_TEST );
	glEnable( GL_TEXTURE_RECTANGLE_EXT );
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _colorTex );

	// Draw a full screen quad
	GLint sMin = 0;
	GLint tMin = 0;
	GLint sMax = _width;
	GLint tMax = _height;
	glBegin( GL_QUADS );
	glColor3f( 1.0f, 1.0f, 1.0f );
	glTexCoord2i( sMin, tMin ); glVertex2i( -1, -1 );
	glTexCoord2i( sMax, tMin ); glVertex2i(  1, -1 );
	glTexCoord2i( sMax, tMax ); glVertex2i(  1,  1 );
	glTexCoord2i( sMin, tMax ); glVertex2i( -1,  1 );
	glEnd();

	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
	glDisable( GL_TEXTURE_RECTANGLE_EXT );
	
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();

	// TEST - optimization due to early unmap() graphics resource from GigaVoxels
	//_volumeTreeRenderer->doPostRender();
	
	// Update GigaVoxels info
	_volumeTreeRenderer->nextFrame();

	CUDAPM_STOP_EVENT( frame );
	CUDAPM_STOP_FRAME;

	// Display the GigaVoxels performance monitor (if it has been activated during GigaVoxels compilation)
	if ( _displayPerfmon )
	{
		GvPerfMon::CUDAPerfMon::getApplicationPerfMon().displayFrameGL( _displayPerfmon - 1 );
	}
}

/******************************************************************************
 * Resize the frame
 *
 * @param width the new width
 * @param height the new height
 ******************************************************************************/
void SampleCore::resize( int width, int height )
{
	_width = width;
	_height = height;

	// Reset default active frame region for rendering
	_volumeTreeRenderer->setProjectedBBox( make_uint4( 0, 0, _width, _height ) );

	// Re-init Perfmon subsystem
	CUDAPM_RESIZE(make_uint2(_width, _height));

	// Disconnect all registered graphics resources
	_volumeTreeRenderer->resetGraphicsResources();

	// Create frame-dependent objects
	if (_depthBuffer)
	{
		glDeleteBuffers(1, &_depthBuffer);
	}

	if (_colorTex)
	{
		glDeleteTextures(1, &_colorTex);
	}
	if (_depthTex)
	{
		glDeleteTextures(1, &_depthTex);
	}

	if (_frameBuffer)
	{
		glDeleteFramebuffers(1, &_frameBuffer);
	}

	glGenTextures(1, &_colorTex);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, _colorTex);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);
	GV_CHECK_GL_ERROR();

	glGenBuffers(1, &_depthBuffer);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, _depthBuffer);
	glBufferData(GL_PIXEL_PACK_BUFFER, width * height * sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	GV_CHECK_GL_ERROR();

	glGenTextures(1, &_depthTex);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, _depthTex);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, GL_DEPTH24_STENCIL8_EXT, width, height, 0, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, NULL);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);
	GV_CHECK_GL_ERROR();

	glGenFramebuffers(1, &_frameBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, _frameBuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_EXT, _colorTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, _depthTex, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, _depthTex, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	GV_CHECK_GL_ERROR();

	// Create CUDA resources from OpenGL objects
	if ( _displayOctree )
	{
		_volumeTreeRenderer->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		_volumeTreeRenderer->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	}
	else
	{
		_volumeTreeRenderer->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
	}
}

/******************************************************************************
 * Clear the GigaVoxels cache
 ******************************************************************************/
void SampleCore::clearCache()
{
	_pipeline->clear();
}

/******************************************************************************
 * Toggle the display of the N-tree (octree) of the data structure
 ******************************************************************************/
void SampleCore::toggleDisplayOctree()
{
	_displayOctree = !_displayOctree;

	// Disconnect all registered graphics resources
	_volumeTreeRenderer->resetGraphicsResources();

	if ( _displayOctree )
	{
		_volumeTreeRenderer->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		_volumeTreeRenderer->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	}
	else
	{
		_volumeTreeRenderer->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
	}
}

/******************************************************************************
 * Toggle the GigaVoxels dynamic update mode
 ******************************************************************************/
void SampleCore::toggleDynamicUpdate()
{
	const bool status = _pipeline->hasDynamicUpdate();
	_pipeline->setDynamicUpdate( ! status );
}

/******************************************************************************
 * Toggle the display of the performance monitor utility if
 * GigaVoxels has been compiled with the Performance Monitor option
 *
 * @param mode The performance monitor mode (1 for CPU, 2 for DEVICE)
 ******************************************************************************/
void SampleCore::togglePerfmonDisplay( uint mode )
{
	if ( _displayPerfmon )
	{
		_displayPerfmon = 0;
	}
	else
	{
		_displayPerfmon = mode;
	}
}

/******************************************************************************
 * Increment the max resolution of the data structure
 ******************************************************************************/
void SampleCore::incMaxVolTreeDepth()
{
	if ( _maxVolTreeDepth < 32 )
	{
		_maxVolTreeDepth++;
	}

	_volumeTree->setMaxDepth( _maxVolTreeDepth );
}

/******************************************************************************
 * Decrement the max resolution of the data structure
 ******************************************************************************/
void SampleCore::decMaxVolTreeDepth()
{
	if ( _maxVolTreeDepth > 0 )
	{
		_maxVolTreeDepth--;
	}

	_volumeTree->setMaxDepth( _maxVolTreeDepth );
}

/******************************************************************************
 * Get the node tile resolution of the data structure.
 *
 * @param pX the X node tile resolution
 * @param pY the Y node tile resolution
 * @param pZ the Z node tile resolution
 ******************************************************************************/
void SampleCore::getDataStructureNodeTileResolution( unsigned int& pX, unsigned int& pY, unsigned int& pZ ) const
{
	const uint3& nodeTileResolution = _volumeTree->getNodeTileResolution().get();

	pX = nodeTileResolution.x;
	pY = nodeTileResolution.y;
	pZ = nodeTileResolution.z;
}

/******************************************************************************
 * Get the brick resolution of the data structure (voxels).
 *
 * @param pX the X brick resolution
 * @param pY the Y brick resolution
 * @param pZ the Z brick resolution
 ******************************************************************************/
void SampleCore::getDataStructureBrickResolution( unsigned int& pX, unsigned int& pY, unsigned int& pZ ) const
{
	const uint3& brickResolution = _volumeTree->getBrickResolution().get();

	pX = brickResolution.x;
	pY = brickResolution.y;
	pZ = brickResolution.z;
}

/******************************************************************************
 * Get the max depth.
 *
 * @return the max depth
 ******************************************************************************/
unsigned int SampleCore::getRendererMaxDepth() const
{
	return _volumeTree->getMaxDepth();
}

/******************************************************************************
 * Set the max depth.
 *
 * @param pValue the max depth
 ******************************************************************************/
void SampleCore::setRendererMaxDepth( unsigned int pValue )
{
	_volumeTree->setMaxDepth( pValue );
}

/******************************************************************************
 * Get the max number of requests of node subdivisions.
 *
 * @return the max number of requests
 ******************************************************************************/
unsigned int SampleCore::getCacheMaxNbNodeSubdivisions() const
{
	return _volumeTreeCache->getMaxNbNodeSubdivisions();
}

/******************************************************************************
 * Set the max number of requests of node subdivisions.
 *
 * @param pValue the max number of requests
 ******************************************************************************/
void SampleCore::setCacheMaxNbNodeSubdivisions( unsigned int pValue )
{
	_volumeTreeCache->setMaxNbNodeSubdivisions( pValue );
}

/******************************************************************************
 * Get the max number of requests of brick of voxel loads.
 *
 * @return the max number of requests
 ******************************************************************************/
unsigned int SampleCore::getCacheMaxNbBrickLoads() const
{
	return _volumeTreeCache->getMaxNbBrickLoads();
}

/******************************************************************************
 * Set the max number of requests of brick of voxel loads.
 *
 * @param pValue the max number of requests
 ******************************************************************************/
void SampleCore::setCacheMaxNbBrickLoads( unsigned int pValue )
{
	_volumeTreeCache->setMaxNbBrickLoads( pValue );
}

/******************************************************************************
 * Set the request strategy indicating if, during data structure traversal,
 * priority of requests is set on brick loads or on node subdivisions first.
 *
 * @param pFlag the flag indicating the request strategy
 ******************************************************************************/
void SampleCore::setRendererPriorityOnBricks( bool pFlag )
{
	_volumeTreeRenderer->setPriorityOnBricks( pFlag );
}

/******************************************************************************
 * Specify color to clear the color buffer
 *
 * @param pRed red component
 * @param pGreen green component
 * @param pBlue blue component
 * @param pAlpha alpha component
 ******************************************************************************/
void SampleCore::setClearColor( unsigned char pRed, unsigned char pGreen, unsigned char pBlue, unsigned char pAlpha )
{
	_volumeTreeRenderer->setClearColor( make_uchar4( pRed, pGreen, pBlue, pAlpha ) );
}

/******************************************************************************
 * Tell wheter or not the pipeline has a light.
 *
 * @return the flag telling wheter or not the pipeline has a light
 ******************************************************************************/
bool SampleCore::hasLight() const
{
	return true;
}

/******************************************************************************
 * Get the light position
 *
 * @param pX the X light position
 * @param pY the Y light position
 * @param pZ the Z light position
 ******************************************************************************/
void SampleCore::getLightPosition( float& pX, float& pY, float& pZ ) const
{
	pX = _lightPosition.x;
	pY = _lightPosition.y;
	pZ = _lightPosition.z;
}

/******************************************************************************
 * Set the light position
 *
 * @param pX the X light position
 * @param pY the Y light position
 * @param pZ the Z light position
 ******************************************************************************/
void SampleCore::setLightPosition( float pX, float pY, float pZ )
{
	// Update DEVICE memory with "light position"
	//
	// WARNING
	// Apply inverse modelisation matrix applied on the GigaVoxels object to set light position correctly.
	// Here a glTranslatef( -0.5f, -0.5f, -0.5f ) has been used.
	_lightPosition.x = pX - _translation[ 0 ];
	_lightPosition.y = pY - _translation[ 1 ];
	_lightPosition.z = pZ - _translation[ 2 ];

	// Update device memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cLightPosition, &_lightPosition, sizeof( _lightPosition ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Get the translation used to position the GigaVoxels data structure
 *
 * @param pX the x componenet of the translation
 * @param pX the y componenet of the translation
 * @param pX the z componenet of the translation
 ******************************************************************************/
void SampleCore::getTranslation( float& pX, float& pY, float& pZ ) const
{
	pX = _translation[ 0 ];
	pY = _translation[ 1 ];
	pZ = _translation[ 2 ];
}

/******************************************************************************
 * Get the clipping plane equation
 *
 * Note : Plane is represented by equation Ax + By + Cz + D = 0
 *
 * @param pNormalX the X clipping plane normal's component
 * @param pNormalY the Y clipping plane normal's component
 * @param pNormalZ the Z clipping plane normal's component
 * @param pDistance the distance
 ******************************************************************************/
void SampleCore::getClippingPlane(float& pNormalX, float& pNormalY, float& pNormalZ, float& pDistance ) const
{
	// Normal
	pNormalX = _clippingPlane.x;
	pNormalY = _clippingPlane.y;
	pNormalZ = _clippingPlane.z;

	// Distance
	pDistance = _clippingPlane.w;
}

/******************************************************************************
 * Set the clipping plane equation
 *
 * Note : Plane is represented by equation Ax + By + Cz + D = 0
 *
 * @param pNormalX the X clipping plane normal's component
 * @param pNormalY the Y clipping plane normal's component
 * @param pNormalZ the Z clipping plane normal's component
 * @param pDistance the distance
 ******************************************************************************/
void SampleCore::setClippingPlane( float pNormalX, float pNormalY, float pNormalZ, float pDistance )
{
	// Update DEVICE memory with "light position"
	
	// Distance
	_clippingPlane.x = pNormalX;
	_clippingPlane.y = pNormalY;
	_clippingPlane.z = pNormalZ;

	// Distance
	_clippingPlane.w = pDistance;

	// Update device memory
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cClippingPlane, &_clippingPlane, sizeof( _clippingPlane ), 0, cudaMemcpyHostToDevice ) );
}

///******************************************************************************
// * Slot called when the clipping plane ManipulatedFrame is manipulated
// ******************************************************************************/
//void SampleCore::onClippingPlaneFrameManipulated()
//{
//	float pos[ 4 ] = { 1.f, 1.f, 1.f, 1.f };
//	_clippingPlaneFrame->getPosition( pos[ 0 ], pos[ 1 ], pos[ 2 ] );
//
//	float planeEquation[ 4 ] = { 0.f, 0.f, 0.f, 0.f };
//	setClippingPlane( planeEquation[ 0 ], planeEquation[ 1 ], planeEquation[ 2 ], planeEquation[ 3 ] );
//
//}