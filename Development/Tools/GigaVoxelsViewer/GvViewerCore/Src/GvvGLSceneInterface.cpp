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

#include "GvvGLSceneInterface.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

// Assimp
#include <assimp/cimport.h>
#include <assimp/postprocess.h>

// STL
#include <limits>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Tag name identifying a space profile element
 */
const char* GvvGLSceneInterface::cTypeName = "GLScene";

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/**
 * ...
 */
GLuint scene_list = 0;

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvvGLSceneInterface::GvvGLSceneInterface()
:	GvvBrowsable()
,	_minX( 0.f )
,	_minY( 0.f )
,	_minZ( 0.f )
,	_maxX( 0.f )
,	_maxY( 0.f )
,	_maxZ( 0.f )
,	_scene( NULL )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvGLSceneInterface::~GvvGLSceneInterface()
{
	// Release graphics resources
	glDeleteLists( scene_list, 1 );
	scene_list = 0;

	// TO DO
	// ... delete aiScene !!
	// - look in GvvGLSceneManager::load() where it has been allocated
}

/******************************************************************************
 * Returns the type of this browsable. The type is used for retrieving
 * the context menu or when requested or assigning an icon to the
 * corresponding item
 *
 * @return the type name of this browsable
 ******************************************************************************/
const char* GvvGLSceneInterface::getTypeName() const
{
	return cTypeName;
}

/******************************************************************************
 * Gets the name of this browsable
 *
 * @return the name of this browsable
 ******************************************************************************/
const char* GvvGLSceneInterface::getName() const
{
	return "GLScene";
}

/******************************************************************************
 * Initialize the scene
 ******************************************************************************/
void GvvGLSceneInterface::initialize()
{
}

/******************************************************************************
 * Finalize the scene
 ******************************************************************************/
void GvvGLSceneInterface::finalize()
{
}

/******************************************************************************
 * ...
 *
 * @param pScene ...
 ******************************************************************************/
void GvvGLSceneInterface::setScene( const aiScene* pScene )
{
	_scene = pScene;

	// Compute mesh bounds
	float minX = +std::numeric_limits< float >::max();
	float minY = +std::numeric_limits< float >::max();
	float minZ = +std::numeric_limits< float >::max();
	float maxX = -std::numeric_limits< float >::max();
	float maxY = -std::numeric_limits< float >::max();
	float maxZ = -std::numeric_limits< float >::max();

	// Iterate through meshes
	for ( unsigned int i = 0; i < _scene->mNumMeshes; ++i )
	{
		// Retrieve current mesh
		const aiMesh* mesh = _scene->mMeshes[ i ];

		// Iterate through vertices
		for ( unsigned int j = 0; j < mesh->mNumVertices; ++j )
		{
			minX = std::min( minX, mesh->mVertices[ j ].x );
			minY = std::min( minY, mesh->mVertices[ j ].y );
			minZ = std::min( minZ, mesh->mVertices[ j ].z );
			maxX = std::max( maxX, mesh->mVertices[ j ].x );
			maxY = std::max( maxY, mesh->mVertices[ j ].y );
			maxZ = std::max( maxZ, mesh->mVertices[ j ].z );
		}
	}

	// Update mesh bounds
	_minX = minX;
	_minY = minY;
	_minZ = minZ;
	_maxX = maxX;
	_maxY = maxY;
	_maxZ = maxZ;
}

/******************************************************************************
 * Draw the scene
 ******************************************************************************/
void GvvGLSceneInterface::draw()
{
	if ( _scene == NULL )
	{
		return;
	}
//	float tmp;

//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

//	glMatrixMode(GL_MODELVIEW);
//	glLoadIdentity();
//	gluLookAt(0.f,0.f,3.f,0.f,0.f,-5.f,0.f,1.f,0.f);

	// rotate it around the y axis
	//	glRotatef(angle,0.f,1.f,0.f);

	//// scale the whole asset to fit into our view frustum 
	//tmp = scene_max.x - scene_min.x;
	//tmp = aisgl_max( scene_max.y - scene_min.y, tmp );
	//tmp = aisgl_max( scene_max.z - scene_min.z, tmp);
	//tmp = 1.f / tmp;
	//glScalef( tmp, tmp, tmp );

//	// center the model
//	glTranslatef( -scene_center.x, -scene_center.y, -scene_center.z );

	// if the display list has not been made yet, create a new one and
	// fill it with scene contents
	if ( scene_list == 0 )
	{
		scene_list = glGenLists( 1 );
		glNewList( scene_list, GL_COMPILE );
		// now begin at the root node of the imported data and traverse
		// the scenegraph by multiplying subsequent local transforms
		// together on GL's matrix stack.
		recursive_render( _scene, _scene->mRootNode );
		glEndList();
	}

	glCallList( scene_list );

	/*glutSwapBuffers();*/

	/*do_motion();*/
}

/******************************************************************************
 * ...
 ******************************************************************************/
void GvvGLSceneInterface::recursive_render( const aiScene* sc, const aiNode* nd )
{
	unsigned int i;
	unsigned int n = 0, t;
	aiMatrix4x4 m = nd->mTransformation;

	// update transform
	aiTransposeMatrix4( &m );
	glPushMatrix();
	glMultMatrixf( (float*)&m );

	// Draw all meshes assigned to this node
	for (; n < nd->mNumMeshes; ++n)
	{
		const aiMesh* mesh = _scene->mMeshes[ nd->mMeshes[ n ] ];

		apply_material(sc->mMaterials[mesh->mMaterialIndex]);

		if ( mesh->mNormals == NULL )
		{
			glDisable( GL_LIGHTING );
		}
		else
		{
			glEnable( GL_LIGHTING );
		}

		for ( t = 0; t < mesh->mNumFaces; ++t )
		{
			const aiFace* face = &mesh->mFaces[ t ];

			GLenum face_mode;
			switch ( face->mNumIndices )
			{
				case 1: face_mode = GL_POINTS;
					break;
				
				case 2: face_mode = GL_LINES;
					break;
				
				case 3: face_mode = GL_TRIANGLES;
					break;
				
				default: face_mode = GL_POLYGON;
					break;
			}

			// Immediate mode rendering
			glBegin( face_mode );
			for ( i = 0; i < face->mNumIndices; i++ )
			{
				int index = face->mIndices[ i ];

				// Color
				if ( mesh->mColors[ 0 ] != NULL )
				{
					glColor4fv( (GLfloat*)&mesh->mColors[ 0 ][ index ] );
				}

				// Normal
				if ( mesh->mNormals != NULL )
				{
					glNormal3fv(&mesh->mNormals[ index ].x);
				}
				
				// Vertex
				glVertex3fv( &mesh->mVertices[ index ].x );
			}
			glEnd();
		}
	}

	// Draw all children
	for ( n = 0; n < nd->mNumChildren; ++n )
	{
		recursive_render( sc, nd->mChildren[ n ] );
	}

	glPopMatrix();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void GvvGLSceneInterface::apply_material( const aiMaterial* mtl )
{
	float c[ 4 ];

	GLenum fill_mode;
	int ret1, ret2;
	aiColor4D diffuse;
	aiColor4D specular;
	aiColor4D ambient;
	aiColor4D emission;
	float shininess, strength;
	int two_sided;
	int wireframe;
	unsigned int max;

	// Diffuse color
	set_float4( c, 0.8f, 0.8f, 0.8f, 1.0f );
	if ( AI_SUCCESS == aiGetMaterialColor( mtl, AI_MATKEY_COLOR_DIFFUSE, &diffuse ) )
	{
		color4_to_float4( &diffuse, c );
	}
	glMaterialfv( GL_FRONT_AND_BACK, GL_DIFFUSE, c );

	// Specular color
	set_float4( c, 0.0f, 0.0f, 0.0f, 1.0f );
	if ( AI_SUCCESS == aiGetMaterialColor( mtl, AI_MATKEY_COLOR_SPECULAR, &specular ) )
	{
		color4_to_float4( &specular, c );
	}
	glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, c );

	// Ambient color
	set_float4( c, 0.2f, 0.2f, 0.2f, 1.0f );
	if ( AI_SUCCESS == aiGetMaterialColor( mtl, AI_MATKEY_COLOR_AMBIENT, &ambient ) )
	{
		color4_to_float4( &ambient, c );
	}
	glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT, c );

	// Emissive color
	set_float4( c, 0.0f, 0.0f, 0.0f, 1.0f );
	if ( AI_SUCCESS == aiGetMaterialColor( mtl, AI_MATKEY_COLOR_EMISSIVE, &emission ) )
	{
		color4_to_float4( &emission, c );
	}
	glMaterialfv( GL_FRONT_AND_BACK, GL_EMISSION, c );

	// Shininess
	max = 1;
	ret1 = aiGetMaterialFloatArray( mtl, AI_MATKEY_SHININESS, &shininess, &max );
	if ( ret1 == AI_SUCCESS )
	{
		max = 1;
		ret2 = aiGetMaterialFloatArray( mtl, AI_MATKEY_SHININESS_STRENGTH, &strength, &max );
		if ( ret2 == AI_SUCCESS )
		{
			glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, shininess * strength );
		}
		else
		{
			glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, shininess );
		}
	}
	else
	{
		glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, 0.0f );
		set_float4( c, 0.0f, 0.0f, 0.0f, 0.0f );
		glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, c );
	}

	// Wireframe mode
	max = 1;
	if ( AI_SUCCESS == aiGetMaterialIntegerArray( mtl, AI_MATKEY_ENABLE_WIREFRAME, &wireframe, &max ) )
	{
		fill_mode = wireframe ? GL_LINE : GL_FILL;
	}
	else
	{
		fill_mode = GL_FILL;
	}
	glPolygonMode( GL_FRONT_AND_BACK, fill_mode );

	// Two-sided mode
	max = 1;
	if ( ( AI_SUCCESS == aiGetMaterialIntegerArray( mtl, AI_MATKEY_TWOSIDED, &two_sided, &max ) ) && two_sided )
	{
		glDisable( GL_CULL_FACE );
	}
	else 
	{
		glEnable( GL_CULL_FACE );
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
void GvvGLSceneInterface::color4_to_float4( const aiColor4D* c, float f[4] )
{
	f[0] = c->r;
	f[1] = c->g;
	f[2] = c->b;
	f[3] = c->a;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void GvvGLSceneInterface::set_float4( float f[4], float a, float b, float c, float d )
{
	f[0] = a;
	f[1] = b;
	f[2] = c;
	f[3] = d;
}
