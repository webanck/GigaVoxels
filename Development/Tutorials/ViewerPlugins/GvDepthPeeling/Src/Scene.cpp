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

#include "Scene.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

#include <iostream>

// System
#include <cfloat>
#include <limits>

// assimp
#include <assimp.h>
#include <aiScene.h>
#include <aiPostProcess.h>

//Debugue
#include <iostream>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

#define BUFFER_OFFSET(i) ((void*)(i))

/**
 * Assimp library object to load 3D model (with a log mechanism)
 */
static struct aiLogStream stream;


/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

struct MyVertex
{
    float x, y, z;        //Vertex
    float nx, ny, nz;     //Normal
};


/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
Scene::Scene()
:      	mScene( NULL ),
	mVBO(0),
	mIBO(0),
	mNbTriangle(0)
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
Scene::~Scene()
{
	if ( mVBO != 0 ) {
		glDeleteBuffers(1, &mVBO);
	}
	if (mIBO != 0) {
		glDeleteBuffers(1, &mIBO);
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Scene::init(char *pSceneFile)
{
	// Import the geometry
	stream = aiGetPredefinedLogStream( aiDefaultLogStream_STDOUT, NULL );
	aiAttachLogStream( &stream );
	mScene = aiImportFile( pSceneFile, 0 );	// aiProcessPreset_TargetRealtime_Fast );

	// Scale the geometry
	float minx = +std::numeric_limits<float>::max();//FLT_MAX;
	float miny = +std::numeric_limits<float>::max();//FLT_MAX;
	float minz = +std::numeric_limits<float>::max();//FLT_MAX;
	float maxx = -std::numeric_limits<float>::max();//-FLT_MAX;
	float maxy = -std::numeric_limits<float>::max();//-FLT_MAX;
	float maxz = -std::numeric_limits<float>::max();//-FLT_MAX;

	for (unsigned int meshIndex = 0; meshIndex < mScene->mNumMeshes; ++meshIndex)
	{
		const aiMesh *pMesh = mScene->mMeshes[meshIndex];

		for (unsigned int vertexIndex = 0; vertexIndex < pMesh->mNumVertices; ++vertexIndex)
		{
			minx = std::min(minx, pMesh->mVertices[vertexIndex].x);
			miny = std::min(miny, pMesh->mVertices[vertexIndex].y);
			minz = std::min(minz, pMesh->mVertices[vertexIndex].z);
			maxx = std::max(maxx, pMesh->mVertices[vertexIndex].x);
			maxy = std::max(maxy, pMesh->mVertices[vertexIndex].y);
			maxz = std::max(maxz, pMesh->mVertices[vertexIndex].z);
		}
	}

	float scale = 0.95f / std::max(std::max(maxx - minx, maxy - miny), maxz - minz);

	for (unsigned int meshIndex = 0; meshIndex < mScene->mNumMeshes; ++meshIndex)
	{
		const aiMesh *pMesh = mScene->mMeshes[meshIndex];

		for (unsigned int vertexIndex = 0; vertexIndex < pMesh->mNumVertices; ++vertexIndex)
		{
			pMesh->mVertices[vertexIndex].x = (pMesh->mVertices[vertexIndex].x - (maxx + minx) * 0.5f) * scale + 0.5f;
			pMesh->mVertices[vertexIndex].y = (pMesh->mVertices[vertexIndex].y - (maxy + miny) * 0.5f) * scale + 0.5f;
			pMesh->mVertices[vertexIndex].z = (pMesh->mVertices[vertexIndex].z - (maxz + minz) * 0.5f) * scale + 0.5f;
		}
	}
	
	// init VBO and IBO : 
	glGenBuffers(1, &mVBO);
	glGenBuffers(1, &mIBO);
	
	// WARNING : we assume here that faces of the mesh are triangle. Plus we don't take of scene tree structure...
		
	// Computing number of vertices and triangles:
	unsigned int nbVertices = 0;
	mNbTriangle = 0;
	
	for (unsigned int meshIndex = 0; meshIndex < mScene->mNumMeshes; ++meshIndex) {
		nbVertices += mScene->mMeshes[meshIndex]->mNumVertices;
		mNbTriangle += mScene->mMeshes[meshIndex]->mNumFaces;
	}

	MyVertex *VBO = new MyVertex[nbVertices];
	unsigned int *IBO = new unsigned int[3*mNbTriangle];
	unsigned int offsetIBO = 0;
	unsigned int offsetVBO = 0;
	
	for (unsigned int meshIndex = 0; meshIndex < mScene->mNumMeshes; ++meshIndex)
		{
			const aiMesh *pMesh = mScene->mMeshes[meshIndex];
			
			// Storing vertices and normals into mVBO : X | Y | Z | Nx | Ny | Nz ... And storing index into IBO 
			
			for (unsigned int faceIndex = 0; faceIndex < pMesh->mNumFaces; ++faceIndex)
				{
					const struct aiFace *pFace = &pMesh->mFaces[faceIndex];
					
					// Remark : we can compute different normal for same vertex, but new one overwrites the old one
					for (unsigned int vertIndex = 0; vertIndex < pFace->mNumIndices; ++vertIndex)
						{
							int index = pFace->mIndices[vertIndex];
							
							float normal[3];

							// TO DO : Normaliser la normal
							if (!pMesh->HasNormals()) {
								// We compute normal with cross product :
								
								// retrieve vertex index of the face
								int a = pFace->mIndices[0];
								int b = pFace->mIndices[1];
								int c = pFace->mIndices[2];
								
								float e1[3] = { pMesh->mVertices[b].x - pMesh->mVertices[a].x,
										pMesh->mVertices[b].y - pMesh->mVertices[a].y,
										pMesh->mVertices[b].z - pMesh->mVertices[a].z };
								
								float e2[3] = { pMesh->mVertices[c].x - pMesh->mVertices[a].x,
										pMesh->mVertices[c].y - pMesh->mVertices[a].y,
										pMesh->mVertices[c].z - pMesh->mVertices[a].z };
								
								VBO[offsetVBO + index].nx = e1[1]*e2[2] - e1[2]*e2[1];
								VBO[offsetVBO + index].ny = e1[2]*e2[0] - e1[0]*e2[2];
								VBO[offsetVBO + index].nz = e1[0]*e2[1] - e1[1]*e2[0];

								// Normalizing the normal 
								float normal = sqrt ( VBO[offsetVBO + index].nx*VBO[offsetVBO + index].nx + 
										      VBO[offsetVBO + index].ny*VBO[offsetVBO + index].ny +
										      VBO[offsetVBO + index].nz*VBO[offsetVBO + index].nz );
								VBO[offsetVBO + index].nx /= normal;
								VBO[offsetVBO + index].ny /= normal;
								VBO[offsetVBO + index].nz /= normal;
							} else {
								VBO[offsetVBO + index].nx = pMesh->mNormals[index].x;
								VBO[offsetVBO + index].ny = pMesh->mNormals[index].y;
								VBO[offsetVBO + index].nz = pMesh->mNormals[index].z;
							}
							
							
							VBO[offsetVBO + index].x = pMesh->mVertices[index].x;
							VBO[offsetVBO + index].y = pMesh->mVertices[index].y;
							VBO[offsetVBO + index].z = pMesh->mVertices[index].z;
							
							IBO[offsetIBO + vertIndex] = index;							
						}
					offsetIBO += 3;
				}
			offsetVBO +=  mScene->mMeshes[meshIndex]->mNumVertices ;
		}
	
	glBindBuffer(GL_ARRAY_BUFFER, mVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(MyVertex)*nbVertices, &VBO[0].x, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int)*3*mNbTriangle, IBO, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	// deleting tab used 
	delete[] VBO;
	delete[] IBO;
}


/******************************************************************************
 * ...
 ******************************************************************************/
void Scene::draw() const
{
	glBindBuffer(GL_ARRAY_BUFFER, mVBO);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, sizeof(MyVertex), BUFFER_OFFSET(0));   //The starting point of the VBO, for the vertices
	glEnableClientState(GL_NORMAL_ARRAY);
	glNormalPointer(GL_FLOAT, sizeof(MyVertex), BUFFER_OFFSET(3*sizeof(float)));   //The starting point of normals
 
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIBO);
	
	glDrawElements(GL_TRIANGLES, 3*mNbTriangle, GL_UNSIGNED_INT, BUFFER_OFFSET(0));   //The starting point of the IBO
	
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
		
}

/******************************************************************************
 * ...
 ******************************************************************************/
void drawProxyRecursive(const struct aiScene *pScene, const struct aiNode *pNode)
{
	struct aiMatrix4x4 m = pNode->mTransformation;

	aiTransposeMatrix4(&m);
	
	glPushMatrix();
    glMultMatrixf((float *)&m);

	for (unsigned int meshIndex = 0; meshIndex < pScene->mNumMeshes; ++meshIndex)
		{
			const aiMesh *pMesh = pScene->mMeshes[meshIndex];

			for (unsigned int faceIndex = 0; faceIndex < pMesh->mNumFaces; ++faceIndex)
				{
					const struct aiFace *pFace = &pMesh->mFaces[faceIndex];

					GLenum face_mode;

					switch (pFace->mNumIndices)
						{
						case 1: face_mode = GL_POINTS; break;
						case 2: face_mode = GL_LINES; break;
						case 3: face_mode = GL_TRIANGLES; break;
						default: face_mode = GL_POLYGON; break;
						}

					glBegin(face_mode);

					for (unsigned int vertIndex = 0; vertIndex < pFace->mNumIndices; ++vertIndex)
						{
							int index = pFace->mIndices[vertIndex];

							if (pMesh->HasNormals()) {
								glNormal3fv(&pMesh->mNormals[index].x);
							} else {
								// We compute normal with cross product :
								
								// retrieve vertex index of the face
								int a = pFace->mIndices[0];
								int b = pFace->mIndices[1];
								int c = pFace->mIndices[2];
								
								float e1[3] = { pMesh->mVertices[b].x - pMesh->mVertices[a].x,
										pMesh->mVertices[b].y - pMesh->mVertices[a].y,
										pMesh->mVertices[b].z - pMesh->mVertices[a].z };
								
								float e2[3] = { pMesh->mVertices[c].x - pMesh->mVertices[a].x,
										pMesh->mVertices[c].y - pMesh->mVertices[a].y,
										pMesh->mVertices[c].z - pMesh->mVertices[a].z };
								
								float normal[3] = { e1[1]*e2[2] - e1[2]*e2[1],
										    e1[2]*e2[0] - e1[0]*e2[2],
										    e1[0]*e2[1] - e1[1]*e2[0] };
																
								glNormal3fv( normal );
							}
							
							glVertex3fv(&pMesh->mVertices[index].x);
						}

					glEnd();
				}
		}

	for (unsigned int childIndex = 0; childIndex < pNode->mNumChildren; ++childIndex)
		drawProxyRecursive(pScene, pNode->mChildren[childIndex]);

	glPopMatrix();
}

