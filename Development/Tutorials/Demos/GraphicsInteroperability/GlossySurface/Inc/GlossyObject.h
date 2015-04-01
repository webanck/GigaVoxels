#ifndef _GLOSSYOBJECT_H_
#define _GLOSSYOBJECT_H_

#include "ShaderManager.h"
#include "Mesh.h"
namespace GvCore
{
	template<typename T>
	class Array3DGPULinear;
}

class GlossyObject {
private:
	bool loadedObject;//true if the user loads an object, false if he creates it manually
	//if the user wants to load a file 
	Mesh* object;
	//if he wants to define the object manually
	GLuint idVBO;
	GLuint idIndices;
	GLuint cubeTexID;
	GLuint vshader;
	GLuint fshader;
	GLuint program;
	//Light and camera positions
	float lightPos[3];
	float worldLight[3];
	float worldCamPos[3];
	//GigaVoxels object casting shadows stuff 
	unsigned int brickCacheSize[3];
	float brickPoolResInv[3];
	unsigned int maxDepth;
	GLuint _childArrayTBO;
	GLuint _dataArrayTBO;
	GvCore::Array3DGPULinear< unsigned int >* volTreeChildArray;
	GvCore::Array3DGPULinear< unsigned int >* volTreeDataArray;
	GLint childBufferName;
	GLint dataBufferName;
	GLint texBufferName;
	float modelMatrix[16];
	float cubeModelMatrix[16];
public:
	GlossyObject();
	void init();
	void render();
	void setLightPosition(float x, float y, float z);
	~GlossyObject();
	void setBrickCacheSize(unsigned int x, unsigned int y, unsigned int z);
	void setBrickPoolResInv(float x, float y, float z);
	void setMaxDepth(unsigned int v);
	void setVolTreeChildArray(GvCore::Array3DGPULinear< unsigned int >* v, GLint id);
	void setVolTreeDataArray(GvCore::Array3DGPULinear< unsigned int >* v, GLint id);
	void setModelMatrix(float m00, float m01, float m02, float m03,
						float m10, float m11, float m12, float m13,
						float m20, float m21, float m22, float m23,
						float m30, float m31, float m32, float m33);
	void setWorldLight(float x, float y, float z);
	void setWorldCameraPosition(float x, float y, float z);
	void setTexBufferName(GLint v);
	void setCubeMapTextureID(GLuint v);
	void setCubeModelMatrix(float m[16]);
};

#endif