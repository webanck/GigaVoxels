#ifndef _OBJECTRECEIVINGSHADOW_H_
#define _OBJECTRECEIVINGSHADOW_H_

#include "ShaderManager.h"
namespace GvCore
{
	template<typename T>
	class Array3DGPULinear;
}

class ObjectReceivingShadow {
private:
	//float vertices[12];
	//float normals[12];
	//GLuint indices[4];
	GLuint idVBO;
	GLuint idIndices;
	GLuint vshader;
	GLuint fshader;
	GLuint program;
	float lightPos[3];
	float worldLight[3];
	unsigned int _brickCacheSize[3];
	float brickPoolResInv[3];
	unsigned int maxDepth;
	GLuint _childArrayTBO;
	GLuint _dataArrayTBO;
	GvCore::Array3DGPULinear< unsigned int >* volTreeChildArray;
	GvCore::Array3DGPULinear< unsigned int >* volTreeDataArray;
	GLint childBufferName;
	GLint dataBufferName;
	float modelMatrix[16];
public:
	ObjectReceivingShadow();
	void init();
	void render();
	void setLightPosition(float x, float y, float z);
	~ObjectReceivingShadow();
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
};

#endif