#ifndef _CUBE_MAP_H_
#define _CUBE_MAP_H_
#include <QImage>
#include <GL/glew.h>
#include <string>
#include <iostream>
#include <fstream>
#include <QGLWidget>
#include "ShaderManager.h"

using namespace std;

class CubeMap {
private:
	string fileNames[6];
	GLuint idVBO;
	GLuint idI;
	GLuint program;
	float cubeModelMatrix[16];
	float center[3]; //3d position of the camera.
public:
	GLuint id;
	CubeMap(const string& PosXFilename,
			const string& NegXFilename,
			const string& PosYFilename,
			const string& NegYFilename,
			const string& PosZFilename,
			const string& NegZFilename);
	bool Load();
	void init();
	void render();
	GLuint getTextureID();
	void getCubeModelMatrix(float m[16]);
	void setCenter(float x, float y, float z);
	~CubeMap();
};

#endif
