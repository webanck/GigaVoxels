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
		GLuint id;
		GLuint idVBO;
		GLuint idI;
		GLuint program;
		float modelMatrix[16];
		float cubeModelMatrix[16];

	public:
		CubeMap(const string& PosXFilename,
				const string& NegXFilename,
				const string& PosYFilename,
				const string& NegYFilename,
				const string& PosZFilename,
				const string& NegZFilename);

		bool Load();
		void init();
		void render();
		~CubeMap();

		GLuint getTextureID();
		void getCubeModelMatrix(float m[16]);
};

#endif
