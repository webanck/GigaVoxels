#ifndef _SHADERMANAGER_H_
#define _SHADERMANAGER_H_

#include <GL/glew.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sstream>
#include <vector>
using namespace std;

pair<char**, int> loadSource(const char* filename);
GLuint useShader(GLenum shaderType, const char* filename);
void linkStatus(GLuint program);

#endif
