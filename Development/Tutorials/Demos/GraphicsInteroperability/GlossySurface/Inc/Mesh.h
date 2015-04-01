#ifndef _MESH_H_
#define _MESH_H_

//for assimp 2
//#include <assimp/Importer.hpp> // C++ importer interface
//#include <assimp/assimp.hpp>
//#include <assimp/aiConfig.h>
#include <assimp/Importer.hpp> // C++ importer interface
#include <assimp/scene.h> // Output data structure
#include <assimp/postprocess.h> // Post processing flags
#include <GL/glew.h>
#include <QDir>
#include <QDirIterator>
#include <vector>
#include <QGLWidget>
#include <iostream>
#include <fstream>
#include "ShaderManager.h"
#include <limits>

using namespace std;

struct oneMesh {
	GLuint VB;//vertex buffer id
	GLuint IB;//index buffer id
	vector<GLfloat> Vertices;
	vector<GLfloat> Normals;
	vector<GLfloat> Textures;
	vector<GLuint> Indices;
	GLenum mode;//GL_QUADS OR GL_TRIANGLES		
	float ambient[4];
	float diffuse[4];
	float specular[4];
	vector<string> texFiles[3];//one for ambient, diffuse, specular 
	vector<GLuint> texIDs[3];
	bool hasATextures;
	bool hasDTextures;
	bool hasSTextures;
	float shininess;
};

class Mesh {
	private:
		float boundingBoxSide;
		float center[3];
		vector<oneMesh> meshes;//all the meshes in the scene
		string Dir;
		GLuint program;
		float lightPos[3];
		float boxMin[3];
		float boxMax[3];
	public:
		Mesh(GLuint p=0);
		void loadTexture(const char* filename, GLuint id);
		void InitFromScene(const aiScene* scene);
		bool chargerMesh(const string& filename); //loads file
		void creerVBO();
		void renderMesh(int i);
		void render(); //renders scene
		vector<oneMesh> getMeshes();
		int getNumberOfMeshes();
		void getAmbient(float tab[4], int i);
		void getDiffuse(float tab[4], int i);
		void getSpecular(float tab[4], int i);
		void getShininess(float &s, int i);
		void setLightPosition(float x, float y, float z);
		bool hasTexture(int i);
		float getScaleFactor();
		void getTranslationFactors(float translation[3]);
		~Mesh();
};

string Directory(const string& filename);
string Filename(const string& path);
#endif
