#ifndef _MESH_H_
#define _MESH_H_


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
		vector<oneMesh> meshes;//all the meshes in the scene
		string Dir;
		GLuint vshader;
		GLuint fshader ;
		GLuint program;
		float lightPos[3];
	public:
		Mesh();
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
		~Mesh();
};

string Directory(const string& filename);
string Filename(const string& path);
#endif
