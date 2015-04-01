#include "Mesh.h"

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>

// STL
#include <iostream>

#define BUFFER_OFFSET(a) ((char*)NULL + (a))

//Importer
Assimp::Importer importer;
//Assimp scene object

const aiScene* scene = NULL;


Mesh::Mesh() {
		lightPos[0] = 1;
		lightPos[1] = 1;
		lightPos[2] = 1;
}

void Mesh::loadTexture(const char* filename, GLuint id) {
	string f;
	//get the right file name
	QDir d(filename);
	QDirIterator it(QDir(QString(Dir.c_str())), QDirIterator::Subdirectories);
	while (it.hasNext()) {
		it.next();
		QString file = it.fileName();
		if (file == QString(Filename(filename).c_str())) {
			f = it.filePath().toStdString();
		}
	}
	cout << f << endl;

	ifstream fin(f.c_str());
	if (!fin.fail()) {
		fin.close();
	} else {
		cout << "Couldn't open texture file." <<endl;
		return;
	}
	QImage img = QGLWidget::convertToGLFormat(QImage(f.c_str()));
	glBindTexture(GL_TEXTURE_2D, id);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA, img.width(), img.height(), 0,
		GL_RGBA, GL_UNSIGNED_BYTE, img.bits());
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glEnable(GL_TEXTURE_2D);
}

string Directory(const string& filename) {
	size_t pos = filename.find_last_of("\\/");
	return (string::npos == pos) ? "" : filename.substr(0, pos + 1);
}

string Filename(const string& path) {
	return path.substr(path.find_last_of("/\\") + 1);
}

void Mesh::InitFromScene(const aiScene* scene) {
	aiColor4D coltemp;
    int materialIndex;
    aiReturn texFound;
    int nbT;
    aiString file;
    float shininess;
	/**Vertices, Normals and Textures**/
	for (int i = 0; i < scene->mNumMeshes; i++) {
		const aiMesh* mesh = scene->mMeshes[i];
		oneMesh M;
		M.hasATextures = false;
		M.hasDTextures = false;
		M.hasSTextures = false;
		M.shininess = 20;
		for (int a = 0; a < 3; a++) {
			M.ambient[a] = 0.5;
			M.diffuse[a] = 0;
			M.specular[a] = 0.75;
		}	
		M.ambient[3] = 1.0;
		M.diffuse[3] = 1.0;
		M.specular[3] = 1.0;

		for (int j = 0; j < mesh->mNumVertices; j++) {
			if (mesh->HasPositions()) {
				const aiVector3D* pos = &(mesh->mVertices[j]);
				M.Vertices.push_back(pos->x/2);//just for our sphere!! remove the /2 for anything else
				M.Vertices.push_back(pos->y/2);//just for our sphere!! remove the /2 for anything else
				M.Vertices.push_back(pos->z/2);//just for our sphere!! remove the /2 for anything else
			}
			if (mesh->HasNormals()) {
				const aiVector3D* normal = &(mesh->mNormals[j]);
				M.Normals.push_back(normal->x);
				M.Normals.push_back(normal->y);
				M.Normals.push_back(normal->z);
			}
			if (mesh->HasTextureCoords(0)) {
	        		M.Textures.push_back(mesh->mTextureCoords[0][j].x);
	        		M.Textures.push_back(mesh->mTextureCoords[0][j].y);
	        	}
		}
		/**Indices**/
		for (int k = 0 ; k < mesh->mNumFaces ; k++) {
	        const aiFace& Face = mesh->mFaces[k];
	        if (Face.mNumIndices == 3) {
	    		M.mode = GL_TRIANGLES;
	        	M.Indices.push_back(Face.mIndices[0]);
	        	M.Indices.push_back(Face.mIndices[1]);
	        	M.Indices.push_back(Face.mIndices[2]);
	        	
	        } else {
	    		cout << "NumVertices != 3." << endl;
	    	}
	    }
	    /**Materials**/
	    if (scene->HasMaterials()) {
		    materialIndex = mesh->mMaterialIndex;
	        aiMaterial* material = scene->mMaterials[materialIndex];

			nbT = material->GetTextureCount(aiTextureType_AMBIENT);
			if (nbT > 0) {
				M.hasATextures = true;
			} 
			for (int j = 0; j < nbT; j++) {
				material->GetTexture(aiTextureType_AMBIENT, j, &file);
				M.texFiles[0].push_back(file.data); 
				GLuint id;
				glGenTextures(1, &id);
				M.texIDs[0].push_back(id);
				loadTexture(file.data, id);
			}
	        material->Get(AI_MATKEY_COLOR_AMBIENT, coltemp);
			if (!(coltemp.r ==0 && coltemp.g == 0 && coltemp.b ==0)) {
				M.ambient[0] = coltemp.r;
				//cout << "M.ambient[0] " << M.ambient[0] << endl;
				M.ambient[1] = coltemp.g;
				//cout << "M.ambient[1] " << M.ambient[1] << endl;
				M.ambient[2] = coltemp.b;
				//cout << "M.ambient[2] " << M.ambient[2] << endl;
				M.ambient[3] = coltemp.a;
				//cout << "M.ambient[3] " << M.ambient[3] << endl;
			}

			nbT = material->GetTextureCount(aiTextureType_DIFFUSE);
			if (nbT > 0) {
				M.hasDTextures = true;
			} 
			for (int j = 0; j < nbT; j++) {
				material->GetTexture(aiTextureType_DIFFUSE, j, &file);
				M.texFiles[1].push_back(file.data); 
				GLuint id;
				glGenTextures(1, &id);
				M.texIDs[1].push_back(id);
				loadTexture(file.data, id);
			}
			material->Get(AI_MATKEY_COLOR_DIFFUSE, coltemp);
			if (!(coltemp.r ==0 && coltemp.g == 0 && coltemp.b ==0)) {
				M.diffuse[0] = coltemp.r;
				M.diffuse[1] = coltemp.g;
				M.diffuse[2] = coltemp.b;
				M.diffuse[3] = coltemp.a;
			}

			nbT = material->GetTextureCount(aiTextureType_SPECULAR);
			if (nbT > 0) {
				M.hasSTextures = true;
			} 
			for (int j = 0; j < nbT; j++) {
				material->GetTexture(aiTextureType_SPECULAR, j, &file);
				M.texFiles[2].push_back(file.data); 
				GLuint id;
				glGenTextures(1, &id);
				M.texIDs[2].push_back(id);
				loadTexture(file.data, id);
			}
			material->Get(AI_MATKEY_COLOR_SPECULAR, coltemp);
			if (!(coltemp.r ==0 && coltemp.g == 0 && coltemp.b ==0)) {
				M.specular[0] = coltemp.r;
				M.specular[1] = coltemp.g;
				M.specular[2] = coltemp.b;
				M.specular[3] = coltemp.a;
			}
			material->Get(AI_MATKEY_SHININESS, shininess);
			if (shininess != 0.f) {
				M.shininess = shininess;
			}
		}
		meshes.push_back(M);
	}
}

bool Mesh::chargerMesh(const string& filename) {
	//check if file exists
	ifstream fin(filename.c_str());
	if (!fin.fail()) {
		fin.close();
	} else {
		cout << "Couldn't open file." <<endl;
		return false;
	}
	Dir = Directory(filename);
	scene = importer.ReadFile(filename, aiProcessPreset_TargetRealtime_MaxQuality);
	QString s(Dir.c_str());
	QDir d(s);
	Dir = d.absolutePath().toStdString();
	if (!scene) {
		cout << "Import failed." << endl;
		return false;
	}
	InitFromScene(scene);
	cout << "Import scene succeeded.\n" << endl;
	return true;
}

void Mesh::creerVBO()
{
	cout <<"nb de meshes: " <<meshes.size()<<endl;
	for (int i = 0; i < meshes.size(); i++) {
		//cout << i << endl;

		glGenBuffers(1, &(meshes[i].VB));
		glGenBuffers(1, &(meshes[i].IB));
		//cout << "apres gen" << endl;
		glBindBuffer(GL_ARRAY_BUFFER, meshes[i].VB); 
		//cout <<"apres bind"<<endl;
		//float* pointer = meshes[i].Vertices.data();
		//the following line doesn't work on Windows after a certain number of loops! :(
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*meshes[i].Vertices.size() 
									+ sizeof(GLfloat)*meshes[i].Normals.size()
									+ sizeof(GLfloat)*meshes[i].Textures.size(), NULL, GL_STATIC_DRAW);
		
		//cout <<"apres gros buffer data"<<endl;
		glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(GLfloat)*meshes[i].Vertices.size(),meshes[i].Vertices.data());
		glBufferSubData(GL_ARRAY_BUFFER,sizeof(GLfloat)*meshes[i].Vertices.size(),sizeof(GLfloat)*meshes[i].Normals.size(),meshes[i].Normals.data());
		glBufferSubData(GL_ARRAY_BUFFER,sizeof(GLfloat)*meshes[i].Vertices.size() 
										+ sizeof(GLfloat)*meshes[i].Normals.size(),sizeof(GLfloat)*meshes[i].Textures.size(),meshes[i].Textures.data());
		//cout << "apres subdatas"<<endl;
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshes[i].IB);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*meshes[i].Indices.size(), meshes[i].Indices.data(), GL_STATIC_DRAW);
		//cout <<"buffer data elt array"<<endl;
		//glDeleteBuffers(1, &(meshes[i].VB));
		//glDeleteBuffers(1, &(meshes[i].IB));
	}

	// Initialize shader program
	//vshader = useShader( GL_VERTEX_SHADER, "/home/maverick/bellouki/Gigavoxels/Development/Tutorials/Demos/GraphicsInteroperability/RendererGLSLbis/Res/vert.glsl" );
	//fshader = useShader( GL_FRAGMENT_SHADER, "/home/maverick/bellouki/Gigavoxels/Development/Tutorials/Demos/GraphicsInteroperability/RendererGLSLbis/Res/frag.glsl" );
	QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	QString vertexShaderFilename = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "GvShadowCasting" ) + QDir::separator() + QString( "vert.glsl" );
	QString fragmentShaderFilename = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "GvShadowCasting" ) + QDir::separator() + QString( "frag.glsl" );
	vshader = useShader( GL_VERTEX_SHADER, vertexShaderFilename.toLatin1().constData() );
	fshader = useShader( GL_FRAGMENT_SHADER, fragmentShaderFilename.toLatin1().constData() );
	program = glCreateProgram();
	glAttachShader(program, vshader);
	glAttachShader(program, fshader);
	glLinkProgram(program);
	linkStatus(program);
	//cout <<"apres for"<<endl;
}

void Mesh::renderMesh(int i) {
	glEnable(GL_TEXTURE_2D);
	/*if (meshes[i].hasATextures) {
		//cout << "hasATextures mesh num"<<i<<endl;
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, meshes[i].texIDs[0][0]);
	}*/
	if (meshes[i].hasDTextures) {
		//cout << "hasDTextures mesh num"<<i<<endl;
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, meshes[i].texIDs[1][0]);
	} //else {glColor3f(0, 0, 0);}
	/*if (meshes[i].hasSTextures) {
		//cout << "hasSTextures mesh num"<<i<<endl;
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, meshes[i].texIDs[2][0]);
	}*/
	glBindBuffer(GL_ARRAY_BUFFER, meshes[i].VB);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshes[i].IB);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glNormalPointer(GL_FLOAT, 0, BUFFER_OFFSET(sizeof(GLfloat)*meshes[i].Vertices.size()));
	glTexCoordPointer(2, GL_FLOAT,0,BUFFER_OFFSET(sizeof(GLfloat)*meshes[i].Vertices.size() + sizeof(GLfloat)*meshes[i].Normals.size()));
	glDrawElements(meshes[i].mode, meshes[i].Indices.size(), GL_UNSIGNED_INT,0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glBindTexture(GL_TEXTURE_2D, 0); 
	glDisable(GL_TEXTURE_2D);
	//glColor3f(1, 1,1);
}

void Mesh::render() {	
	glEnable(GL_CULL_FACE);
	//glEnable( GL_DEPTH_TEST );
	//glCullFace(GL_BACK);
	glUseProgram(program);
	glUniform1i(glGetUniformLocation(program, "samplerd"), 0);
	glUniform3f(glGetUniformLocation(program, "lightPos"), lightPos[0], lightPos[1], lightPos[2]);
	//cout <<"light: "<< lightPos[0]<<" "<<lightPos[1]<<" "<<lightPos[2]<<endl;
	for (int i = 0; i < meshes.size(); i++) {	
		if (hasTexture(i)) {
		glUniform1i(glGetUniformLocation(program, "hasTex"), 1);
	} else {
		glUniform1i(glGetUniformLocation(program, "hasTex"), 0);
	}
		//cout <<"amb: "<< meshes[i].ambient[0]<<" "<<meshes[i].ambient[1]<<" "<<meshes[i].ambient[2]<<endl;
		glUniform3f(glGetUniformLocation(program, "ambientLight"), meshes[i].ambient[0], meshes[i].ambient[1], meshes[i].ambient[2]);
		glUniform3f(glGetUniformLocation(program, "specularColor"), meshes[i].specular[0], meshes[i].specular[1], meshes[i].specular[2]);
		glUniform1f(glGetUniformLocation(program, "shininess"), meshes[i].shininess);
		
    	renderMesh(i);		
		
	}
	glDisable(GL_CULL_FACE);
	//glDisable( GL_DEPTH_TEST );
	glUseProgram(0);
}

vector<oneMesh> Mesh::getMeshes() {
	return meshes;
}

int Mesh::getNumberOfMeshes() {
	return meshes.size();
}

void Mesh::getAmbient(float tab[4], int i) {
	tab[0] = meshes[i].ambient[0];
	tab[1] = meshes[i].ambient[1];
	tab[2] = meshes[i].ambient[2];
	tab[3] = meshes[i].ambient[3];
}

void Mesh::getDiffuse(float tab[4], int i) {
	tab[0] = meshes[i].diffuse[0];
	tab[1] = meshes[i].diffuse[1];
	tab[2] = meshes[i].diffuse[2];
	tab[3] = meshes[i].diffuse[3];	
}

void Mesh::getSpecular(float tab[4], int i) {
	tab[0] = meshes[i].specular[0];
	tab[1] = meshes[i].specular[1];
	tab[2] = meshes[i].specular[2];
	tab[3] = meshes[i].specular[3];
}

void Mesh::getShininess(float &s, int i) {
	s = meshes[i].shininess;
}

bool Mesh::hasTexture(int i) {
	return (meshes[i].hasATextures || meshes[i].hasDTextures || meshes[i].hasSTextures);
}

void Mesh::setLightPosition(float x, float y, float z) {
	lightPos[0] = x;
	lightPos[1] = y;
	lightPos[2] = z;
}

Mesh::~Mesh() {}
