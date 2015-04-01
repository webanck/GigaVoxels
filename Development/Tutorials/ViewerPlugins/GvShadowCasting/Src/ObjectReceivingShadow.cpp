#include "ObjectReceivingShadow.h"

// GigaVoxels
#include <GvCore/Array3D.h>
#include <GvCore/Array3DGPULinear.h>
#include <GvCore/GvError.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>

// STL
#include <iostream>

#define BUFFER_OFFSET(a) ((char*)NULL + (a))

		//vertices = new float(12);
		float vertices[12] = {100.0, 100.0, -3.0,
					-100.0, 100.0, -3.0, 
					-100.0, -100.0, -3.0,
					100.0, -100.0, -3.0};
		//normals = new float(12);
	    float normals[12] = {0.0, 0.0, 1.0,
					0.0, 0.0, 1.0,
					0.0, 0.0, 1.0,
					0.0, 0.0, 1.0};
		//indices = new GLuint(4);
		GLuint indices[4] = {0, 1, 2, 3};
	ObjectReceivingShadow::ObjectReceivingShadow() {
		
		lightPos[0] = 1;
		lightPos[1] = 1;
		lightPos[2] = 1;
		
	}
	void ObjectReceivingShadow::init() {

		// Initialize shader program
		//vshader = useShader(GL_VERTEX_SHADER, "/home/maverick/bellouki/Gigavoxels/Development/Tutorials/Demos/GraphicsInteroperability/RendererGLSLbis/Res/objectReceivingShadowVert.glsl");
		//fshader = useShader(GL_FRAGMENT_SHADER, "/home/maverick/bellouki/Gigavoxels/Development/Tutorials/Demos/GraphicsInteroperability/RendererGLSLbis/Res/objectReceivingShadowFrag.glsl");
		QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
		QString vertexShaderFilename = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "GvShadowCasting" ) + QDir::separator() + QString( "objectReceivingShadowVert.glsl" );
		QString fragmentShaderFilename = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "GvShadowCasting" ) + QDir::separator() + QString( "objectReceivingShadowFrag.glsl" );
		vshader = useShader( GL_VERTEX_SHADER, vertexShaderFilename.toLatin1().constData() );
		fshader = useShader( GL_FRAGMENT_SHADER, fragmentShaderFilename.toLatin1().constData() );
		program = glCreateProgram();
		glAttachShader(program, vshader);
		glAttachShader(program, fshader);
		glLinkProgram(program);
		linkStatus(program);
		GV_CHECK_GL_ERROR();
		//VBO stuff
		glGenBuffers(1, &idVBO);
		glGenBuffers(1, &idIndices); 
		glBindBuffer(GL_ARRAY_BUFFER, idVBO);		
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices)+sizeof(normals), &vertices[0], GL_STATIC_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(vertices),vertices);
		glBufferSubData(GL_ARRAY_BUFFER,sizeof(vertices),sizeof(normals),normals);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, idIndices);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
		GV_CHECK_GL_ERROR();
		
		glGenTextures( 1, &_childArrayTBO );
		glBindTexture( GL_TEXTURE_BUFFER, _childArrayTBO );
		// Attach the storage of buffer object to buffer texture
		glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, childBufferName );
		glBindTexture( GL_TEXTURE_BUFFER, 0 );
		GV_CHECK_GL_ERROR();
		glGenTextures( 1, &_dataArrayTBO );
		glBindTexture( GL_TEXTURE_BUFFER, _dataArrayTBO );
		// Attach the storage of buffer object to buffer texture
		glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, dataBufferName );
		glBindTexture( GL_TEXTURE_BUFFER, 0 );
		GV_CHECK_GL_ERROR();

		//Texture buffer arrays linked to GigaVoxels
		//cout << volTreeChildArray->getBufferName()<<endl;
		glBindImageTextureEXT(3, _childArrayTBO, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI );
		glBindImageTextureEXT(4, _dataArrayTBO, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI );
	}

	void ObjectReceivingShadow::render() {
		glPushMatrix();
		//glScalef(3, 3, 1);
		//uniform info
		glProgramUniform3fEXT( program, glGetUniformLocation( program, "lightPos" ), lightPos[0], lightPos[1], lightPos[2]);
		glProgramUniform3fEXT( program, glGetUniformLocation( program, "worldLight" ), worldLight[0], worldLight[1], worldLight[2]);
		glProgramUniformMatrix4fvEXT( program, glGetUniformLocation( program, "modelMatrix" ), 1, GL_FALSE, modelMatrix );
		glProgramUniform3uiEXT( program, glGetUniformLocation( program, "uBrickCacheSize" ), _brickCacheSize[0], _brickCacheSize[1], _brickCacheSize[2]);
		glProgramUniform3fEXT( program, glGetUniformLocation( program, "uBrickPoolResInv" ), brickPoolResInv[0], brickPoolResInv[1], brickPoolResInv[2]);
		glProgramUniform1uiEXT( program, glGetUniformLocation( program, "uMaxDepth" ), maxDepth);
		//glProgramUniform1iEXT( program, glGetUniformLocation( program, "uDataPool" ), 0);
		GV_CHECK_GL_ERROR();
		//glBindImageTextureEXT(3, _childArrayTBO, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI );
		glProgramUniform1iEXT( program, glGetUniformLocation( program, "uNodePoolChildArray" ), 3 );
		//glBindImageTextureEXT(4, _dataArrayTBO, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI );
		glProgramUniform1iEXT( program, glGetUniformLocation( program, "uNodePoolDataArray" ), 4 );
		GV_CHECK_GL_ERROR();
		//using program
		glUseProgram(program);
		glUniform1i( glGetUniformLocation( program, "uDataPool" ), 0);
		//rendering
		glBindBuffer(GL_ARRAY_BUFFER, idVBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, idIndices);
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);
		glVertexPointer(3, GL_FLOAT, 0, 0);
		glNormalPointer(GL_FLOAT, 0, BUFFER_OFFSET(sizeof(vertices)));
		glDrawElements(GL_QUADS, 4, GL_UNSIGNED_INT,0);//4 is the number of indices
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);
		GV_CHECK_GL_ERROR();
		glUseProgram(0);
		glPopMatrix();
	}

	void ObjectReceivingShadow::setLightPosition(float x, float y, float z) {
		lightPos[0] = x;
		lightPos[1] = y;
		lightPos[2] = z;
	}

	void ObjectReceivingShadow::setBrickCacheSize(unsigned int x, unsigned int y, unsigned int z) {
		_brickCacheSize[0] = x;
		_brickCacheSize[1] = y;
		_brickCacheSize[2] = z;
	}

	void ObjectReceivingShadow::setBrickPoolResInv(float x, float y, float z) {
		brickPoolResInv[0] = x;
		brickPoolResInv[1] = y;
		brickPoolResInv[2] = z;
	}

	void ObjectReceivingShadow::setMaxDepth(unsigned int v) {
		maxDepth = v;
	}

	void ObjectReceivingShadow::setVolTreeChildArray(GvCore::Array3DGPULinear< uint >* v, GLint id) {
		volTreeChildArray = v;
		childBufferName = id;
	}

	void ObjectReceivingShadow::setVolTreeDataArray(GvCore::Array3DGPULinear< uint >* v, GLint id) {
		volTreeDataArray = v;
		dataBufferName = id;
	}

	void ObjectReceivingShadow::setModelMatrix(float m00, float m01, float m02, float m03,
						float m10, float m11, float m12, float m13,
						float m20, float m21, float m22, float m23,
						float m30, float m31, float m32, float m33) {
							modelMatrix[0] = m00;
							modelMatrix[1] = m01;
							modelMatrix[2] = m02;
							modelMatrix[3] = m03;

							modelMatrix[4] = m10;
							modelMatrix[5] = m11;
							modelMatrix[6] = m12;
							modelMatrix[7] = m13;

							modelMatrix[8] = m20;
							modelMatrix[9] = m21;
							modelMatrix[10] = m22;
							modelMatrix[11] = m23;

							modelMatrix[12] = m30;
							modelMatrix[13] = m31;
							modelMatrix[14] = m32;
							modelMatrix[15] = m33;


	}

	void ObjectReceivingShadow::setWorldLight(float x, float y, float z) {
		worldLight[0] = x;
		worldLight[1] = y;
		worldLight[2] = z;
	}

	ObjectReceivingShadow::~ObjectReceivingShadow() {
		//delete vertices;
		//delete normals;
	}

