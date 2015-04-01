#include "GlossyObject.h"
#include <GvCore/Array3D.h>
#include <GvCore/Array3DGPULinear.h>
#include <GvCore/GvError.h>

#define BUFFER_OFFSET(a) ((char*)NULL + (a))

	float vertices[12] = {15.0, 15.0, -3.0,
				-15.0, 15.0, -3.0, 
				-15.0, -15.0, -3.0,
				15.0, -15.0, -3.0};
    float normals[12] = {0.0, 0.0, 1.0,
				0.0, 0.0, 1.0,
				0.0, 0.0, 1.0,
				0.0, 0.0, 1.0};
	GLuint indices[4] = {0, 1, 2, 3};

	GlossyObject::GlossyObject() {
		
		lightPos[0] = 1;
		lightPos[1] = 1;
		lightPos[2] = 1;

		object = NULL;
		loadedObject = false;//************set to true if the model should be loaded from an OBJ file*************//
		
	}
	void GlossyObject::init() {
		//Shader stuff		
		vshader = useShader(GL_VERTEX_SHADER, "../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/Res/glossyObjectVert.glsl");
		fshader = useShader(GL_FRAGMENT_SHADER, "../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/Res/glossyObjectFrag.glsl");
		program = glCreateProgram();
		glAttachShader(program, vshader);
		glAttachShader(program, fshader);
		glLinkProgram(program);
		linkStatus(program);
		GV_CHECK_GL_ERROR();
		//Texture buffer arrays linked to GigaVoxels
		glGenTextures( 1, &_childArrayTBO );
		glBindTexture( GL_TEXTURE_BUFFER, _childArrayTBO );
		glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, childBufferName );
		GV_CHECK_GL_ERROR();
		glGenTextures( 1, &_dataArrayTBO );
		glBindTexture( GL_TEXTURE_BUFFER, _dataArrayTBO );
		glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, dataBufferName );
		glBindTexture(GL_TEXTURE_CUBE_MAP, cubeTexID);		
		//VBO stuff
		if (loadedObject) {
			object = new Mesh(program);
			object->chargerMesh("Data/3DModels/Butterfly/Butterfly.obj");
			object->creerVBO();
		} else {
			glGenBuffers(1, &idVBO);
			glGenBuffers(1, &idIndices); 
			glBindBuffer(GL_ARRAY_BUFFER, idVBO);		
			glBufferData(GL_ARRAY_BUFFER, sizeof(vertices)+sizeof(normals), &vertices[0], GL_STATIC_DRAW);
			glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(vertices),vertices);
			glBufferSubData(GL_ARRAY_BUFFER,sizeof(vertices),sizeof(normals),normals);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, idIndices);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);			
			GV_CHECK_GL_ERROR();			
		}
		GV_CHECK_GL_ERROR();
	}

	void GlossyObject::render() {
		//retrieveing the object model matrix
		float objectModelMatrix[16];
		glMatrixMode( GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glTranslatef(-150, -150, -100.0);
		glScalef(80, 80, 80);
		glGetFloatv( GL_MODELVIEW_MATRIX, objectModelMatrix );
		glPopMatrix();
		//start of rendering process
		glPushMatrix();
		glTranslatef(-150, -150, -100.0);		
		glScalef(80, 80, 80);
		//uniform info
		glProgramUniform3fEXT( program, glGetUniformLocation( program, "lightPos" ), lightPos[0], lightPos[1], lightPos[2]);
		glProgramUniform3fEXT( program, glGetUniformLocation( program, "worldCamPos" ), worldCamPos[0], worldCamPos[1], worldCamPos[2]);
		glProgramUniform3fEXT( program, glGetUniformLocation( program, "worldLight" ), worldLight[0], worldLight[1], worldLight[2]);
		glProgramUniformMatrix4fvEXT( program, glGetUniformLocation( program, "gvModelMatrix" ), 1, GL_FALSE, modelMatrix );
		glProgramUniformMatrix4fvEXT( program, glGetUniformLocation( program, "cubeModelMatrix" ), 1, GL_FALSE, cubeModelMatrix );
		glProgramUniformMatrix4fvEXT( program, glGetUniformLocation( program, "objectModelMatrix" ), 1, GL_FALSE, objectModelMatrix );
		glProgramUniform3uiEXT( program, glGetUniformLocation( program, "uBrickCacheSize" ), brickCacheSize[0], brickCacheSize[1], brickCacheSize[2]);
		glProgramUniform3fEXT( program, glGetUniformLocation( program, "uBrickPoolResInv" ), brickPoolResInv[0], brickPoolResInv[1], brickPoolResInv[2]);
		glProgramUniform1uiEXT( program, glGetUniformLocation( program, "uMaxDepth" ), maxDepth);
		GV_CHECK_GL_ERROR();
		glProgramUniform1iEXT( program, glGetUniformLocation( program, "uNodePoolChildArray" ), 3 );
		glProgramUniform1iEXT( program, glGetUniformLocation( program, "uNodePoolDataArray" ), 4 );
		GV_CHECK_GL_ERROR();
		//using program
		glUseProgram(program);
		glActiveTexture( GL_TEXTURE0 );
		glBindTexture( GL_TEXTURE_3D, texBufferName );
		glUniform1i( glGetUniformLocation( program, "uDataPool" ), 0);
		glActiveTexture( GL_TEXTURE1 );
		glBindTexture(GL_TEXTURE_CUBE_MAP, cubeTexID);
		glUniform1i( glGetUniformLocation( program, "s" ), 1);
		GV_CHECK_GL_ERROR();
		glBindImageTextureEXT(3, _childArrayTBO, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI );
		glBindImageTextureEXT(4, _dataArrayTBO, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI );
		GV_CHECK_GL_ERROR();
		//rendering
		if (loadedObject) {
			GV_CHECK_GL_ERROR();
			object->render();
			GV_CHECK_GL_ERROR();
		} else {
			glUniform4f(glGetUniformLocation(program, "ambientLight"), 0.75, 0.75, 0.75, 1.0);
			glUniform4f(glGetUniformLocation(program, "specularColor"), 0.7, 0.7, 0.7, 1.0);
			glUniform1f(glGetUniformLocation(program, "shininess"), 80);
			glUniform1i(glGetUniformLocation(program, "hasTex"), 0);
			glBindBuffer(GL_ARRAY_BUFFER, idVBO);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, idIndices);
			glEnableClientState(GL_VERTEX_ARRAY);
			glEnableClientState(GL_NORMAL_ARRAY);
			glVertexPointer(3, GL_FLOAT, 0, 0);
			glNormalPointer(GL_FLOAT, 0, BUFFER_OFFSET(sizeof(vertices)));
			glDrawElements(GL_QUADS, 4, GL_UNSIGNED_INT,0);//4 is the number of indices
			GV_CHECK_GL_ERROR();
			glDisableClientState(GL_VERTEX_ARRAY);
			glDisableClientState(GL_NORMAL_ARRAY);
			glUseProgram(0);
		}
		glPopMatrix();
	}

	void GlossyObject::setLightPosition(float x, float y, float z) {
		lightPos[0] = x;
		lightPos[1] = y;
		lightPos[2] = z;
	}

	void GlossyObject::setBrickCacheSize(unsigned int x, unsigned int y, unsigned int z) {
		brickCacheSize[0] = x;
		brickCacheSize[1] = y;
		brickCacheSize[2] = z;
	}

	void GlossyObject::setBrickPoolResInv(float x, float y, float z) {
		brickPoolResInv[0] = x;
		brickPoolResInv[1] = y;
		brickPoolResInv[2] = z;
	}

	void GlossyObject::setMaxDepth(unsigned int v) {
		maxDepth = v;
	}

	void GlossyObject::setVolTreeChildArray(GvCore::Array3DGPULinear< uint >* v, GLint id) {
		volTreeChildArray = v;
		childBufferName = id;
	}

	void GlossyObject::setVolTreeDataArray(GvCore::Array3DGPULinear< uint >* v, GLint id) {
		volTreeDataArray = v;
		dataBufferName = id;
	}

	void GlossyObject::setModelMatrix(float m00, float m01, float m02, float m03,
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

	void GlossyObject::setWorldLight(float x, float y, float z) {
		worldLight[0] = x;
		worldLight[1] = y;
		worldLight[2] = z;
	}

	void GlossyObject::setTexBufferName(GLint v) {
		texBufferName = v;
	}

	void GlossyObject::setWorldCameraPosition(float x, float y, float z) {
		worldCamPos[0] = x;
		worldCamPos[1] = y;
		worldCamPos[2] = z;
	}

	void GlossyObject::setCubeMapTextureID(GLuint v) {
		cubeTexID = v;
	}

	void GlossyObject::setCubeModelMatrix(float m[16]) {
		cubeModelMatrix[0] = m[0];
		cubeModelMatrix[1] = m[1];
		cubeModelMatrix[2] = m[2];
		cubeModelMatrix[3] = m[3];

		cubeModelMatrix[4] = m[4];
		cubeModelMatrix[5] = m[5];
		cubeModelMatrix[6] = m[6];
		cubeModelMatrix[7] = m[7];

		cubeModelMatrix[8] = m[8];
		cubeModelMatrix[9] = m[9];
		cubeModelMatrix[10] = m[10];
		cubeModelMatrix[11] = m[11];

		cubeModelMatrix[12] = m[12];
		cubeModelMatrix[13] = m[13];
		cubeModelMatrix[14] = m[14];
		cubeModelMatrix[15] = m[15];
	}

	GlossyObject::~GlossyObject() {
	}

