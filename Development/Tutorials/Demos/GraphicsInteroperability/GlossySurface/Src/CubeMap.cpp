#include "CubeMap.h"

// GigaVoxels
#include <GvCore/GvError.h>


GLfloat vertex[24] = {10.0,10.0,10.0,
		-10.0,10.0,10.0,
		-10.0,-10.0,10.0,
		10.0,-10.0,10.0,
		10.0,10.0,-10.0,
		-10.0,10.0,-10.0,
		-10.0,-10.0,-10.0,
		10.0,-10.0,-10.0};

GLuint indice[24] = {
0, 1, 2, 3,
0, 3, 7, 4, 
0, 1, 5, 4,
1, 2, 6, 5,
3, 2, 6, 7,
5, 4, 7, 6	
};

GLenum types[6] = {GL_TEXTURE_CUBE_MAP_POSITIVE_X,
					GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
					GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
					GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
					GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
					GL_TEXTURE_CUBE_MAP_NEGATIVE_Z
				};

CubeMap::CubeMap(const string& PosXFilename, const string& NegXFilename, const string& PosYFilename,	const string& NegYFilename,	const string& PosZFilename,	const string& NegZFilename) {
	fileNames[0] = PosXFilename;
	fileNames[1] = NegXFilename;
	fileNames[3] = PosYFilename;
	fileNames[2] = NegYFilename;
	fileNames[4] = PosZFilename;
	fileNames[5] = NegZFilename;

}

bool CubeMap::Load() {
	glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_CUBE_MAP,id);
	for (int i = 0; i < 6; i++) {
		QImage img = QGLWidget::convertToGLFormat(QImage(fileNames[i].c_str()));
		glTexImage2D(types[i], 0, GL_RGBA, img.width(), img.height(), 0,
		GL_RGBA, GL_UNSIGNED_BYTE, img.bits());
		ifstream fin(fileNames[i].c_str());
		if (!fin.fail()) {
			fin.close();
		} else {
			cout << "Couldn't open cubemap texture file." <<endl;
			return false;
		}
	}
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	return true;
	GV_CHECK_GL_ERROR();
}

void CubeMap::init() {
	

	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	glColor4f(0.0, 1.0 ,0.0, 1.0);
	glGenBuffers(1, &idVBO);	
	glGenBuffers(1, &idI);	
	glBindBuffer(GL_ARRAY_BUFFER, idVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertex), NULL, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(vertex),vertex);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, idI);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indice), &indice[0], GL_STATIC_DRAW);
	GLuint vshader;
	GLuint fshader;
	vshader = useShader(GL_VERTEX_SHADER, "../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/Res/cubeVert.glsl");
	fshader = useShader(GL_FRAGMENT_SHADER, "../../Development/Tutorials/Demos/GraphicsInteroperability/GlossySurface/Res/cubeFrag.glsl");

	program = glCreateProgram();
	glAttachShader(program, vshader);
	glAttachShader(program, fshader);
	glLinkProgram(program);
	linkStatus(program);
	GV_CHECK_GL_ERROR();

}


void CubeMap::render() {
	
	glMatrixMode( GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();	
	glRotatef(-90, 1.0, 0.0, 0.0);
	glScalef(140, 140, 140);
	glGetFloatv( GL_MODELVIEW_MATRIX, cubeModelMatrix );
	glPopMatrix();

	glPushMatrix();
	glRotatef(-90, 1.0, 0.0, 0.0);
	glScalef(140, 140, 140);
	glMatrixMode(GL_MODELVIEW);

	glEnable(GL_DEPTH_TEST);

	glEnable(GL_TEXTURE_CUBE_MAP);
	glClientActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, id);

	GLint ids = glGetUniformLocation(program, "s");
	glProgramUniformMatrix4fvEXT( program, glGetUniformLocation( program, "cubeModelMatrix" ), 1, GL_FALSE, cubeModelMatrix );
	glUseProgram(program);
	glUniform1i(ids, 0);


	glBindBuffer(GL_ARRAY_BUFFER, idVBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, idI);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, 0);



	
	
	
	glDrawElements(GL_QUADS, 24, GL_UNSIGNED_INT,0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindTexture(GL_TEXTURE_CUBE_MAP, 0); 
	glDisable(GL_TEXTURE_CUBE_MAP);
	glDisable(GL_DEPTH_TEST);
glPopMatrix();
GV_CHECK_GL_ERROR();
}

GLuint CubeMap::getTextureID() {
	return id;
}

void CubeMap::getCubeModelMatrix(float m[16]) {
		m[0] = cubeModelMatrix[0];
		m[1] = cubeModelMatrix[1];
		m[2] = cubeModelMatrix[2];
		m[3] = cubeModelMatrix[3];

		m[4] = cubeModelMatrix[4];
		m[5] = cubeModelMatrix[5];
		m[6] = cubeModelMatrix[6];
		m[7] = cubeModelMatrix[7];

		m[8] = cubeModelMatrix[8];
		m[9] = cubeModelMatrix[9];
		m[10] = cubeModelMatrix[10];
		m[11] = cubeModelMatrix[11];

		m[12] = cubeModelMatrix[12];
		m[13] = cubeModelMatrix[13];
		m[14] = cubeModelMatrix[14];
		m[15] = cubeModelMatrix[15];
}

CubeMap::~CubeMap() {}
