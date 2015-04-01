#include "CubeMap.h"
#include <GvCore/GvError.h>

GLfloat vertex[24] = {50.0,50.0,50.0,
		-50.0,50.0,50.0,
		-50.0,-50.0,50.0,
		50.0,-50.0,50.0,
		50.0,50.0,-50.0,
		-50.0,50.0,-50.0,
		-50.0,-50.0,-50.0,
		50.0,-50.0,-50.0};

GLuint indice[24] = {
	0, 1, 2, 3,
	0, 3, 7, 4, 
	0, 4, 5, 1,
	1, 5, 6, 2,
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

/**
 * Constructor: 6 file names for the six faces of the cube map.
 **/
CubeMap::CubeMap(const string& PosXFilename, const string& NegXFilename, const string& PosYFilename,	const string& NegYFilename,	const string& PosZFilename,	const string& NegZFilename) {
	fileNames[0] = PosXFilename;
	fileNames[1] = NegXFilename;
	fileNames[3] = PosYFilename;
	fileNames[2] = NegYFilename;
	fileNames[4] = PosZFilename;
	fileNames[5] = NegZFilename;

}

/**
 * Creating the cube map texture. Returns true if successful, false if not.
 **/
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
	GV_CHECK_GL_ERROR();
	return true;
}

/**
 * Initialization: creating the shaders and the VBOs.
 **/
void CubeMap::init() {	
	glEnable(GL_TEXTURE_CUBE_MAP);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

	//VBO stuff
	glGenBuffers(1, &idVBO);	
	glGenBuffers(1, &idI);	
	glBindBuffer(GL_ARRAY_BUFFER, idVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertex), NULL, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(vertex),vertex);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, idI);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indice), &indice[0], GL_STATIC_DRAW);

	//Shader stuff
	GLuint vshader;
	GLuint fshader;
	vshader = useShader(GL_VERTEX_SHADER, "/home/maverick/bellouki/Gigavoxels/Development/Tutorials/Demos/GraphicsInteroperability/RefractingLayer/Res/cubeVert.glsl");
	fshader = useShader(GL_FRAGMENT_SHADER, "/home/maverick/bellouki/Gigavoxels/Development/Tutorials/Demos/GraphicsInteroperability/RefractingLayer/Res/cubeFrag.glsl");
	program = glCreateProgram();
	glAttachShader(program, vshader);
	glAttachShader(program, fshader);
	glLinkProgram(program);
	linkStatus(program);

	GV_CHECK_GL_ERROR();
}

/**
 * Rendering the cube map.
 **/
void CubeMap::render() {
	glEnable(GL_CULL_FACE);
	glCullFace(GL_FRONT);
    glDepthFunc(GL_LEQUAL);

    //Creating the cube map's model matrix
    //Transformations must be reported when rendering the cube map ( a few lines down)
	glMatrixMode( GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();	
	glTranslatef(center[0], center[1], center[2]);// moves with the camera, which is always at its center.
	glRotatef(-90, 1.0, 0.0, 0.0);// Rotated for our example. MUST BE CHANGED IN RAYCASTFRAG.GLSL, see mat3 rotCube.
	glGetFloatv( GL_MODELVIEW_MATRIX, cubeModelMatrix );
	glPopMatrix();

	glPushMatrix();
	glTranslatef(center[0], center[1], center[2]);
	glRotatef(-90, 1.0, 0.0, 0.0);
	glMatrixMode(GL_MODELVIEW);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_CUBE_MAP);

	glClientActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, id);
	GLint ids = glGetUniformLocation(program, "s");
	glUseProgram(program);
	glUniform1i(ids, 0);
	glBindBuffer(GL_ARRAY_BUFFER, idVBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, idI);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, 0);	
	glDrawElements(GL_QUADS, 24, GL_UNSIGNED_INT,0);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindTexture(GL_TEXTURE_CUBE_MAP, 0); 
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glUseProgram(0);
	glDisable(GL_TEXTURE_CUBE_MAP);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
    glDepthFunc(GL_LESS);
	glPopMatrix();

	GV_CHECK_GL_ERROR();
}

/**
 * Returns cube map texture ID.
 **/
GLuint CubeMap::getTextureID() {
	return id;
}

/**
 * Returns the cube map model matrix
 **/
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

/**
 * Sets the center of the cube map, ie the 3d position of the camera.
 * @param x 
 * @param y
 * @param z
 **/
void CubeMap::setCenter(float x, float y, float z) {
	center[0] = x;
	center[1] = y;
	center[2] = z;
}

/** 
 * Destructor.
 **/
CubeMap::~CubeMap() {}
