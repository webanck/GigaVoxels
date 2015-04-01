#include "ProxyGeometry.h"
#include <vector_types.h>
#include <GvCore/GvError.h>


/**
 * Constructor
 * @param w boolean expressing whether the animated outer surface is being created
 **/
ProxyGeometry::ProxyGeometry(bool w) {
	depthMinTex = 0; 
	depthMaxTex = 0;
	depthTex = 0;
	normalTex = 0;
	frameBuffer = 0;
	innerDistance = 0.0f;
	mesh = NULL;
	water = w;
}

/**
 * Initialization: creating the shaders and the mesh object from which to retrieve 
 * the proxy geometry
 **/
void ProxyGeometry::init() {
	//Shader to compute depth and normals (according to the activated buffers)
	GLuint vshader = useShader(GL_VERTEX_SHADER, "/home/maverick/bellouki/Gigavoxels/Development/Tutorials/Demos/GraphicsInteroperability/RefractingLayer/Res/proxyVert.glsl");
	GLuint fshader = useShader(GL_FRAGMENT_SHADER, "/home/maverick/bellouki/Gigavoxels/Development/Tutorials/Demos/GraphicsInteroperability/RefractingLayer/Res/proxyFrag.glsl");
	program = glCreateProgram();
	glAttachShader(program, vshader);
	glAttachShader(program, fshader);
	glLinkProgram(program);
	linkStatus(program);

	//Creating the mesh object associated with the proxy geometry 
	mesh = new Mesh();
	mesh->setInnerShellDistance(innerDistance);
	mesh->chargerMesh(filename);
	mesh->creerVBO();
} 

/**
 * Creating the frame buffer composed of 4 buffers: 
 * - depth buffer for the front face of the object
 * - depth buffer for the back face of the object
 * - normals
 * - default depth buffer
 **/
void ProxyGeometry::initBuffers() {

	//deleting previous buffers in case the buffers' dimensions have changed (resized window)
	if ( frameBuffer )
	{
		glDeleteFramebuffers( 1, &frameBuffer );
	}
	if ( depthMaxTex )
	{
		glDeleteTextures( 1, &depthMaxTex );
	}
	if ( depthMinTex )
	{
		glDeleteTextures( 1, &depthMinTex );
	}

	//Front face of the object (depths)
	glGenTextures( 1, &depthMinTex );
	glBindTexture( GL_TEXTURE_RECTANGLE, depthMinTex );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexImage2D( GL_TEXTURE_RECTANGLE, 0/*level*/, GL_R32F/*internal format*/, width, height, 0/*border*/, GL_RED/*format*/, GL_FLOAT/*type*/, NULL );
	glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

	//Back face of the object (depths)
	glGenTextures( 1, &depthMaxTex );
	glBindTexture( GL_TEXTURE_RECTANGLE, depthMaxTex );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexImage2D( GL_TEXTURE_RECTANGLE, 0/*level*/, GL_R32F/*internal format*/, width, height, 0/*border*/, GL_RED/*format*/, GL_FLOAT/*type*/, NULL );
	glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

	//Per pixel normals
	glGenTextures( 1, &normalTex );
	glBindTexture( GL_TEXTURE_RECTANGLE, normalTex );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexImage2D( GL_TEXTURE_RECTANGLE, 0/*level*/, GL_RGB32F/*internal format*/, width, height, 0/*border*/, GL_RGB/*format*/, GL_FLOAT/*type*/, NULL );
	glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

	//Default depth buffer associated to the FBO
	glGenTextures( 1, &depthTex );
	glBindTexture( GL_TEXTURE_RECTANGLE, depthTex );
	glTexImage2D( GL_TEXTURE_RECTANGLE, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glBindTexture( GL_TEXTURE_RECTANGLE, 0 );


	//Creating the frame buffer object (FBO)
	glGenFramebuffers( 1, &frameBuffer );
	glBindFramebuffer( GL_FRAMEBUFFER, frameBuffer );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, depthMinTex, 0);
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_RECTANGLE, depthMaxTex, 0);
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_RECTANGLE, normalTex, 0);
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_RECTANGLE, depthTex, 0/*level*/ );
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );	
	GV_CHECK_GL_ERROR();
}

/**
 * Filling the minimum depths of the object (front face). Not used (see renderMinAndNorm())
 **/
void ProxyGeometry::renderMin() {
	static unsigned int time = 0;	
	//Initializing stuff	
	glColorMask( GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE );
	glClearColor( 0.0f, 0.1f, 0.3f, 0.0f );	// check if not clamp to [ 0.0; 1.0 ]
	glEnable( GL_DEPTH_TEST );
	glDepthMask(GL_TRUE);
	glDisable( GL_CULL_FACE );
	glBindFramebuffer( GL_FRAMEBUFFER, frameBuffer );
	glBindTexture( GL_TEXTURE_RECTANGLE, depthMinTex );

	//Setting the depth test parameters
	glClearDepth( 1.0f );
	glDepthFunc( GL_LESS );
	//Setting the buffer to write the shader output into
	GLenum drawBuffers[ 1 ] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers( 1, drawBuffers );//fragment shader output
	glClear( GL_COLOR_BUFFER_BIT |GL_DEPTH_BUFFER_BIT );
	//Use shader
	glUseProgram(program);
	glUniform1f( glGetUniformLocation( program, "time" ), time );
	glUniform1i( glGetUniformLocation(program, "water"), water);
	mesh->setProgram(program);
	mesh->render();
	glUseProgram(0);
	
	//Unbinding
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );
	glBindTexture( GL_TEXTURE_RECTANGLE, 0);
	glDepthMask(GL_FALSE);
	glDisable( GL_DEPTH_TEST );
	glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );
	glDisable( GL_DEPTH_TEST );

	time++;
	GV_CHECK_GL_ERROR();
}

/**
 * Filling the maximum depths of the object (back face). 
 **/
void ProxyGeometry::renderMax() {
	static unsigned int time = 0;
	//Initializing stuff
	glColorMask( GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE );
	glClearColor( 0.0f, 0.1f, 0.3f, 0.0f );	// check if not clamp to [ 0.0; 1.0 ]
	glEnable( GL_DEPTH_TEST );
	glDepthMask(GL_TRUE);
	glDisable( GL_CULL_FACE );
	glBindFramebuffer( GL_FRAMEBUFFER, frameBuffer );
	glBindTexture( GL_TEXTURE_RECTANGLE, depthMaxTex );

	//Setting the depth test parameters
	glClearDepth( 0.0f );
	glDepthFunc( GL_GREATER );
	//Setting the buffer to write the shader output into
	GLenum drawBuffers[ 1 ] = { GL_COLOR_ATTACHMENT1 };
    glDrawBuffers( 1, drawBuffers );
	glClear(GL_COLOR_BUFFER_BIT |GL_DEPTH_BUFFER_BIT );
	//Use shader
    glUseProgram(program);
    glUniform1f( glGetUniformLocation( program, "time" ), time );
    glUniform1i( glGetUniformLocation(program, "water"), water);
    mesh->setProgram(program);
    mesh->render();
    glUseProgram(0);

    //Unbinding
    glBindFramebuffer( GL_FRAMEBUFFER, 0 );
    glBindTexture( GL_TEXTURE_RECTANGLE,0 );
	glDepthMask(GL_FALSE);
	glDisable( GL_DEPTH_TEST );
	glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );

	time++;
	GV_CHECK_GL_ERROR();
}

/**
 * Fills the front face and the normals in one pass.
 **/
void ProxyGeometry::renderMinAndNorm() {
	static unsigned int time = 0;
	//Initializing stuff
	glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE );
	glClearColor( 0.0f, 0.1f, 0.3f, 0.0f );	// check if not clamp to [ 0.0; 1.0 ]
	glEnable( GL_DEPTH_TEST );
	glDepthMask(GL_TRUE);
	glEnable( GL_CULL_FACE );
	glBindFramebuffer( GL_FRAMEBUFFER, frameBuffer );
	glBindTexture( GL_TEXTURE_RECTANGLE, depthMinTex );
	glBindTexture( GL_TEXTURE_RECTANGLE, normalTex );

	//Setting the depth test parameters
	glClearDepth( 1.0f );
	glDepthFunc( GL_LESS );
	//Setting the buffer to write the shader outputs into
	GLenum drawBuffers[ 2 ] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT2 };
    glDrawBuffers( 2, drawBuffers );
	glClear(GL_COLOR_BUFFER_BIT |GL_DEPTH_BUFFER_BIT );
	//Use shader
    glUseProgram(program);
    glUniform1f( glGetUniformLocation( program, "time" ), time );
    glUniform1i( glGetUniformLocation(program, "water"), water);
    mesh->setProgram(program);
    mesh->render();
    glUseProgram(0);

    //Unbinding
    glBindFramebuffer( GL_FRAMEBUFFER, 0 );
    glBindTexture( GL_TEXTURE_RECTANGLE,0 );
	glDepthMask(GL_FALSE);
	glDisable( GL_DEPTH_TEST );
	glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );
	
	time++;
	GV_CHECK_GL_ERROR();
}

/** 
 * Sets buffer width at every resize
 * @param w new width
 **/
void  ProxyGeometry::setBufferWidth(int w) {
	width = w;
}

/** 
 * Sets buffer height at every resize
 * @param h new height
 **/
void  ProxyGeometry::setBufferHeight(int h) {
	height = h;
}

/**
 * Sets the obj filename. Must be done before a call to init()!
 * @param s filename
 **/
void  ProxyGeometry::setFilename(string s) {
	filename = s;
}

/**
 * Returns the mesh object used for the proxy geometry
 **/
Mesh* ProxyGeometry::getMesh() {
	return mesh;
}

/**
 * Sets inner distance of the mesh object (used to 'dig' into the original mesh)
 * @param d new inner distance
 **/ 
void ProxyGeometry::setInnerDistance(float d) {
	innerDistance = d;
}

/**
 * Destructor.
 **/
ProxyGeometry::~ProxyGeometry() {

}
