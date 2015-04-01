#----------------------------------------------------------------
# DEMO CMake file
# Main user file
#----------------------------------------------------------------

#----------------------------------------------------------------
# Project name
#----------------------------------------------------------------

project (GvViewerScene)

MESSAGE (STATUS "")
MESSAGE (STATUS "PROJECT : ${PROJECT_NAME}")

#----------------------------------------------------------------
# Target yype
#----------------------------------------------------------------

# Can be GV_EXE or GV_SHARED_LIB
SET (GV_TARGET_TYPE "GV_SHARED_LIB")

#----------------------------------------------------------------
# Add library dependencies
#----------------------------------------------------------------

# add custom PROJECT LINK LIBRARY
SET (projectLibList "GvViewerCore")
INCLUDE (Project_CMakeImport)

INCLUDE (OpenGL_CMakeImport)
INCLUDE (GLU_CMakeImport)
INCLUDE (glew_CMakeImport)
INCLUDE (freeglut_CMakeImport)

# Add IHM third party dependencies
INCLUDE (QGLViewer_CMakeImport)
INCLUDE (Qt_CMakeImport)

# Qwt : plot
INCLUDE (Qwt_CMakeImport)
add_definitions(-DQWT_DLL)

# Add 3D format load library
INCLUDE (assimp_CMakeImport)

# Add math 3D maths, matrix/vector library
INCLUDE (GLM_CMakeImport)

# Linux special features
if (WIN32)
else ()
	INCLUDE (dl_CMakeImport)
endif()

# GigaSpace library
SET (gigaspaceLib "GsGraphics")
INCLUDE (GigaVoxels_CMakeImport)

#----------------------------------------------------------------
# Main CMake file used for project generation
#----------------------------------------------------------------

INCLUDE (GV_CMakeCommonTools)
