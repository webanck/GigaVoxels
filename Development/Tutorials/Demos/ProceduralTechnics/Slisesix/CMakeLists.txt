#----------------------------------------------------------------
# DEMO CMake file
# Main user file
#----------------------------------------------------------------

#----------------------------------------------------------------
# Project name
#----------------------------------------------------------------

PROJECT (SliseSixDemo)

MESSAGE (STATUS "")
MESSAGE (STATUS "PROJECT : ${PROJECT_NAME}")

#----------------------------------------------------------------
# Target yype
#----------------------------------------------------------------

# Can be GV_EXE or GV_PLUGIN
SET (GV_TARGET_TYPE "GV_EXE")

#----------------------------------------------------------------
# Add library dependencies
#----------------------------------------------------------------

# GigaVoxels library
SET (gigaspaceLib "GsGraphics" "GsCompute" "GigaSpace")
INCLUDE (GigaVoxels_CMakeImport)

# Add GigaVoxels - main third party - dependencies LINK directory
#LINK_DIRECTORIES (${GV_EXTERNAL}/lib)

# Set third party dependencies LINK library name if needed
#     . Example :
#     . When using the Qt_CMakeImport.cmake file to add Qt dependency with the command
#     .     INCLUDE (Qt_CMakeImport)
#     . the file use the ${qtLib} value to test if the user has set name different names to default ones.
#     . The DEFAULT library name is
#     .     set (qtLib "QtCore" "QtGui" "QtOpenGL" "QtXml")
#     . but user can only use
#     .     set (qtLib "QtCore" "QtGui")
#     . to reduce dependencies if needed
#     . or add dependencies
#     .     set (qtLib "QtCore" "QtGui" "QtOpenGL" "QtXml" "QtSvg")

# Add GigaVoxels third party dependencies
INCLUDE (Loki_CMakeImport)
INCLUDE (GPU_COMPUTING_SDK_CMakeImport)
INCLUDE (OpenGL_CMakeImport)
INCLUDE (GLU_CMakeImport)
INCLUDE (glew_CMakeImport)
INCLUDE (freeglut_CMakeImport)

# By default, cudpp is not used
INCLUDE (CUDPP_CMakeImport)

# Linux special features
if (WIN32)
else ()
	INCLUDE (dl_CMakeImport)
	INCLUDE (rt_CMakeImport)
endif()

# Add IHM third party dependencies
INCLUDE (Qt_CMakeImport)
INCLUDE (QGLViewer_CMakeImport)

#----------------------------------------------------------------
# Main CMake file used for project generation
#----------------------------------------------------------------

INCLUDE (GV_CMakeCommonTutorial)
