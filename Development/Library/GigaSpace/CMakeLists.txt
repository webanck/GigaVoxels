#----------------------------------------------------------------
# GigaSpace LIBRARY CMake file
# Main user file
#----------------------------------------------------------------

#----------------------------------------------------------------
# Project name
#----------------------------------------------------------------

project (GigaSpace)

#----------------------------------------------------------------
# Target Type
#----------------------------------------------------------------

# Can be GV_STATIC_BUILD or GV_SHARED_LIB
SET (GV_TARGET_TYPE "GV_SHARED_LIB")

#----------------------------------------------------------------
# Add library dependencies
#----------------------------------------------------------------

# Add GigaSpace source directory to INCLUDE path
INCLUDE_DIRECTORIES (${CMAKE_SOURCE_DIR}/GigaSpace)

# GigaSpace library
SET (gigaspaceLib "GsGraphics" "GsCompute")
INCLUDE (GigaVoxels_CMakeImport)

# Add GigaSpace - main third party - dependencies LINK directory
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

# Add GigaSpace third party dependencies
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

#----------------------------------------------------------------
# Main CMake file used for project generation
#----------------------------------------------------------------

INCLUDE (GV_CMakeCommon)
