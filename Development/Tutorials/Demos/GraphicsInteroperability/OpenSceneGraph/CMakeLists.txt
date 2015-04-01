##############################################################################
# 
#  GigaVoxels CMake file
#
##############################################################################


# DO NOT REMOVE
# PLUG_IN NAME / DIRECTORY_NAME IN TREATMENT
#_______________________________

project (OpenSceneGraphInteroperability)


# TARGET TYPE
#_______________________________

set(TARGET_TYPE "EXE")


#
# NAMESPACE
#_______________________________

#set(MODULE "GvAPI")


# LINK LIBRARY
#_______________________________


# PROJECT LINK LIBRARY
#_______________________________

set(gigavoxelsLibList "GigaVoxels")

#set(gpuComputingSDKLibList "")
set(cudppLibList "cudpp32")
set(qglviewerLibList "QGLViewer")
set(qtLibList "QtCore" "QtGui" "QtOpenGL" "QtXml")
set(glewLibList "glew32")
set(freeglutLibList "freeglut")

# INCLUDES
#____________________________

INCLUDE(GigaVoxels_CMakeImport)

INCLUDE(Loki_CMakeImport)
INCLUDE(GPU_COMPUTING_SDK_CMakeImport)
INCLUDE(CUDPP_CMakeImport)
INCLUDE(glew_CMakeImport)
INCLUDE(freeglut_CMakeImport)
INCLUDE(QGLViewer_CMakeImport)
INCLUDE(Qt_CMakeImport)


# COMPILATION
#____________________________

INCLUDE(CMakeCommonTutorial)
