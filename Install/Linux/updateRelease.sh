#!/bin/sh

# **************************************************************************
# Setups environment variables
# **************************************************************************

# PATH : GigaVoxels RELEASE directory
# export GV_RELEASE=$PWD/../../Release
GV_RELEASE=../../Release

# PATH : GigaVoxels EXTERNALS directory (third party dependencies)
# 32 bits mode :
# export GV_EXTERNAL=$PWD/../../External/Linux/x86
# GV_EXTERNAL=$PWD/../../External/Linux/x86
# 64 bits mode :
# export GV_EXTERNAL=$PWD/../../External/Linux/x64
GV_EXTERNAL=../../External/Linux/x64

# **************************************************************************
# cudpp
# **************************************************************************

cp -v -L -u -R $GV_EXTERNAL/cudpp/lib/libcudpp.so $GV_RELEASE/Bin/.

# **************************************************************************
# freeglut
# **************************************************************************

cp -v -L -u -R $GV_EXTERNAL/freeglut/lib/libglut.so $GV_RELEASE/Bin/.

# **************************************************************************
# glew
# **************************************************************************

cp -v -L -u -R $GV_EXTERNAL/glew/lib/libGLEW.so $GV_RELEASE/Bin/.

# **************************************************************************
# assimp
# **************************************************************************

cp -v -L -u -R $GV_EXTERNAL/assimp/lib/libassimp.so $GV_RELEASE/Bin/.

# **************************************************************************
# QGLViewer
# **************************************************************************

cp -v -L -u -R $GV_EXTERNAL/QGLViewer/lib/libQGLViewer.so $GV_RELEASE/Bin/.

# **************************************************************************
# Qt
# **************************************************************************

cp -v -L -u -R $GV_EXTERNAL/Qt/lib/libQtCore.so $GV_RELEASE/Bin/.
cp -v -L -u -R $GV_EXTERNAL/Qt/lib/libQtGui.so $GV_RELEASE/Bin/.
cp -v -L -u -R $GV_EXTERNAL/Qt/lib/libQtOpenGL.so $GV_RELEASE/Bin/.
cp -v -L -u -R $GV_EXTERNAL/Qt/lib/libQtXml.so $GV_RELEASE/Bin/.

# **************************************************************************
# Data
# TO DO :
# -- This is a temporary solution.
# -- Find a way to store and load data with a GvRessourceManager singleton.
# **************************************************************************
