#!/bin/sh

# **************************************************************************
# CMAKE GENERATION
# **************************************************************************

CURRENT_PATH=$PWD

# Generates standard UNIX makefiles
mkdir ../../Generated_Linux
mkdir ../../Generated_Linux/Tutorials
mkdir ../../Generated_Linux/Tutorials/ViewerPlugins

# Generates standard UNIX makefiles
cd ../../Generated_Linux/Tutorials/ViewerPlugins
cmake -G "Unix Makefiles" ../../../Development/Tutorials/ViewerPlugins "$CMAKE_DEBUG_OPTIONS"

# Exit
cd $CURRENT_PATH

