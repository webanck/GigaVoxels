#!/bin/sh

# **************************************************************************
# CMAKE GENERATION
# **************************************************************************

CURRENT_PATH=$PWD

# Generates standard UNIX makefiles
mkdir ../../Generated_Linux
mkdir ../../Generated_Linux/Library

# Generates standard UNIX makefiles
cd ../../Generated_Linux/Library
cmake -G "Unix Makefiles" ../../Development/Library "$CMAKE_DEBUG_OPTIONS"

# Exit
cd $CURRENT_PATH

