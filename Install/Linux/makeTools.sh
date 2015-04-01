#!/bin/sh

# **************************************************************************
# CMAKE GENERATION
# **************************************************************************

CURRENT_PATH=$PWD

# Generates standard UNIX makefiles
mkdir ../../Generated_Linux
mkdir ../../Generated_Linux/Tools

# Generates standard UNIX makefiles
cd ../../Generated_Linux/Tools
cmake -G "Unix Makefiles" ../../Development/Tools

# Exit
cd $CURRENT_PATH

