#!/bin/sh

# **************************************************************************
# CMAKE GENERATION
# **************************************************************************

CURRENT_PATH=$PWD

# Generates standard UNIX makefiles
mkdir ../../Generated_Linux
mkdir ../../Generated_Linux/Tests

# Generates standard UNIX makefiles
cd ../../Generated_Linux/Tests
cmake -G "Unix Makefiles" ../../Development/Tests

# Exit
cd $CURRENT_PATH

