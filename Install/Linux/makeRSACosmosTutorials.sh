#!/bin/sh

# **************************************************************************
# CMAKE GENERATION
# **************************************************************************

CURRENT_PATH=$PWD

# Generates standard UNIX makefiles
mkdir ../../Generated_Linux
mkdir ../../Generated_Linux/Tutorials
mkdir ../../Generated_Linux/Tutorials/RSACosmos

# Generates standard UNIX makefiles
cd ../../Generated_Linux/Tutorials/RSACosmos
cmake -G "Unix Makefiles" ../../../Development/Tutorials/RSACosmos

# Exit
cd $CURRENT_PATH
