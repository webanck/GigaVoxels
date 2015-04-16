#!/bin/sh

# **************************************************************************
# FULL INSTALLATION
# **************************************************************************

CURRENT_PATH=$PWD

#Selection of the QT version for systems with multiple versions.
export QT_SELECT=4

#Creation of the compilation directory.
sh ./makeDemoTutorials.sh
sh ./makeDocumentation.sh
sh ./makeLibrary.sh
sh ./makeRSACosmosTutorials.sh
sh ./makeTests.sh
sh ./makeTools.sh
sh ./makeViewerPluginTutorials.sh

#Copy of the required stuff for the binaries directory.
sh ./updateData.sh
sh ./updateRelease.sh
sh ./updateShaders.sh

#Compilation of the library and the tools.
cd ../../Generated_Linux/Library
make
cd $CURRENT_PATH
cd ../../Generated_Linux/Tools
make
cd $CURRENT_PATH

#Compilation of the DynamicLoad ViewerPlugin.
cd ../../Generated_Linux/Tutorials/ViewerPlugins
make GvDynamicLoad
cd $CURRENT_PATH
