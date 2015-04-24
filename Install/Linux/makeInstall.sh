#!/bin/sh

# **************************************************************************
# FULL INSTALLATION
# **************************************************************************

CURRENT_PATH=$PWD

#Selection of the QT version for systems with multiple versions.
export QT_SELECT=4

#Copy of the required stuff for the binaries directory.
sh ./updateData.sh && \
sh ./updateShaders.sh && \

#Compilation of the library.
sh ./makeLibrary.sh && \
cd ../../Generated_Linux/Library && \
make clean && \
make -j5 && \
cd $CURRENT_PATH && \

#Compilation of the tools.
sh ./makeTools.sh && \
cd ../../Generated_Linux/Tools && \
make clean && \
make -j5 && \
cd $CURRENT_PATH && \

#Compilation of the DynamicLoad ViewerPlugin.
sh ./makeDemoTutorials.sh && \
sh ./makeViewerPluginTutorials.sh && \
cd ../../Generated_Linux/Tutorials/ViewerPlugins && \
make clean && \
make -j5 GvDynamicLoad && \
cd $CURRENT_PATH

#~ sh ./updateRelease.sh
#~ sh ./makeDocumentation.sh
#~ sh ./makeRSACosmosTutorials.sh
#~ sh ./makeTests.sh
