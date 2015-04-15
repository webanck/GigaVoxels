#!/bin/sh

CURRENTSCRIPTPATH=$PWD

cd ..
cd ..
cd Development
cd Documents
cd Doxygen

doxygen DoxyfileGigaSpace.cfg

cd $CURRENTSCRIPTPATH
