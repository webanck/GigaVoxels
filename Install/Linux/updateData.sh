#!/bin/sh

# **************************************************************************
# Setups environment variables
# **************************************************************************

# PATH : GigaVoxels RELEASE directory
# export GV_RELEASE=$PWD/../../Release
GV_RELEASE=../../Release

# PATH : GigaVoxels DATA directory
GV_DATA=../../Data

# **************************************************************************
# Icons
# **************************************************************************

mkdir -p $GV_RELEASE/Bin/Icons
cp -v -L -u -R $GV_DATA/Icons/*.* $GV_RELEASE/Bin/Icons/.

# **************************************************************************
# Shaders
# **************************************************************************

mkdir -p $GV_RELEASE/Bin/Data/Shaders
cp -v -L -u -R $GV_DATA/Shaders/*.* $GV_RELEASE/Bin/Data/Shaders/.

# **************************************************************************
# TransferFunctions
# **************************************************************************

mkdir -p $GV_RELEASE/Bin/Data/TransferFunctions
cp -v -L -u -R $GV_DATA/TransferFunctions/*.* $GV_RELEASE/Bin/Data/TransferFunctions/.

# **************************************************************************
# Voxels
# **************************************************************************

mkdir -p $GV_RELEASE/Bin/Data/Voxels/xyzrgb_dragon512_BR8_B1
cp -v -L -u -R $GV_DATA/Voxels/xyzrgb_dragon512_BR8_B1/*.* $GV_RELEASE/Bin/Data/Voxels/xyzrgb_dragon512_BR8_B1/.

mkdir -p $GV_RELEASE/Bin/Data/Voxels/Dino
cp -v -L -u -R $GV_DATA/Voxels/Dino/*.* $GV_RELEASE/Bin/Data/Voxels/Dino/.

mkdir -p $GV_RELEASE/Bin/Data/Voxels/vd4
cp -v -L -u -R $GV_DATA/Voxels/vd4/*.* $GV_RELEASE/Bin/Data/Voxels/vd4/.

mkdir -p $GV_RELEASE/Bin/Data/Voxels/Raw/aneurism
cp -v -L -u -R $GV_DATA/Voxels/Raw/aneurism/*.* $GV_RELEASE/Bin/Data/Voxels/Raw/aneurism/.
mkdir -p $GV_RELEASE/Bin/Data/Voxels/Raw/bonsai
cp -v -L -u -R $GV_DATA/Voxels/Raw/bonsai/*.* $GV_RELEASE/Bin/Data/Voxels/Raw/bonsai/.
mkdir -p $GV_RELEASE/Bin/Data/Voxels/Raw/foot
cp -v -L -u -R $GV_DATA/Voxels/Raw/foot/*.* $GV_RELEASE/Bin/Data/Voxels/Raw/foot/.
mkdir -p $GV_RELEASE/Bin/Data/Voxels/Raw/hydrogenAtom
cp -v -L -u -R $GV_DATA/Voxels/Raw/hydrogenAtom/*.* $GV_RELEASE/Bin/Data/Voxels/Raw/hydrogenAtom/.
mkdir -p $GV_RELEASE/Bin/Data/Voxels/Raw/neghip
cp -v -L -u -R $GV_DATA/Voxels/Raw/neghip/*.* $GV_RELEASE/Bin/Data/Voxels/Raw/neghip/.
mkdir -p $GV_RELEASE/Bin/Data/Voxels/Raw/skull
cp -v -L -u -R $GV_DATA/Voxels/Raw/skull/*.* $GV_RELEASE/Bin/Data/Voxels/Raw/skull/.

# **************************************************************************
# 3D Models
# **************************************************************************

mkdir -p $GV_RELEASE/Bin/Data/3DModels
cp -v -L -u -R $GV_DATA/3DModels/*.* $GV_RELEASE/Bin/Data/3DModels/.

mkdir -p $GV_RELEASE/Bin/Data/3DModels/stanford_buddha
cp -v -L -u -R $GV_DATA/3DModels/stanford_buddha/*.* $GV_RELEASE/Bin/Data/3DModels/stanford_buddha/.
mkdir -p $GV_RELEASE/Bin/Data/3DModels/stanford_bunny
cp -v -L -u -R $GV_DATA/3DModels/stanford_bunny/*.* $GV_RELEASE/Bin/Data/3DModels/stanford_bunny/.
cp -v -L -u -R $GV_DATA/3DModels/stanford_bunny/bunny.obj $GV_RELEASE/Bin/Data/3DModels/.
mkdir -p $GV_RELEASE/Bin/Data/3DModels/stanford_dragon
cp -v -L -u -R $GV_DATA/3DModels/stanford_dragon/*.* $GV_RELEASE/Bin/Data/3DModels/stanford_dragon/.

# **************************************************************************
# Videos
# **************************************************************************

mkdir -p $GV_RELEASE/Bin/Data/Videos
