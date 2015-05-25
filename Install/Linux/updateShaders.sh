#!/bin/sh

# **************************************************************************
# Setups environment variables
# **************************************************************************

# PATH : GigaVoxels RELEASE directory
# export GV_RELEASE=$PWD/../../Release
GV_RELEASE=../../Release

# PATH : GigaVoxels Development directory
GV_DATA=../../Development

# **************************************************************************
# GLSL shaders
# **************************************************************************

# Demos
mkdir -p $GV_RELEASE/Bin/Data/Shaders/SimpleSphere
cp -v -L -u -R $GV_DATA/Tutorials/Demos/ProceduralTechnics/SimpleSphere/Res/*.* $GV_RELEASE/Bin/Data/Shaders/SimpleSphere/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/ProxyGeometry
cp -v -L -u -R $GV_DATA/Tutorials/Demos/GraphicsInteroperability/ProxyGeometry/Res/*.* $GV_RELEASE/Bin/Data/Shaders/ProxyGeometry/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/RendererGLSL
cp -v -L -u -R $GV_DATA/Tutorials/Demos/GraphicsInteroperability/RendererGLSL/Res/*.* $GV_RELEASE/Bin/Data/Shaders/RendererGLSL/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/RendererGLSLSphere
cp -v -L -u -R $GV_DATA/Tutorials/Demos/GraphicsInteroperability/RendererGLSLSphere/Res/*.* $GV_RELEASE/Bin/Data/Shaders/RendererGLSLSphere/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/Voxelization
cp -v -L -u -R $GV_DATA/Tutorials/Demos/Voxelization/Voxelization/Res/*.* $GV_RELEASE/Bin/Data/Shaders/Voxelization/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/VoxelizationSignedDistanceField
cp -v -L -u -R $GV_DATA/Tutorials/Demos/Voxelization/VoxelizationSignedDistanceField/Res/*.* $GV_RELEASE/Bin/Data/Shaders/VoxelizationSignedDistanceField/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/CastShadows
cp -v -L -u -R $GV_DATA/Tutorials/Demos/GraphicsInteroperability/CastShadows/Res/*.* $GV_RELEASE/Bin/Data/Shaders/CastShadows/.

# Viewer plugins
mkdir -p $GV_RELEASE/Bin/Data/Shaders/GvDepthPeeling
cp -v -L -u -R $GV_DATA/Tutorials/ViewerPlugins/GvDepthPeeling/Res/*.* $GV_RELEASE/Bin/Data/Shaders/GvDepthPeeling/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/GvProxyGeometryManager
cp -v -L -u -R $GV_DATA/Tutorials/ViewerPlugins/GvProxyGeometryManager/Res/*.* $GV_RELEASE/Bin/Data/Shaders/GvProxyGeometryManager/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/GvRayMapGenerator
cp -v -L -u -R $GV_DATA/Tutorials/ViewerPlugins/GvRayMapGenerator/Res/*.* $GV_RELEASE/Bin/Data/Shaders/GvRayMapGenerator/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/GvRendererGLSL
cp -v -L -u -R $GV_DATA/Tutorials/ViewerPlugins/GvRendererGLSL/Res/*.* $GV_RELEASE/Bin/Data/Shaders/GvRendererGLSL/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/GvSimpleShapeGLSL
cp -v -L -u -R $GV_DATA/Tutorials/ViewerPlugins/GvSimpleShapeGLSL/Res/*.* $GV_RELEASE/Bin/Data/Shaders/GvSimpleShapeGLSL/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/GvShadowMap
cp -v -L -u -R $GV_DATA/Tutorials/ViewerPlugins/GvShadowMap/Res/*.* $GV_RELEASE/Bin/Data/Shaders/GvShadowMap/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/GvSignedDistanceFieldVoxelization
cp -v -L -u -R $GV_DATA/Tutorials/ViewerPlugins/GvSignedDistanceFieldVoxelization/Res/*.* $GV_RELEASE/Bin/Data/Shaders/GvSignedDistanceFieldVoxelization/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/GvSimpleSphere
cp -v -L -u -R $GV_DATA/Tutorials/ViewerPlugins/GvSimpleSphere/Res/*.* $GV_RELEASE/Bin/Data/Shaders/GvSimpleSphere/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/GvAnimatedCylinders
cp -v -L -u -R $GV_DATA/Tutorials/ViewerPlugins/GvAnimatedCylinders/Res/*.* $GV_RELEASE/Bin/Data/Shaders/GvAnimatedCylinders/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/GvVBOGenerator
cp -v -L -u -R $GV_DATA/Tutorials/ViewerPlugins/GvVBOGenerator/Res/*.* $GV_RELEASE/Bin/Data/Shaders/GvVBOGenerator/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/GvAnimatedLUT
cp -v -L -u -R $GV_DATA/Tutorials/ViewerPlugins/GvAnimatedLUT/Res/*.* $GV_RELEASE/Bin/Data/Shaders/GvAnimatedLUT/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/GvInstancing
cp -v -L -u -R $GV_DATA/Tutorials/ViewerPlugins/GvInstancing/Res/*.* $GV_RELEASE/Bin/Data/Shaders/GvInstancing/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/GvSlisesix
cp -v -L -u -R $GV_DATA/Tutorials/ViewerPlugins/GvInstancing/Res/*.* $GV_RELEASE/Bin/Data/Shaders/GvSlisesix/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/GvEnvironmentMapping
cp -v -L -u -R $GV_DATA/Tutorials/ViewerPlugins/GvEnvironmentMapping/Res/*.* $GV_RELEASE/Bin/Data/Shaders/GvEnvironmentMapping/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/GvAnimatedSnake
cp -v -L -u -R $GV_DATA/Tutorials/ViewerPlugins/GvAnimatedSnake/Res/*.* $GV_RELEASE/Bin/Data/Shaders/GvAnimatedSnake/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/GvShadowCasting
cp -v -L -u -R $GV_DATA/Tutorials/ViewerPlugins/GvShadowCasting/Res/*.* $GV_RELEASE/Bin/Data/Shaders/GvShadowCasting/.
mkdir -p $GV_RELEASE/Bin/Data/Shaders/GvCastShadows
cp -v -L -u -R $GV_DATA/Tutorials/ViewerPlugins/GvCastShadows/Res/*.* $GV_RELEASE/Bin/Data/Shaders/GvCastShadows/.

