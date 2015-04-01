#----------------------------------------------------------------
# GigaVoxels LIBRARY CMake file
# Main user file
#----------------------------------------------------------------

# Check CMAKE version
cmake_minimum_required (VERSION 2.8) # needed to have the findCUDA module

# Use colored output
set (CMAKE_COLOR_MAKEFILE ON)

# Force CMake to not put in the rules that re-run CMake 
# CMakeLists.txt files are not added and not used when (re)building projects
# set (CMAKE_SUPPRESS_REGENERATION ON)

#----------------------------------------------------------------
# Project
#----------------------------------------------------------------

# Invite message
MESSAGE ( STATUS "" )
MESSAGE ( STATUS "PROJECT GENERATION : GigaVoxels Tutorials" )
MESSAGE ( STATUS "" )

# Project name
project (ViewerPluginTutorials)

#----------------------------------------------------------------
# Check architecture
#
# Note :
# CMAKE_SIZEOF_VOID_P is undefined if used before
# a call to the "project" command.
#----------------------------------------------------------------

IF ( CMAKE_SIZEOF_VOID_P EQUAL 4 )
    MESSAGE ( STATUS "ARCHITECTURE : 32 bits" )
    SET (GV_DESTINATION_ARCH "x86")
ENDIF()
IF (CMAKE_SIZEOF_VOID_P EQUAL 8)
    MESSAGE ( STATUS "ARCHITECTURE : 64 bits" )
    SET ( GV_DESTINATION_ARCH "x64" )
ENDIF()
MESSAGE ( STATUS "" )

#----------------------------------------------------------------
# Required packages
#----------------------------------------------------------------

# Search for CUDA
MESSAGE ( STATUS "REQUIRED PACKAGE : CUDA 5.0" )
find_package (CUDA REQUIRED) # TO DO : utiliser FindCUDA à la place
if (CUDA_FOUND)
	message (STATUS "system has CUDA")
	message (STATUS "CUDA version : ${CUDA_VERSION_STRING}")
	message (STATUS "CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
	message (STATUS "CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}")
	message (STATUS "CUDA_LIBRARIES = ${CUDA_LIBRARIES}")
endif()
MESSAGE ( STATUS "" )

# Search for OpenGL
MESSAGE ( STATUS "REQUIRED PACKAGE : OpenGL" )
find_package (OpenGL REQUIRED) # TO DO : utiliser FindOpenGL à la place
if (OPENGL_FOUND)
	message (STATUS "system has OpenGL")
endif()
if (OPENGL_GLU_FOUND)
	message (STATUS "system has GLU")
endif()
if (OPENGL_XMESA_FOUND)
	message (STATUS "system has XMESA")
endif()
message (STATUS "OPENGL_INCLUDE_DIR = ${OPENGL_INCLUDE_DIR}")
message (STATUS "OPENGL_LIBRARIES = ${OPENGL_LIBRARIES}")
message (STATUS "OPENGL_gl_LIBRARY = ${OPENGL_gl_LIBRARY}")
message (STATUS "OPENGL_glu_LIBRARY = ${OPENGL_glu_LIBRARY}")
MESSAGE ( STATUS "" )

# Search for ImageMagick
MESSAGE ( STATUS "REQUIRED PACKAGE : ImageMagick" )
find_package (ImageMagick COMPONENTS Magick++)
MESSAGE ( STATUS "" )

#----------------------------------------------------------------
# GigaVoxels Settings
#----------------------------------------------------------------

# SYSTEM_NAME should be Windows or Linux
SET (GV_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}")

# Set Third Party Dependencies path
MESSAGE ( STATUS "THIRD PARTY DEPENDENCIES" )
IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
    SET (GV_EXTERNAL ${CMAKE_SOURCE_DIR}/../../../External/${CMAKE_SYSTEM_NAME}/x86)
ENDIF()
IF ( ${GV_DESTINATION_ARCH} STREQUAL "x64" )
    SET (GV_EXTERNAL ${CMAKE_SOURCE_DIR}/../../../External/${CMAKE_SYSTEM_NAME}/x64)
ENDIF()
MESSAGE ( STATUS "path : ${GV_EXTERNAL}" )
MESSAGE ( STATUS "" )

# Set Main GigaVoxels RELEASE directory
# It will contain all generated executables, demos, tools, etc...
SET (GV_RELEASE ${CMAKE_SOURCE_DIR}/../../../Release)

#----------------------------------------------------------------
# Defines
#----------------------------------------------------------------

if (MSVC)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
	add_definitions(-DNOMINMAX)
endif ()

#----------------------------------------------------------------
# Initialize some environment variables
#----------------------------------------------------------------

# Tell CMake to first search in directories listed in CMAKE_MODULE_PATH when 
# using FIND_PACKAGE() or INCLUDE()
SET (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../../CMake")

SET (RELEASE_BIN_DIR ${GV_RELEASE}/Bin)
#SET (RELEASE_LIB_DIR ${GV_RELEASE}/Lib)
#SET (RELEASE_INC_DIR ${GV_RELEASE}/Inc)

#----------------------------------------------------------------
# Options
#----------------------------------------------------------------

option (GV_ENABLE_DEBUGABLE_DEVICE_CODE "Enable/disable generation of debug-able device code" OFF)

#set (CUDA_NVCC_FLAGS "-arch=sm_20;-G0")
#list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_20,code=\"sm_20,compute_20\")
#list(APPEND CUDA_NVCC_FLAGS -G)
#list(APPEND CUDA_NVCC_FLAGS --keep-dir;"Debug")
#list(APPEND CUDA_NVCC_FLAGS -g)
#list(APPEND CUDA_NVCC_FLAGS --compile)
#list(APPEND CUDA_NVCC_FLAGS -maxrregcount=0)

#set (CUDA_NVCC_FLAGS "-arch=sm_20")
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20")
#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30")

# NVCC verbose mode : set this flag to see detailed NVCC statistics (registers usage, memory, etc...)
#list(APPEND CUDA_NVCC_FLAGS -Xptxas;-v)

# Generate line-number information for device code
#list(APPEND CUDA_NVCC_FLAGS -lineinfo)

# Max nb registers
#list(APPEND CUDA_NVCC_FLAGS -maxrregcount=32)

# Use fast math
#list(APPEND CUDA_NVCC_FLAGS -use_fast_math)

# NSight Debugging
# - debug on CPU
#list(APPEND CUDA_NVCC_FLAGS -g)
# - debug on GPU
#list(APPEND CUDA_NVCC_FLAGS -G)

# Set this flag to see detailed NVCC command lines
#set (CUDA_VERBOSE_BUILD "ON")

#########################################################################
# Add all the sub projects
# SDK Viewer : plugin integration
#########################################################################

####
# Demos / Tutorials
####

# Procedural generation
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvSimpleSphere")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvMandelbrotSet")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvSlisesix")

# Voxel data synthesis
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvMengerSponge")

# Streaming
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvSimpleSphereCPU")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvDynamicLoad")
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvRAWDataLoader")

# Noise
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvAmplifiedSurface")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvAmplifiedVolume")

# Graphics interoperability
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvRendererGLSL")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvSimpleShapeGLSL")
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvRayMapGenerator")
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvShadowMap")
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvClippingPlane")
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvDepthPeeling")

# Effects
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvDepthOfField")

# Hypertexture
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvHypertexture")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvDynamicHypertexture")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvNoiseInheritance")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvLazyHypertexture")
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvShellNoise")

# Environment Mapping
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvEnvironmentMapping")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvCastShadows")

# Demos
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvVideoGame")
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvInfiniteSky")

# Proxy geometry
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvProxyGeometryManager")

# GigaSpace
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvSphereRayTracing")
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvVBOGenerator")

# Animation
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvAnimatedLUT")

# Voxelization
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvSignedDistanceFieldVoxelization")

# Instancing
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvInstancing")

# BVH
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvSimpleTriangles")

# Collisions
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvCollisionBBox")
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvCollision")

# Animation
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvAnimatedSnake")
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvAnimatedFan")

# Environment mapping
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvNoiseInAShellGLSL")

#Others
#add_subdirectory ("${CMAKE_SOURCE_DIR}/GvVoxelsVersusVBOPoints")
