#----------------------------------------------------------------
# GigaSpace TESTS CMake file
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
MESSAGE ( STATUS "PROJECT GENERATION : GigaSpace Tests" )
MESSAGE ( STATUS "" )

# Project name
project (Tests)

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
# GigaSpace Settings
#----------------------------------------------------------------

# SYSTEM_NAME should be Windows or Linux
SET (GV_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}")

# Set Third Party Dependencies path
MESSAGE ( STATUS "THIRD PARTY DEPENDENCIES" )
IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
    SET (GV_EXTERNAL ${CMAKE_SOURCE_DIR}/../../External/${CMAKE_SYSTEM_NAME}/x86)
ENDIF()
IF ( ${GV_DESTINATION_ARCH} STREQUAL "x64" )
    SET (GV_EXTERNAL ${CMAKE_SOURCE_DIR}/../../External/${CMAKE_SYSTEM_NAME}/x64)
ENDIF()
MESSAGE ( STATUS "path : ${GV_EXTERNAL}" )
MESSAGE ( STATUS "" )

# Set Main GigaSpace RELEASE directory
# It will contain all generated executables, demos, tools, etc...
SET (GV_RELEASE ${CMAKE_SOURCE_DIR}/../../Release)

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
SET (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../CMake")

#set (RELEASE_BIN_DIR ${GV_RELEASE}/Tests/Bin)
#set (RELEASE_LIB_DIR ${GV_RELEASE}/Tests/Lib)
#set (RELEASE_INC_DIR ${GV_RELEASE}/Tests/Inc)

#----------------------------------------------------------------
# Options
#----------------------------------------------------------------

option (GV_ENABLE_DEBUGABLE_DEVICE_CODE "Enable/disable generation of debug-able device code" OFF)

#set (CUDA_NVCC_FLAGS "-arch=sm_20;-G0")
#list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_20,code=\"sm_20,compute_20\")
#list(APPEND CUDA_NVCC_FLAGS -G0)
#list(APPEND CUDA_NVCC_FLAGS --keep-dir;"Debug")
#list(APPEND CUDA_NVCC_FLAGS -g)
#list(APPEND CUDA_NVCC_FLAGS --compile)
#list(APPEND CUDA_NVCC_FLAGS -maxrregcount=0)

# Set your compute capability version
#
# GiagVoxels requires 2.0 at least
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20")
#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_21")
#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30")
#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35")

# NVCC verbose mode
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

# CUDAcompiler verbose mode (NVCC)
#set (CUDA_VERBOSE_BUILD "ON")

#########################################################################
# Add all the sub projects
#########################################################################

# Compression
add_subdirectory ("${CMAKE_SOURCE_DIR}/RLECompression")

# Time Budget
add_subdirectory ("${CMAKE_SOURCE_DIR}/FrameTimeBudget")

# Cache Management
add_subdirectory ("${CMAKE_SOURCE_DIR}/CacheInspector")

# Serialization
add_subdirectory ("${CMAKE_SOURCE_DIR}/Serialization")

# Function Pointers
add_subdirectory ("${CMAKE_SOURCE_DIR}/FunctionPointers")

# Studying silhouettes impact
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvSilhouetteBenchmark")

# Studying grand father proxy impact
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvGFProxy")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvTraversalPolicyBenchmark")

add_subdirectory ("${CMAKE_SOURCE_DIR}/GvGFProxySphere")

add_subdirectory ("${CMAKE_SOURCE_DIR}/GvNoiseInheritanceBenchmark")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GvLazyTest")

