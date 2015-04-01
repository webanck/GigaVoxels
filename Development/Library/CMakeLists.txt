#----------------------------------------------------------------
# GigaSpace LIBRARY CMake file
# Main user file
#----------------------------------------------------------------

# Check CMAKE version
cmake_minimum_required (VERSION 2.8)

# Use colored output
set (CMAKE_COLOR_MAKEFILE ON)

# Force CMake to not put in the rules that re-run CMake 
# CMakeLists.txt files are not added and not used when (re)building projects
# set (CMAKE_SUPPRESS_REGENERATION ON)

#----------------------------------------------------------------
# Project
#----------------------------------------------------------------

# Invite message
MESSAGE ( STATUS "-----------------------------------------------------" )
MESSAGE ( STATUS "- PROJECT GENERATION : GigaVoxels/GigaSpace Library -" )
MESSAGE ( STATUS "-----------------------------------------------------" )

# Project name
project (Library)
	
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
find_package (CUDA 5.0 REQUIRED) # TO DO : utiliser FindCUDA à la place
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
# Third Library Dependencies
#----------------------------------------------------------------

# USER OPTION
OPTION (GV_USE_GIGASPACE_EXTERNALS "By default, use the 3rd library dependencies provided by GigaSpace. If not, user has to set PATH to its libraries." ON)

IF (GV_USE_GIGASPACE_EXTERNALS)

ELSE ()

ENDIF ()

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

SET (RELEASE_BIN_DIR ${GV_RELEASE}/Bin)
SET (RELEASE_LIB_DIR ${GV_RELEASE}/Lib)
SET (RELEASE_INC_DIR ${GV_RELEASE}/Inc)

#----------------------------------------------------------------
# Options
#----------------------------------------------------------------

option (GV_ENABLE_DEBUGABLE_DEVICE_CODE "Enable/disable generation of debug-able device code" OFF)
option (GV_USE_CUDPP_LIBRARY "Enable/disable use of CUDPP library to speed up some device kernels. If OFF, it uses Thrust library." ON)

option ( GV_USE_64_BITS "Enable/disable compilation in 64 bits." ON )
if ( GV_USE_64_BITS )
    set ( GV_SYSTEM_PROCESSOR "x64" )
else()
	set ( GV_SYSTEM_PROCESSOR "x86" )
endif()

#----------------------------------------------------------------
# CUDA : additional NVCC command line arguments
# NOTE: multiple arguments must be semi-colon delimited (e.g. --compiler-options;-Wall)
#----------------------------------------------------------------

#list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_20,code=\"sm_20,compute_20\") ==> produit une erreur à cause de compute_20
#list(APPEND CUDA_NVCC_FLAGS -G0)
#list(APPEND CUDA_NVCC_FLAGS --keep-dir;"Debug")
#list(APPEND CUDA_NVCC_FLAGS --compile)
#list(APPEND CUDA_NVCC_FLAGS -g)
#list(APPEND CUDA_NVCC_FLAGS -maxrregcount=0)
#set (CUDA_NVCC_FLAGS "-arch=sm_20")

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
list(APPEND CUDA_NVCC_FLAGS -lineinfo)

# Max nb registers
#list(APPEND CUDA_NVCC_FLAGS -maxrregcount=32)

# Instruction Optimization
# - Arithmetic Instructions
# --- Precision-related Compiler Flags
# ---- use fast math
#list(APPEND CUDA_NVCC_FLAGS -use_fast_math)
# Others :
# ---- denormalized numbers are flushed to zero)
#list(APPEND CUDA_NVCC_FLAGS -ftz=true)
# ---- less precise division
#list(APPEND CUDA_NVCC_FLAGS -prec-div=false)
# ---- less precise square root
#list(APPEND CUDA_NVCC_FLAGS -prec-sqrt=false)

# NSight Debugging
# - debug on CPU
#list(APPEND CUDA_NVCC_FLAGS -g)
# - debug on GPU
#list(APPEND CUDA_NVCC_FLAGS -G)

# CUDAcompiler verbose mode (NVCC)
#set (CUDA_VERBOSE_BUILD "ON")

#----------------------------------------------------------------
# Add GigaSpace Library path
#----------------------------------------------------------------

add_subdirectory ("${CMAKE_SOURCE_DIR}/GigaSpace")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GsGraphics")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GsCompute")
