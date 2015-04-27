#----------------------------------------------------------------
# PROJECT common CMake file
#----------------------------------------------------------------

message (STATUS "")
message (STATUS "PROJECT : ${PROJECT_NAME}")

# Set the build types.
# Supports Debug, Release, MinSizeRel, and RelWithDebInfo, anything else will be ignored.
set (CMAKE_BUILD_TYPES Debug Release)
set (CMAKE_CONFIGURATION_TYPES "${CMAKE_BUILD_TYPES}" CACHE STRING "" FORCE)
mark_as_advanced (CMAKE_CONFIGURATION_TYPES)

set (BUILD_TYPELIST "${CMAKE_BUILD_TYPES}" CACHE STRING "" FORCE)

set (LIBRARY_PATH ${CMAKE_CURRENT_BINARY_DIR})
file (TO_NATIVE_PATH ${LIBRARY_PATH} LIBRARY_PATH)

# set (EXECUTABLE_OUTPUT_PATH ${LIBRARY_PATH})
# set (LIBRARY_OUTPUT_PATH ${LIBRARY_PATH})

#--------------------
#--------------------
#MESSAGE (STATUS "LIBRARY_PATH : ${LIBRARY_PATH}")
#--------------------
#--------------------

#-----------------------------------------------
# Defines files lists
#-----------------------------------------------

#add_subdirectory ("${CMAKE_SOURCE_DIR}/API/Core")

#FILE(GLOB myincList "${CMAKE_SOURCE_DIR}/API/Core/Inc/*.h*")
#FILE(GLOB myinlList "${CMAKE_SOURCE_DIR}/API/Core/Inc/*.inl")
#FILE(GLOB mysrcList "${CMAKE_SOURCE_DIR}/API/Core/Src/*.c*")

#source_group ("Core\\Inc" FILES ${myincList})
#source_group ("Core\\Inl" FILES ${myinlList})
#source_group ("Core\\Src" FILES ${mysrcList})

#-----------------------------------------------
# Declare a macro to define subgroups for source files
# in Visual Studio by filtering them uppon filenames
#-----------------------------------------------

#-- Macro used to add subgroups
MACRO (BUILD_GROUP package)
file (GLOB incList "${CMAKE_SOURCE_DIR}/GigaSpace/${package}/*.h*")
file (GLOB inlList "${CMAKE_SOURCE_DIR}/GigaSpace/${package}/*.inl")
file (GLOB srcList "${CMAKE_SOURCE_DIR}/GigaSpace/${package}/*.c*")
file (GLOB glslList "${CMAKE_SOURCE_DIR}/GigaSpace/${package}/*.*glsl*")
source_group ("${package}\\Inc" FILES ${incList})
source_group ("${package}\\Inl" FILES ${inlList})
source_group ("${package}\\Src" FILES ${srcList})
source_group ("${package}\\Res" FILES ${glslList})
ENDMACRO (BUILD_GROUP)

#-- Add subgroups
BUILD_GROUP(GvCore)
BUILD_GROUP(GvStructure)
BUILD_GROUP(GvCache)
BUILD_GROUP(GvRendering)
BUILD_GROUP(GvUtils)
BUILD_GROUP(GvPerfMon)
BUILD_GROUP(GvVoxelizer)

FILE (GLOB_RECURSE incList "*.h*")
FILE (GLOB_RECURSE inlList "*.inl")
FILE (GLOB_RECURSE cppList "*.cpp")
FILE (GLOB_RECURSE cList "*.c")
FILE (GLOB_RECURSE cuList "*.cu")
FILE (GLOB_RECURSE glslList "*.*glsl")

SET (srcList ${cppList} ${cList} ${cuList})

#-----------------------------------------------
# Manage Visual studio group files 
#-----------------------------------------------

#source_group (Inc FILES ${incList})
#source_group (Inl FILES ${inlList})
#source_group (Src FILES ${srcList})


#-----------------------------------------------
# Manage generated files 
#-----------------------------------------------

# Add the header files to the project (only for Win32???)
#INCLUDE_DIRECTORIES ( ${CMAKE_CURRENT_SOURCE_DIR}/Inc/)
SET (resList ${resList} ${incList})
SET (resList ${resList} ${inlList})
SET (resList ${resList} ${glslList})


#-----------------------------------------------
# Manage Win32 definitions
#-----------------------------------------------
if (WIN32)
	ADD_DEFINITIONS (-DWIN32 -D_WINDOWS)
	if (NOT USE_FULL_WIN32_H)
		ADD_DEFINITIONS (-DWIN32_LEAN_AND_MEAN)
	endif (NOT USE_FULL_WIN32_H)
endif (WIN32)

SET (CMAKE_CXX_STANDARD_LIBRARIES "")


#-----------------------------------------------
# Define targets
#-----------------------------------------------

STRING (REGEX MATCH "GV_EXE" _matchExe "${GV_TARGET_TYPE}")
STRING (REGEX MATCH "GV_STATIC_LIB" _matchStaticLib "${GV_TARGET_TYPE}")
STRING (REGEX MATCH "GV_SHARED_LIB" _matchSharedLib "${GV_TARGET_TYPE}")
STRING (REGEX MATCH "GV_PLUGIN" _matchGigaSpacePlugin "${GV_TARGET_TYPE}")

STRING (TOUPPER ${PROJECT_NAME} PROJECTDLL)

#--
if (_matchExe)
	add_definitions (-D${PROJECTDLL}_MAKEDLL)
	ADD_EXECUTABLE (${PROJECT_NAME} ${srcList} ${resList})
endif (_matchExe)

#--
if (_matchStaticLib)
	add_definitions (-D${PROJECTDLL}_MAKELIB)
#	ADD_LIBRARY (${PROJECT_NAME} STATIC ${srcList} ${resList})
	CUDA_ADD_LIBRARY (${PROJECT_NAME} STATIC ${srcList} ${resList})
endif (_matchStaticLib)

#--
if (_matchSharedLib)
	add_definitions (-D${PROJECTDLL}_MAKEDLL)
	#ADD_LIBRARY (${PROJECT_NAME} SHARED ${srcList} ${resList})
	CUDA_ADD_LIBRARY (${PROJECT_NAME} SHARED ${srcList} ${resList})
endif (_matchSharedLib)

#--
if (_matchGigaSpacePlugin)
	add_definitions (-D${PROJECTDLL}_MAKEDLL)
    ADD_LIBRARY (${PROJECT_NAME} SHARED ${srcList} ${resList})
	if ( NOT PLUGIN_EXTENSION )
		set( PLUGIN_EXTENSION "gvp" )
	endif ( NOT PLUGIN_EXTENSION )
endif (_matchGigaSpacePlugin)

	
SET_TARGET_PROPERTIES (${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX ".d")

foreach (it ${projectLibList})
	ADD_DEPENDENCIES (${PROJECT_NAME} ${it})
	TARGET_LINK_LIBRARIES( ${PROJECT_NAME} debug ${it}.d optimized ${it})
endforeach (it)

ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
#	@echo Copying files to release directory...)
	echo Copying files to release directory...)

#-----------------------------------------------
# A macro for post build copy
#-----------------------------------------------
MACRO(POST_BUILD_COPY Src Dst)
FILE(TO_NATIVE_PATH ${Src} SrcNative)
FILE(TO_NATIVE_PATH ${Dst} DstNative)

IF (WIN32)
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD
	COMMAND if EXIST \"${SrcNative}\" \(
	COMMAND 	echo F | xcopy /d /y /i \"${SrcNative}\" \"${DstNative}\" 
	COMMAND 	if errorlevel 1 \( 
	COMMAND			echo Error can't copy \"${SrcNative}\" to \"${DstNative}\"
	COMMAND			exit 1 
	COMMAND		\)
	COMMAND \)
	)

ELSE (WIN32)

## If the Src is of the form /my/path/*.*, then need to iterate over the list
## cp does not work as copy under win32
#FILE(GLOB myListOfFiles ${Src})
#LIST(LENGTH myListOfFiles myListLength)
#
#IF(myListLength EQUAL 0)
#	SET(myListOfFiles ${Src}) # of there was only one file, ie. /my/path/mycodec.vip, not GLOBED !
#ENDIF(myListLength EQUAL 0)
#
#FOREACH( myFile ${myListOfFiles})
		#MESSAGE(STATUS "Processing ${Src}")
		#MESSAGE( STATUS "for myFile in ${Src} \; do if [ -e $myFile ] \;  then cp -u $myFile ${Dst} \; fi; done")
		ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD
		COMMAND for myFile in ${Src} \; do if [ -e $$myFile ] \;  then cp -f -s $$myFile ${Dst} \; fi; done)
#ENDFOREACH( myFile)

ENDIF (WIN32)

ENDMACRO(POST_BUILD_COPY)


# Copy binary files
#-----------------------------------------------
if (RELEASE_BIN_DIR)

MAKE_DIRECTORY(${RELEASE_BIN_DIR})

# There's a bug with VC (at least the 7.1 version): the generated pdb in debug has the name ${PROJECT_NAME}.pdb instead of ${PROJECT_NAME}.d.pdb
# So we copy both files
#ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
#	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.pdb\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.pdb\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.pdb\")
#ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
#	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.pdb\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.pdb\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.pdb\")

IF (WIN32)

IF(_matchExe)
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.exe\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.exe\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.exe\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.pdb\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.pdb\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.pdb\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.pdb\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.pdb\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.pdb\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.exe\" copy \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.exe\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.exe\")

#POST_BUILD_COPY(${LIBRARY_PATH}/Release/${PROJECT_NAME}.d.exe ${RELEASE_DIR_BIN})
#POST_BUILD_COPY(${LIBRARY_PATH}/Release/${PROJECT_NAME}.exe ${RELEASE_DIR_BIN})

ELSE(_matchExe)

IF(_matchSharedLib)
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.dll\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.dll\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.dll\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.pdb\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.pdb\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.pdb\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.pdb\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.pdb\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.pdb\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.dll\" copy \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.dll\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.dll\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.so\" copy \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.dll\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.so\")
ENDIF(_matchSharedLib)

IF(_matchGigaSpacePlugin)
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.dll\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.dll\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.${PLUGIN_EXTENSION}\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.pdb\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.pdb\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.pdb\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.pdb\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.pdb\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.pdb\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.dll\" copy \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.dll\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.${PLUGIN_EXTENSION}\")
ENDIF(_matchGigaSpacePlugin)

ENDIF(_matchExe)



ELSE (WIN32)

IF(_matchExe )# AND RELEASE_DIR_BIN)
    #FILE( MAKE_DIRECTORY ${RELEASE_DIR_BIN})
    POST_BUILD_COPY( ${LIBRARY_PATH}/${PROJECT_NAME}* ${RELEASE_BIN_DIR})
ENDIF (_matchExe )# AND RELEASE_DIR_BIN)

#IF(NOT _matchExe AND RELEASE_DIR_LIB)

IF(_matchSharedLib)
    #FILE( MAKE_DIRECTORY ${RELEASE_DIR_LIB})
    
	POST_BUILD_COPY( ${LIBRARY_PATH}/lib${PROJECT_NAME}.* ${RELEASE_BIN_DIR} )
	
	if ( RELEASE_LIB_DIR )
		MAKE_DIRECTORY (${RELEASE_LIB_DIR})
		POST_BUILD_COPY( ${LIBRARY_PATH}/lib${PROJECT_NAME}.* ${RELEASE_LIB_DIR} )
	endif ()
	
ENDIF(_matchSharedLib)

IF(_matchStaticLib)
    #FILE( MAKE_DIRECTORY ${RELEASE_DIR_LIB})
    POST_BUILD_COPY( ${LIBRARY_PATH}/lib${PROJECT_NAME}.* ${RELEASE_BIN_DIR} )
ENDIF(_matchStaticLib)

#ENDIF(NOT _matchExe AND RELEASE_DIR_LIB)

IF(_matchPlugin)
#	FILE( MAKE_DIRECTORY ${RELEASE_DIR_BIN})
	POST_BUILD_COPY(${LIBRARY_PATH}/lib${PROJECT_NAME}.d.${PLUGIN_EXTENSION} ${RELEASE_BIN_DIR})
	POST_BUILD_COPY(${LIBRARY_PATH}/lib${PROJECT_NAME}.${PLUGIN_EXTENSION} ${RELEASE_BIN_DIR})
ENDIF(_matchPlugin)


ENDIF (WIN32)



endif( RELEASE_BIN_DIR )

# Copy library files
#-----------------------------------------------
if ( RELEASE_LIB_DIR )

MAKE_DIRECTORY (${RELEASE_LIB_DIR})
IF (WIN32)
    ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	    IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.lib\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.lib\" \"${RELEASE_LIB_DIR}/${PROJECT_NAME}.d.lib\")
    ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	    IF EXIST \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.lib\" copy \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.lib\" \"${RELEASE_LIB_DIR}/${PROJECT_NAME}.lib\")
ENDIF (WIN32)

endif ( RELEASE_LIB_DIR )

# Copy header files
#-----------------------------------------------

# Macro used to copy header files
MACRO(GV_COPY_ROOT_HEADER)
#ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
#	copy \"${CMAKE_SOURCE_DIR}\\GigaSpace\\GvConfig.h\" \"${RELEASE_INC_DIR}\")
POST_BUILD_COPY(${CMAKE_SOURCE_DIR}/GigaSpace/GvConfig.h ${RELEASE_INC_DIR})
ENDMACRO(GV_COPY_ROOT_HEADER)

MACRO(GV_COPY_HEADER package)
MAKE_DIRECTORY(${RELEASE_INC_DIR}\\${package})
#ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
#	copy \"${CMAKE_SOURCE_DIR}\\GigaSpace\\${package}\\*.h*\" \"${RELEASE_INC_DIR}\\${package}\")
POST_BUILD_COPY(${CMAKE_SOURCE_DIR}/GigaSpace/${package}/*.h* ${RELEASE_INC_DIR}/${package})
IF (inlList)
#ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
#	copy \"${CMAKE_SOURCE_DIR}\\GigaSpace\\${package}\\*.inl\" \"${RELEASE_INC_DIR}\\${package}\")
POST_BUILD_COPY(${CMAKE_SOURCE_DIR}/GigaSpace/${package}/*.inl ${RELEASE_INC_DIR}/${package})
ENDIF (inlList)
ENDMACRO(GV_COPY_HEADER)

MACRO(GS_COPY_HEADER package)
MAKE_DIRECTORY(${RELEASE_INC_DIR}\\${package})
#ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
#	copy \"${CMAKE_SOURCE_DIR}\\GsGraphics\\${package}\\*.h*\" \"${RELEASE_INC_DIR}\\${package}\")
#POST_BUILD_COPY(${CMAKE_SOURCE_DIR}/GsGraphics/${package}/*.h* ${RELEASE_INC_DIR}/${package})
POST_BUILD_COPY(${CMAKE_SOURCE_DIR}/${package}/${package}/*.h* ${RELEASE_INC_DIR}/${package})
IF (inlList)
#ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
#	copy \"${CMAKE_SOURCE_DIR}\\GsGraphics\\${package}\\*.inl\" \"${RELEASE_INC_DIR}\\${package}\")
#POST_BUILD_COPY(${CMAKE_SOURCE_DIR}/GsGraphics/${package}/*.inl ${RELEASE_INC_DIR}/${package})
POST_BUILD_COPY(${CMAKE_SOURCE_DIR}/${package}/${package}/*.inl ${RELEASE_INC_DIR}/${package})
ENDIF (inlList)
ENDMACRO(GS_COPY_HEADER)

if ( RELEASE_INC_DIR )

message (STATUS "${RELEASE_INC_DIR}")

file( MAKE_DIRECTORY(${RELEASE_INC_DIR}) )

IF(MODULE)
    MAKE_DIRECTORY(${RELEASE_INC_DIR}/${MODULE})
    #ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
    #	copy \"${LIBRARY_PATH}\\Inc\\${MODULE}\\*.h\" \"${RELEASE_INC_DIR}/${MODULE}\")
    POST_BUILD_COPY(${LIBRARY_PATH}/Inc/${MODULE}/*.h* ${RELEASE_INC_DIR}/${MODULE})
IF (inlList)
    #ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
    #	copy \"${LIBRARY_PATH}\\Inc\\${MODULE}\\*.inl\" \"${RELEASE_INC_DIR}/${MODULE}\")
    POST_BUILD_COPY(${LIBRARY_PATH}/Inc/${MODULE}/*.inl ${RELEASE_INC_DIR}/${MODULE})
ENDIF (inlList)
ELSE(MODULE)
GV_COPY_ROOT_HEADER()
GV_COPY_HEADER(GvCore)
GV_COPY_HEADER(GvCache)
GV_COPY_HEADER(GvStructure)
GV_COPY_HEADER(GvRendering)
GV_COPY_HEADER(GvUtils)
GV_COPY_HEADER(GvPerfMon)
GV_COPY_HEADER(GvVoxelizer)
GS_COPY_HEADER(GsGraphics)
GS_COPY_HEADER(GsCompute)
ENDIF(MODULE)
	
endif( RELEASE_INC_DIR )

ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
#	@echo Finish copying files...)
	echo Finish copying files...)

set (CMAKE_SUPPRESS_REGENERATION true)

