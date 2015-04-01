# PROJECT
#____________________________

#MESSAGE(STATUS "PROJECT : ${PROJECT_NAME}")

# Set the build types.
# Supports Debug, Release, MinSizeRel, and RelWithDebInfo, anything else will be ignored.
SET(CMAKE_BUILD_TYPES Debug Release)
SET(CMAKE_CONFIGURATION_TYPES "${CMAKE_BUILD_TYPES}" CACHE STRING "" FORCE)
MARK_AS_ADVANCED(CMAKE_CONFIGURATION_TYPES)

SET(BUILD_TYPELIST "${CMAKE_BUILD_TYPES}" CACHE STRING "" FORCE)

set (LIBRARY_PATH ${CMAKE_CURRENT_BINARY_DIR})
FILE(TO_NATIVE_PATH ${LIBRARY_PATH} LIBRARY_PATH)

# set (EXECUTABLE_OUTPUT_PATH ${LIBRARY_PATH})
# set (LIBRARY_OUTPUT_PATH ${LIBRARY_PATH})


#--------------------
#--------------------
MESSAGE(STATUS "LIBRARY_PATH : ${LIBRARY_PATH}")
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
MACRO(BUILD_GROUP tutorial)
FILE(GLOB incList "${CMAKE_SOURCE_DIR}/Inc/*.h*")
FILE(GLOB inlList "${CMAKE_SOURCE_DIR}/Inc/*.inl")
FILE(GLOB srcList "${CMAKE_SOURCE_DIR}/Src/*.c*")
source_group (Inc FILES ${incList})
source_group (Inl FILES ${inlList})
source_group (Src FILES ${srcList})
ENDMACRO(BUILD_GROUP)

#-- Add subgroups
#BUILD_GROUP(${PROJECT_NAME})
#FILE(GLOB incList "${CMAKE_SOURCE_DIR}/Inc/*.h*")
#FILE(GLOB inlList "${CMAKE_SOURCE_DIR}/Inc/*.inl")
#FILE(GLOB srcList "${CMAKE_SOURCE_DIR}/Src/*.c*")
#source_group (Inc FILES ${incList})
#source_group (Inl FILES ${inlList})
#source_group (Src FILES ${srcList})

FILE(GLOB incList "${CMAKE_CURRENT_SOURCE_DIR}/Inc/*.h*")
FILE(GLOB inlList "${CMAKE_CURRENT_SOURCE_DIR}/Inc/*.inl")
FILE(GLOB cppList "${CMAKE_CURRENT_SOURCE_DIR}/Src/*.cpp")
FILE(GLOB cList "${CMAKE_CURRENT_SOURCE_DIR}/Src/*.c")
FILE(GLOB cuList "${CMAKE_CURRENT_SOURCE_DIR}/Src/*.cu")
FILE(GLOB uiList "${CMAKE_CURRENT_SOURCE_DIR}/Ui/*.ui")
FILE(GLOB rcList "${CMAKE_CURRENT_SOURCE_DIR}/Ui/*.qrc")
file(GLOB glslList "${CMAKE_CURRENT_SOURCE_DIR}/Res/*.*glsl*")

#FILE(GLOB_RECURSE incList "*.h*")
#FILE(GLOB_RECURSE inlList "*.inl")
#FILE(GLOB_RECURSE cppList "*.cpp")
#FILE(GLOB_RECURSE cList "*.c")
#FILE(GLOB_RECURSE cuList "*.cu")

SET(srcList ${cppList} ${cList} ${cuList})

#MESSAGE(STATUS "incList : ${incList}")
#MESSAGE(STATUS "inlList : ${inlList}")
#MESSAGE(STATUS "cppList : ${cppList}")
#MESSAGE(STATUS "cList : ${cList}")
#MESSAGE(STATUS "cuList : ${cuList}")
#MESSAGE(STATUS "srcList : ${srcList}")

source_group (Inc FILES ${incList})
source_group (Inl FILES ${inlList})
source_group (Src FILES ${srcList})
source_group (Res FILES ${glslList})

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
INCLUDE_DIRECTORIES ( ${CMAKE_CURRENT_SOURCE_DIR}/Inc)
SET( resList ${resList} ${incList} )
SET( resList ${resList} ${inlList} )
SET( resList ${resList} ${glslList} )

# Qt if used
IF (GV_QT_UIC_EXECUTABLE)
INCLUDE_DIRECTORIES ( BEFORE ${LIBRARY_PATH}/Inc)
GV_QT4_WRAP_UI (genList ${uiList})
GV_QT4_ADD_RESOURCES (genList ${rcList})
GV_QT4_AUTOMOC (genList ${incList})
ENDIF (GV_QT_UIC_EXECUTABLE)

# Add generated files in a folder (Visual Studio)
source_group (Generated FILES ${genList})

#-----------------------------------------------
# Manage Win32 definitions
#-----------------------------------------------
if (WIN32)
	ADD_DEFINITIONS ( -DWIN32 -D_WINDOWS )
	if (NOT USE_FULL_WIN32_H)
		ADD_DEFINITIONS (-DWIN32_LEAN_AND_MEAN)
	endif (NOT USE_FULL_WIN32_H)
endif (WIN32)

SET(CMAKE_CXX_STANDARD_LIBRARIES "")

STRING (TOUPPER ${PROJECT_NAME} PROJECTDLL)
add_definitions (-D${PROJECTDLL}_MAKEDLL)


#-----------------------------------------------
# Define targets
#-----------------------------------------------
STRING (REGEX MATCH "GV_EXE" _matchExe "${GV_TARGET_TYPE}")
STRING (REGEX MATCH "GV_STATIC_LIB" _matchStaticLib "${GV_TARGET_TYPE}")
STRING (REGEX MATCH "GV_SHARED_LIB" _matchSharedLib "${GV_TARGET_TYPE}")
STRING (REGEX MATCH "GV_PLUGIN" _matchGigaSpacePlugin "${GV_TARGET_TYPE}")

#--
if (_matchExe)
	add_definitions (-D${PROJECTDLL}_MAKEDLL)
	#ADD_EXECUTABLE (${PROJECT_NAME} ${srcList} ${resList})
	CUDA_ADD_EXECUTABLE (${PROJECT_NAME} ${srcList} ${genList} ${resList})
endif (_matchExe)

#--
if (_matchStaticLib)
	add_definitions (-D${PROJECTDLL}_MAKELIB)
#	ADD_LIBRARY (${PROJECT_NAME} STATIC ${srcList} ${resList})
	CUDA_ADD_LIBRARY (${PROJECT_NAME} STATIC ${srcList} ${genList} ${resList})
endif (_matchStaticLib)

#--
if (_matchSharedLib)
	add_definitions (-D${PROJECTDLL}_MAKEDLL)
	#ADD_LIBRARY (${PROJECT_NAME} SHARED ${srcList} ${resList})
	CUDA_ADD_LIBRARY (${PROJECT_NAME} SHARED ${srcList} ${genList} ${resList})
endif (_matchSharedLib)

#--
if (_matchGigaSpacePlugin)
	add_definitions (-D${PROJECTDLL}_MAKEDLL)
    #ADD_LIBRARY (${PROJECT_NAME} SHARED ${srcList} ${resList})
	CUDA_ADD_LIBRARY (${PROJECT_NAME} SHARED ${srcList} ${genList} ${resList})
	if ( NOT PLUGIN_EXTENSION )
		set( PLUGIN_EXTENSION "gvp" )
	endif ( NOT PLUGIN_EXTENSION )
endif (_matchGigaSpacePlugin)

	
SET_TARGET_PROPERTIES (${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX ".d")

foreach (it ${projectLibList})
	ADD_DEPENDENCIES (${PROJECT_NAME} ${it})
	TARGET_LINK_LIBRARIES( ${PROJECT_NAME} debug ${it}.d.lib optimized ${it}.lib)
endforeach (it)

ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
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
		COMMAND for myFile in ${Src} \; do if [ -e $$myFile ] \;  then cp -u $$myFile ${Dst} \; fi; done)
#ENDFOREACH( myFile)

ENDIF (WIN32)

ENDMACRO(POST_BUILD_COPY)

# Copy binary files
#-----------------------------------------------
if ( RELEASE_BIN_DIR )

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
ENDIF(_matchSharedLib)

IF(_matchStaticLib)
    #FILE( MAKE_DIRECTORY ${RELEASE_DIR_LIB})
    POST_BUILD_COPY( ${LIBRARY_PATH}/lib${PROJECT_NAME}.* ${RELEASE_BIN_DIR} )
ENDIF(_matchStaticLib)

#ENDIF(NOT _matchExe AND RELEASE_DIR_LIB)

IF(_matchGigaSpacePlugin)
#	FILE( MAKE_DIRECTORY ${RELEASE_DIR_BIN})
#	POST_BUILD_COPY(${LIBRARY_PATH}/lib${PROJECT_NAME}.d.${PLUGIN_EXTENSION} ${RELEASE_BIN_DIR})
#	POST_BUILD_COPY(${LIBRARY_PATH}/lib${PROJECT_NAME}.${PLUGIN_EXTENSION} ${RELEASE_BIN_DIR})
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	${CMAKE_COMMAND} copy -E ${LIBRARY_PATH}/lib${PROJECT_NAME}.so ${RELEASE_BIN_DIR}/lib${PROJECT_NAME}.${PLUGIN_EXTENSION})
#ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
#	${CMAKE_COMMAND} copy -E ${LIBRARY_PATH}/lib${PROJECT_NAME}.d.so ${RELEASE_BIN_DIR}/lib${PROJECT_NAME}.d.${PLUGIN_EXTENSION})
ENDIF(_matchGigaSpacePlugin)


ENDIF (WIN32)

endif( RELEASE_BIN_DIR )

# Copy library files
#-----------------------------------------------
if ( RELEASE_LIB_DIR )

MAKE_DIRECTORY(${RELEASE_LIB_DIR})

ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.lib\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.lib\" \"${RELEASE_LIB_DIR}/${PROJECT_NAME}.d.lib\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.lib\" copy \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.lib\" \"${RELEASE_LIB_DIR}/${PROJECT_NAME}.lib\")

endif ( RELEASE_LIB_DIR )

# Copy header files
#-----------------------------------------------

# Macro used to copy header files
MACRO(GV_COPY_HEADER package)
MAKE_DIRECTORY(${RELEASE_INC_DIR}\\${package})
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	copy \"${CMAKE_SOURCE_DIR}\\Library\\${package}\\*.h*\" \"${RELEASE_INC_DIR}\\${package}\")
IF (inlList)
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	copy \"${CMAKE_SOURCE_DIR}\\Library\\${package}\\*.inl\" \"${RELEASE_INC_DIR}\\${package}\")
ENDIF (inlList)
ENDMACRO(GV_COPY_HEADER)


if ( RELEASE_INC_DIR )

MAKE_DIRECTORY(${RELEASE_INC_DIR})

IF(MODULE)
MAKE_DIRECTORY(${RELEASE_INC_DIR}/${MODULE})
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	copy \"${LIBRARY_PATH}\\Inc\\${MODULE}\\*.h\" \"${RELEASE_INC_DIR}/${MODULE}\")
IF (inlList)
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	copy \"${LIBRARY_PATH}\\Inc\\${MODULE}\\*.inl\" \"${RELEASE_INC_DIR}/${MODULE}\")
ENDIF (inlList)
ELSE(MODULE)
#GV_COPY_HEADER(GvCore)
#GV_COPY_HEADER(GvCache)
#GV_COPY_HEADER(GvStructure)
#GV_COPY_HEADER(GvRenderer)
#GV_COPY_HEADER(GvUtils)
ENDIF(MODULE)
	
endif( RELEASE_INC_DIR )

ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	echo Finish copying files...)

set (CMAKE_SUPPRESS_REGENERATION true)

