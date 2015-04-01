# PROJECT
#____________________________

MESSAGE(STATUS "PROJECT : ${PROJECT_NAME}")

# Set the build types.
# Supports Debug, Release, MinSizeRel, and RelWithDebInfo, anything else will be ignored.
SET(CMAKE_BUILD_TYPES Debug Release)
SET(CMAKE_CONFIGURATION_TYPES "${CMAKE_BUILD_TYPES}" CACHE STRING "" FORCE)
MARK_AS_ADVANCED(CMAKE_CONFIGURATION_TYPES)

SET(BUILD_TYPELIST "${CMAKE_BUILD_TYPES}" CACHE STRING "" FORCE)

set (LIBRARY_PATH ${CMAKE_CURRENT_BINARY_DIR})
FILE(TO_NATIVE_PATH ${LIBRARY_PATH} LIBRARY_PATH)

#-----------------------------------------------
# Defines files lists
#-----------------------------------------------

FILE(GLOB incList "${CMAKE_CURRENT_SOURCE_DIR}/Inc/*.h")
FILE(GLOB inlList "${CMAKE_CURRENT_SOURCE_DIR}/Inc/*.inl")
FILE(GLOB cppList "${CMAKE_CURRENT_SOURCE_DIR}/Src/*.cpp")
FILE(GLOB cList "${CMAKE_CURRENT_SOURCE_DIR}/Src/*.c")
FILE(GLOB cuList "${CMAKE_CURRENT_SOURCE_DIR}/Src/*.cu")
FILE(GLOB uiList  "${CMAKE_CURRENT_SOURCE_DIR}/Ui/*.ui")
FILE(GLOB rcList  "${CMAKE_CURRENT_SOURCE_DIR}/Ui/*.qrc")

SET(srcList ${cppList} ${cList} ${cuList})

#-----------------------------------------------
# Manage Visual studio group files 
#-----------------------------------------------

source_group (Inc FILES ${incList})
source_group (Inl FILES ${inlList})
source_group (Src FILES ${srcList})
source_group (Ui  REGULAR_EXPRESSION .*//.ui$|.*//.qrc$  FILES ${uiList})
source_group (Rc  REGULAR_EXPRESSION .*//.ui$|.*//.qrc$  FILES ${rcList})

#-----------------------------------------------
# Manage generated files 
#-----------------------------------------------

# Add the header files to the project (only for Win32???)
INCLUDE_DIRECTORIES ( BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/Inc)
SET( resList ${resList} ${incList} )
SET( resList ${resList} ${inlList} )

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
STRING (REGEX MATCH "GV_CUDA_EXE" _matchCudaExe "${GV_TARGET_TYPE}")
STRING (REGEX MATCH "GV_SHARED_LIB" _matchSharedLib "${GV_TARGET_TYPE}")
STRING (REGEX MATCH "GV_CUDA_SHARED_LIB" _matchCudaSharedLib "${GV_TARGET_TYPE}")

if (_matchExe)
	ADD_EXECUTABLE (${PROJECT_NAME} ${srcList} ${genList} ${resList})
endif (_matchExe)

if (_matchCudaExe)
	CUDA_ADD_EXECUTABLE (${PROJECT_NAME} ${srcList} ${genList} ${resList})
endif (_matchCudaExe)

if (_matchSharedLib)
	add_definitions (-D${PROJECTDLL}_MAKEDLL)
	ADD_LIBRARY (${PROJECT_NAME} SHARED ${srcList} ${genList} ${resList})
endif (_matchSharedLib)

if (_matchCudaSharedLib)
	add_definitions (-D${PROJECTDLL}_MAKEDLL)
	CUDA_ADD_LIBRARY (${PROJECT_NAME} SHARED ${srcList} ${genList} ${resList})
endif (_matchCudaSharedLib)

SET_TARGET_PROPERTIES (${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX ".d")

foreach (it ${projectLibList})
	ADD_DEPENDENCIES (${PROJECT_NAME} ${it})
	TARGET_LINK_LIBRARIES( ${PROJECT_NAME} debug ${it}.d optimized ${it})
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














# Workaround to prevent cmake to create additional ${PROJECT_NAME}_UTILITY target per project
# see http://www.cmake.org/pipermail/cmake-commits/2007-April/001153.html
SET_TARGET_PROPERTIES (${PROJECT_NAME} PROPERTIES ALTERNATIVE_DEPENDENCY_NAME ${PROJECT_NAME}) 
	
# Copy binary files
#-----------------------------------------------
if ( RELEASE_BIN_DIR )

file( MAKE_DIRECTORY(${RELEASE_BIN_DIR}) )

# WINDOWS Operating System

IF (WIN32)

# There's a bug with VC (at least the 7.1 version): the generated pdb in debug has the name ${PROJECT_NAME}.pdb instead of ${PROJECT_NAME}.d.pdb
# So we copy both files
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.pdb\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.pdb\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.pdb\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.pdb\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.pdb\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.pdb\")

# Executable files
IF(_matchExe)
#Copy files to Release directory (i.e. directory for distribution)
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.exe\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.exe\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.exe\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.exe\" copy \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.exe\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.exe\")
#Copy files to main GigaSpace directory (i.e. Release)
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.exe\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.exe\" \"${GIGASPACE_RELEASE_BIN_DIR}/${PROJECT_NAME}.d.exe\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.exe\" copy \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.exe\" \"${GIGASPACE_RELEASE_BIN_DIR}/${PROJECT_NAME}.exe\")
ENDIF(_matchExe)

# Executable CUDA files
IF(_matchCudaExe)
#Copy files to Release directory (i.e. directory for distribution)
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.exe\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.exe\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.exe\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.exe\" copy \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.exe\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.exe\")
#Copy files to main GigaSpace directory (i.e. Release)
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.exe\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.exe\" \"${GIGASPACE_RELEASE_BIN_DIR}/${PROJECT_NAME}.d.exe\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.exe\" copy \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.exe\" \"${GIGASPACE_RELEASE_BIN_DIR}/${PROJECT_NAME}.exe\")
ENDIF(_matchCudaExe)

# Shared library files
IF(_matchSharedLib)
#Copy files to Release directory (i.e. directory for distribution)
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.dll\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.dll\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.dll\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.pdb\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.pdb\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.pdb\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.pdb\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.pdb\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.pdb\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.dll\" copy \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.dll\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.dll\")
#Copy files to main GigaSpace directory (i.e. Release)
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.dll\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.dll\" \"${GIGASPACE_RELEASE_BIN_DIR}/${PROJECT_NAME}.d.dll\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.pdb\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.pdb\" \"${GIGASPACE_RELEASE_BIN_DIR}/${PROJECT_NAME}.d.pdb\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.pdb\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.pdb\" \"${GIGASPACE_RELEASE_BIN_DIR}/${PROJECT_NAME}.d.pdb\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.dll\" copy \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.dll\" \"${GIGASPACE_RELEASE_BIN_DIR}/${PROJECT_NAME}.dll\")
ENDIF(_matchSharedLib)

# Shared library CUDA files
IF(_matchCudaSharedLib)
#Copy files to Release directory (i.e. directory for distribution)
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.dll\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.dll\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.dll\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.pdb\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.pdb\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.pdb\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.pdb\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.pdb\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.d.pdb\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.dll\" copy \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.dll\" \"${RELEASE_BIN_DIR}/${PROJECT_NAME}.dll\")
#Copy files to main GigaSpace directory (i.e. Release)
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.dll\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.dll\" \"${GIGASPACE_RELEASE_BIN_DIR}/${PROJECT_NAME}.d.dll\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.pdb\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.pdb\" \"${GIGASPACE_RELEASE_BIN_DIR}/${PROJECT_NAME}.d.pdb\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.pdb\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.pdb\" \"${GIGASPACE_RELEASE_BIN_DIR}/${PROJECT_NAME}.d.pdb\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.dll\" copy \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.dll\" \"${GIGASPACE_RELEASE_BIN_DIR}/${PROJECT_NAME}.dll\")
ENDIF(_matchCudaSharedLib)

ELSE (WIN32)

# LINUX Operating System

IF(_matchExe )# AND RELEASE_DIR_BIN)
    #FILE( MAKE_DIRECTORY ${RELEASE_DIR_BIN})
	
	#Copy files to Release directory (i.e. directory for distribution)
	POST_BUILD_COPY( ${LIBRARY_PATH}/${PROJECT_NAME}* ${RELEASE_BIN_DIR})
	
	#Copy files to main GigaSpace directory (i.e. Release)
	POST_BUILD_COPY( ${LIBRARY_PATH}/${PROJECT_NAME}* ${GIGASPACE_RELEASE_BIN_DIR} )
	
ENDIF (_matchExe )# AND RELEASE_DIR_BIN)

#IF(NOT _matchExe AND RELEASE_DIR_LIB)

IF(_matchSharedLib)
    #FILE( MAKE_DIRECTORY ${RELEASE_DIR_LIB})
    
	#Copy files to Release directory (i.e. directory for distribution)
	POST_BUILD_COPY( ${LIBRARY_PATH}/lib${PROJECT_NAME}.* ${RELEASE_BIN_DIR} )
	
	if ( RELEASE_LIB_DIR )
		file( MAKE_DIRECTORY (${RELEASE_LIB_DIR}) )
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
IF (WIN32)
file( MAKE_DIRECTORY(${RELEASE_LIB_DIR}) )

ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.lib\" copy \"${LIBRARY_PATH}\\DEBUG\\${PROJECT_NAME}.d.lib\" \"${RELEASE_LIB_DIR}/${PROJECT_NAME}.d.lib\")
ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	IF EXIST \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.lib\" copy \"${LIBRARY_PATH}\\RELEASE\\${PROJECT_NAME}.lib\" \"${RELEASE_LIB_DIR}/${PROJECT_NAME}.lib\")
ENDIF (WIN32)
endif ( RELEASE_LIB_DIR )

# Copy header files
#-----------------------------------------------

# Macro used to copy header files
MACRO(GV_COPY_HEADER)

MESSAGE(STATUS "CMAKE_SOURCE_DIR : ${CMAKE_SOURCE_DIR}")
MESSAGE(STATUS "CMAKE_CURRENT_SOURCE_DIR : ${CMAKE_CURRENT_SOURCE_DIR}")

file( MAKE_DIRECTORY(${RELEASE_INC_DIR}) )
#ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
#	copy \"${CMAKE_CURRENT_SOURCE_DIR}\\Inc\\*.h*\" \"${RELEASE_INC_DIR}\")
POST_BUILD_COPY(${CMAKE_CURRENT_SOURCE_DIR}/Inc/*.h* ${RELEASE_INC_DIR})
	
IF (inlList)
#ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
#	copy \"${CMAKE_CURRENT_SOURCE_DIR}\\Inc\\*.inl\" \"${RELEASE_INC_DIR}\")
POST_BUILD_COPY(${CMAKE_CURRENT_SOURCE_DIR}/Inc/*.inl ${RELEASE_INC_DIR})
ENDIF (inlList)

IF (uiList)
#ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
#	copy \"${CMAKE_CURRENT_BINARY_DIR}\\Inc\\UI_*.h\" \"${RELEASE_INC_DIR}\")
POST_BUILD_COPY(${CMAKE_CURRENT_BINARY_DIR}/Inc/UI_*.h ${RELEASE_INC_DIR})
ENDIF (uiList)

ENDMACRO(GV_COPY_HEADER)


if ( RELEASE_INC_DIR )

file( MAKE_DIRECTORY(${RELEASE_INC_DIR}) )

IF(MODULE)
file( MAKE_DIRECTORY(${RELEASE_INC_DIR}/${MODULE}) )
#ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
#	copy \"${LIBRARY_PATH}\\Inc\\${MODULE}\\*.h\" \"${RELEASE_INC_DIR}/${MODULE}\")
POST_BUILD_COPY(${LIBRARY_PATH}/Inc/${MODULE}/*.h ${RELEASE_INC_DIR}/${MODULE})
IF (inlList)
#ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
#	copy \"${LIBRARY_PATH}\\Inc\\${MODULE}\\*.inl\" \"${RELEASE_INC_DIR}/${MODULE}\")
POST_BUILD_COPY(${LIBRARY_PATH}/Inc/${MODULE}/*.inl ${RELEASE_INC_DIR}/${MODULE})
ENDIF (inlList)
ELSE(MODULE)
	GV_COPY_HEADER()
ENDIF(MODULE)
	
endif( RELEASE_INC_DIR )

ADD_CUSTOM_COMMAND(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
	echo Finish copying files...)

set (CMAKE_SUPPRESS_REGENERATION true)
