#----------------------------------------------------------------
# DEMO CMake file
# Main user file
#----------------------------------------------------------------

#----------------------------------------------------------------
# Project name
#----------------------------------------------------------------

project (GvVoxelizer)

MESSAGE (STATUS "")
MESSAGE (STATUS "PROJECT : ${PROJECT_NAME}")

#----------------------------------------------------------------
# Target yype
#----------------------------------------------------------------

# Can be GV_EXE or GV_SHARED_LIB
SET (GV_TARGET_TYPE "GV_EXE")

SET(RELEASE_BIN_DIR ${GV_RELEASE}/Tools/GigaVoxelsVoxelizer/Bin)
SET(RELEASE_LIB_DIR ${GV_RELEASE}/Tools/GigaVoxelsVoxelizer/Lib)
SET(RELEASE_INC_DIR ${GV_RELEASE}/Tools/GigaVoxelsVoxelizer/Inc)

SET(GIGASPACE_RELEASE_BIN_DIR ${GV_RELEASE}/Bin)

#----------------------------------------------------------------
# Add library dependencies
#----------------------------------------------------------------

# Set third party dependencies LINK library name if needed
#     . Example :
#     . When using the Qt_CMakeImport.cmake file to add Qt dependency with the command
#     .     INCLUDE (Qt_CMakeImport)
#     . the file use the ${qtLib} value to test if the user has set name different names to default ones.
#     . The DEFAULT library name is
#     .     set (qtLib "QtCore" "QtGui" "QtOpenGL" "QtXml")
#     . but user can only use
#     .     set (qtLib "QtCore" "QtGui")
#     . to reduce dependencies if needed
#     . or add dependencies
#     .     set (qtLib "QtCore" "QtGui" "QtOpenGL" "QtXml" "QtSvg")

# Add 3D format load library
INCLUDE (assimp_CMakeImport)

# Add image loading library
INCLUDE (CImg_CMakeImport)
# CImg is extended with ImageMagick support
INCLUDE (ImageMagick_CMakeImport)

# Add XML parsing library
INCLUDE (TinyXML_CMakeImport)

# Add IHM third party dependencies
INCLUDE (Qt_CMakeImport)

# Linux special features
if (WIN32)
else ()
	INCLUDE (X11_CMakeImport)
	INCLUDE (pthread_CMakeImport)
endif()

#----------------------------------------------------------------
# Main CMake file used for project generation
#----------------------------------------------------------------

IF (WIN32)
	# Add Windows dependencies
	LINK_LIBRARIES (user32)
	LINK_LIBRARIES (gdi32)
	LINK_LIBRARIES (shell32)
ENDIF ()

# Add the common CMAKE seetings to generate a GigaVoxels tool
INCLUDE (GV_CMakeCommonTools)
