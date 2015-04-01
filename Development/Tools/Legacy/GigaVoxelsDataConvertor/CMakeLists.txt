#----------------------------------------------------------------
# DEMO CMake file
# Main user file
#----------------------------------------------------------------

#----------------------------------------------------------------
# Project name
#----------------------------------------------------------------

project (GvDataConverter)

MESSAGE (STATUS "")
MESSAGE (STATUS "PROJECT : ${PROJECT_NAME}")

#----------------------------------------------------------------
# Target yype
#----------------------------------------------------------------

# Can be GV_EXE or GV_SHARED_LIB
SET (GV_TARGET_TYPE "GV_EXE")

SET(RELEASE_BIN_DIR ${GV_RELEASE}/Tools/Legacy/GigaVoxelsDataConverter/Bin)
SET(RELEASE_LIB_DIR ${GV_RELEASE}/Tools/Legacy/GigaVoxelsDataConverter/Lib)
SET(RELEASE_INC_DIR ${GV_RELEASE}/Tools/Legacy/GigaVoxelsDataConverter/Inc)

SET(GIGASPACE_RELEASE_BIN_DIR ${GV_RELEASE}/Bin)

#----------------------------------------------------------------
# Add library dependencies
#----------------------------------------------------------------

#----------------------------------------------------------------
# Main CMake file used for project generation
#----------------------------------------------------------------

INCLUDE (GV_CMakeCommonTools)
