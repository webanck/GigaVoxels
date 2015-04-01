SET(RELEASE_BIN_DIR ${GV_RELEASE}/Tools/GigaVoxelsViewer/Bin)
SET(RELEASE_LIB_DIR ${GV_RELEASE}/Tools/GigaVoxelsViewer/Lib)
SET(RELEASE_INC_DIR ${GV_RELEASE}/Tools/GigaVoxelsViewer/Inc)

SET(GIGASPACE_RELEASE_BIN_DIR ${GV_RELEASE}/Bin)

add_subdirectory ("${CMAKE_SOURCE_DIR}/GigaVoxelsViewer/GvViewer")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GigaVoxelsViewer/GvViewerCore")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GigaVoxelsViewer/GvViewerScene")
add_subdirectory ("${CMAKE_SOURCE_DIR}/GigaVoxelsViewer/GvViewerGui")
