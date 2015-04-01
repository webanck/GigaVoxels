#----------------------------------------------------------------
# Import library
#----------------------------------------------------------------

MESSAGE (STATUS "IMPORT : nvTools library")

#----------------------------------------------------------------
# SET library PATH
#----------------------------------------------------------------

INCLUDE (GvSettings_CMakeImport)

#----------------------------------------------------------------
# Add INCLUDE library directories
#----------------------------------------------------------------

INCLUDE_DIRECTORIES (${GV_NVTOOLS_INC})

#----------------------------------------------------------------
# Add LINK library directories
#----------------------------------------------------------------

IF (WIN32)
	IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
		LINK_DIRECTORIES ("${GV_NVTOOLS_LIB}/Win32")
	ELSE ()
		LINK_DIRECTORIES ("${GV_NVTOOLS_LIB}/x64")
	ENDIF ()
ELSE ()
#	IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
#		LINK_DIRECTORIES ("${GV_NVTOOLS_LIB}/Win32")
#	ELSE ()
#		LINK_DIRECTORIES ("${GV_NVTOOLS_LIB}/x64")
#	ENDIF ()
ENDIF ()
	
#----------------------------------------------------------------
# Set LINK libraries if not defined by user
#----------------------------------------------------------------

IF ( "${nvtoolsLib}" STREQUAL "" )
	IF (WIN32)
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (nvtoolsLib "nvToolsExt32_1")
		ELSE ()
			SET (nvtoolsLib "nvToolsExt64_1")
		ENDIF ()
	ELSE ()
#		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
#			SET (nvtoolsLib "assimp")
#		ELSE ()
#			SET (nvtoolsLib "assimp")
#		ENDIF ()
	ENDIF ()
ENDIF ()

#----------------------------------------------------------------
# Add LINK libraries
#----------------------------------------------------------------

FOREACH (it ${nvtoolsLib})
	IF (WIN32)
		LINK_LIBRARIES (optimized ${it} debug ${it})
	ELSE ()
#		LINK_LIBRARIES (optimized ${it} debug ${it})
	ENDIF ()
ENDFOREACH (it)
