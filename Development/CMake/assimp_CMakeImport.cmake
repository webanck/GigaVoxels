#----------------------------------------------------------------
# Import library
#----------------------------------------------------------------

MESSAGE (STATUS "IMPORT : Assimp library")

#----------------------------------------------------------------
# SET library PATH
#----------------------------------------------------------------

INCLUDE (GvSettings_CMakeImport)

#----------------------------------------------------------------
# Add INCLUDE library directories
#----------------------------------------------------------------

INCLUDE_DIRECTORIES (${GV_ASSIMP_INC})

#----------------------------------------------------------------
# Add LINK library directories
#----------------------------------------------------------------

LINK_DIRECTORIES (${GV_ASSIMP_LIB})
	
#----------------------------------------------------------------
# Set LINK libraries if not defined by user
#----------------------------------------------------------------

IF ( "${assimpLib}" STREQUAL "" )
	IF (WIN32)
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (assimpLib "assimp")
		ELSE ()
			SET (assimpLib "assimp")
		ENDIF ()
	ELSE ()
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (assimpLib "assimp")
		ELSE ()
			SET (assimpLib "assimp")
		ENDIF ()
	ENDIF ()
ENDIF ()

#----------------------------------------------------------------
# Add LINK libraries
#----------------------------------------------------------------

FOREACH (it ${assimpLib})
	IF (WIN32)
		LINK_LIBRARIES (optimized ${it} debug ${it}d)
	ELSE ()
		LINK_LIBRARIES (optimized ${it} debug ${it})
	ENDIF ()
ENDFOREACH (it)
