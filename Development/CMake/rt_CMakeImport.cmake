#----------------------------------------------------------------
# Import library
#----------------------------------------------------------------
	
MESSAGE (STATUS "IMPORT : rt library")

#----------------------------------------------------------------
# SET library PATH
#----------------------------------------------------------------

INCLUDE (GvSettings_CMakeImport)
	
#----------------------------------------------------------------
# Add INCLUDE library directories
#----------------------------------------------------------------

#INCLUDE_DIRECTORIES (${GV_GLU_INC})

#----------------------------------------------------------------
# Add LINK library directories
#----------------------------------------------------------------

#LINK_DIRECTORIES (${GV_GLU_LIB})
	
#----------------------------------------------------------------
# Set LINK libraries if not defined by user
#----------------------------------------------------------------

IF ( "${rtLib}" STREQUAL "" )
	IF (WIN32)
	ELSE ()
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (rtLib "rt")
		ELSE ()
			SET (rtLib "rt")
		ENDIF ()
	ENDIF ()
ENDIF ()

#----------------------------------------------------------------
# Add LINK libraries
#----------------------------------------------------------------

FOREACH (it ${rtLib})
	IF (WIN32)
		LINK_LIBRARIES (optimized ${it} debug ${it})
	ELSE ()
		LINK_LIBRARIES (optimized ${it} debug ${it})
	ENDIF ()
ENDFOREACH (it)
