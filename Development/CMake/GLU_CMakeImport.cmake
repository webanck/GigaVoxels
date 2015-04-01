#----------------------------------------------------------------
# Import library
#----------------------------------------------------------------
	
MESSAGE (STATUS "IMPORT : GLU library")

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

IF ( "${gluLib}" STREQUAL "" )
	IF (WIN32)
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (gluLib "glu32")
		ELSE ()
			SET (gluLib "glu32")
		ENDIF ()
	ELSE ()
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (gluLib "GLU")
		ELSE ()
			SET (gluLib "GLU")
		ENDIF ()
	ENDIF ()
ENDIF ()

#----------------------------------------------------------------
# Add LINK libraries
#----------------------------------------------------------------

FOREACH (it ${gluLib})
	IF (WIN32)
		LINK_LIBRARIES (optimized ${it} debug ${it})
	ELSE ()
		LINK_LIBRARIES (optimized ${it} debug ${it})
	ENDIF ()
ENDFOREACH (it)
