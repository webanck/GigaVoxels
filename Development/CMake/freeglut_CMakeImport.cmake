#----------------------------------------------------------------
# Import library
#----------------------------------------------------------------
	
MESSAGE (STATUS "IMPORT : freeglut library")

#----------------------------------------------------------------
# SET library PATH
#----------------------------------------------------------------

INCLUDE (GvSettings_CMakeImport)
	
#----------------------------------------------------------------
# Add INCLUDE library directories
#----------------------------------------------------------------

INCLUDE_DIRECTORIES (${GV_FREEGLUT_INC})

#----------------------------------------------------------------
# Add LINK library directories
#----------------------------------------------------------------

LINK_DIRECTORIES (${GV_FREEGLUT_LIB})
	
#----------------------------------------------------------------
# Set LINK libraries if not defined by user
#----------------------------------------------------------------

IF ( "${freeglutLib}" STREQUAL "" )
	IF (WIN32)
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (freeglutLib "freeglut")
		ELSE ()
			SET (freeglutLib "freeglut")
		ENDIF ()
	ELSE ()
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (freeglutLib "glut")
		ELSE ()
			SET (freeglutLib "glut")
		ENDIF ()
	ENDIF ()
ENDIF ()
	
#----------------------------------------------------------------
# Add LINK libraries
#----------------------------------------------------------------

FOREACH (it ${freeglutLib})
	IF (WIN32)
		LINK_LIBRARIES (optimized ${it} debug ${it})
	ELSE ()
		LINK_LIBRARIES (optimized ${it} debug ${it})
	ENDIF ()
ENDFOREACH (it)
