#----------------------------------------------------------------
# Import library
#----------------------------------------------------------------

MESSAGE (STATUS "IMPORT : Qwt library")

#----------------------------------------------------------------
# SET library PATH
#----------------------------------------------------------------

INCLUDE (GvSettings_CMakeImport)
	
#----------------------------------------------------------------
# Add INCLUDE library directories
#----------------------------------------------------------------

INCLUDE_DIRECTORIES (${GV_QWT_INC})

#----------------------------------------------------------------
# Add LINK library directories
#----------------------------------------------------------------

LINK_DIRECTORIES (${GV_QWT_LIB})
	
#----------------------------------------------------------------
# Set LINK libraries if not defined by user
#----------------------------------------------------------------

IF ( "${qwtLib}" STREQUAL "" )
	IF (WIN32)
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (qwtLib "qwt")
		ELSE ()
			SET (qwtLib "qwt")
		ENDIF ()
	ELSE ()
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (qwtLib "qwt")
		ELSE ()
			SET (qwtLib "qwt")
		ENDIF ()
	ENDIF ()
ENDIF ()
	
#----------------------------------------------------------------
# Add LINK libraries
#----------------------------------------------------------------

FOREACH (it ${qwtLib})
	IF (WIN32)
		LINK_LIBRARIES (optimized ${it} debug ${it}d)
	ELSE ()
		LINK_LIBRARIES (optimized ${it} debug ${it})
	ENDIF ()
ENDFOREACH (it)
