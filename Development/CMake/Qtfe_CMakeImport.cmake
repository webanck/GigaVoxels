#----------------------------------------------------------------
# Import library
#----------------------------------------------------------------
	
MESSAGE (STATUS "IMPORT : Qtfe library")

#----------------------------------------------------------------
# SET library PATH
#----------------------------------------------------------------

INCLUDE (GvSettings_CMakeImport)
	
#----------------------------------------------------------------
# Add INCLUDE library directories
#----------------------------------------------------------------

INCLUDE_DIRECTORIES (${GV_QTFE_INC})

#----------------------------------------------------------------
# Add LINK library directories
#----------------------------------------------------------------

LINK_DIRECTORIES (${GV_QTFE_LIB})
	
#----------------------------------------------------------------
# Set LINK libraries if not defined by user
#----------------------------------------------------------------
	
IF ( "${qtfeLib}" STREQUAL "" )
	IF (WIN32)
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (qtfeLib "Qtfe")
		ELSE ()
			SET (qtfeLib "Qtfe")
		ENDIF ()
	ELSE ()
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (qtfeLib "Qtfe")
		ELSE ()
			SET (qtfeLib "Qtfe")
		ENDIF ()
	ENDIF ()
ENDIF ()

#----------------------------------------------------------------
# Add LINK libraries
#----------------------------------------------------------------

FOREACH (it ${qtfeLib})
	IF (WIN32)
		LINK_LIBRARIES (optimized ${it} debug ${it}.d)
	ELSE ()
		LINK_LIBRARIES (optimized ${it} debug ${it}d)
	ENDIF ()
ENDFOREACH (it)
