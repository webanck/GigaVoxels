#----------------------------------------------------------------
# Import library
#----------------------------------------------------------------

MESSAGE (STATUS "IMPORT : GigaSpace library")

#----------------------------------------------------------------
# SET library PATH
#----------------------------------------------------------------

#----------------------------------------------------------------
# Add INCLUDE library directories
#----------------------------------------------------------------

INCLUDE_DIRECTORIES (${GV_RELEASE}/Inc)

#----------------------------------------------------------------
# Add LINK library directories
#----------------------------------------------------------------

LINK_DIRECTORIES (${GV_RELEASE}/Lib)

#----------------------------------------------------------------
# Set LINK libraries if not defined by user
#----------------------------------------------------------------

IF ( "${gigaspaceLib}" STREQUAL "" )
	IF (WIN32)
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (gigaspaceLib "GigaSpace")
		ELSE ()
			SET (gigaspaceLib "GigaSpace")
		ENDIF ()
	ELSE ()
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (gigaspaceLib "GigaSpace")
		ELSE ()
			SET (gigaspaceLib "GigaSpace")
		ENDIF ()
	ENDIF ()
ENDIF ()
		
#----------------------------------------------------------------
# Add LINK libraries
#----------------------------------------------------------------

FOREACH (it ${gigaspaceLib})
	IF (WIN32)
		LINK_LIBRARIES (optimized ${it} debug ${it}.d)
	ELSE ()
		LINK_LIBRARIES (optimized ${it} debug ${it}.d)
	ENDIF ()
ENDFOREACH (it)
