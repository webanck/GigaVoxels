#----------------------------------------------------------------
# Import library
#----------------------------------------------------------------
	 
MESSAGE (STATUS "IMPORT : GvViewer library")

#----------------------------------------------------------------
# SET library PATH
#----------------------------------------------------------------

#----------------------------------------------------------------
# Add INCLUDE library directories
#----------------------------------------------------------------

INCLUDE_DIRECTORIES (${GV_RELEASE}/Tools/GigaVoxelsViewer/Inc)

#----------------------------------------------------------------
# Add LINK library directories
#----------------------------------------------------------------

LINK_DIRECTORIES (${GV_RELEASE}/Tools/GigaVoxelsViewer/Lib)

#----------------------------------------------------------------
# Set LINK libraries if not defined by user
#----------------------------------------------------------------

IF ( "${gvviewerLib}" STREQUAL "" )
	IF (WIN32)
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (gvviewerLib "GvViewerCore" "GvViewerScene" "GvViewerGui")
		ELSE ()
			SET (gvviewerLib "GvViewerCore" "GvViewerScene" "GvViewerGui")
		ENDIF ()
	ELSE ()
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (gvviewerLib "GvViewerCore" "GvViewerScene" "GvViewerGui")
		ELSE ()
			SET (gvviewerLib "GvViewerCore" "GvViewerScene" "GvViewerGui")
		ENDIF ()
	ENDIF ()
ENDIF ()

#----------------------------------------------------------------
# Add LINK libraries
#----------------------------------------------------------------
		
FOREACH (it ${gvviewerLib})
	IF (WIN32)
		LINK_LIBRARIES (optimized ${it} debug ${it}.d)
	ELSE ()
		LINK_LIBRARIES (optimized ${it} debug ${it}.d)
	ENDIF ()
ENDFOREACH (it)
