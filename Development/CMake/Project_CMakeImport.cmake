#----------------------------------------------------------------
# Import library
#----------------------------------------------------------------
	 
MESSAGE (STATUS "IMPORT : Project library")

# Add Definitions
	
#----------------------------------------------------------------
# Add INCLUDE library directories
#----------------------------------------------------------------

INCLUDE_DIRECTORIES (${RELEASE_INC_DIR})

#----------------------------------------------------------------
# Set LINK libraries if not defined by user
#----------------------------------------------------------------

#----------------------------------------------------------------
# Add LINK libraries
#----------------------------------------------------------------

LINK_DIRECTORIES (${RELEASE_LIB_DIR})
	
FOREACH (it ${projectLibList})
	IF (WIN32)
		LINK_LIBRARIES (optimized ${it} debug ${it}.d)
	ELSE ()
		LINK_LIBRARIES (optimized ${it} debug ${it}.d)
	ENDIF ()
ENDFOREACH (it)
