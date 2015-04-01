#----------------------------------------------------------------
# Import library
#----------------------------------------------------------------

MESSAGE (STATUS "IMPORT : ImageMagick library")

#----------------------------------------------------------------
# SET library PATH
#----------------------------------------------------------------

INCLUDE (GvSettings_CMakeImport)

#----------------------------------------------------------------
# Add INCLUDE library directories
#----------------------------------------------------------------

INCLUDE_DIRECTORIES (${GV_IMAGEMAGICK_INC})

#----------------------------------------------------------------
# Add LINK library directories
#----------------------------------------------------------------

LINK_DIRECTORIES (${GV_IMAGEMAGICK_LIB})
	
#----------------------------------------------------------------
# Set LINK libraries if not defined by user
#----------------------------------------------------------------

IF ( "${imagemagickLib}" STREQUAL "" )
	IF (WIN32)
		# Windows name of the library (CORE_RL_Magick++_.lib)
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (imagemagickLib "CORE_RL_Magick++_")
		ELSE ()
			SET (imagemagickLib "CORE_RL_Magick++_")
		ENDIF ()
	ELSE ()
		# Linux name of the library (libMagick++.so)
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (imagemagickLib "Magick++")
		ELSE ()
			SET (imagemagickLib "Magick++")
		ENDIF ()
	ENDIF ()
ENDIF ()

#----------------------------------------------------------------
# Add LINK libraries
#----------------------------------------------------------------

FOREACH (it ${imagemagickLib})
	IF (WIN32)
		LINK_LIBRARIES (optimized ${it} debug ${it})
	ELSE ()
		LINK_LIBRARIES (optimized ${it} debug ${it})
	ENDIF ()
ENDFOREACH (it)
