#----------------------------------------------------------------
# Import library
#----------------------------------------------------------------

MESSAGE (STATUS "IMPORT : Ogre3D library")

#----------------------------------------------------------------
# SET library PATH
#----------------------------------------------------------------

INCLUDE (GvSettings_CMakeImport)
	
#----------------------------------------------------------------
# Add INCLUDE library directories
#----------------------------------------------------------------

INCLUDE_DIRECTORIES (${GV_OGRE3D_INC})

#----------------------------------------------------------------
# Add LINK library directories
#----------------------------------------------------------------

LINK_DIRECTORIES (${GV_OGRE3D_LIB})

#----------------------------------------------------------------
# Set LINK libraries if not defined by user
#----------------------------------------------------------------

IF ( "${ogre3dLib}" STREQUAL "" )
	IF (WIN32)
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (ogre3dLib "OgreMain")
		ELSE ()
			SET (ogre3dLib "OgreMain")
		ENDIF ()
	ELSE ()
		IF ( ${GV_DESTINATION_ARCH} STREQUAL "x86" )
			SET (ogre3dLib "OgreMain")
		ELSE ()
			SET (ogre3dLib "OgreMain")
		ENDIF ()
	ENDIF ()
ENDIF ()

#----------------------------------------------------------------
# Add LINK libraries
#----------------------------------------------------------------

FOREACH (it ${ogre3dLib})
	IF (WIN32)
		LINK_LIBRARIES (optimized ${it} debug ${it}_d)
	ELSE ()
		LINK_LIBRARIES (optimized ${it} debug ${it}_d)
	ENDIF ()
ENDFOREACH (it)
