#----------------------------------------------------------------
# Global GigaSpace Library Settings
#----------------------------------------------------------------

# Include guard
if (GvSettings_Included)
	return()
endif ()
set (GvSettings_Included true)

message (STATUS "IMPORT : GigaSpace Library Settings")

#----------------------------------------------------------------
# ASSIMP library settings
#----------------------------------------------------------------

if (WIN32)
	set (GV_ASSIMP_RELEASE "${GV_EXTERNAL}/assimp")
	set (GV_ASSIMP_INC "${GV_ASSIMP_RELEASE}/include")
	set (GV_ASSIMP_LIB "${GV_ASSIMP_RELEASE}/lib")
#	set (GV_ASSIMP_BIN "${GV_ASSIMP_RELEASE}/bin")
else ()
	set (GV_ASSIMP_RELEASE "/usr")
	set (GV_ASSIMP_INC "${GV_ASSIMP_RELEASE}/include/assimp")
	set (GV_ASSIMP_LIB "${GV_ASSIMP_RELEASE}/lib")
#	set (GV_ASSIMP_BIN "${GV_ASSIMP_RELEASE}/bin")
endif ()

#----------------------------------------------------------------
# CIMG library settings
#----------------------------------------------------------------

if (WIN32)
	set (GV_CIMG_RELEASE "${GV_EXTERNAL}/CImg")
	set (GV_CIMG_INC "${GV_CIMG_RELEASE}")
#	set (GV_CIMG_LIB "")
#	set (GV_CIMG_BIN "")
else ()
	set (GV_CIMG_RELEASE "/usr")
	set (GV_CIMG_INC "${GV_CIMG_RELEASE}/include")
#	set (GV_CIMG_LIB "")
#	set (GV_CIMG_BIN "")
endif ()

#----------------------------------------------------------------
# CUDPP library settings
#----------------------------------------------------------------

if (WIN32)
	set (GV_CUDPP_RELEASE "${GV_EXTERNAL}/cudpp")
	set (GV_CUDPP_INC "${GV_CUDPP_RELEASE}/include")
	set (GV_CUDPP_LIB "${GV_CUDPP_RELEASE}/lib")
#	set (GV_CUDPP_BIN "${GV_CUDPP_RELEASE}/bin")
else ()
	set (GV_CUDPP_RELEASE "/usr/local")
	set (GV_CUDPP_INC "${GV_CUDPP_RELEASE}/include")
	set (GV_CUDPP_LIB "${GV_CUDPP_RELEASE}/lib")
#	set (GV_CUDPP_BIN "${GV_CUDPP_RELEASE}/bin")
endif ()

#----------------------------------------------------------------
# FREEGLUT library settings
#----------------------------------------------------------------

if (WIN32)
	set (GV_FREEGLUT_RELEASE "${GV_EXTERNAL}/freeglut")
	set (GV_FREEGLUT_INC "${GV_FREEGLUT_RELEASE}/include")
	set (GV_FREEGLUT_LIB "${GV_FREEGLUT_RELEASE}/lib")
#	set (GV_FREEGLUT_BIN "${GV_FREEGLUT_RELEASE}/bin")
else ()
	set (GV_FREEGLUT_RELEASE "/usr")
	set (GV_FREEGLUT_INC "${GV_FREEGLUT_RELEASE}/include")
	set (GV_FREEGLUT_LIB "${GV_FREEGLUT_RELEASE}/lib/x86_64-linux-gnu")
#	set (GV_FREEGLUT_BIN "${GV_FREEGLUT_RELEASE}/bin")
endif ()

#----------------------------------------------------------------
# GLEW library settings
#----------------------------------------------------------------

if (WIN32)
	set (GV_GLEW_RELEASE "${GV_EXTERNAL}/glew")
	set (GV_GLEW_INC "${GV_GLEW_RELEASE}/include")
	set (GV_GLEW_LIB "${GV_GLEW_RELEASE}/lib")
#	set (GV_GLEW_BIN "${GV_GLEW_RELEASE}/bin")
else ()
# TO DO : modify the followinf paths...
	set (GV_GLEW_RELEASE "/usr")
#	set (GV_GLEW_RELEASE "/home/artis/guehl/Projects/glew-1.10.0/Generated")
	set (GV_GLEW_INC "${GV_GLEW_RELEASE}/include")
#	set (GV_GLEW_LIB "${GV_GLEW_RELEASE}/lib/x86_64-linux-gnu")
	set (GV_GLEW_LIB "${GV_GLEW_RELEASE}/lib64")
	set (GV_GLEW_BIN "${GV_GLEW_RELEASE}/bin")
endif ()

#----------------------------------------------------------------
# CUDA SDK library settings
#----------------------------------------------------------------

if (WIN32)
	set (GV_NVIDIAGPUCOMPUTINGSDK_RELEASE "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v6.0")
	set (GV_NVIDIAGPUCOMPUTINGSDK_INC "${GV_NVIDIAGPUCOMPUTINGSDK_RELEASE}/common/inc")
#	set (GV_NVIDIAGPUCOMPUTINGSDK_LIB "")
#	set (GV_NVIDIAGPUCOMPUTINGSDK_BIN "")
else ()
	set (GV_NVIDIAGPUCOMPUTINGSDK_RELEASE "/usr/local/cuda-6.0/samples")
	set (GV_NVIDIAGPUCOMPUTINGSDK_INC "${GV_NVIDIAGPUCOMPUTINGSDK_RELEASE}/common/inc")
#	set (GV_NVIDIAGPUCOMPUTINGSDK_LIB "")
#	set (GV_NVIDIAGPUCOMPUTINGSDK_BIN "")
endif ()

#----------------------------------------------------------------
# NV TOOLS library settings
#----------------------------------------------------------------

if (WIN32)
	set (GV_NVTOOLS_RELEASE "${CUDA_TOOLKIT_ROOT_DIR}/../../../NVIDIA Corporation/NvToolsExt")
	set (GV_NVTOOLS_INC "${GV_NVTOOLS_RELEASE}/include")
	set (GV_NVTOOLS_LIB "${GV_NVTOOLS_RELEASE}/lib")
	set (GV_NVTOOLS_BIN "${GV_NVTOOLS_RELEASE}/bin")
else ()
#	set (GV_NVTOOLS_RELEASE "${CUDA_TOOLKIT_ROOT_DIR}/../../../NVIDIA Corporation/NvToolsExt")
#	set (GV_NVTOOLS_INC "${GV_NVTOOLS_RELEASE}/include")
#	set (GV_NVTOOLS_LIB "${GV_NVTOOLS_RELEASE}/lib")
#	set (GV_NVTOOLS_BIN "${GV_NVTOOLS_RELEASE}/bin")
endif ()

#----------------------------------------------------------------
# IMAGEMAGICK library settings
#----------------------------------------------------------------

# The CMake FindImageMagick module define the following INCLUDE path
# ImageMagick_Magick++_INCLUDE_DIR

# The CMake FindImageMagick module define the ImageMagick_Magick++_LIBRARY filepath
# to the library, but not the path to the directory, that's why we use
# the IMAGEMAGICK_BINARY_PATH path on Windows.

if (WIN32)
#	set (GV_IMAGEMAGICK_RELEASE "")
	set (GV_IMAGEMAGICK_INC "${ImageMagick_Magick++_INCLUDE_DIR}")
	#set (GV_IMAGEMAGICK_LIB "${IMAGEMAGICK_BINARY_PATH}/lib")
	set (GV_IMAGEMAGICK_LIB "${ImageMagick_EXECUTABLE_DIR}/lib")
#	set (GV_IMAGEMAGICK_BIN "")
else ()
#	set (GV_IMAGEMAGICK_RELEASE "")
	set (GV_IMAGEMAGICK_INC "${ImageMagick_Magick++_INCLUDE_DIR}")
	set (GV_IMAGEMAGICK_LIB "/usr/lib")
#	set (GV_IMAGEMAGICK_BIN "")
endif ()

# Check directory
#message (STATUS "DEBUG : Image Magick : GV_IMAGEMAGICK_INC : ${GV_IMAGEMAGICK_INC}")
#message (STATUS "DEBUG : Image Magick : GV_IMAGEMAGICK_LIB : ${GV_IMAGEMAGICK_LIB}")

#----------------------------------------------------------------
# LOKI library settings
#
# NOTE
# GigaVoxels uses a modified version of the library in order to
# be able to use it in device code (i.e. on GPU).
#----------------------------------------------------------------

set (GV_LOKI_RELEASE "${GV_EXTERNAL}/Loki")
set (GV_LOKI_INC "${GV_LOKI_RELEASE}/include")
#set (GV_LOKI_LIB "${GV_LOKI_RELEASE}/lib")
#set (GV_LOKI_BIN "${GV_LOKI_RELEASE}/bin")

#----------------------------------------------------------------
# NEMOGRAPHICS library settings
#----------------------------------------------------------------

set (GV_NEMOGRAPHICS_RELEASE "${GV_EXTERNAL}/NemoGraphics")
set (GV_NEMOGRAPHICS_INC "${GV_NEMOGRAPHICS_RELEASE}/include")
#set (GV_NEMOGRAPHICS_LIB "")
#set (GV_NEMOGRAPHICS_BIN "")

#----------------------------------------------------------------
# QGLVIEWER library settings
#----------------------------------------------------------------

if (WIN32)
	set (GV_QGLVIEWER_RELEASE "${GV_EXTERNAL}/QGLViewer")
	set (GV_QGLVIEWER_INC "${GV_QGLVIEWER_RELEASE}/include")
	set (GV_QGLVIEWER_LIB "${GV_QGLVIEWER_RELEASE}/lib")
#	set (GV_QGLVIEWER_BIN "${GV_QGLVIEWER_RELEASE}/bin")
else ()
	set (GV_QGLVIEWER_RELEASE "/usr")
	set (GV_QGLVIEWER_INC "${GV_QGLVIEWER_RELEASE}/include")
	set (GV_QGLVIEWER_LIB "${GV_QGLVIEWER_RELEASE}/lib/x86_64-linux-gnu")
#	set (GV_QGLVIEWER_BIN "${GV_QGLVIEWER_RELEASE}/bin")
endif ()

#----------------------------------------------------------------
# QT library settings
#----------------------------------------------------------------

if (WIN32)
	set (GV_QT_RELEASE "${GV_EXTERNAL}/Qt")
	#set (GV_QT_RELEASE "C:/Qt/Qt5.2.1/5.2.1/msvc2012_64_opengl")
	set (GV_QT_INC "${GV_QT_RELEASE}/include")
	set (GV_QT_LIB "${GV_QT_RELEASE}/lib")
	set (GV_QT_BIN "${GV_QT_RELEASE}/bin")
else ()
	set (GV_QT_RELEASE "/usr")
	set (GV_QT_INC "${GV_QT_RELEASE}/include/qt4")
	set (GV_QT_LIB "${GV_QT_RELEASE}/lib/x86_64-linux-gnu")
	set (GV_QT_BIN "${GV_QT_RELEASE}/bin")
endif ()

# Where to find the uic, moc and rcc tools

set (GV_QT_UIC_EXECUTABLE ${GV_QT_BIN}/uic)
set (GV_QT_MOC_EXECUTABLE ${GV_QT_BIN}/moc)
set (GV_QT_RCC_EXECUTABLE ${GV_QT_BIN}/rcc)

#message (STATUS "Qt UIC : ${GV_QT_UIC_EXECUTABLE}")
#message (STATUS "Qt MOC : ${GV_QT_MOC_EXECUTABLE}")
#message (STATUS "Qt RCC : ${GV_QT_RCC_EXECUTABLE}")

#----------------------------------------------------------------
# QTFE library settings
#----------------------------------------------------------------

set (GV_QTFE_RELEASE "${GV_EXTERNAL}/Qtfe")
set (GV_QTFE_INC "${GV_QTFE_RELEASE}/include")
set (GV_QTFE_LIB "${GV_QTFE_RELEASE}/lib")
#set (GV_QTFE_BIN "${GV_QTFE_RELEASE}/bin")

#----------------------------------------------------------------
# QWT library settings
#----------------------------------------------------------------

if (WIN32)
	set (GV_QWT_RELEASE "${GV_EXTERNAL}/Qwt")
	set (GV_QWT_INC "${GV_QWT_RELEASE}/include")
	set (GV_QWT_LIB "${GV_QWT_RELEASE}/lib")
#	set (GV_QWT_BIN "${GV_QWT_RELEASE}/bin")
else ()
	set (GV_QWT_RELEASE "/usr")
	set (GV_QWT_INC "${GV_QWT_RELEASE}/include/qwt")
	set (GV_QWT_LIB "${GV_QWT_RELEASE}/lib")
#	set (GV_QWT_BIN "${GV_QWT_RELEASE}/bin")
endif ()

#----------------------------------------------------------------
# GLM( OpenGL Mathematics) library settings
#----------------------------------------------------------------

if (WIN32)
	set (GV_GLM_RELEASE "${GV_EXTERNAL}/glm")
	set (GV_GLM_INC "${GV_GLM_RELEASE}")
#	set (GV_GLM_LIB "")
#	set (GV_GLM_BIN "")
else ()
	set (GV_GLM_RELEASE "${GV_EXTERNAL}/glm")
	set (GV_GLM_INC "${GV_GLM_RELEASE}")
#	set (GV_GLM_LIB "")
#	set (GV_GLM_BIN "")
endif ()

#----------------------------------------------------------------
# OGRE3D library settings
#----------------------------------------------------------------

if (WIN32)
	set (GV_OGRE3D_RELEASE "${GV_EXTERNAL}/Ogre3D")
	set (GV_OGRE3D_INC "${GV_OGRE3D_RELEASE}/include")
	set (GV_OGRE3D_LIB "${GV_OGRE3D_RELEASE}/lib")
#	set (GV_OGRE3D_BIN "${GV_OGRE3D_RELEASE}/bin")
else ()
	set (GV_OGRE3D_RELEASE "/usr/local")
	set (GV_OGRE3D_INC "${GV_OGRE3D_RELEASE}/include")
	set (GV_OGRE3D_LIB "${GV_OGRE3D_RELEASE}/lib")
#	set (GV_OGRE3D_BIN "${GV_OGRE3D_RELEASE}/bin")
endif ()

#----------------------------------------------------------------
# TinyXML library settings
#----------------------------------------------------------------

if (WIN32)
	set (GV_TINYXML_RELEASE "${GV_EXTERNAL}/tinyXML")
	set (GV_TINYXML_INC "${GV_TINYXML_RELEASE}/include")
	set (GV_TINYXML_LIB "${GV_TINYXML_RELEASE}/lib")
#	set (GV_TINYXML_BIN "${GV_TINYXML_RELEASE}/bin")
else ()
	set (GV_TINYXML_RELEASE "/usr/local")
	set (GV_TINYXML_INC "${GV_TINYXML_RELEASE}/include")
	set (GV_TINYXML_LIB "${GV_TINYXML_RELEASE}/lib")
#	set (GV_TINYXML_BIN "${GV_TINYXML_RELEASE}/bin")
endif ()
