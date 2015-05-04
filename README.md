#The GigaVoxels Library!

The Gigavoxels is a render pipeline that let you render complex volumetric scenes containing billions of voxels as a representation of volumetric. The library has been written by a [team of the INRIA](http://gigavoxels.inrialpes.fr/) after the work of [Cyrill Crassin](http://maverick.inria.fr/Membres/Cyril.Crassin/).

The main components are implemented in C++ using mainly CUDA and a bit of OpenGL and the interface uses the Qt API.

##Prerequisites

###A Nvidia GPU

A Nvidia GPU is required in order to run the CUDA executable on the GPU side.
Be sure to check the compute capability of your specific model because you will need this information later during the configuration of the library.

Warning: if your GPU is to old, it might not even be compatible with CUDA! So, again, check the compute capability before going further.

###CUDA

Be sure to have CUDA installed. Detailed explanations are given on the [Nvidia official site](http://docs.nvidia.com/cuda/index.html#getting-started-guides) for Windows, Linux and Mac.

###CMake

The library is cross platform thanks to [CMake](http://www.cmake.org/).
On Ubuntu 14.04 you can install the maintained package:
```
sudo apt-get install -y cmake cmake-gui
```

###CUDPP (CUDA Data Parallel Primitives Library)

The GigaVoxels library uses [CUDPP](http://cudpp.github.io/) currently in the 2.2 version.
Unfortunately you need to compile it from source but it is available on GitHub [here](https://github.com/cudpp/cudpp).

On Linux, you can use git to download the source and the project's submodules:
```
git clone https://github.com/cudpp/cudpp.git
cd cudpp
git submodule init
git submodule update
```

Once you have the source code you can run the CMake GUI:
```
cmake-gui
```
Firstly, set the paths for the source code directory and where you want the binaries to be built.
Click on `Configure` and  choose your type of code generator: `Unix Makefiles` for Linux.

Then, set the specific variables:
- add the `-fPIC` parameter to `CMAKE_CXX_FLAGS` and `CMAKE_C_FLAGS`
- verify the path of `CUDA_TOOLKIT_ROOT_DIR` (`/usr/local/cuda/include` for me)
- check `CUDPP_BUILD_SHARED_LIBS`
- check the compute capability corresponding to your GPU (`CUDPP_GENCODE_SM20` for me)

You can now click on `Generate` and go to the next step: actually compiling CUDPP.
Go to the path you set before (where to build the binaries) and compile.
On Linux with the `Unix Makefiles`:
```
make
make install
```

###Libraries

You also need additional libraries.

On Ubuntu 14.04 you can install the maintained packages:
```
sudo apt-get install -y \
  libqglviewer-dev \
  libmagick++-dev \
  freeglut3-dev \
  libqt4-dev \
  libtinyxml-dev \
  libqwt-dev \
  cimg-dev \
  libassimp-dev \
  libglm-dev
```

##Installation

###Download
Once you have all the prerequisites, download the files of the repository: the source code and some sample data.
```
git clone https://github.com/webanck/GigaVoxels.git
```
The source code and the config files for CMake are in the [Development](Development) directory .

###Configure
You need to adapt some settings to your environment.
Firstly, if any, give the path of the external required libraries in the file [GvSettings_CMakeImport.cmake](Development/CMake/GvSettings_CMakeImport.cmake).
Secondly, uncomment the compute capability corresponding to your GPU in the following files:
- [Development/Library/CMakeLists.txt](Development/Library/CMakeLists.txt)
- [Development/Tools/CMakeLists.txt](Development/Tools/CMakeLists.txt)
- [Development/Tutorials/](Development/Tutorials/)
- [Development/Tutorials/Demos/CMakeLists.txt](Development/Tutorials/Demos/CMakeLists.txt)

To help you, the lines of interest are looking like that:
```cmake
# Set your compute capability version (comment/uncomment with #)
#
# GiagVoxels requires 2.0 at least
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20")
#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_21")
#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30")
#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35")
```

###Compile
I added a [shell script](Install/Linux/makeInstall.sh) for the Linux users so they only have to launch it to clean and recompile all the library, the tools and the specific example I am working on: the [GvDynamicLoad](Development/Tutorials/ViewerPlugins/GvDynamicLoad).
