This software can be built using CMake with the following options: <br/>
-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="conan_provider.cmake" -DCMAKE_CXX_FLAGS="/EHsc"

If any dependencies are missing, building will not work properly. To install missing dependencies, install [Conan](https://conan.io/downloads) and run ```conan install```.
