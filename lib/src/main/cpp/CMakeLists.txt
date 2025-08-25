# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("FaceRecognitionDlib")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(DLIB_NO_GUI_SUPPORT ON CACHE BOOL "Disable GUI" FORCE)
set(DLIB_USE_BLAS OFF CACHE BOOL "" FORCE)
set(DLIB_USE_LAPACK OFF CACHE BOOL "" FORCE)
set(DLIB_USE_FFMPEG OFF CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_SOURCE_DIR}/../../../../external/dlib dlib_build)


# Add the native source code
add_library(
        ${CMAKE_PROJECT_NAME}
        SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        DlibLandmarks.cpp
        FaceRecognition.cpp
)

include_directories(
        ${CMAKE_SOURCE_DIR}/onnxruntime/include
)

add_library(
        onnxruntime
        SHARED
        IMPORTED
)

set_target_properties(
        onnxruntime
        PROPERTIES
        IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libonnxruntime.so
)

find_library(log-lib log)
find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/../../../../external/dlib
)

#target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
#        DLIB_NO_GUI_SUPPORT
#        DLIB_USE_BLAS=0
#        DLIB_USE_LAPACK=0
#)

# Link libraries to your native code
target_link_libraries(
        ${CMAKE_PROJECT_NAME}
        dlib::dlib
        onnxruntime
        ${log-lib}
        ${android-lib}
        ${jnigraphics-lib}
)