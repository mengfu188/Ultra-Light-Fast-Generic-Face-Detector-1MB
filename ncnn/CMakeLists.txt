cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_STANDARD 11)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Debug)" FORCE)
endif()

project(Ultra-Light-Fast-Generic-Face-Detector-1MB)


find_package(OpenCV REQUIRED PATHS /home/cmf/dev/opencv-4.2.0/build_extra/install)

message(${OpenCV_INCLUDE_DIRS})

#add_subdirectory(3rdparty/ncnn ncnn_build)
include_directories(${OpenCV_INCLUDE_DIRS}
        /home/cmf/dev/ncnn-20200413/build_without_omp/install/include/ncnn)
link_directories(${OpenCV_LIBS}
        /home/cmf/dev/ncnn-20200413/build_without_omp/install/lib)


add_executable(main src/main.cpp src/UltraFace.cpp)
target_link_libraries(main ncnn ${OpenCV_LIBS})
