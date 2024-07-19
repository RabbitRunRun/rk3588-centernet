cmake_minimum_required(VERSION 3.4.1)

project(test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

# skip 3rd-party lib dependencies
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/bin)

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

# set to release 
# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/3rdparty/librknn_api)
set(RKNN_RT_LIB ${RKNN_API_PATH}/lib/librknnrt.so)

include_directories(${RKNN_API_PATH}/include)

# opencv
# set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/share/OpenCV)
# find_package(OpenCV REQUIRED)

set(OpenCV_ROOT /workingspace/vizvision_rk3588_linux_sdk_v0.1.19/vizvision_rk3588_linux_sdk_v0.1.19)
include_directories(${OpenCV_ROOT}/include/opencv4)
find_library(CORE NAMES libopencv_core.so PATHS ${OpenCV_ROOT}/lib)
find_library(HIGHGUI NAMES libopencv_highgui.so PATHS ${OpenCV_ROOT}/lib)
find_library(IMGPROC NAMES libopencv_imgproc.so PATHS ${OpenCV_ROOT}/lib)
find_library(VIDEOIO NAMES libopencv_videoio.so PATHS ${OpenCV_ROOT}/lib)
find_library(CODEC NAMES libopencv_imgcodecs.so PATHS ${OpenCV_ROOT}/lib)
find_library(JPEG NAMES libjpeg.so PATHS ${OpenCV_ROOT}/lib)

set(OPENCVLIBS ${CORE} ${HIGHGUI} ${IMGPROC} ${VIDEOIO} ${CODEC} ${JPEG})


set(CMAKE_INSTALL_RPATH "lib")

# MODetector so file
include_directories( ${CMAKE_SOURCE_DIR}/include)
file(GLOB LIB_SOURCES src/*.cpp)
add_library(MODetector SHARED ${LIB_SOURCES})
target_link_libraries(MODetector ${RKNN_RT_LIB} ${OPENCVLIBS})
# target_link_libraries(MODetector ${RKNN_RT_LIB} ${OpenCV_LIBS})

# EXE
set(MAIN test/main.cpp)
add_executable(test_3_serial ${MAIN})
# target_link_libraries(test_3_serial PRIVATE MODetector ${OpenCV_LIBS})
target_link_libraries(test_3_serial PRIVATE MODetector ${OPENCVLIBS})
