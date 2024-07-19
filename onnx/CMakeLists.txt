cmake_minimum_required(VERSION 3.17)

set(PROJECT_NAME  demo)
project(${PROJECT_NAME})

set(CMAKE_BUILD_TYPE DEBUG)
include_directories(
    include
    thirdlibrarys/include/opencv4
    thirdlibrarys/include/onnx190/include

)
link_directories(
    thirdlibrarys/libs/opencv4/lib
    thirdlibrarys/libs/onnx190/lib
)
link_libraries(opencv_world onnxruntime onnxruntime_providers_cuda)

# add source cpp files
aux_source_directory(. SRC_LIST)
aux_source_directory(./src SRC_LIST2)

add_executable(${PROJECT_NAME} ${SRC_LIST} ${SRC_LIST2})

add_library(centernet SHARED ${SRC_LIST} ${SRC_LIST2})