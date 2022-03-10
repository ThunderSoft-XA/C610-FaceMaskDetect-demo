#Cmake lowest  version 
CMAKE_MINIMUM_REQUIRED(VERSION 3.4)
 
#project name
PROJECT(tflite_test CXX)
 
#allow gdb debug
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 $ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 $ENV{CXXFLAGS} -O3 -Wall")
 
#add C++11 support and other options
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -g -ftest-coverage -fprofile-arcs -Wno-deprecated")
#set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS -g -ftest-coverage -fprofile-arcs"})
 
# OpenCV Version: 3.4.3
find_package(PkgConfig REQUIRED)
set(OpenCV_DIR /usr/local/opencv/share/OpenCV)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV Include path: ${OpenCV_INCLUDE_DIRS}")
 
MESSAGE(STSTUS "###################################")
# Protobuf
find_package(Protobuf REQUIRED)
message(STATUS "Protobuf status:")
message(STATUS "Protobuf Version: ${PROTOBUF_VERSION}")
message(STATUS "Protobuf Libraries: ${PROTOBUF_LIBRARIES}")
message(STATUS "Protobuf Include path: ${PROTOBUF_INCLUDE_DIRS}")

#PB config
set(PB_CONFIG_INCLUDE "${CMAKE_SOURCE_DIR}/pb_conf")
set(PB_CONFIG_LIB_DIR "${CMAKE_SOURCE_DIR}/pb_conf/libs")

#Tensorflow
set(TENSORFLOW_INCLUDE "${CMAKE_SOURCE_DIR}/inference/tflite")
set(TFLITE_LIBS tflite)
IF (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    MESSAGE(STATUS "current hardware platform: x86_64 ")
    set(TENSORFLOW_LIBRARY_DIRS "${CMAKE_SOURCE_DIR}/../libs/tflite/x86_64/libtensorflowlite.so")
ELSEIF (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    MESSAGE(STATUS "current hardware platform: aarch64")
    set(TENSORFLOW_LIBRARY_DIRS "${CMAKE_SOURCE_DIR}/../libs/tflite/aarch64/libtensorflowlite.so")
ENDIF ()

MESSAGE(STATUS ${CMAKE_SOURCE_DIR})

# include file dir
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${PROTOBUF_DIRS}
    ${PB_CONFIG_INCLUDE}
    ${CMAKE_SOURCE_DIR}/tensorflow/external/flat/include
    ${CMAKE_SOURCE_DIR}/RunTflite
    ${CMAKE_SOURCE_DIR}/../utils
)
 
#add sub dir
#add_subdirectory(dict) 
 
#set all sources files in current dir to DIR_SRCS
#AUX_SOURCE_DIRECTORY(. DIR_SRCS) 
 
#get a executabl file
add_executable(tflite_test 
    ${CMAKE_SOURCE_DIR}/RunTflite/tflite_inference.cpp
    ${CMAKE_SOURCE_DIR}/tflite_test.cpp
) 
 
#add link libs
target_link_libraries(tflite_test
    ${OpenCV_LIBS}
    ${TFLITE_TARGET_NAME}
    ${PROTOBUF_LIBRARIES}
    ${TENSORFLOW_LIBRARY_DIRS}
    pthread
    m
    z
)