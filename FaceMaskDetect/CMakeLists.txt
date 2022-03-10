cmake_minimum_required (VERSION 3.4.1)
set(CMAKE_VERBOSE_MAKEFILE on)
project (facemask_detect)

MESSAGE(STATUS "operation system is ${CMAKE_SYSTEM}")
      
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    MESSAGE(STATUS "current platform: Linux ")
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Windows")
    MESSAGE(STATUS "current platform: Windows")
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    MESSAGE(STATUS "current platform: FreeBSD")
ELSE ()
    MESSAGE(STATUS "other platform: ${CMAKE_SYSTEM_NAME}")
ENDIF (CMAKE_SYSTEM_NAME MATCHES "Linux")
  
MESSAGE(STSTUS "###################################")

# OpenCV Version: 3.4.3
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBRARIES}")
message(STATUS "OpenCV Include path: ${OpenCV_INCLUDE_DIRS}")

MESSAGE(STSTUS "###################################")
# Protobuf
find_package(Protobuf REQUIRED)
message(STATUS "Protobuf status:")
message(STATUS "Protobuf Version: ${PROTOBUF_VERSION}")
message(STATUS "Protobuf Libraries: ${PROTOBUF_LIBRARIES}")
message(STATUS "Protobuf Include path: ${PROTOBUF_INCLUDE_DIRS}")

# Fastcv

# for avoid missing glib header file
# glib-2.0, version 2.58.0
pkg_check_modules(GLIB REQUIRED glib-2.0)

# gstreamer
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
set(GSTREAMER_INCLUDE_DIRS "/usr/include/gstreamer-1.0;")
set(GSTREAMER_INCLUDE_DIRS ${GSTREAMER_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/gst_pipe/glib-2.0;")
set(GSTREAMER_INCLUDE_DIRS ${GSTREAMER_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/gst_pipe/glib-2.0/include;")
set(GSTREAMER_INCLUDE_DIRS ${GSTREAMER_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/gst_pipe/gobject;")
set(GSTREAMER_INCLUDE_DIRS ${GSTREAMER_INCLUDE_DIRS} "${CMAKE_SOURCE_DIR}/gst_pipe/gst;")
set(GSTREAMER_INCLUDE_DIRS ${GSTREAMER_INCLUDE_DIRS} "/usr/lib/gstreamer-1.0/include;")
set(GSTREAMER_INCLUDE_DIRS ${GSTREAMER_INCLUDE_DIRS} "/usr/include/glib-2.0;")
set(GSTREAMER_INCLUDE_DIRS ${GSTREAMER_INCLUDE_DIRS} "/usr/lib/glib-2.0/include;")
set(GSTREAMER_LIBS  "gstapp-1.0;")
set(GSTREAMER_LIBS ${GSTREAMER_LIBS} "gstbase-1.0;")
set(GSTREAMER_LIBS ${GSTREAMER_LIBS} "gstrtsp-1.0;")
set(GSTREAMER_LIBS ${GSTREAMER_LIBS} "gstreamer-1.0;")
set(GSTREAMER_LIBS ${GSTREAMER_LIBS} "gstsdp-1.0;")
set(GSTREAMER_LIBS ${GSTREAMER_LIBS} "gio-2.0;")
set(GSTREAMER_LIBS ${GSTREAMER_LIBS} "gobject-2.0;")
set(GSTREAMER_LIBS ${GSTREAMER_LIBS} "glib-2.0;")
set(GSTREAMER_LIBS ${GSTREAMER_LIBS} "gstvideo-1.0;")
set(GSTREAMER_LIBRARY_DIRS  "/usr/lib/;")

message(STATUS "GST DEBUG:###########################")
message(STATUS "libraries: ${GSTREAMER_LIBS}")
message(STATUS "include path: ${GSTREAMER_INCLUDE_DIRS}")
message(STATUS "library path: ${GSTREAMER_LIBRARY_DIRS}")


#SNPE
set(SNPE_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/inference/snpe")
set(SNPE_LIBS  "SNPE")
set(SNPE_LIBRARY_DIRS  "${CMAKE_SOURCE_DIR}/inference/libs/snpe")

message(STATUS "SNPE DEBUG:###########################")
message(STATUS "libraries: ${SNPE_LIBS}")
message(STATUS "include path: ${SNPE_INCLUDE_DIRS}")
message(STATUS "library path: ${SNPE_LIBRARY_DIRS}")

#Tensorflow
set(TENSORFLOW_INCLUDE "${CMAKE_SOURCE_DIR}/inference/tflite")
set(TFLITE_LIBS tflite)
set(TENSORFLOW_LIBRARY_DIRS "${CMAKE_SOURCE_DIR}/inference/libs/tflite")

#PB config
set(PB_CONFIG_INCLUDE "${CMAKE_SOURCE_DIR}/pb_conf")
set(PB_CONFIG_LIB_DIR "${CMAKE_SOURCE_DIR}/pb_conf/libs")

aux_source_directory(${CMAKE_SOURCE_DIR} DIR_SRCS)

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTREAMER_APP_INCLUDE_DIRS}
    ${GSTREAMER_RTSP_INCLUDE_DIRS}
    ${GLIB_INCLUDE_DIRS}
    ${SNPE_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/utils
    ${CMAKE_SOURCE_DIR}/examples/face_mask
    ${TENSORFLOW_INCLUDE}
    ${TENSORFLOW_INCLUDE}/tensorflow/external/flat/include
    ${TENSORFLOW_INCLUDE}/RunTflite
    ${PROTOBUF_DIRS}
    ${PB_CONFIG_INCLUDE}
)

link_directories(
    ${OpenCV_LIBRARY_DIRS}
    ${GSTREAMER_LIBRARY_DIRS}
    ${SNPE_LIBRARY_DIRS}
    ${TENSORFLOW_LIBRARY_DIRS}
)

add_executable(
    ${PROJECT_NAME}
    ${CMAKE_SOURCE_DIR}/pb_conf/gstreamer.pb.cc
    ${CMAKE_SOURCE_DIR}/pb_conf/aiconf.pb.cc
    ${CMAKE_SOURCE_DIR}/inference/tflite/RunTflite/tflite_inference.cpp

    ${CMAKE_SOURCE_DIR}/examples/face_mask/facemask_copy.cpp
)

target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${GSTREAMER_LIBS}
    ${SNPE_LIBRARY_DIRS}/1.37/libSNPE.so
    ${PROTOBUF_LIBRARIES}
    ${PROTOBUF_LIBRARY}
    ${TFLITE_TARGET_NAME}
    ${CMAKE_SOURCE_DIR}/inference/libs/tflite/aarch64/libtensorflow-lite.so
    evaluation_utils
    nnapi_delegate
    time
    memory_info
    pthread
    fastcvopt
    m
    z
)

# c++11 required
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lpthread  -lm -lrt -ljpeg -lfastcvopt -lprotobuf -lstdc++ -DSNPE_ENGINE_SUPPORTED")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb -std=c++11 -ljpeg -lstdc++ -lm -lrt -ljpeg -lfastcvopt -lpthread -lprotobuf -lstdc++ -DSNPE_ENGINE_SUPPORTED")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall -lstdc++")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib/)
