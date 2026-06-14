cmake_minimum_required(VERSION 3.16)
project(uniad_sparse_lidar LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

set(DEFAULT_SPARSE_CONV_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/3DSparseConvolution)
if(NOT SPARSE_CONV_ROOT)
  if(DEFINED ENV{SPARSE_CONV_ROOT})
    set(SPARSE_CONV_ROOT $ENV{SPARSE_CONV_ROOT})
  elseif(EXISTS ${DEFAULT_SPARSE_CONV_ROOT})
    set(SPARSE_CONV_ROOT ${DEFAULT_SPARSE_CONV_ROOT})
  elseif(LIDAR_AI_SOLUTION_PATH)
    set(SPARSE_CONV_ROOT ${LIDAR_AI_SOLUTION_PATH}/libraries/3DSparseConvolution)
  elseif(DEFINED ENV{LIDAR_AI_SOLUTION_PATH})
    set(SPARSE_CONV_ROOT $ENV{LIDAR_AI_SOLUTION_PATH}/libraries/3DSparseConvolution)
  else()
    message(FATAL_ERROR
      "Need: -DSPARSE_CONV_ROOT=/path/to/3DSparseConvolution "
      "(or place it at dependencies/3DSparseConvolution).")
  endif()
endif()

if(NOT SPCONV_CUDA_VERSION)
  set(SPCONV_CUDA_VERSION "11.4")
endif()

if(NOT EXISTS ${SPARSE_CONV_ROOT}/src/onnx-parser.cpp)
  message(FATAL_ERROR "Invalid 3DSparseConvolution path: ${SPARSE_CONV_ROOT}")
endif()
message(STATUS "Using 3DSparseConvolution: ${SPARSE_CONV_ROOT}")

find_package(Protobuf REQUIRED)

if(NOT CUDA_TOOLKIT_ROOT_DIR)
  if(DEFINED ENV{CUDA_HOME})
    set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_HOME})
  elseif(DEFINED ENV{CUDA_PATH})
    set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_PATH})
  else()
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
  endif()
endif()

set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/include)
set(CUDA_LIBRARY_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUDART_LIBRARY cudart PATHS ${CUDA_LIBRARY_DIRS} NO_DEFAULT_PATH)
if(NOT EXISTS ${CUDA_INCLUDE_DIRS}/cuda_runtime.h OR NOT CUDART_LIBRARY)
  message(FATAL_ERROR "Invalid CUDA toolkit path: ${CUDA_TOOLKIT_ROOT_DIR}")
endif()

set(ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
set(SPCONV_LIB_DIR ${SPARSE_CONV_ROOT}/libspconv/lib/${ARCH}_cuda${SPCONV_CUDA_VERSION})
if(NOT EXISTS ${SPCONV_LIB_DIR}/libspconv.so)
  message(FATAL_ERROR "Missing libspconv.so: ${SPCONV_LIB_DIR}/libspconv.so")
endif()

set(GENERATED_PROTO_ROOT ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(GENERATED_PROTO_SRCS
  ${GENERATED_PROTO_ROOT}/onnx/onnx-ml.pb.cc
  ${GENERATED_PROTO_ROOT}/onnx/onnx-operators-ml.pb.cc
)
set(GENERATED_PROTO_HDRS
  ${GENERATED_PROTO_ROOT}/onnx/onnx-ml.pb.h
  ${GENERATED_PROTO_ROOT}/onnx/onnx-operators-ml.pb.h
)
add_custom_command(
  OUTPUT ${GENERATED_PROTO_SRCS} ${GENERATED_PROTO_HDRS}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_PROTO_ROOT}/onnx
  COMMAND ${Protobuf_PROTOC_EXECUTABLE}
          --proto_path=${SPARSE_CONV_ROOT}/src/onnx
          --cpp_out=${GENERATED_PROTO_ROOT}/onnx
          onnx-ml.proto
          onnx-operators-ml.proto
  DEPENDS
    ${SPARSE_CONV_ROOT}/src/onnx/onnx-ml.proto
    ${SPARSE_CONV_ROOT}/src/onnx/onnx-operators-ml.proto
  WORKING_DIRECTORY ${SPARSE_CONV_ROOT}/src
  COMMENT "Generating ONNX protobuf sources"
  VERBATIM
)

configure_file(
  ${SPARSE_CONV_ROOT}/src/onnx-parser.cpp
  ${GENERATED_PROTO_ROOT}/onnx-parser.cpp
  COPYONLY
)

add_executable(validate_sparse
  src/validate_sparse.cpp
  ${GENERATED_PROTO_ROOT}/onnx-parser.cpp
  ${GENERATED_PROTO_SRCS}
)

set(SPARSE_RUNTIME_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${GENERATED_PROTO_ROOT}
  ${SPARSE_CONV_ROOT}/src
  ${SPARSE_CONV_ROOT}/libspconv/include
  ${CUDA_INCLUDE_DIRS}
  ${Protobuf_INCLUDE_DIRS}
)

set(SPARSE_RUNTIME_LIB_DIRS
  ${SPCONV_LIB_DIR}
)

set(SPARSE_RUNTIME_LIBS
  spconv
  protobuf::libprotobuf
  ${CUDART_LIBRARY}
  dl
  pthread
)

target_include_directories(validate_sparse PRIVATE ${SPARSE_RUNTIME_INCLUDES})
target_link_directories(validate_sparse PRIVATE ${SPARSE_RUNTIME_LIB_DIRS})
target_link_libraries(validate_sparse PRIVATE ${SPARSE_RUNTIME_LIBS})

set_target_properties(validate_sparse PROPERTIES
  BUILD_RPATH "${SPCONV_LIB_DIR};${CUDA_LIBRARY_DIRS}"
  INSTALL_RPATH "${SPCONV_LIB_DIR};${CUDA_LIBRARY_DIRS}"
)

if(TENSORRT_PATH)
  set(TENSORRT_INCLUDE_DIRS ${TENSORRT_PATH}/include)
  set(TENSORRT_LIBRARY_DIRS ${TENSORRT_PATH}/lib)
  find_library(NVINFER_LIBRARY nvinfer PATHS ${TENSORRT_LIBRARY_DIRS} NO_DEFAULT_PATH)
  find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin PATHS ${TENSORRT_LIBRARY_DIRS} NO_DEFAULT_PATH)
  if(NOT EXISTS ${TENSORRT_INCLUDE_DIRS}/NvInfer.h OR
     NOT NVINFER_LIBRARY OR
     NOT NVINFER_PLUGIN_LIBRARY)
    message(FATAL_ERROR "Invalid TensorRT path: ${TENSORRT_PATH}")
  endif()

  set(ENQUEUE_V3_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../enqueueV3)
  add_library(sparse_lidar_trt_runtime STATIC
    src/lidar_runtime.cpp
    src/lidar_postprocess.cpp
    src/lidar_bev_visualizer.cpp
    src/lidar_metadata.cpp
    ${GENERATED_PROTO_ROOT}/onnx-parser.cpp
    ${GENERATED_PROTO_SRCS}
    ${ENQUEUE_V3_ROOT}/src/tensorrt.cpp
  )
  target_include_directories(sparse_lidar_trt_runtime PUBLIC
    ${SPARSE_RUNTIME_INCLUDES}
    ${ENQUEUE_V3_ROOT}/include
    ${TENSORRT_INCLUDE_DIRS}
  )
  target_link_directories(sparse_lidar_trt_runtime PUBLIC
    ${SPARSE_RUNTIME_LIB_DIRS}
    ${TENSORRT_LIBRARY_DIRS}
  )
  target_link_libraries(sparse_lidar_trt_runtime PUBLIC
    ${SPARSE_RUNTIME_LIBS}
    ${NVINFER_LIBRARY}
    ${NVINFER_PLUGIN_LIBRARY}
  )

  add_executable(validate_sparse_trt
    src/validate_sparse_trt.cpp
  )
  target_link_libraries(validate_sparse_trt PRIVATE sparse_lidar_trt_runtime)
  add_executable(uniad_lidar
    src/uniad_lidar.cpp
  )
  target_link_libraries(uniad_lidar PRIVATE sparse_lidar_trt_runtime)
  add_executable(uniad_lidar_track
    src/uniad_lidar_track.cpp
  )
  target_link_libraries(uniad_lidar_track PRIVATE sparse_lidar_trt_runtime)
  add_executable(uniad_lidar_e2e
    src/uniad_lidar_e2e.cpp
  )
  target_link_libraries(uniad_lidar_e2e PRIVATE sparse_lidar_trt_runtime)
  set_target_properties(validate_sparse_trt PROPERTIES
    BUILD_RPATH "${SPCONV_LIB_DIR};${CUDA_LIBRARY_DIRS};${TENSORRT_LIBRARY_DIRS}"
    INSTALL_RPATH "${SPCONV_LIB_DIR};${CUDA_LIBRARY_DIRS};${TENSORRT_LIBRARY_DIRS}"
  )
  set_target_properties(uniad_lidar PROPERTIES
    BUILD_RPATH "${SPCONV_LIB_DIR};${CUDA_LIBRARY_DIRS};${TENSORRT_LIBRARY_DIRS}"
    INSTALL_RPATH "${SPCONV_LIB_DIR};${CUDA_LIBRARY_DIRS};${TENSORRT_LIBRARY_DIRS}"
  )
  set_target_properties(uniad_lidar_track PROPERTIES
    BUILD_RPATH "${SPCONV_LIB_DIR};${CUDA_LIBRARY_DIRS};${TENSORRT_LIBRARY_DIRS}"
    INSTALL_RPATH "${SPCONV_LIB_DIR};${CUDA_LIBRARY_DIRS};${TENSORRT_LIBRARY_DIRS}"
  )
  set_target_properties(uniad_lidar_e2e PROPERTIES
    BUILD_RPATH "${SPCONV_LIB_DIR};${CUDA_LIBRARY_DIRS};${TENSORRT_LIBRARY_DIRS}"
    INSTALL_RPATH "${SPCONV_LIB_DIR};${CUDA_LIBRARY_DIRS};${TENSORRT_LIBRARY_DIRS}"
  )
else()
  message(STATUS "TENSORRT_PATH is not set; validate_sparse_trt target is disabled")
endif()
