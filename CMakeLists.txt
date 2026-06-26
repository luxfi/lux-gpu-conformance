cmake_minimum_required(VERSION 3.20)
project(lux-gpu-conformance VERSION 0.1.0 LANGUAGES CXX)

# Cross-backend kernel-parity harness. It only needs backend_api.h from
# lux-accel; it dlopens the lux_<name>.plugin shared objects at runtime.
find_package(lux-accel CONFIG QUIET)
if(NOT lux-accel_FOUND)
  if(NOT LUXACCEL_INCLUDE_DIR)
    set(LUXACCEL_INCLUDE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/../lux-accel/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/../install/include")
  endif()
endif()

add_executable(lux_gpu_conformance conformance.cpp)
target_compile_features(lux_gpu_conformance PRIVATE cxx_std_17)
target_include_directories(lux_gpu_conformance PRIVATE ${LUXACCEL_INCLUDE_DIR})
if(lux-accel_FOUND AND TARGET lux::accel)
  target_include_directories(lux_gpu_conformance PRIVATE
    $<TARGET_PROPERTY:lux::accel,INTERFACE_INCLUDE_DIRECTORIES>)
endif()
if(NOT WIN32)
  target_link_libraries(lux_gpu_conformance PRIVATE ${CMAKE_DL_LIBS})
endif()

enable_testing()
# Discovers plugins from sibling build trees (../lux-<name>/build) or
# $LUX_PLUGIN_DIR; pass "<name>=<path>" args to override individual plugins.
add_test(NAME lux_gpu_conformance COMMAND lux_gpu_conformance)
