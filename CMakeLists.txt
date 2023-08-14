cmake_minimum_required(VERSION 3.5)
project(PARLAY_ATOMIC_SHARED_PTR VERSION 2.0
        DESCRIPTION "A lock-free atomic shared pointer for C++"
        LANGUAGES CXX)

include(FetchContent)

# Set a default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

message(STATUS "--------------- General configuration -------------")
message(STATUS "CMake Generator:                ${CMAKE_GENERATOR}")
message(STATUS "Compiler:                       ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build type:                     ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_FLAGS:                ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG:          ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE:        ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS          ${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "CMAKE_INSTALL_PREFIX:           ${CMAKE_INSTALL_PREFIX}" )

# Make sure -fno-omit-frame-pointer is set for profiling
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer")

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(LINUX TRUE)
endif()

# -------------------------------------------------------------------
#                       Library definition

add_library(parlay_atomic_shared_ptr INTERFACE)
set(PARLAY_ATOMIC_SHARED_PTR_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
target_include_directories(parlay_atomic_shared_ptr INTERFACE ${PARLAY_ATOMIC_SHARED_PTR_INCLUDE_DIR})
target_compile_features(parlay_atomic_shared_ptr INTERFACE cxx_std_20)

# Link against system threads
find_package(Threads REQUIRED)
target_link_libraries(parlay_atomic_shared_ptr INTERFACE Threads::Threads)


# -------------------------------------------------------------------
#                             Folly

#FetchContent_Declare(folly
#  GIT_REPOSITORY https://github.com/facebook/folly
#  GIT_TAG main
#)

#if(NOT folly_POPULATED)
#  FetchContent_Populate(folly)
#  add_subdirectory(${folly_SOURCE_DIR} ${folly_BINARY_DIR})
#endif()

find_package(folly REQUIRED)
target_link_libraries(parlay_atomic_shared_ptr INTERFACE folly glog dl pthread double-conversion fmt gflags)

# -------------------------------------------------------------------
#                         Unit tests

message(STATUS "---------------------------------- Unit Tests ----------------------------------")


# Set CMake options for GoogleTest
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)

# Download and configure GoogleTest
include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY  https://github.com/google/googletest.git
  GIT_TAG         main
)
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  message(STATUS "testing: Configuring GoogleTest")
  FetchContent_Populate(googletest)  
  set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE BOOL "")
  add_subdirectory(${googletest_SOURCE_DIR}
                   ${googletest_BINARY_DIR}
                   EXCLUDE_FROM_ALL)
endif()

# Include test targets
message(STATUS "testing: Enabled")
include(CTest)
add_subdirectory(test)

# -------------------------------------------------------------------
#                       Benchmarks

message(STATUS "---------------------------------- Benchmarks ----------------------------------")

# Benchmark should not run its own unit tests
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

# Download Benchmark library
include(FetchContent)
FetchContent_Declare(benchmark
  GIT_REPOSITORY  https://github.com/google/benchmark.git
  GIT_TAG         main
)
FetchContent_GetProperties(benchmark)
if(NOT benchmark_POPULATED)
  message(STATUS "benchmarks: Configuring Google Benchmark")
  FetchContent_Populate(benchmark)
  set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE BOOL "")
  add_subdirectory(${benchmark_SOURCE_DIR}
          ${benchmark_BINARY_DIR}
          EXCLUDE_FROM_ALL)
endif()

add_subdirectory(benchmark)
