cmake_minimum_required(VERSION 3.20)
project(Warpmate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Warnings for C++ files only (the CUDA compiler takes different flags).
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)

find_package(Threads REQUIRED)

# GPU support is optional: it's built only when a CUDA compiler (nvcc) is
# found, so the project still builds and tests on machines without NVIDIA GPUs.
option(WARPMATE_CUDA "Build the GPU solver when a CUDA compiler is available" ON)
set(WARPMATE_HAS_CUDA OFF)
if(WARPMATE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # Compile for the GPU in this machine when CMake can detect it.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
      else()
        set(CMAKE_CUDA_ARCHITECTURES 75)  # Tesla T4 (Colab's free GPU)
      endif()
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    set(WARPMATE_HAS_CUDA ON)
  endif()
endif()
message(STATUS "Warpmate GPU support: ${WARPMATE_HAS_CUDA}")

# Shared chess and solver code.
add_library(warpmate_core
  src/core/notation.cpp
  src/io/tablebase.cpp
  src/solver/krk_solver.cpp
)
target_include_directories(warpmate_core PUBLIC src)
target_link_libraries(warpmate_core PUBLIC Threads::Threads)

# GPU solver
if(WARPMATE_HAS_CUDA)
  add_library(warpmate_gpu src/gpu/krk_gpu.cu)
  target_link_libraries(warpmate_gpu PUBLIC warpmate_core CUDA::cudart)
  target_compile_definitions(warpmate_gpu PUBLIC WARPMATE_HAS_CUDA=1)
  set(WARPMATE_SOLVER_LIBS warpmate_gpu)
else()
  set(WARPMATE_SOLVER_LIBS warpmate_core)
endif()

# Command-line tools
add_executable(warpmate-solve tools/solve.cpp)
target_link_libraries(warpmate-solve PRIVATE ${WARPMATE_SOLVER_LIBS})

add_executable(warpmate-probe tools/probe.cpp)
target_link_libraries(warpmate-probe PRIVATE warpmate_core)

add_executable(warpmate-bench tools/bench.cpp)
target_link_libraries(warpmate-bench PRIVATE ${WARPMATE_SOLVER_LIBS})

# CUDA learning examples
if(WARPMATE_HAS_CUDA)
  add_executable(cuda-basics examples/cuda_basics/basics.cu)
endif()

# Tests
enable_testing()

add_executable(warpmate-test-core tests/test_core.cpp)
target_link_libraries(warpmate-test-core PRIVATE warpmate_core)
add_test(NAME core COMMAND warpmate-test-core)

add_executable(warpmate-test-solver tests/test_solver.cpp)
target_link_libraries(warpmate-test-solver PRIVATE warpmate_core)
add_test(NAME solver COMMAND warpmate-test-solver)

if(WARPMATE_HAS_CUDA)
  add_executable(warpmate-test-gpu tests/test_gpu.cpp)
  target_link_libraries(warpmate-test-gpu PRIVATE warpmate_gpu)
  add_test(NAME gpu COMMAND warpmate-test-gpu)
endif()
