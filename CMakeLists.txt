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
  src/egtb/notation.cpp
  src/egtb/solver.cpp
  src/io/tablebase.cpp
)
target_include_directories(warpmate_core PUBLIC src)
target_link_libraries(warpmate_core PUBLIC Threads::Threads)

# GPU solver
if(WARPMATE_HAS_CUDA)
  add_library(warpmate_gpu src/gpu/egtb_gpu.cu)
  target_link_libraries(warpmate_gpu PUBLIC warpmate_core CUDA::cudart)
  target_compile_definitions(warpmate_gpu PUBLIC WARPMATE_HAS_CUDA=1)
  set(WARPMATE_SOLVER_LIBS warpmate_gpu)
else()
  set(WARPMATE_SOLVER_LIBS warpmate_core)
endif()

# Inference: Warpmate's own network inference code (CPU reference + CUDA
# kernels). warpmate_infer_core is used by the C++ tests; the shared library
# libwarpmate_infer is what Python loads (ml/inference.py). Without CUDA it is
# still built, with the CPU reference only.
set(WARPMATE_INFER_SOURCES src/inference/npz.cpp src/inference/network.cpp)
if(WARPMATE_HAS_CUDA)
  list(APPEND WARPMATE_INFER_SOURCES src/inference/infer_gpu.cu)
endif()
add_library(warpmate_infer_core STATIC ${WARPMATE_INFER_SOURCES})
# The 8-bit CPU reference must round exactly like NumPy and the GPU kernel:
# stop the compiler from fusing a * b + c into one instruction (an FMA, which
# rounds once instead of twice).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set_source_files_properties(src/inference/network.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
endif()
target_include_directories(warpmate_infer_core PUBLIC src)
set_target_properties(warpmate_infer_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(WARPMATE_HAS_CUDA)
  target_link_libraries(warpmate_infer_core PUBLIC CUDA::cudart_static)
  target_compile_definitions(warpmate_infer_core PUBLIC WARPMATE_HAS_CUDA=1)
endif()

add_library(warpmate_infer SHARED src/inference/c_api.cpp)
target_link_libraries(warpmate_infer PRIVATE warpmate_infer_core)

# Command-line tools
add_executable(warpmate-probe tools/probe.cpp)
target_link_libraries(warpmate-probe PRIVATE warpmate_core)

add_executable(warpmate-gen tools/gen.cpp)
target_link_libraries(warpmate-gen PRIVATE ${WARPMATE_SOLVER_LIBS})

add_executable(warpmate-bench tools/bench.cpp)
target_link_libraries(warpmate-bench PRIVATE ${WARPMATE_SOLVER_LIBS})

# CUDA learning examples
if(WARPMATE_HAS_CUDA)
  add_executable(cuda-basics examples/cuda_basics/basics.cu)
endif()

# Tests
enable_testing()

add_executable(warpmate-test-egtb tests/test_egtb.cpp)
target_link_libraries(warpmate-test-egtb PRIVATE warpmate_core)
add_test(NAME egtb COMMAND warpmate-test-egtb)

add_executable(warpmate-test-inference tests/test_inference.cpp)
target_link_libraries(warpmate-test-inference PRIVATE warpmate_infer_core)
add_test(NAME inference COMMAND warpmate-test-inference)

if(WARPMATE_HAS_CUDA)
  add_executable(warpmate-test-egtb-gpu tests/test_egtb_gpu.cpp)
  target_link_libraries(warpmate-test-egtb-gpu PRIVATE warpmate_gpu)
  add_test(NAME egtb-gpu COMMAND warpmate-test-egtb-gpu)
endif()
