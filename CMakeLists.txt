cmake_minimum_required(VERSION 3.20)
project(Warpmate LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

# Shared chess and solver code.
add_library(warpmate_core
  src/core/notation.cpp
)
target_include_directories(warpmate_core PUBLIC src)

# Tests
enable_testing()

add_executable(warpmate-test-core tests/test_core.cpp)
target_link_libraries(warpmate-test-core PRIVATE warpmate_core)
add_test(NAME core COMMAND warpmate-test-core)
