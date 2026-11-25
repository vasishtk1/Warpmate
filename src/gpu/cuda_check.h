// Error checking for CUDA calls, shared by the GPU solvers.
//
// Every CUDA API call returns an error code. Ignoring them is the most common
// CUDA bug: a failed call silently leaves garbage behind and the program
// crashes (or gives wrong answers) somewhere else. CUDA_CHECK turns any
// failure into an exception that names the call and the line.
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        const cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                             \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_) + \
                                     " at " __FILE__ ":" + std::to_string(__LINE__) + " (" #call \
                                     ")");                                                     \
        }                                                                                      \
    } while (0)
