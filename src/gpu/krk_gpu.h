// GPU (CUDA) version of the KRvK solver.
//
// This header deliberately contains no CUDA code, so ordinary C++ files
// (tools, tests) can include it and call the GPU solver. All CUDA lives in
// krk_gpu.cu, which only builds when a CUDA compiler is available.
#pragma once

#include <string>

#include "solver/krk_solver.h"

namespace wm::krk {

struct GpuTiming {
    double kernel_ms = 0.0;  // init + all rounds + finalize, measured on the GPU
    double copy_ms = 0.0;    // copying the finished table back to the CPU
};

// Name of the GPU in use, e.g. "Tesla T4". Throws if no GPU is available.
std::string gpu_name();

// Solves every position on the GPU. Produces exactly the same table as
// solve() and solve_parallel(). `timing` (optional) receives a breakdown.
SolveResult solve_gpu(const RoundCallback& on_round = nullptr, GpuTiming* timing = nullptr);

}  // namespace wm::krk
