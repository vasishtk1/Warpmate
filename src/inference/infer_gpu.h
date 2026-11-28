// Runs the Warpmate network on an NVIDIA GPU with Warpmate's own CUDA kernels.
//
// Versions ("variants"), each one a step in making inference faster:
//   kNaive: one kernel per step (expand the input to 769 numbers, layer 1,
//           layer 2, layer 3, softmax), passing results through GPU memory.
//           The straightforward version, and the GPU reference.
//
// No CUDA types appear here, so ordinary C++ files can include it.
#pragma once

#include <cstdint>
#include <string>

#include "inference/network.h"

namespace wm::infer {

enum class Variant : int {
    kNaive = 0,
};
constexpr int kVariantCount = 1;

const char* variant_name(Variant v);

class GpuEngine {
public:
    // Copies the weights to the GPU (once; they stay there).
    explicit GpuEngine(const FloatWeights& weights);
    ~GpuEngine();
    GpuEngine(const GpuEngine&) = delete;
    GpuEngine& operator=(const GpuEngine&) = delete;

    // Everything is a GPU pointer: codes (n x slots), stm (n), probs (n x 3),
    // logits (n x 3, may be null). Kernels are queued on `stream` (a
    // cudaStream_t; null = the default stream) and the call returns without
    // waiting for them. It doesn't allocate memory once a batch of this size
    // has run before, so it can be recorded into a CUDA graph.
    void run(Variant v, const int16_t* codes, int slots, const uint8_t* stm, int n, float* probs,
             float* logits, void* stream = nullptr);

    // Same, with CPU pointers: copies the boards in, runs, waits, copies back.
    void run_host(Variant v, const int16_t* codes, int slots, const uint8_t* stm, int n,
                  float* probs, float* logits);

    // Name of the GPU in use, e.g. "Tesla T4". Empty if there's no usable GPU.
    static std::string device_name();

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace wm::infer
