// A plain C interface to the inference library, so Python can call it with
// ctypes (see ml/inference.py). Built as a shared library: libwarpmate_infer.
//
// Every function that can fail returns 0 on success, or non-zero after
// writing a message into `err` (a buffer of `errlen` bytes).
//
// Boards are passed in the compact form described in network.h: `codes`
// (n x slots int16 piece codes) and `stm` (n uint8, 0 = White to move).
// Outputs: `probs` (n x 3: loss, draw, win for the side to move) and,
// optionally, `logits` (n x 3, or null).
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wm_engine wm_engine;

// Loads the 32-bit weights (warpmate_net.npz) and, if `int8_npz` isn't null,
// the 8-bit ones (warpmate_net_int8.npz). Copies them to the GPU when the
// library was built with CUDA and a GPU is present. Returns null on failure.
wm_engine* wm_create(const char* float_npz, const char* int8_npz, char* err, int errlen);
void wm_destroy(wm_engine* engine);

// 1 if this engine has a GPU to run on, 0 otherwise.
int wm_has_gpu(const wm_engine* engine);
// 1 if the 8-bit weights were loaded.
int wm_has_int8(const wm_engine* engine);
// Name of the GPU (e.g. "Tesla T4"), or "" if none. Always succeeds.
void wm_device_name(char* buf, int len);
// Name of a variant number (0 = "naive", ...), or null past the last one.
const char* wm_variant_name(int variant);

// CPU reference implementation. int8 = 0: 32-bit weights; 1: 8-bit weights.
int wm_eval_cpu(wm_engine* engine, int int8, const int16_t* codes, int slots,
                const uint8_t* stm, int n, float* probs, float* logits, char* err, int errlen);

// GPU, CPU pointers: copies in, runs, waits, copies out.
int wm_eval_gpu(wm_engine* engine, int variant, const int16_t* codes, int slots,
                const uint8_t* stm, int n, float* probs, float* logits, char* err, int errlen);

// GPU, GPU pointers (passed as integers, e.g. torch's tensor.data_ptr()),
// queued on `stream` (a cudaStream_t as an integer, 0 = default stream).
// Returns without waiting for the GPU, like a PyTorch operation.
int wm_eval_gpu_device(wm_engine* engine, int variant, uint64_t codes, int slots, uint64_t stm,
                       int n, uint64_t probs, uint64_t logits, uint64_t stream, char* err,
                       int errlen);

#ifdef __cplusplus
}
#endif
