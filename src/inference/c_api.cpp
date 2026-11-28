#include "inference/c_api.h"

#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

#include "inference/network.h"
#ifdef WARPMATE_HAS_CUDA
#include "inference/infer_gpu.h"
#endif

using namespace wm::infer;

struct wm_engine {
    FloatWeights weights;
    std::unique_ptr<Int8Weights> int8;
#ifdef WARPMATE_HAS_CUDA
    std::unique_ptr<GpuEngine> gpu;
#endif
};

namespace {

void set_error(char* err, int errlen, const char* message) {
    if (err && errlen > 0) std::snprintf(err, static_cast<size_t>(errlen), "%s", message);
}

// Runs `body`, turning any C++ exception into an error code and message
// (exceptions must never cross into Python).
template <class F>
int guarded(char* err, int errlen, F&& body) {
    try {
        body();
        return 0;
    } catch (const std::exception& e) {
        set_error(err, errlen, e.what());
    } catch (...) {
        set_error(err, errlen, "unknown error");
    }
    return 1;
}

#ifdef WARPMATE_HAS_CUDA
constexpr int variant_count() { return kVariantCount; }

GpuEngine& gpu_of(wm_engine* engine, int variant) {
    if (!engine->gpu) throw std::runtime_error("no GPU available");
    if (variant < 0 || variant >= variant_count())
        throw std::invalid_argument("unknown variant " + std::to_string(variant));
    return *engine->gpu;
}
#endif

}  // namespace

extern "C" {

wm_engine* wm_create(const char* float_npz, const char* int8_npz, char* err, int errlen) {
    auto engine = std::make_unique<wm_engine>();
    const int status = guarded(err, errlen, [&] {
        engine->weights = load_float_weights(float_npz);
        if (int8_npz) engine->int8 = std::make_unique<Int8Weights>(load_int8_weights(int8_npz));
#ifdef WARPMATE_HAS_CUDA
        if (!GpuEngine::device_name().empty())
            engine->gpu = std::make_unique<GpuEngine>(engine->weights, engine->int8.get());
#endif
    });
    return status == 0 ? engine.release() : nullptr;
}

void wm_destroy(wm_engine* engine) { delete engine; }

int wm_has_gpu(const wm_engine* engine) {
#ifdef WARPMATE_HAS_CUDA
    return engine && engine->gpu ? 1 : 0;
#else
    (void)engine;
    return 0;
#endif
}

int wm_has_int8(const wm_engine* engine) { return engine && engine->int8 ? 1 : 0; }

void wm_device_name(char* buf, int len) {
#ifdef WARPMATE_HAS_CUDA
    set_error(buf, len, GpuEngine::device_name().c_str());
#else
    set_error(buf, len, "");
#endif
}

const char* wm_variant_name(int variant) {
#ifdef WARPMATE_HAS_CUDA
    if (variant >= 0 && variant < variant_count())
        return variant_name(static_cast<Variant>(variant));
#else
    (void)variant;
#endif
    return nullptr;
}

int wm_eval_cpu(wm_engine* engine, int int8, const int16_t* codes, int slots,
                const uint8_t* stm, int n, float* probs, float* logits, char* err, int errlen) {
    return guarded(err, errlen, [&] {
        if (int8) {
            if (!engine->int8) throw std::runtime_error("8-bit weights weren't loaded");
            forward_cpu_int8(*engine->int8, codes, slots, stm, n, probs, logits);
        } else {
            forward_cpu(engine->weights, codes, slots, stm, n, probs, logits);
        }
    });
}

int wm_eval_gpu(wm_engine* engine, int variant, const int16_t* codes, int slots,
                const uint8_t* stm, int n, float* probs, float* logits, char* err, int errlen) {
    return guarded(err, errlen, [&] {
#ifdef WARPMATE_HAS_CUDA
        gpu_of(engine, variant)
            .run_host(static_cast<Variant>(variant), codes, slots, stm, n, probs, logits);
#else
        (void)engine; (void)variant; (void)codes; (void)slots; (void)stm; (void)n;
        (void)probs; (void)logits;
        throw std::runtime_error("this library was built without CUDA");
#endif
    });
}

int wm_eval_gpu_device(wm_engine* engine, int variant, uint64_t codes, int slots, uint64_t stm,
                       int n, uint64_t probs, uint64_t logits, uint64_t stream, char* err,
                       int errlen) {
    return guarded(err, errlen, [&] {
#ifdef WARPMATE_HAS_CUDA
        gpu_of(engine, variant)
            .run(static_cast<Variant>(variant), reinterpret_cast<const int16_t*>(codes), slots,
                 reinterpret_cast<const uint8_t*>(stm), n, reinterpret_cast<float*>(probs),
                 reinterpret_cast<float*>(logits), reinterpret_cast<void*>(stream));
#else
        (void)engine; (void)variant; (void)codes; (void)slots; (void)stm; (void)n;
        (void)probs; (void)logits; (void)stream;
        throw std::runtime_error("this library was built without CUDA");
#endif
    });
}

}  // extern "C"
