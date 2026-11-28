// Tests for the inference code. Needs no trained model: it makes up random
// weights and boards, so it runs anywhere (the GPU part only with CUDA).
//
//   1. .npy parsing on a hand-made file
//   2. the CPU's sparse layer 1 (add up the weight rows of the pieces) gives
//      the same answers as the plain dense calculation over all 769 inputs,
//      for both the 32-bit and the 8-bit weights
//   3. (with CUDA) every GPU variant matches the CPU reference (the 8-bit one
//      the 8-bit reference), for batch sizes that do and don't fill whole
//      blocks and for different slot counts
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "inference/network.h"
#include "inference/npz.h"
#ifdef WARPMATE_HAS_CUDA
#include "inference/infer_gpu.h"
#endif

using namespace wm::infer;

namespace {

// Small deterministic random numbers (the same on every machine).
struct Rng {
    uint64_t s;
    uint32_t next() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(s >> 33);
    }
    float uniform(float lo, float hi) { return lo + (hi - lo) * (next() / 2147483648.0f); }
    int below(int n) { return static_cast<int>(next() % static_cast<uint32_t>(n)); }
};

FloatWeights random_float_weights(Rng& rng) {
    FloatWeights w;
    auto fill = [&](std::vector<float>& v, size_t n, float scale) {
        v.resize(n);
        for (auto& x : v) x = rng.uniform(-scale, scale);
    };
    fill(w.w1t, static_cast<size_t>(kInputs) * kH1, 0.3f);
    fill(w.b1, kH1, 0.1f);
    fill(w.w2t, static_cast<size_t>(kH1) * kH2, 0.15f);
    fill(w.b2, kH2, 0.1f);
    fill(w.w3, static_cast<size_t>(kOutputs) * kH2, 0.3f);
    fill(w.b3, kOutputs, 0.1f);
    return w;
}

Int8Weights random_int8_weights(Rng& rng) {
    Int8Weights w;
    auto fill_q = [&](std::vector<int8_t>& v, size_t n) {
        v.resize(n);
        for (auto& x : v) x = static_cast<int8_t>(rng.below(255) - 127);
    };
    auto fill = [&](std::vector<float>& v, size_t n, float lo, float hi) {
        v.resize(n);
        for (auto& x : v) x = rng.uniform(lo, hi);
    };
    fill_q(w.q1t, static_cast<size_t>(kInputs) * kH1);
    fill(w.s1, kH1, 0.001f, 0.003f);
    fill(w.b1, kH1, -0.1f, 0.2f);
    w.a1 = 0.01f;
    fill_q(w.q2, static_cast<size_t>(kH2) * kH1);
    fill(w.m2, kH2, 0.00001f, 0.00003f);
    fill(w.b2, kH2, -0.1f, 0.1f);
    fill(w.w3, static_cast<size_t>(kOutputs) * kH2, -0.3f, 0.3f);
    fill(w.b3, kOutputs, -0.1f, 0.1f);
    return w;
}

// n random boards with 2..min(slots, 6) distinct piece codes each (or up to
// `slots` when `crowded`), the rest padded. Some padding uses other
// out-of-range values to check they are skipped too.
void random_boards(Rng& rng, int n, int slots, bool crowded, std::vector<int16_t>& codes,
                   std::vector<uint8_t>& stm) {
    codes.assign(static_cast<size_t>(n) * slots, kPad);
    stm.resize(n);
    const int16_t pads[] = {kPad, -1, 800};
    for (int b = 0; b < n; ++b) {
        const int max_pieces = crowded ? slots : std::min(slots, 6);
        const int pieces = 2 + rng.below(std::max(1, max_pieces - 1));
        std::vector<int16_t> used;
        for (int s = 0; s < slots; ++s) {
            int16_t c = pads[rng.below(3)];
            if (s < pieces) {
                do c = static_cast<int16_t>(rng.below(kStmInput));
                while (std::find(used.begin(), used.end(), c) != used.end());
                used.push_back(c);
            }
            codes[static_cast<size_t>(b) * slots + s] = c;
        }
        // Shuffle the slots so pieces aren't always first.
        for (int s = slots - 1; s > 0; --s)
            std::swap(codes[static_cast<size_t>(b) * slots + s],
                      codes[static_cast<size_t>(b) * slots + rng.below(s + 1)]);
        stm[b] = static_cast<uint8_t>(rng.below(2));
    }
}

// The 769 inputs of one board, as ml/model.py's features() builds them.
std::vector<double> dense_input(const int16_t* c, int slots, uint8_t stm) {
    std::vector<double> x(kInputs, 0.0);
    for (int s = 0; s < slots; ++s)
        if (is_piece(c[s])) x[c[s]] = 1.0;
    x[kStmInput] = stm == 0 ? 1.0 : 0.0;
    return x;
}

// Plain dense 32-bit network in double precision: the textbook calculation.
void dense_reference(const FloatWeights& w, const std::vector<double>& x, double* logits) {
    std::vector<double> h1(kH1), h2(kH2);
    for (int j = 0; j < kH1; ++j) {
        double acc = w.b1[j];
        for (int k = 0; k < kInputs; ++k) acc += x[k] * w.w1t[static_cast<size_t>(k) * kH1 + j];
        h1[j] = std::max(acc, 0.0);
    }
    for (int j = 0; j < kH2; ++j) {
        double acc = w.b2[j];
        for (int k = 0; k < kH1; ++k) acc += h1[k] * w.w2t[static_cast<size_t>(k) * kH2 + j];
        h2[j] = std::max(acc, 0.0);
    }
    for (int o = 0; o < kOutputs; ++o) {
        double acc = w.b3[o];
        for (int k = 0; k < kH2; ++k) acc += h2[k] * w.w3[o * kH2 + k];
        logits[o] = acc;
    }
}

// Plain dense 8-bit network: integer matrix products over all 769 inputs.
void dense_reference_int8(const Int8Weights& w, const std::vector<double>& x, double* logits) {
    std::vector<int> h1q(kH1);
    std::vector<double> h2(kH2);
    for (int j = 0; j < kH1; ++j) {
        int acc = 0;
        for (int k = 0; k < kInputs; ++k)
            acc += static_cast<int>(x[k]) * w.q1t[static_cast<size_t>(k) * kH1 + j];
        const float h = std::max(w.s1[j] * static_cast<float>(acc) + w.b1[j], 0.0f);
        h1q[j] = static_cast<int>(std::min(127.0f, std::nearbyint(h / w.a1)));
    }
    for (int j = 0; j < kH2; ++j) {
        int acc = 0;
        for (int k = 0; k < kH1; ++k) acc += w.q2[static_cast<size_t>(j) * kH1 + k] * h1q[k];
        h2[j] = std::max(static_cast<double>(w.m2[j] * static_cast<float>(acc) + w.b2[j]), 0.0);
    }
    for (int o = 0; o < kOutputs; ++o) {
        double acc = w.b3[o];
        for (int k = 0; k < kH2; ++k) acc += h2[k] * w.w3[o * kH2 + k];
        logits[o] = acc;
    }
}

[[maybe_unused]] double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0;
    for (size_t i = 0; i < a.size(); ++i) d = std::max(d, std::fabs(double(a[i]) - b[i]));
    return d;
}

void test_npy_parsing() {
    // Build a version-1 .npy file by hand: a (2, 3) float32 array.
    const std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
    std::string file = std::string("\x93NUMPY\x01\x00", 8);
    std::string padded = header;
    while ((10 + padded.size() + 1) % 64 != 0) padded += ' ';
    padded += '\n';
    file += static_cast<char>(padded.size() & 0xFF);
    file += static_cast<char>(padded.size() >> 8);
    file += padded;
    const float values[6] = {1.5f, -2.0f, 3.25f, 0.0f, 7.0f, -0.5f};
    file.append(reinterpret_cast<const char*>(values), sizeof(values));

    NpyArray a = parse_npy(file.data(), file.size(), "hand-made");
    CHECK(a.dtype == "<f4");
    CHECK((a.shape == std::vector<int64_t>{2, 3}));
    CHECK(a.size() == 6);
    const auto f = a.as<float>("<f4");
    CHECK(f.size() == 6 && f[2] == 3.25f && f[5] == -0.5f);

    bool threw = false;
    try {
        a.as<int8_t>("|i1");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);  // wrong type requested

    threw = false;
    try {
        parse_npy(file.data(), file.size() - 4, "truncated");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);  // data cut short
}

void test_cpu_matches_dense(const FloatWeights& fw, const Int8Weights& qw) {
    Rng rng{7};
    const int n = 500, slots = 8;
    std::vector<int16_t> codes;
    std::vector<uint8_t> stm;
    random_boards(rng, n, slots, false, codes, stm);

    std::vector<float> probs(n * 3), logits(n * 3), qprobs(n * 3), qlogits(n * 3);
    forward_cpu(fw, codes.data(), slots, stm.data(), n, probs.data(), logits.data());
    forward_cpu_int8(qw, codes.data(), slots, stm.data(), n, qprobs.data(), qlogits.data());

    double worst = 0, worst_q = 0, prob_sum_err = 0;
    for (int b = 0; b < n; ++b) {
        const auto x = dense_input(&codes[static_cast<size_t>(b) * slots], slots, stm[b]);
        double ref[3], qref[3];
        dense_reference(fw, x, ref);
        dense_reference_int8(qw, x, qref);
        for (int o = 0; o < 3; ++o) {
            worst = std::max(worst, std::fabs(ref[o] - logits[b * 3 + o]));
            worst_q = std::max(worst_q, std::fabs(qref[o] - qlogits[b * 3 + o]));
        }
        prob_sum_err = std::max(prob_sum_err,
                                std::fabs(probs[b * 3] + probs[b * 3 + 1] + probs[b * 3 + 2] - 1.0));
    }
    std::printf("  CPU sparse vs dense: max logit difference %.2e (32-bit), %.2e (8-bit)\n",
                worst, worst_q);
    CHECK(worst < 1e-4);
    CHECK(worst_q < 1e-4);
    CHECK(prob_sum_err < 1e-5);
}

#ifdef WARPMATE_HAS_CUDA
void test_gpu_matches_cpu(const FloatWeights& fw, const Int8Weights& qw) {
    if (GpuEngine::device_name().empty()) {
        std::printf("  no GPU found: skipping the GPU checks\n");
        return;
    }
    std::printf("  GPU: %s\n", GpuEngine::device_name().c_str());
    GpuEngine engine(fw, &qw);
    Rng rng{11};
    struct Case { int n, slots; bool crowded; };
    const Case cases[] = {{1, 5, false},  {7, 5, false},   {8, 5, false},  {9, 5, false},
                          {100, 8, false}, {5000, 5, false}, {333, 32, true}, {64, 1, false}};
    for (int v = 0; v < kVariantCount; ++v) {
        const Variant variant = static_cast<Variant>(v);
        double worst_p = 0, worst_l = 0;
        for (const Case& c : cases) {
            std::vector<int16_t> codes;
            std::vector<uint8_t> stm;
            random_boards(rng, c.n, c.slots, c.crowded, codes, stm);
            std::vector<float> p(c.n * 3), l(c.n * 3), gp(c.n * 3), gl(c.n * 3);
            if (variant == Variant::kInt8)
                forward_cpu_int8(qw, codes.data(), c.slots, stm.data(), c.n, p.data(), l.data());
            else
                forward_cpu(fw, codes.data(), c.slots, stm.data(), c.n, p.data(), l.data());
            engine.run_host(variant, codes.data(), c.slots, stm.data(), c.n, gp.data(), gl.data());
            worst_p = std::max(worst_p, max_diff(p, gp));
            worst_l = std::max(worst_l, max_diff(l, gl));
            // Probabilities only (no logits requested) must give the same answer.
            std::vector<float> gp2(c.n * 3);
            engine.run_host(variant, codes.data(), c.slots, stm.data(), c.n, gp2.data(), nullptr);
            CHECK(max_diff(gp, gp2) == 0.0);
        }
        std::printf("  GPU %-6s vs CPU: max difference %.2e (probabilities), %.2e (logits)\n",
                    variant_name(variant), worst_p, worst_l);
        CHECK(worst_p < 1e-5);
        CHECK(worst_l < 1e-4);
    }
}
#endif

}  // namespace

int main() {
    test_npy_parsing();
    Rng rng{2026};
    const FloatWeights fw = random_float_weights(rng);
    const Int8Weights qw = random_int8_weights(rng);
    test_cpu_matches_dense(fw, qw);
#ifdef WARPMATE_HAS_CUDA
    test_gpu_matches_cpu(fw, qw);
#else
    std::printf("  built without CUDA: GPU checks skipped\n");
#endif
    return wm::test::finish("inference");
}
