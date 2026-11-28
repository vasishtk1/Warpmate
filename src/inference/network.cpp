#include "inference/network.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "inference/npz.h"

namespace wm::infer {

namespace {

const NpyArray& get(const std::map<std::string, NpyArray>& arrays, const std::string& name,
                    std::vector<int64_t> shape, const std::string& path) {
    auto it = arrays.find(name);
    if (it == arrays.end()) throw std::runtime_error(path + ": missing array " + name);
    if (it->second.shape != shape) {
        std::string want, got;
        for (auto d : shape) want += std::to_string(d) + " ";
        for (auto d : it->second.shape) got += std::to_string(d) + " ";
        throw std::runtime_error(path + ": " + name + " has shape ( " + got +
                                 "), expected ( " + want + "); this code is built for the " +
                                 "769 -> 256 -> 64 -> 3 network");
    }
    return it->second;
}

// (rows, cols) row-major -> (cols, rows) row-major.
template <class T>
std::vector<T> transpose(const std::vector<T>& a, int rows, int cols) {
    std::vector<T> t(a.size());
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) t[static_cast<size_t>(c) * rows + r] = a[static_cast<size_t>(r) * cols + c];
    return t;
}

void check_slots(int slots) {
    if (slots < 1 || slots > kMaxSlots)
        throw std::invalid_argument("slots must be between 1 and " + std::to_string(kMaxSlots));
}

// The last layer and softmax, shared by both CPU versions.
void output_layer(const std::vector<float>& w3, const std::vector<float>& b3, const float* h2,
                  float* probs, float* logits) {
    float out[kOutputs];
    for (int o = 0; o < kOutputs; ++o) {
        float acc = b3[o];
        for (int k = 0; k < kH2; ++k) acc += w3[o * kH2 + k] * h2[k];
        out[o] = acc;
    }
    softmax3(out, probs);
    if (logits)
        for (int o = 0; o < kOutputs; ++o) logits[o] = out[o];
}

}  // namespace

void softmax3(const float* logits, float* probs) {
    const float m = std::max(logits[0], std::max(logits[1], logits[2]));
    float e[kOutputs], sum = 0.0f;
    for (int o = 0; o < kOutputs; ++o) {
        e[o] = std::exp(logits[o] - m);
        sum += e[o];
    }
    for (int o = 0; o < kOutputs; ++o) probs[o] = e[o] / sum;
}

FloatWeights load_float_weights(const std::string& path) {
    const auto a = load_npz(path);
    FloatWeights w;
    w.w1t = transpose(get(a, "w0", {kH1, kInputs}, path).as<float>("<f4"), kH1, kInputs);
    w.b1 = get(a, "b0", {kH1}, path).as<float>("<f4");
    w.w2t = transpose(get(a, "w1", {kH2, kH1}, path).as<float>("<f4"), kH2, kH1);
    w.b2 = get(a, "b1", {kH2}, path).as<float>("<f4");
    w.w3 = get(a, "w2", {kOutputs, kH2}, path).as<float>("<f4");
    w.b3 = get(a, "b2", {kOutputs}, path).as<float>("<f4");
    return w;
}

Int8Weights load_int8_weights(const std::string& path) {
    const auto a = load_npz(path);
    Int8Weights w;
    w.q1t = transpose(get(a, "q1", {kH1, kInputs}, path).as<int8_t>("|i1"), kH1, kInputs);
    w.s1 = get(a, "s1", {kH1}, path).as<float>("<f4");
    w.b1 = get(a, "b1", {kH1}, path).as<float>("<f4");
    w.a1 = get(a, "a1", {}, path).as<float>("<f4")[0];
    w.q2 = get(a, "q2", {kH2, kH1}, path).as<int8_t>("|i1");
    w.m2 = get(a, "m2", {kH2}, path).as<float>("<f4");
    w.b2 = get(a, "b2", {kH2}, path).as<float>("<f4");
    w.w3 = get(a, "w3", {kOutputs, kH2}, path).as<float>("<f4");
    w.b3 = get(a, "b3", {kOutputs}, path).as<float>("<f4");
    if (!(w.a1 > 0.0f)) throw std::runtime_error(path + ": a1 must be positive");
    return w;
}

void forward_cpu(const FloatWeights& w, const int16_t* codes, int slots, const uint8_t* stm,
                 int n, float* probs, float* logits) {
    check_slots(slots);
    float h1[kH1], h2[kH2];
    for (int b = 0; b < n; ++b) {
        // Layer 1, sparse: start from the bias and add the weight row of every
        // input that is on. Same result as multiplying all 769 inputs, since
        // the other 763+ inputs are 0.
        for (int j = 0; j < kH1; ++j) h1[j] = w.b1[j];
        const int16_t* c = codes + static_cast<size_t>(b) * slots;
        for (int s = 0; s < slots; ++s) {
            if (!is_piece(c[s])) continue;
            const float* row = &w.w1t[static_cast<size_t>(c[s]) * kH1];
            for (int j = 0; j < kH1; ++j) h1[j] += row[j];
        }
        if (stm[b] == 0) {
            const float* row = &w.w1t[static_cast<size_t>(kStmInput) * kH1];
            for (int j = 0; j < kH1; ++j) h1[j] += row[j];
        }
        for (int j = 0; j < kH1; ++j) h1[j] = std::max(h1[j], 0.0f);

        // Layer 2, dense.
        for (int j = 0; j < kH2; ++j) h2[j] = w.b2[j];
        for (int k = 0; k < kH1; ++k) {
            const float* row = &w.w2t[static_cast<size_t>(k) * kH2];
            for (int j = 0; j < kH2; ++j) h2[j] += h1[k] * row[j];
        }
        for (int j = 0; j < kH2; ++j) h2[j] = std::max(h2[j], 0.0f);

        output_layer(w.w3, w.b3, h2, probs + 3 * b, logits ? logits + 3 * b : nullptr);
    }
}

void forward_cpu_int8(const Int8Weights& w, const int16_t* codes, int slots, const uint8_t* stm,
                      int n, float* probs, float* logits) {
    check_slots(slots);
    int32_t acc1[kH1];
    int8_t h1q[kH1];
    float h2[kH2];
    for (int b = 0; b < n; ++b) {
        // Layer 1: integer sums of the 8-bit weight rows, then one scale.
        for (int j = 0; j < kH1; ++j) acc1[j] = 0;
        const int16_t* c = codes + static_cast<size_t>(b) * slots;
        for (int s = 0; s < slots; ++s) {
            if (!is_piece(c[s])) continue;
            const int8_t* row = &w.q1t[static_cast<size_t>(c[s]) * kH1];
            for (int j = 0; j < kH1; ++j) acc1[j] += row[j];
        }
        if (stm[b] == 0) {
            const int8_t* row = &w.q1t[static_cast<size_t>(kStmInput) * kH1];
            for (int j = 0; j < kH1; ++j) acc1[j] += row[j];
        }
        for (int j = 0; j < kH1; ++j) {
            const float h = std::max(w.s1[j] * static_cast<float>(acc1[j]) + w.b1[j], 0.0f);
            // Round to the nearest integer (ties to even, like np.rint and the GPU).
            h1q[j] = static_cast<int8_t>(std::min(127.0f, std::nearbyint(h / w.a1)));
        }

        // Layer 2: exact integer dot products, then one scale per neuron.
        for (int j = 0; j < kH2; ++j) {
            int32_t acc = 0;
            const int8_t* row = &w.q2[static_cast<size_t>(j) * kH1];
            for (int k = 0; k < kH1; ++k) acc += static_cast<int32_t>(row[k]) * h1q[k];
            h2[j] = std::max(w.m2[j] * static_cast<float>(acc) + w.b2[j], 0.0f);
        }

        output_layer(w.w3, w.b3, h2, probs + 3 * b, logits ? logits + 3 * b : nullptr);
    }
}

}  // namespace wm::infer
