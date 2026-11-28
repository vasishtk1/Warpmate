// The Warpmate network for inference: its weights, loaded from the files
// ml/train.py and ml/quantize.py write, plus a plain CPU implementation that
// every GPU version is checked against.
//
//     769 inputs -> 256 (ReLU) -> 64 (ReLU) -> 3 logits -> softmax
//
// Boards come in the compact form the dataset uses instead of 769 numbers:
//   * codes: `slots` 16-bit piece codes per board, code = (colour*6 + type)*64
//     + square, which is the index of that piece's input. Unused slots hold
//     kPad (any value outside 0..767 is skipped). A board never lists the same
//     code twice (a square holds one piece).
//   * stm:   side to move per board, 0 = White. Input 768 is 1 when White moves.
// A board switches on only 3 to 6 of the 769 inputs, which is what the
// "sparse" first layer exploits: instead of multiplying 769 numbers it adds
// up the weight columns of the inputs that are on.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace wm::infer {

constexpr int kInputs = 769;    // 12 piece kinds x 64 squares + side to move
constexpr int kStmInput = 768;  // the side-to-move input
constexpr int kH1 = 256;        // first hidden layer
constexpr int kH2 = 64;         // second hidden layer
constexpr int kOutputs = 3;     // loss, draw, win (for the side to move)
constexpr int16_t kPad = 769;   // "no piece" (as in ml/model.py)
constexpr int kMaxSlots = 32;   // a full chess board

inline bool is_piece(int code) { return code >= 0 && code < kStmInput; }

// 32-bit weights, stored "transposed" (input-major): w1t[k * kH1 + j] is the
// weight from input k to neuron j. This puts the 256 weights of one input
// next to each other, so the sparse layer reads one contiguous row per piece,
// and so neighbouring GPU threads (neighbouring j) read neighbouring
// addresses (a "coalesced" read).
struct FloatWeights {
    std::vector<float> w1t;  // kInputs x kH1
    std::vector<float> b1;   // kH1
    std::vector<float> w2t;  // kH1 x kH2
    std::vector<float> b2;   // kH2
    std::vector<float> w3;   // kOutputs x kH2 (not transposed: one row per output)
    std::vector<float> b3;   // kOutputs
};

// 8-bit version (see ml/quantize.py for how it is made):
//   layer 1: h1[j] = relu(s1[j] * sum of q1t[code][j] over the board's inputs + b1[j])
//            then h1 is stored as an 8-bit number: h1q = min(127, rint(h1 / a1))
//   layer 2: h2[j] = relu(m2[j] * dot(q2[j], h1q) + b2[j]), where m2 = s2 * a1
//   layer 3: 32-bit, as in FloatWeights (it is tiny: 64 x 3)
// q1t and q2 hold weights divided by a per-neuron scale (s1, s2) and rounded
// to -127..127. Sums are done in exact integer arithmetic; the scales are
// applied once at the end.
struct Int8Weights {
    std::vector<int8_t> q1t;  // kInputs x kH1 (transposed, like w1t)
    std::vector<float> s1;    // kH1
    std::vector<float> b1;    // kH1
    float a1 = 1.0f;          // scale of the 8-bit h1 values
    std::vector<int8_t> q2;   // kH2 x kH1 (one row per neuron)
    std::vector<float> m2;    // kH2: s2 * a1
    std::vector<float> b2;    // kH2
    std::vector<float> w3;    // kOutputs x kH2
    std::vector<float> b3;    // kOutputs
};

// warpmate_net.npz from ml/train.py: w0 (256, 769), b0, w1 (64, 256), b1, w2 (3, 64), b2.
FloatWeights load_float_weights(const std::string& npz_path);

// warpmate_net_int8.npz from ml/quantize.py.
Int8Weights load_int8_weights(const std::string& npz_path);

// CPU reference implementations. Write n x 3 probabilities to `probs` and,
// if `logits` isn't null, the n x 3 raw scores before softmax.
void forward_cpu(const FloatWeights& w, const int16_t* codes, int slots, const uint8_t* stm,
                 int n, float* probs, float* logits);
void forward_cpu_int8(const Int8Weights& w, const int16_t* codes, int slots,
                      const uint8_t* stm, int n, float* probs, float* logits);

// Softmax of 3 logits, computed the numerically safe way (subtract the max
// first, so exp never overflows).
void softmax3(const float* logits, float* probs);

}  // namespace wm::infer
