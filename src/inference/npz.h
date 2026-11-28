// Reads NumPy's .npy and .npz files, which is how ml/train.py saves the
// network's weights (np.savez: a zip archive holding one .npy file per array).
//
// Only what Warpmate needs: little-endian float32 / int8 / int16 / int32 /
// uint8 arrays in C order, inside an uncompressed archive (np.savez, not
// np.savez_compressed). Anything else is rejected with a clear error rather
// than misread.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace wm::infer {

struct NpyArray {
    std::string dtype;            // NumPy type code, e.g. "<f4" or "|i1"
    std::vector<int64_t> shape;   // () for a single number
    std::vector<char> data;       // raw bytes, C order

    int64_t size() const;         // number of elements
    // The data as T, after checking that the type and element count match.
    template <class T>
    std::vector<T> as(const char* expected_dtype) const;
};

// Parses one .npy file held in memory. `what` names it in error messages.
NpyArray parse_npy(const char* bytes, size_t size, const std::string& what);

// Every array in a .npz file, keyed by name ("w0", "b0", ...).
std::map<std::string, NpyArray> load_npz(const std::string& path);

}  // namespace wm::infer
