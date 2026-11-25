// Saving and loading solved tables.
//
// File layout (little-endian):
//   bytes 0-3    magic "WMTB"
//   bytes 4-7    format version (uint32, currently 1)
//   bytes 8-23   endgame name, e.g. "KRvK", zero-padded (16 bytes)
//   bytes 24-31  number of positions (uint64)
//   then         one byte per position (see core/values.h)
#pragma once

#include <string>
#include <vector>

#include "core/values.h"

namespace wm {

struct Tablebase {
    std::string name;
    std::vector<Value> values;
};

// Both return false and fill `error` on failure.
bool save_tablebase(const std::string& path, const Tablebase& tb, std::string* error);
bool load_tablebase(const std::string& path, Tablebase* tb, std::string* error);

}  // namespace wm
