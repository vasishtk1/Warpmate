// Small printing helpers for the command-line tools.
#pragma once

#include <cstdint>
#include <string>

namespace wm {

// 524288 -> "524,288"
inline std::string with_commas(uint64_t n) {
    std::string digits = std::to_string(n);
    std::string out;
    int count = 0;
    for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
        if (count && count % 3 == 0) out.insert(out.begin(), ',');
        out.insert(out.begin(), *it);
        ++count;
    }
    return out;
}

}  // namespace wm
