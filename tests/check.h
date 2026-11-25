// A minimal test helper: CHECK(cond) records a failure with file/line instead
// of stopping, and finish() prints a summary and returns the exit code.
#pragma once

#include <cstdio>

namespace wm::test {

inline int& failures() {
    static int count = 0;
    return count;
}

inline int finish(const char* suite) {
    if (failures() == 0) {
        std::printf("[%s] all checks passed\n", suite);
        return 0;
    }
    std::printf("[%s] %d check(s) FAILED\n", suite, failures());
    return 1;
}

}  // namespace wm::test

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
            ++wm::test::failures();                                          \
        }                                                                    \
    } while (0)
