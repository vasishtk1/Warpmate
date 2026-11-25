#include "io/tablebase.h"

#include <cstdint>
#include <cstring>
#include <fstream>

namespace wm {

namespace {

constexpr char kMagic[4] = {'W', 'M', 'T', 'B'};
constexpr uint32_t kVersion = 1;
constexpr size_t kNameBytes = 16;

void set_error(std::string* error, const std::string& msg) {
    if (error) *error = msg;
}

}  // namespace

bool save_tablebase(const std::string& path, const Tablebase& tb, std::string* error) {
    if (tb.name.size() >= kNameBytes) {
        set_error(error, "endgame name too long");
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        set_error(error, "can't open " + path + " for writing");
        return false;
    }
    char name[kNameBytes] = {};
    std::memcpy(name, tb.name.data(), tb.name.size());
    const uint64_t count = tb.values.size();

    out.write(kMagic, sizeof kMagic);
    out.write(reinterpret_cast<const char*>(&kVersion), sizeof kVersion);
    out.write(name, kNameBytes);
    out.write(reinterpret_cast<const char*>(&count), sizeof count);
    out.write(reinterpret_cast<const char*>(tb.values.data()),
              static_cast<std::streamsize>(tb.values.size()));
    if (!out) {
        set_error(error, "write failed for " + path);
        return false;
    }
    return true;
}

bool load_tablebase(const std::string& path, Tablebase* tb, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        set_error(error, "can't open " + path);
        return false;
    }
    char magic[4];
    uint32_t version = 0;
    char name[kNameBytes];
    uint64_t count = 0;
    in.read(magic, sizeof magic);
    in.read(reinterpret_cast<char*>(&version), sizeof version);
    in.read(name, kNameBytes);
    in.read(reinterpret_cast<char*>(&count), sizeof count);
    if (!in || std::memcmp(magic, kMagic, sizeof magic) != 0) {
        set_error(error, path + " is not a Warpmate tablebase");
        return false;
    }
    if (version != kVersion) {
        set_error(error, "unsupported tablebase version " + std::to_string(version));
        return false;
    }
    name[kNameBytes - 1] = '\0';
    tb->name = name;
    tb->values.resize(count);
    in.read(reinterpret_cast<char*>(tb->values.data()), static_cast<std::streamsize>(count));
    if (!in) {
        set_error(error, path + " is truncated");
        return false;
    }
    return true;
}

}  // namespace wm
